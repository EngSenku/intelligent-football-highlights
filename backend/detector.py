"""
detector.py — v6 (complete rewrite)
------------------------------------
The fundamental problem with v4/v5 was using a single fused score and
point-peak detection. Football events are:

  GOAL      → sustained audio peak (30-60s), crowd ratio spike, scene changes
  CHANCE    → motion spike + audio bump + density spike, short duration
  FOUL/CARD → sudden audio spike then drop, scene change (camera cut to ref)
  CORNER    → density high (players in box), moderate audio, set-play pattern
  PENALTY   → long pause, then goal-like sequence

Strategy:
1. Use SEPARATE specialized signals per event type — not one fused score.
2. For goals: look for SUSTAINED audio + high crowd ratio (not just one loud frame).
3. For fouls: look for audio spike that DECAYS quickly (not sustained).
4. For corners: density + scene change without strong audio.
5. Refinement: move the timestamp BACKWARD to the action start (was moving forward).
6. NO cap on clips — every qualifying event gets a clip.
7. Strong rejection of midfield passes (high motion, low audio, low crowd ratio).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


EVENT_PRIORITY = {
    "goal_candidate": 6,
    "penalty_candidate": 5,
    "big_chance_candidate": 4,
    "chance_candidate": 3,
    "foul_or_card_candidate": 3,
    "corner_or_set_piece_candidate": 2,
    "highlight_candidate": 1,
}

# Pre/post seconds for each event type.
# Pre = how far back from detected peak to start the clip (capture the action).
# Post = how far forward from detected peak to end the clip (capture the reaction).
# Pre/post seconds for each event type.
# IMPORTANT — anchor semantics (v7):
#   timestamp = the crowd/audio REACTION peak, NOT the action start.
#   _find_action_onset() computes how far back the action began, stored as
#   action_onset_offset_seconds (always <= 0). The clipper adds abs(offset)
#   on top of "pre" to ensure the action is always captured.
#
#   So "pre" here is the buffer BEFORE the reaction peak, not before the action.
#   For goals: reaction peak ~3-8s after ball crosses line.
#   "pre": 20 + abs(action_onset_offset) covers the shot + attacking move.
#   "post": seconds of celebration to keep after the reaction peak.
CLIP_RULES = {
    # Goal: anchor = crowd reaction peak (~3-8s after goal).
    # pre covers the shot itself; onset_offset extends back to catch the build-up.
    # post covers celebration — kept shorter because onset_offset already added pre.
    "goal_candidate": {"pre": 20, "post": 35, "min": 55, "max": 95},
    # Penalty: anchor near ref decision / VAR. Post covers full penalty sequence.
    "penalty_candidate": {"pre": 15, "post": 90, "min": 90, "max": 125},
    # Big chance: anchor at peak motion/audio. onset_offset adds build-up.
    "big_chance_candidate": {"pre": 15, "post": 18, "min": 35, "max": 65},
    # Chance: shorter, onset_offset handles build-up.
    "chance_candidate": {"pre": 12, "post": 15, "min": 28, "max": 55},
    # Foul/card: anchor at spike. Small pre, medium post for ref + card show.
    "foul_or_card_candidate": {"pre": 8, "post": 22, "min": 22, "max": 42},
    # Corner: anchor at delivery. Pre covers set-up, post covers outcome.
    "corner_or_set_piece_candidate": {"pre": 15, "post": 25, "min": 32, "max": 55},
    # Generic
    "highlight_candidate": {"pre": 12, "post": 16, "min": 22, "max": 40},
}

# Quality gates — minimum ranking_score to be included in output
MIN_SCORE = {
    "goal_candidate": 0.18,  # FIX: was 0.20 — keep almost all goals
    "penalty_candidate": 0.18,
    "big_chance_candidate": 0.20,  # FIX: was 0.24 — misclassified goals often land here
    "chance_candidate": 0.22,  # FIX: was 0.26
    "foul_or_card_candidate": 0.22,  # FIX: was 0.24
    "corner_or_set_piece_candidate": 0.20,  # FIX: was 0.22
    "highlight_candidate": 0.26,  # FIX: was 0.35 — too high, cut misclassified goals
}


def _norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _smooth(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if n <= 1 or arr.size == 0:
        return arr
    n = min(n, len(arr))
    return np.convolve(arr, np.ones(n) / n, mode="same").astype(np.float32)


def _step(times: np.ndarray, fallback: float = 0.5) -> float:
    if len(times) > 1:
        v = float(np.median(np.diff(times)))
        if v > 0:
            return v
    return fallback


def _fit(arr, length: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == length:
        return arr
    if len(arr) > length:
        return arr[:length]
    if len(arr) == 0:
        return np.zeros(length, dtype=np.float32)
    return np.concatenate(
        [arr, np.full(length - len(arr), float(arr[-1]), dtype=np.float32)]
    )


def _interp(src_t, src_v, dst_t) -> np.ndarray:
    if not src_t or not src_v or not dst_t:
        return np.zeros(len(dst_t), dtype=np.float32)
    return np.interp(
        np.asarray(dst_t, dtype=np.float32),
        np.asarray(src_t, dtype=np.float32),
        np.asarray(src_v, dtype=np.float32),
    ).astype(np.float32)


def _percentile_threshold(arr: np.ndarray, pct: float, std_factor: float) -> float:
    if arr.size == 0:
        return 1.0
    return max(
        float(np.percentile(arr, pct)), float(arr.mean() + std_factor * arr.std())
    )


def _peaks(
    signal: np.ndarray, times: np.ndarray, pct: float, std_f: float, dist_s: float
) -> list[int]:
    if signal.size < 3:
        return []
    dist = max(1, int(round(dist_s / _step(times))))
    thr = _percentile_threshold(signal, pct, std_f)
    idxs, _ = find_peaks(signal, height=thr, distance=dist)
    return idxs.tolist()


def _refine_backward(
    idx: int, times: np.ndarray, signal: np.ndarray, back_s: float, ahead_s: float
) -> int:
    """
    Find the local action-signal peak in the window [idx-back_s, idx+ahead_s].
    Returns an index near the action start (for reading signal values).
    NOTE: do NOT use this as the clip anchor — use the original peak idx for that.
    """
    step = _step(times)
    back = max(0, int(round(back_s / step)))
    ahead = max(1, int(round(ahead_s / step)))
    start = max(0, idx - back)
    end = min(len(times), idx + ahead + 1)
    if end <= start:
        return idx
    return int(start + int(np.argmax(signal[start:end])))


def _find_action_onset(
    peak_idx: int,
    times: np.ndarray,
    motion_s: np.ndarray,
    audio_s: np.ndarray,
    scene_s: np.ndarray,
    back_s: float,
) -> float:
    """
    Walk BACKWARD from the peak to find where the action actually began.
    Returns the offset in SECONDS from peak_idx (always negative or zero).

    Strategy:
    - Scan backward up to back_s seconds from the peak.
    - Find the last frame where BOTH motion AND audio drop below their
      local baseline — that is the calm before the storm (the moment just
      before the attacking move started).
    - If no such frame is found, return -back_s (go back the full window).

    This offset is stored per-event so the clipper can apply it as an
    adjustment to the pre-roll, rather than misusing it as the timestamp.
    """
    step = _step(times)
    back = max(1, int(round(back_s / step)))
    start = max(0, peak_idx - back)

    if start >= peak_idx:
        return 0.0

    window_motion = motion_s[start:peak_idx]
    window_audio = audio_s[start:peak_idx]

    if window_motion.size == 0:
        return 0.0

    # Local baseline: bottom quartile of motion in this window
    motion_base = float(np.percentile(window_motion, 25))
    audio_base = float(np.percentile(window_audio, 25))

    # Walk backward from peak: find last index where both signals are calm
    calm_idx = None
    for i in range(len(window_motion) - 1, -1, -1):
        if (
            window_motion[i] <= motion_base * 1.4
            and window_audio[i] <= audio_base * 1.6
        ):
            calm_idx = i
            break

    if calm_idx is None:
        # No calm found — go back the full window
        return float(times[start] - times[peak_idx])

    abs_idx = start + calm_idx
    return float(times[abs_idx] - times[peak_idx])  # negative seconds


def fuse_scores(
    video_timestamps,
    audio_times,
    audio_scores,
    motion_scores,
    scene_change_scores,
    optical_flow_scores,
    foreground_density_scores,
    weights=None,
):
    """Keep for backward compat with processor.py."""
    if weights is None:
        weights = {
            "audio": 0.55,
            "motion": 0.14,
            "flow": 0.08,
            "density": 0.11,
            "scene": 0.12,
        }
    timestamps = np.asarray(video_timestamps, dtype=np.float32)
    n = len(timestamps)
    audio = _norm(_interp(audio_times, audio_scores, video_timestamps))
    motion = _norm(_fit(motion_scores, n))
    scene = _norm(_fit(scene_change_scores, n))
    flow = _norm(_fit(optical_flow_scores, n))
    density = _norm(_fit(foreground_density_scores, n))
    fused = (
        weights.get("audio", 0.55) * audio
        + weights.get("motion", 0.14) * motion
        + weights.get("flow", 0.08) * flow
        + weights.get("density", 0.11) * density
        + weights.get("scene", 0.12) * scene
    )
    return timestamps, _norm(fused)


def detect_highlight_peaks(
    timestamps: np.ndarray,
    fused_scores: np.ndarray,
    audio_scores_at_timestamps: np.ndarray,
    motion_scores: np.ndarray,
    optical_flow_scores: np.ndarray,
    foreground_density_scores: np.ndarray,
    scene_change_scores: np.ndarray,
    # Extra audio signals passed via kwargs in processor.py
    sustained_scores_at_timestamps: np.ndarray | None = None,
    spike_scores_at_timestamps: np.ndarray | None = None,
    crowd_ratio_scores_at_timestamps: np.ndarray | None = None,
    min_distance_seconds: int = 18,
    percentile: float = 78.0,
    std_factor: float = 0.85,
    max_events: int = 200,
    refinement_lookback_seconds: int = 14,
    refinement_lookahead_seconds: int = 10,
) -> list[dict]:

    times = np.asarray(timestamps, dtype=np.float32)
    n = len(times)
    if n < 3:
        return []

    dt = _step(times)
    sp3 = max(1, int(round(3.0 / dt)))  # 3-second smoothing window
    sp8 = max(1, int(round(8.0 / dt)))  # 8-second smoothing window

    # ── Prepare all signals ───────────────────────────────────────────────────
    audio = _norm(_fit(audio_scores_at_timestamps, n))
    motion = _norm(_fit(motion_scores, n))
    flow = _norm(_fit(optical_flow_scores, n))
    density = _norm(_fit(foreground_density_scores, n))
    scene = _norm(_fit(scene_change_scores, n))

    # Sustained crowd signal — most reliable for goals
    if (
        sustained_scores_at_timestamps is not None
        and len(sustained_scores_at_timestamps) > 0
    ):
        sustained = _norm(_fit(sustained_scores_at_timestamps, n))
    else:
        # Approximate from audio if not provided
        sustained = _norm(_smooth(audio, sp8))

    # Crowd ratio — separates crowd cheer from commentary
    if (
        crowd_ratio_scores_at_timestamps is not None
        and len(crowd_ratio_scores_at_timestamps) > 0
    ):
        crowd_ratio = _norm(_fit(crowd_ratio_scores_at_timestamps, n))
    else:
        crowd_ratio = audio  # fallback

    # Spike signal — sudden transient (whistle, tackle)
    if spike_scores_at_timestamps is not None and len(spike_scores_at_timestamps) > 0:
        spike = _norm(_fit(spike_scores_at_timestamps, n))
    else:
        # Approximate: difference between raw and smoothed
        spike = _norm(np.maximum(audio - _smooth(audio, sp8), 0))

    # Smooth versions for stability
    audio_s = _norm(0.6 * _smooth(audio, sp3) + 0.4 * audio)
    motion_s = _norm(0.6 * _smooth(motion, sp3) + 0.4 * motion)
    scene_s = _norm(0.5 * _smooth(scene, sp3) + 0.5 * scene)
    density_s = _norm(0.5 * _smooth(density, sp3) + 0.5 * density)
    sustained_s = _norm(sustained)
    crowd_s = _norm(0.7 * _smooth(crowd_ratio, sp3) + 0.3 * crowd_ratio)
    spike_s = _norm(0.5 * _smooth(spike, sp3) + 0.5 * spike)

    # Percentile references for classification
    def pct(arr):
        pts = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95]
        return {f"p{p}": float(np.percentile(arr, p)) for p in pts}

    aq = pct(audio_s)
    mq = pct(motion_s)
    dq = pct(density_s)
    sq = pct(scene_s)
    suq = pct(sustained_s)
    cq = pct(crowd_s)

    # ── Specialized detection signals ─────────────────────────────────────────

    # GOAL signal: sustained cheering + crowd ratio + audio
    # A real goal has ALL THREE elevated for an extended period.
    goal_signal = _norm(0.50 * sustained_s + 0.25 * crowd_s + 0.25 * audio_s)

    # BIG CHANCE signal: motion + audio burst + density (players near goal)
    # Has elevated audio and motion but NOT the sustained 30s crowd reaction of a goal.
    chance_signal = _norm(
        0.35 * audio_s
        + 0.28 * motion_s
        + 0.20 * density_s
        + 0.12 * scene_s
        + 0.05 * crowd_s
    )

    # FOUL/CARD signal: audio spike (whistle) + scene change + low sustained
    # Key: spike but NOT sustained (crowd doesn't cheer for 30s after a foul).
    foul_signal = _norm(
        0.40 * spike_s
        + 0.30 * scene_s
        + 0.20 * audio_s
        + 0.10 * (1.0 - np.minimum(sustained_s, 1.0))  # negative weight on sustained
    )

    # CORNER/SET-PIECE signal: density (players clustered) + scene + moderate audio
    corner_signal = _norm(
        0.35 * density_s + 0.28 * scene_s + 0.20 * audio_s + 0.17 * motion_s
    )

    # Backward-biased refinement signal — finds the start of the action.
    # Uses motion (active play) more than audio (reaction) so it moves to
    # the moment just before the ball enters the box / foul occurs.
    action_signal = _norm(0.45 * motion_s + 0.30 * audio_s + 0.25 * scene_s)

    # ── Detect peaks per signal ───────────────────────────────────────────────
    # Minimum distance between events: 18s (prevent double-counting same event).
    dist_goal = max(25, min_distance_seconds)  # goals need more separation
    dist_other = max(16, min_distance_seconds)

    raw_candidates: list[dict] = []

    detectors = [
        # (signal, name, percentile, std_factor, distance_s)
        (goal_signal, "goal", 78.0, 0.80, dist_goal),
        (chance_signal, "chance", 75.0, 0.75, dist_other),
        (foul_signal, "foul", 76.0, 0.78, dist_other),
        (corner_signal, "corner", 76.0, 0.78, dist_other),
        # Extra audio-only sweep to catch goals the goal_signal might have missed
        (audio_s, "audio", 82.0, 0.90, dist_goal),
        (sustained_s, "sustained", 80.0, 0.85, dist_goal),
    ]

    for signal, src_name, pct_val, std_val, dist in detectors:
        for idx in _peaks(signal, times, pct_val, std_val, dist):
            # ref_idx: used ONLY for reading signal values for classification.
            # The peak idx is the crowd/audio reaction peak — this is the correct
            # ANCHOR for clip timing (pre/post measured from this point).
            # We do NOT move the timestamp to ref_idx — that was the core timing bug.
            if src_name in {"goal", "audio", "sustained"}:
                ref_idx = _refine_backward(
                    idx,
                    times,
                    action_signal,
                    back_s=refinement_lookback_seconds,
                    ahead_s=refinement_lookahead_seconds,
                )
            else:
                ref_idx = _refine_backward(
                    idx,
                    times,
                    action_signal,
                    back_s=refinement_lookback_seconds * 0.7,
                    ahead_s=min(8, refinement_lookahead_seconds),
                )

            # Compute how far back the action started (used to extend pre-roll
            # in the clipper — NOT to shift the anchor timestamp).
            if src_name in {"goal", "audio", "sustained"}:
                action_onset_offset = _find_action_onset(
                    peak_idx=idx,
                    times=times,
                    motion_s=motion_s,
                    audio_s=audio_s,
                    scene_s=scene_s,
                    back_s=refinement_lookback_seconds,
                )
            else:
                action_onset_offset = _find_action_onset(
                    peak_idx=idx,
                    times=times,
                    motion_s=motion_s,
                    audio_s=audio_s,
                    scene_s=scene_s,
                    back_s=refinement_lookback_seconds * 0.6,
                )

            # Read signal values at ref_idx (near action start) for classification.
            # For sustained/crowd (reaction signals) prefer the peak idx — they
            # are higher there and give better classification accuracy.
            a = float(audio_s[idx])  # peak audio level
            sus = float(sustained_s[idx])  # peak sustained level
            crd = float(crowd_s[idx])  # peak crowd ratio
            spk = float(spike_s[ref_idx])  # spike at action point
            m = float(motion_s[ref_idx])  # motion at action point
            d = float(density_s[ref_idx])  # density at action point
            sc = float(scene_s[ref_idx])  # scene change at action point

            # ── Classify event type ───────────────────────────────────────────
            event_type = _classify_event(
                src_name, a, sus, crd, spk, m, d, sc, aq, mq, dq, sq, suq, cq
            )

            # Hard reject: boring midfield pass
            # High motion + low audio + low crowd ratio + low sustained = passing
            if (
                event_type == "highlight_candidate"
                and m > mq["p65"]
                and a < aq["p50"]
                and crd < cq["p50"]
                and sus < suq["p55"]
            ):
                continue

            ranking = _rank(
                event_type, src_name, a, sus, crd, spk, m, d, sc, aq, suq, cq
            )

            rule = dict(CLIP_RULES.get(event_type, CLIP_RULES["highlight_candidate"]))

            raw_candidates.append(
                {
                    # ANCHOR: the crowd/audio reaction peak — pre/post measured from here.
                    # The clipper uses action_onset_offset to extend the pre-roll to
                    # cover the actual action start without shifting the anchor.
                    "timestamp": float(times[idx]),
                    "original_peak_timestamp": float(times[idx]),
                    "action_onset_offset_seconds": round(action_onset_offset, 3),
                    "event_type": event_type,
                    "ranking_score": ranking,
                    "confidence": round(ranking, 4),
                    "audio_score": round(a, 4),
                    "sustained_score": round(sus, 4),
                    "crowd_ratio_score": round(crd, 4),
                    "spike_score": round(spk, 4),
                    "motion_score": round(m, 4),
                    "foreground_density_score": round(d, 4),
                    "scene_change_score": round(sc, 4),
                    "final_score": round(
                        float(_fit(fused_scores, n)[ref_idx])
                        if len(fused_scores) > 0
                        else a,
                        4,
                    ),
                    "detection_method": "multi_event_v6",
                    "detection_sources": [src_name],
                    "recommended_pre_seconds": rule["pre"],
                    "recommended_post_seconds": rule["post"],
                    "recommended_min_clip_seconds": rule["min"],
                    "recommended_max_clip_seconds": rule["max"],
                    "event_label_note": "probable heuristic label, not guaranteed",
                }
            )

    # ── Merge, rank, select ───────────────────────────────────────────────────
    # FIX: was gap_s=10 — two detectors firing on the same goal 11s apart
    # produced two candidates, one got dropped. 18s collapses same-goal duplicates.
    merged = _merge(raw_candidates, gap_s=18)
    selected = _select(merged, max_events)
    return sorted(selected, key=lambda x: x["timestamp"])


def _classify_event(src, a, sus, crd, spk, m, d, sc, aq, mq, dq, sq, suq, cq) -> str:
    """
    Classify a detected event using the multi-signal profile.

    Key distinguishing features:
    - GOAL:    high sustained + high crowd_ratio + high audio — ALL elevated
    - PENALTY: very high sustained + density spike before, audio after
    - CHANCE:  motion + audio bump, but sustained NOT elevated for 30s
    - FOUL:    audio spike + scene change, sustained stays low
    - CORNER:  density high, scene change, audio moderate
    """
    # GOAL: sustained crowd cheering (>30s elevated) + crowd ratio
    # FIX: relaxed crowd_ratio from p70->p60 — commentary-heavy broadcasts suppress
    # crowd_ratio even during real goals. Also added high-sustained-alone path.
    if sus >= suq["p75"] and crd >= cq["p60"] and a >= aq["p60"]:
        if d >= dq["p65"] and sus >= suq["p85"]:
            return "penalty_candidate"
        return "goal_candidate"

    # FIX: second goal path — very high sustained alone is reliable even without
    # crowd_ratio (covers broadcasts where commentary drowns crowd freq bands)
    if sus >= suq["p80"] and a >= aq["p65"]:
        return "goal_candidate"

    # BIG CHANCE: high audio + motion + density, sustained moderate
    # FIX: relaxed thresholds slightly to catch events misclassified as generic
    if a >= aq["p65"] and m >= mq["p60"] and d >= dq["p55"]:
        if sus >= suq["p60"]:
            return "big_chance_candidate"
        return "chance_candidate"

    # GOAL from audio/sustained sources (less certain but include it)
    if src in {"goal", "audio", "sustained"}:
        # FIX: relaxed sus p65->p60 and crowd_ratio p60->p55
        if sus >= suq["p60"] and (crd >= cq["p55"] or a >= aq["p65"]):
            return "goal_candidate"
        if a >= aq["p70"] and sc >= sq["p60"]:
            return "big_chance_candidate"

    # FOUL/CARD: spike (whistle/gasp) + scene change, sustained stays LOW
    if src == "foul" or (spk >= 0.45 and sc >= sq["p65"] and sus < suq["p60"]):
        if a >= aq["p55"] or sc >= sq["p70"]:
            return "foul_or_card_candidate"

    # CORNER/SET-PIECE: density high (players clustered in box) + scene change
    if src == "corner" or (d >= dq["p70"] and sc >= sq["p60"]):
        if a >= aq["p45"] or m >= mq["p55"]:
            return "corner_or_set_piece_candidate"

    # Catch-all for chance-like events
    if a >= aq["p65"] and m >= mq["p60"]:
        return "chance_candidate"

    if sc >= sq["p70"] and a >= aq["p55"]:
        return "foul_or_card_candidate"

    return "highlight_candidate"


def _rank(event_type, src, a, sus, crd, spk, m, d, sc, aq, suq, cq) -> float:
    if event_type in {"goal_candidate", "penalty_candidate"}:
        score = 0.40 * sus + 0.25 * crd + 0.25 * a + 0.05 * d + 0.05 * sc

    elif event_type == "big_chance_candidate":
        score = 0.35 * a + 0.25 * m + 0.20 * d + 0.10 * sus + 0.10 * sc

    elif event_type == "chance_candidate":
        score = 0.30 * a + 0.25 * m + 0.20 * d + 0.15 * sc + 0.10 * sus

    elif event_type == "foul_or_card_candidate":
        score = 0.35 * spk + 0.30 * sc + 0.20 * a + 0.10 * d + 0.05 * (1 - min(sus, 1))

    elif event_type == "corner_or_set_piece_candidate":
        score = 0.30 * d + 0.25 * sc + 0.25 * a + 0.20 * m

    else:
        score = 0.40 * a + 0.20 * sus + 0.20 * sc + 0.10 * d + 0.10 * m

    # HARD PENALTY for plain midfield passing sequences:
    # high motion, low crowd engagement, low sustained = nothing happened
    if m > 0.60 and sus < suq["p50"] and crd < cq["p45"] and a < aq["p50"]:
        score *= 0.15  # 85% penalty — almost certainly boring

    # Small boost for events from specialized detectors
    if src in {"foul", "corner"}:
        score *= 1.10

    return float(np.clip(score, 0.0, 1.0))


def _merge(candidates: list[dict], gap_s: int) -> list[dict]:
    """Merge candidates within gap_s seconds of each other — keep the best one."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda x: x["timestamp"])
    groups: list[list[dict]] = [[ordered[0]]]
    for c in ordered[1:]:
        if c["timestamp"] - groups[-1][-1]["timestamp"] <= gap_s:
            groups[-1].append(c)
        else:
            groups.append([c])
    merged = []
    for group in groups:
        best = max(
            group,
            key=lambda x: (
                x["ranking_score"],
                EVENT_PRIORITY.get(x["event_type"], 0),
                x.get("audio_score", 0),
            ),
        )
        best["merged_candidates"] = len(group)
        best["merged_event_types"] = sorted({g["event_type"] for g in group})
        best["merged_detection_sources"] = sorted(
            {s for g in group for s in g.get("detection_sources", [])}
        )
        merged.append(best)
    return merged


def _select(candidates: list[dict], max_events: int) -> list[dict]:
    """
    Include ALL events above quality threshold — no artificial count cap.
    The quality gate is calibrated so goals almost never get cut.
    """
    selected = []
    for c in candidates:
        if len(selected) >= max_events:
            break
        gate = MIN_SCORE.get(c["event_type"], 0.25)
        if c["ranking_score"] >= gate:
            selected.append(c)
            c.pop("ranking_score", None)
    return selected


def compute_adaptive_threshold(scores, percentile, std_factor):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return 1.0
    return max(
        float(np.percentile(scores, percentile)),
        float(np.mean(scores) + std_factor * np.std(scores)),
    )


def detect_audio_peaks(
    times, scores, min_distance_seconds, percentile, std_factor, max_events
) -> list[dict]:
    """Audio-only fallback when video analysis fails."""
    if not times or not scores:
        return []
    t = np.asarray(times, dtype=np.float32)
    s = _norm(np.asarray(scores, dtype=np.float32))
    if len(t) != len(s) or len(s) < 3:
        return []
    sp = max(1, int(round(3.0 / _step(t))))
    s_s = _norm(0.70 * _smooth(s, sp) + 0.30 * s)
    dist = max(1, int(round(min_distance_seconds / _step(t))))
    thr = compute_adaptive_threshold(s_s, percentile, std_factor)
    peaks, _ = find_peaks(s_s, height=thr, distance=dist)
    qualified = sorted(peaks.tolist(), key=lambda i: float(t[i]))[:max_events]
    return [
        {
            "timestamp": float(t[i]),
            "confidence": round(float(s_s[i]), 4),
            "final_score": round(float(s_s[i]), 4),
            "audio_score": round(float(s_s[i]), 4),
            "event_type": "goal_candidate",
            "event_label_note": "audio-only fallback",
            "detection_method": "audio_peak_fallback_v6",
            "threshold_used": round(float(thr), 4),
            "recommended_pre_seconds": 30,
            "recommended_post_seconds": 40,
            "recommended_min_clip_seconds": 45,
            "recommended_max_clip_seconds": 85,
        }
        for i in qualified
    ]

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
CLIP_RULES = {
    "goal_candidate": {"pre": 20, "post": 35, "min": 55, "max": 95},
    "penalty_candidate": {"pre": 15, "post": 90, "min": 90, "max": 125},
    "big_chance_candidate": {"pre": 15, "post": 18, "min": 35, "max": 65},
    "chance_candidate": {"pre": 12, "post": 15, "min": 28, "max": 55},
    "foul_or_card_candidate": {"pre": 8, "post": 22, "min": 22, "max": 42},
    "corner_or_set_piece_candidate": {"pre": 15, "post": 25, "min": 32, "max": 55},
    "highlight_candidate": {"pre": 12, "post": 16, "min": 22, "max": 40},
}

MIN_SCORE = {
    "goal_candidate": 0.18,
    "penalty_candidate": 0.18,
    "big_chance_candidate": 0.20,
    "chance_candidate": 0.22,
    "foul_or_card_candidate": 0.22,
    "corner_or_set_piece_candidate": 0.20,
    "highlight_candidate": 0.26,
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
    step = _step(times)
    back = max(1, int(round(back_s / step)))
    start = max(0, peak_idx - back)

    if start >= peak_idx:
        return 0.0

    window_motion = motion_s[start:peak_idx]
    window_audio = audio_s[start:peak_idx]

    if window_motion.size == 0:
        return 0.0

    motion_base = float(np.percentile(window_motion, 25))
    audio_base = float(np.percentile(window_audio, 25))

    calm_idx = None
    for i in range(len(window_motion) - 1, -1, -1):
        if (
            window_motion[i] <= motion_base * 1.4
            and window_audio[i] <= audio_base * 1.6
        ):
            calm_idx = i
            break

    if calm_idx is None:
        return float(times[start] - times[peak_idx])

    abs_idx = start + calm_idx
    return float(times[abs_idx] - times[peak_idx])


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
    sp3 = max(1, int(round(3.0 / dt)))
    sp8 = max(1, int(round(8.0 / dt)))

    audio = _norm(_fit(audio_scores_at_timestamps, n))
    motion = _norm(_fit(motion_scores, n))
    flow = _norm(_fit(optical_flow_scores, n))
    density = _norm(_fit(foreground_density_scores, n))
    scene = _norm(_fit(scene_change_scores, n))

    if (
        sustained_scores_at_timestamps is not None
        and len(sustained_scores_at_timestamps) > 0
    ):
        sustained = _norm(_fit(sustained_scores_at_timestamps, n))
    else:
        sustained = _norm(_smooth(audio, sp8))

    if (
        crowd_ratio_scores_at_timestamps is not None
        and len(crowd_ratio_scores_at_timestamps) > 0
    ):
        crowd_ratio = _norm(_fit(crowd_ratio_scores_at_timestamps, n))
    else:
        crowd_ratio = audio

    if spike_scores_at_timestamps is not None and len(spike_scores_at_timestamps) > 0:
        spike = _norm(_fit(spike_scores_at_timestamps, n))
    else:
        spike = _norm(np.maximum(audio - _smooth(audio, sp8), 0))

    audio_s = _norm(0.6 * _smooth(audio, sp3) + 0.4 * audio)
    motion_s = _norm(0.6 * _smooth(motion, sp3) + 0.4 * motion)
    scene_s = _norm(0.5 * _smooth(scene, sp3) + 0.5 * scene)
    density_s = _norm(0.5 * _smooth(density, sp3) + 0.5 * density)
    sustained_s = _norm(sustained)
    crowd_s = _norm(0.7 * _smooth(crowd_ratio, sp3) + 0.3 * crowd_ratio)
    spike_s = _norm(0.5 * _smooth(spike, sp3) + 0.5 * spike)

    def pct(arr):
        pts = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95]
        return {f"p{p}": float(np.percentile(arr, p)) for p in pts}

    aq = pct(audio_s)
    mq = pct(motion_s)
    dq = pct(density_s)
    sq = pct(scene_s)
    suq = pct(sustained_s)
    cq = pct(crowd_s)

    goal_signal = _norm(0.50 * sustained_s + 0.25 * crowd_s + 0.25 * audio_s)

    chance_signal = _norm(
        0.35 * audio_s
        + 0.28 * motion_s
        + 0.20 * density_s
        + 0.12 * scene_s
        + 0.05 * crowd_s
    )

    foul_signal = _norm(
        0.40 * spike_s
        + 0.30 * scene_s
        + 0.20 * audio_s
        + 0.10 * (1.0 - np.minimum(sustained_s, 1.0))
    )

    corner_signal = _norm(
        0.35 * density_s + 0.28 * scene_s + 0.20 * audio_s + 0.17 * motion_s
    )

    action_signal = _norm(0.45 * motion_s + 0.30 * audio_s + 0.25 * scene_s)


    dist_goal = max(25, min_distance_seconds)
    dist_other = max(16, min_distance_seconds)

    raw_candidates: list[dict] = []

    detectors = [
        (goal_signal, "goal", 78.0, 0.80, dist_goal),
        (chance_signal, "chance", 75.0, 0.75, dist_other),
        (foul_signal, "foul", 76.0, 0.78, dist_other),
        (corner_signal, "corner", 76.0, 0.78, dist_other),
        (audio_s, "audio", 82.0, 0.90, dist_goal),
        (sustained_s, "sustained", 80.0, 0.85, dist_goal),
    ]

    for signal, src_name, pct_val, std_val, dist in detectors:
        for idx in _peaks(signal, times, pct_val, std_val, dist):
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

            a = float(audio_s[idx])
            sus = float(sustained_s[idx])
            crd = float(crowd_s[idx])
            spk = float(spike_s[ref_idx])
            m = float(motion_s[ref_idx])
            d = float(density_s[ref_idx])
            sc = float(scene_s[ref_idx])

            event_type = _classify_event(
                src_name, a, sus, crd, spk, m, d, sc, aq, mq, dq, sq, suq, cq
            )

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

    merged = _merge(raw_candidates, gap_s=18)
    selected = _select(merged, max_events)
    return sorted(selected, key=lambda x: x["timestamp"])


def _classify_event(src, a, sus, crd, spk, m, d, sc, aq, mq, dq, sq, suq, cq) -> str:
    if sus >= suq["p75"] and crd >= cq["p60"] and a >= aq["p60"]:
        if d >= dq["p65"] and sus >= suq["p85"]:
            return "penalty_candidate"
        return "goal_candidate"

    if sus >= suq["p80"] and a >= aq["p65"]:
        return "goal_candidate"

    if a >= aq["p65"] and m >= mq["p60"] and d >= dq["p55"]:
        if sus >= suq["p60"]:
            return "big_chance_candidate"
        return "chance_candidate"

    if src in {"goal", "audio", "sustained"}:
        if sus >= suq["p60"] and (crd >= cq["p55"] or a >= aq["p65"]):
            return "goal_candidate"
        if a >= aq["p70"] and sc >= sq["p60"]:
            return "big_chance_candidate"

    if src == "foul" or (spk >= 0.45 and sc >= sq["p65"] and sus < suq["p60"]):
        if a >= aq["p55"] or sc >= sq["p70"]:
            return "foul_or_card_candidate"

    if src == "corner" or (d >= dq["p70"] and sc >= sq["p60"]):
        if a >= aq["p45"] or m >= mq["p55"]:
            return "corner_or_set_piece_candidate"

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

    if m > 0.60 and sus < suq["p50"] and crd < cq["p45"] and a < aq["p50"]:
        score *= 0.15

    if src in {"foul", "corner"}:
        score *= 1.10

    return float(np.clip(score, 0.0, 1.0))


def _merge(candidates: list[dict], gap_s: int) -> list[dict]:
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

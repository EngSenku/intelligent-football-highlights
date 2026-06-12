from pathlib import Path

from .ffmpeg_tools import concatenate_clips, extract_clip_fast
from .utils import seconds_to_hhmmss


def _event_clip_rule(
    event: dict,
    default_pre: int,
    default_post: int,
    default_min: int,
    default_max: int | None,
) -> dict[str, float]:
    max_value = default_max if default_max is not None else 95

    return {
        "pre": float(event.get("recommended_pre_seconds", default_pre)),
        "post": float(event.get("recommended_post_seconds", default_post)),
        "min": float(event.get("recommended_min_clip_seconds", default_min)),
        "max": float(event.get("recommended_max_clip_seconds", max_value)),
    }


def _best_event(events: list[dict]) -> dict:
    return max(
        events,
        key=lambda e: (
            e.get("confidence", 0.0),
            e.get("audio_score", 0.0),
            e.get("scene_change_score", 0.0),
        ),
    )


def _build_single_window(
    event: dict,
    video_duration: float,
    default_pre: int,
    default_post: int,
    default_min: int,
    default_max: int | None,
) -> dict:

    rule = _event_clip_rule(
        event=event,
        default_pre=default_pre,
        default_post=default_post,
        default_min=default_min,
        default_max=default_max,
    )

    timestamp = float(event["timestamp"])

    action_onset_offset = float(event.get("action_onset_offset_seconds", 0.0))
    extra_pre = abs(min(action_onset_offset, 0.0))  # always >= 0

    extra_pre = min(extra_pre, rule["pre"] * 2.0)
    total_pre = rule["pre"] + extra_pre

    start = max(0.0, timestamp - total_pre)
    end = min(float(video_duration), timestamp + rule["post"])
    duration = end - start

    if duration < rule["min"]:
        missing = rule["min"] - duration
        end = min(float(video_duration), end + missing)
        duration = end - start

    return {
        "start": start,
        "end": end,
        "timestamps": [timestamp],
        "source_events": [event],
        "event_rule": rule,
        "_from_merge": False,
        "_extra_pre_used": round(extra_pre, 2),
    }


def _cap_window_around_best_event(
    window: dict,
    video_duration: float,
    default_pre: int,
    default_post: int,
    default_min: int,
    default_max: int | None,
) -> dict:

    best = _best_event(window["source_events"])
    best_rule = _event_clip_rule(
        event=best,
        default_pre=default_pre,
        default_post=default_post,
        default_min=default_min,
        default_max=default_max,
    )

    current_duration = float(window["end"] - window["start"])

    extra_pre = abs(min(float(best.get("action_onset_offset_seconds", 0.0)), 0.0))
    extra_pre = min(extra_pre, best_rule["pre"] * 2.0)
    natural_duration = best_rule["pre"] + extra_pre + best_rule["post"]

    if (
        not window.get("_from_merge", False)
        or current_duration <= natural_duration * 1.3
    ):
        window["event_rule"] = best_rule
        return window

    max_duration = float(best_rule["max"])

    if current_duration <= max_duration:
        window["event_rule"] = best_rule
        return window

    timestamp = float(best["timestamp"])
    post = float(best_rule["post"])

    end = min(float(video_duration), timestamp + post)
    start = max(0.0, end - max_duration)

    if start < 0.0:
        start = 0.0
        end = min(float(video_duration), max_duration)

    return {
        "start": start,
        "end": end,
        "timestamps": [timestamp],
        "source_events": [best],
        "event_rule": best_rule,
        "_from_merge": True,
        "was_capped_from_long_merge": True,
        "original_merged_duration": round(current_duration, 3),
    }


def build_clip_windows(
    events: list[dict],
    video_duration: float,
    pre_seconds: int,
    post_seconds: int,
    merge_gap_seconds: int,
    min_clip_duration_seconds: int,
    max_clip_duration_seconds: int | None = None,
) -> list[dict]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e["timestamp"])

    raw_windows = [
        _build_single_window(
            event=event,
            video_duration=float(video_duration),
            default_pre=pre_seconds,
            default_post=post_seconds,
            default_min=min_clip_duration_seconds,
            default_max=max_clip_duration_seconds,
        )
        for event in sorted_events
    ]

    merged: list[dict] = []
    for window in raw_windows:
        if not merged:
            merged.append(window)
            continue

        previous = merged[-1]

        if window["start"] <= previous["end"] + merge_gap_seconds:
            previous["end"] = max(previous["end"], window["end"])
            previous["timestamps"].extend(window["timestamps"])
            previous["source_events"].extend(window["source_events"])
            previous["_from_merge"] = True
        else:
            merged.append(window)

    valid_windows = []
    for window in merged:
        window = _cap_window_around_best_event(
            window=window,
            video_duration=float(video_duration),
            default_pre=pre_seconds,
            default_post=post_seconds,
            default_min=min_clip_duration_seconds,
            default_max=max_clip_duration_seconds,
        )

        best = _best_event(window["source_events"])
        rule = _event_clip_rule(
            event=best,
            default_pre=pre_seconds,
            default_post=post_seconds,
            default_min=min_clip_duration_seconds,
            default_max=max_clip_duration_seconds,
        )

        clip_dur = float(window["end"] - window["start"])

        if clip_dur >= rule["min"] * 0.75:
            valid_windows.append(window)

    return valid_windows


def extract_clips_for_windows(
    input_video_path: Path,
    windows: list[dict],
    output_dir: Path,
) -> list[dict]:
    extracted_events = []

    for index, window in enumerate(windows, start=1):
        clip_filename = f"clip_{index:03d}.mp4"
        clip_path = output_dir / clip_filename

        start = float(window["start"])
        end = float(window["end"])
        duration = end - start

        extract_clip_fast(
            input_video_path=input_video_path,
            output_clip_path=clip_path,
            start_seconds=start,
            duration_seconds=duration,
        )

        source_events = window["source_events"]
        best_event = _best_event(source_events)
        event_rule = window.get("event_rule", {})

        extracted_events.append(
            {
                "clip_index": index,
                "event_type": best_event.get("event_type", "highlight_candidate"),
                "event_label_note": best_event.get(
                    "event_label_note", "probable heuristic label"
                ),
                "timestamp": best_event["timestamp"],
                "timestamp_hhmmss": seconds_to_hhmmss(best_event["timestamp"]),
                "original_peak_timestamp": best_event.get("original_peak_timestamp"),
                "confidence": best_event["confidence"],
                "clip_start": round(start, 3),
                "clip_end": round(end, 3),
                "clip_duration": round(duration, 3),
                "clip_filename": clip_filename,
                "clip_path": str(clip_path),
                "dynamic_clip_rule": {
                    "pre": event_rule.get("pre"),
                    "post": event_rule.get("post"),
                    "min": event_rule.get("min"),
                    "max": event_rule.get("max"),
                },
                "merged_event_timestamps": [
                    round(float(ts), 3) for ts in window["timestamps"]
                ],
                "merged_candidates": len(source_events),
                "was_capped_from_long_merge": window.get(
                    "was_capped_from_long_merge", False
                ),
                "original_merged_duration": window.get("original_merged_duration"),
                "detection_method": best_event.get("detection_method", "unknown"),
                "detection_sources": best_event.get("detection_sources"),
                "merged_event_types": best_event.get("merged_event_types"),
                "merged_detection_sources": best_event.get("merged_detection_sources"),
                "audio_score": best_event.get("audio_score"),
                "sustained_score": best_event.get("sustained_score"),
                "crowd_ratio_score": best_event.get("crowd_ratio_score"),
                "motion_score": best_event.get("motion_score"),
                "foreground_density_score": best_event.get("foreground_density_score"),
                "scene_change_score": best_event.get("scene_change_score"),
                "final_score": best_event.get("final_score"),
            }
        )

    return extracted_events


def create_final_highlights_video(
    clip_paths: list[Path],
    output_path: Path,
    concat_list_path: Path,
) -> Path | None:
    return concatenate_clips(
        clip_paths=clip_paths,
        output_video_path=output_path,
        concat_list_path=concat_list_path,
    )

"""
processor.py — v6
------------------
Passes the new multi-signal audio timeline (sustained, spike, crowd_ratio)
into detect_highlight_peaks so the detector can use them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import analyze_audio
from .clipper import (
    build_clip_windows,
    create_final_highlights_video,
    extract_clips_for_windows,
)
from .config import settings
from .detector import (
    detect_audio_peaks,
    detect_highlight_peaks,
    fuse_scores,
)
from .ffmpeg_tools import extract_audio, get_video_metadata
from .utils import seconds_to_hhmmss
from .video_features import analyze_video_frames


logger = logging.getLogger(__name__)
ProgressCallback = Callable[..., None]


def _progress(cb, stage: str, value: int) -> None:
    if cb:
        cb(stage=stage, progress=value)


def _adaptive_max_events(duration_seconds: float) -> int:
    """No hard cap — let quality gates in detector decide."""
    if duration_seconds >= 80 * 60:
        return settings.MAX_EVENTS_LONG_MATCH   # 200
    if duration_seconds >= 45 * 60:
        return settings.MAX_EVENTS_MEDIUM        # 100
    if duration_seconds >= 10 * 60:
        return settings.MAX_EVENTS_SHORT         # 50
    return settings.MAX_EVENTS_VERY_SHORT        # 20


def _event_type_counts(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        t = e.get("event_type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def process_video(
    job_id: str,
    input_video_path: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    cb = progress_callback

    job_temp_dir   = settings.TEMP_DIR   / job_id
    job_output_dir = settings.OUTPUT_DIR / job_id
    job_temp_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    _progress(cb, "Extracting metadata", 2)
    metadata = get_video_metadata(input_video_path)
    duration = metadata.get("duration")
    if duration is None:
        raise ValueError("Could not determine video duration.")
    if not metadata.get("has_audio"):
        raise ValueError("The video does not contain an audio stream.")

    adaptive_max_events = _adaptive_max_events(float(duration))

    _progress(cb, "Extracting audio track", 5)
    wav_path = job_temp_dir / "audio.wav"
    extract_audio(
        video_path=input_video_path,
        output_wav_path=wav_path,
        sample_rate=settings.AUDIO_SAMPLE_RATE,
    )

    _progress(cb, "Analyzing audio", 12)
    audio_timeline = analyze_audio(
        wav_path=wav_path,
        sample_rate=settings.AUDIO_SAMPLE_RATE,
        hop_length=settings.AUDIO_HOP_LENGTH,
    )

    video_timeline = None
    pipeline_version = "fused_av_v6"

    try:
        _progress(cb, "Sampling video frames", 18)

        def _vf_progress(msg: str, sub_pct: int) -> None:
            _progress(cb, f"Video frames: {msg}", 18 + int(sub_pct * 0.40))

        video_timeline = analyze_video_frames(
            video_path=input_video_path,
            sample_fps=settings.FRAME_SAMPLE_FPS,
            motion_threshold_for_flow=settings.OPTICAL_FLOW_MOTION_THRESHOLD,
            progress_callback=_vf_progress,
        )
        logger.info("Video analysis complete: %d frames", video_timeline["total_sampled_frames"])
    except Exception as exc:
        logger.warning("Video feature extraction failed (%s). Using audio-only.", exc)
        pipeline_version = "audio_only_fallback_v6"

    fusion_weights = {
        "audio":   settings.WEIGHT_AUDIO,
        "motion":  settings.WEIGHT_MOTION,
        "flow":    settings.WEIGHT_FLOW,
        "density": settings.WEIGHT_DENSITY,
        "scene":   settings.WEIGHT_SCENE,
    }

    if video_timeline is not None:
        _progress(cb, "Fusing audio + video signals", 60)

        video_ts    = video_timeline["timestamps"]
        motion_arr  = np.asarray(video_timeline["motion_scores"],              dtype=np.float32)
        scene_arr   = np.asarray(video_timeline["scene_change_scores"],        dtype=np.float32)
        flow_arr    = np.asarray(video_timeline["optical_flow_scores"],        dtype=np.float32)
        density_arr = np.asarray(video_timeline["foreground_density_scores"],  dtype=np.float32)

        fused_ts, fused_scores = fuse_scores(
            video_timestamps=video_ts,
            audio_times=audio_timeline["times"],
            audio_scores=audio_timeline["audio_scores"],
            motion_scores=motion_arr.tolist(),
            scene_change_scores=scene_arr.tolist(),
            optical_flow_scores=flow_arr.tolist(),
            foreground_density_scores=density_arr.tolist(),
            weights=fusion_weights,
        )

        # Interpolate all audio signals onto video timestamps
        def _at_video(key: str) -> np.ndarray:
            return np.interp(
                np.asarray(video_ts, dtype=np.float32),
                np.asarray(audio_timeline["times"], dtype=np.float32),
                np.asarray(audio_timeline.get(key, audio_timeline["audio_scores"]), dtype=np.float32),
            ).astype(np.float32)

        audio_at_video     = _at_video("audio_scores")
        sustained_at_video = _at_video("sustained_scores")
        spike_at_video     = _at_video("spike_scores")
        crowd_at_video     = _at_video("crowd_ratio_scores")

        _progress(cb, "Detecting highlight events", 68)

        raw_events = detect_highlight_peaks(
            timestamps=fused_ts,
            fused_scores=fused_scores,
            audio_scores_at_timestamps=audio_at_video,
            motion_scores=motion_arr,
            optical_flow_scores=flow_arr,
            foreground_density_scores=density_arr,
            scene_change_scores=scene_arr,
            sustained_scores_at_timestamps=sustained_at_video,
            spike_scores_at_timestamps=spike_at_video,
            crowd_ratio_scores_at_timestamps=crowd_at_video,
            min_distance_seconds=settings.PEAK_DISTANCE_SECONDS,
            percentile=settings.PEAK_PERCENTILE,
            std_factor=settings.PEAK_STD_FACTOR,
            max_events=adaptive_max_events,
            refinement_lookback_seconds=settings.EVENT_REFINEMENT_LOOKBACK_SECONDS,
            refinement_lookahead_seconds=settings.EVENT_REFINEMENT_LOOKAHEAD_SECONDS,
        )

    else:
        _progress(cb, "Detecting audio peaks (fallback)", 62)
        raw_events = detect_audio_peaks(
            times=audio_timeline["times"],
            scores=audio_timeline["audio_scores"],
            min_distance_seconds=settings.PEAK_DISTANCE_SECONDS,
            percentile=settings.PEAK_PERCENTILE,
            std_factor=settings.PEAK_STD_FACTOR,
            max_events=adaptive_max_events,
        )

    _progress(cb, "Building clip windows", 74)
    windows = build_clip_windows(
        events=raw_events,
        video_duration=float(duration),
        pre_seconds=settings.CLIP_PRE_SECONDS,
        post_seconds=settings.CLIP_POST_SECONDS,
        merge_gap_seconds=settings.MERGE_EVENTS_CLOSER_THAN_SECONDS,
        min_clip_duration_seconds=settings.MIN_CLIP_DURATION_SECONDS,
        max_clip_duration_seconds=settings.MAX_CLIP_DURATION_SECONDS,
    )

    _progress(cb, "Extracting highlight clips", 80)
    events = extract_clips_for_windows(
        input_video_path=input_video_path,
        windows=windows,
        output_dir=job_output_dir,
    )

    _progress(cb, "Concatenating final highlights video", 93)
    final_video = create_final_highlights_video(
        clip_paths=[Path(e["clip_path"]) for e in events],
        output_path=job_output_dir / "final_highlights.mp4",
        concat_list_path=job_temp_dir / "concat_list.txt",
    )

    highlights_duration = sum(float(e["clip_duration"]) for e in events)
    compression_ratio   = (
        round(float(duration) / highlights_duration, 2) if highlights_duration > 0 else None
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "pipeline_version": pipeline_version + "_v6",
        "message": "Highlight candidates detected by audio-video heuristics.",
        "metadata": metadata,
        "original_duration": round(float(duration), 3),
        "original_duration_hhmmss": seconds_to_hhmmss(float(duration)),
        "highlights_duration": round(highlights_duration, 3),
        "highlights_duration_hhmmss": seconds_to_hhmmss(highlights_duration),
        "compression_ratio": compression_ratio,
        "events_detected": len(events),
        "event_type_counts": _event_type_counts(events),
        "events": events,
        "final_video_path": str(final_video) if final_video else None,
        "output_dir": str(job_output_dir),
        "video_analysis": {
            "sample_fps": video_timeline["sample_fps"] if video_timeline else None,
            "total_sampled_frames": (
                video_timeline["total_sampled_frames"] if video_timeline else 0
            ),
        },
        "fusion_weights": fusion_weights,
        "parameters": {
            "peak_distance_seconds": settings.PEAK_DISTANCE_SECONDS,
            "peak_percentile": settings.PEAK_PERCENTILE,
            "peak_std_factor": settings.PEAK_STD_FACTOR,
            "adaptive_max_events": adaptive_max_events,
        },
    }

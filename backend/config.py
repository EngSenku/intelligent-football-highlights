"""
config.py — v6 (fixed)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    TEMP_DIR: Path = DATA_DIR / "temp"
    OUTPUT_DIR: Path = DATA_DIR / "outputs"

    ALLOWED_VIDEO_EXTENSIONS: tuple[str, ...] = (
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".m4v",
    )

    AUDIO_SAMPLE_RATE: int = 22_050
    AUDIO_HOP_LENGTH: int = 512

    FRAME_SAMPLE_FPS: float = 2.0
    OPTICAL_FLOW_MOTION_THRESHOLD: float = 0.30

    WEIGHT_AUDIO: float = 0.55
    WEIGHT_MOTION: float = 0.14
    WEIGHT_FLOW: float = 0.08
    WEIGHT_DENSITY: float = 0.11
    WEIGHT_SCENE: float = 0.12

    # Lower distance → can catch events closer together (foul then free kick)
    PEAK_DISTANCE_SECONDS: int = 18

    # Lower thresholds → detect more, let quality gates filter
    PEAK_PERCENTILE: float = 75.0  # FIX: was 78.0 — too high, misses quieter goals
    PEAK_STD_FACTOR: float = 0.70  # FIX: was 0.85 — too aggressive, filters valid peaks

    # Very high caps — quality gates in detector control the real count
    MAX_EVENTS_VERY_SHORT: int = 20  # < 10 min
    MAX_EVENTS_SHORT: int = 50  # 10-45 min
    MAX_EVENTS_MEDIUM: int = 100  # 45-80 min
    MAX_EVENTS_LONG_MATCH: int = 200  # >= 80 min
    MAX_EVENTS: int = 200

    # FIX: was 25 — not far enough back to reach the shot when peak=crowd reaction
    EVENT_REFINEMENT_LOOKBACK_SECONDS: int = 35
    # FIX: was 8 — too small, sometimes the shot is slightly after the audio onset
    EVENT_REFINEMENT_LOOKAHEAD_SECONDS: int = 12

    # FIX: was 18 — 18s pre-roll clips the shot for goals. 35s is the fallback
    # (per-event rules in detector.py still take priority over this)
    CLIP_PRE_SECONDS: int = 35
    CLIP_POST_SECONDS: int = 35

    # FIX: was 4 — too small. Two detectors firing on the same goal 5-8s apart
    # produce two candidates; only one survives _merge(). Use 12s so they collapse.
    MERGE_EVENTS_CLOSER_THAN_SECONDS: int = 12

    MIN_CLIP_DURATION_SECONDS: int = 20
    MAX_CLIP_DURATION_SECONDS: int = 130  # full penalty sequence

    MAX_WORKERS: int = 1


settings = Settings()


def ensure_data_directories() -> None:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    PEAK_DISTANCE_SECONDS: int = 18

    PEAK_PERCENTILE: float = 75.0
    PEAK_STD_FACTOR: float = 0.70

    MAX_EVENTS_VERY_SHORT: int = 20
    MAX_EVENTS_SHORT: int = 50
    MAX_EVENTS_MEDIUM: int = 100
    MAX_EVENTS_LONG_MATCH: int = 200
    MAX_EVENTS: int = 200

    EVENT_REFINEMENT_LOOKBACK_SECONDS: int = 35

    EVENT_REFINEMENT_LOOKAHEAD_SECONDS: int = 12

    CLIP_PRE_SECONDS: int = 35
    CLIP_POST_SECONDS: int = 35

    MERGE_EVENTS_CLOSER_THAN_SECONDS: int = 12

    MIN_CLIP_DURATION_SECONDS: int = 20
    MAX_CLIP_DURATION_SECONDS: int = 130

    MAX_WORKERS: int = 1


settings = Settings()


def ensure_data_directories() -> None:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

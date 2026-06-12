import re
from pathlib import Path
from typing import Any


def safe_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename)
    return filename or "uploaded_video.mp4"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

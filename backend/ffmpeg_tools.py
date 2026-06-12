import json
import subprocess
from pathlib import Path
from typing import Any


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg or FFprobe command fails."""


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise FFmpegError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\nSTDERR:\n"
            + result.stderr[-4000:]
        )

    return result


def get_video_metadata(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = run_command(command)
    probe = json.loads(result.stdout)

    streams = probe.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise FFmpegError("No video stream found in the uploaded file.")

    duration = None
    if probe.get("format", {}).get("duration"):
        duration = float(probe["format"]["duration"])
    elif video_stream.get("duration"):
        duration = float(video_stream["duration"])

    fps = None
    avg_frame_rate = video_stream.get("avg_frame_rate")
    if avg_frame_rate and avg_frame_rate != "0/0":
        numerator, denominator = avg_frame_rate.split("/")
        fps = float(numerator) / float(denominator)

    return {
        "duration": duration,
        "fps": fps,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "format_name": probe.get("format", {}).get("format_name"),
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
    }


def extract_audio(video_path: Path, output_wav_path: Path, sample_rate: int) -> Path:
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav_path),
    ]
    run_command(command)
    return output_wav_path


def extract_clip_fast(
    input_video_path: Path,
    output_clip_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(input_video_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_clip_path),
    ]
    run_command(command)
    return output_clip_path


def concatenate_clips(
    clip_paths: list[Path],
    output_video_path: Path,
    concat_list_path: Path,
) -> Path | None:
    if not clip_paths:
        return None

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for path in clip_paths:
        escaped = str(path.resolve()).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    concat_list_path.write_text("\n".join(lines), encoding="utf-8")

    copy_command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(output_video_path),
    ]

    try:
        run_command(copy_command)
        return output_video_path
    except FFmpegError:
        reencode_command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_video_path),
        ]
        run_command(reencode_command)
        return output_video_path

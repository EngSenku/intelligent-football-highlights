"""
video_features.py
-----------------
Lightweight image-processing pipeline for the Intelligent Football
Highlight Detection System.

Stages implemented here:
  Stage 2  — Frame sampling at a low FPS (default 2 fps) so we never
             load the full 30fps stream into RAM.
  Stage 3  — Scene-change detection via histogram difference.
  Stage 4  — Motion score via grayscale frame differencing.
  Stage 5  — Optical-flow score (Lucas-Kanade) on high-motion windows.
  Stage 6  — Non-green activity density (foreground_density_score).

None of these stages use a deep-learning model.
All scores are normalised to [0, 1].

The public entry point is `analyze_video_frames()`, which returns a
dict of parallel lists — one value per sampled frame — ready to be
fused by detector.py.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────


def _normalize_01(values: np.ndarray) -> np.ndarray:
    """Normalise an array to [0, 1].  All-zero input stays all-zero."""
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _bgr_to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# ── Stage 2: frame sampling ───────────────────────────────────────────────────


def sample_frames_from_video(
    video_path: Path,
    sample_fps: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Extract one frame every (1 / sample_fps) seconds using OpenCV.

    Frames are yielded sequentially; only the current and previous frame
    are kept in RAM at any time.  Debug storage of frames is NOT done here
    by default — callers that need frames for debugging should collect them
    from the returned list.

    Returns:
        timestamps : list of float  (seconds from start)
        frames     : list of np.ndarray (BGR uint8)

    For a 90-min match at 2 fps → 10 800 frames, each ~0.5–2 MB uncompressed.
    That is feasible; at 3 fps it reaches ~16 200 frames and may approach
    RAM limits on small machines.  Default is 2 fps.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval: int = max(1, round(native_fps / sample_fps))

    timestamps: list[float] = []
    frames: list[np.ndarray] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / native_fps
            timestamps.append(timestamp)
            frames.append(frame)

            if progress_callback and total_frames > 0:
                progress_callback(frame_idx, total_frames)

        frame_idx += 1

    cap.release()
    return timestamps, frames


# ── Stage 3: scene-change detection ──────────────────────────────────────────


def compute_scene_change_scores(frames: list[np.ndarray]) -> np.ndarray:
    """
    Detect abrupt scene changes via histogram difference between consecutive
    sampled frames.

    A high score means the shot changed (replay, crowd shot, celebration).
    These are NOT excluded from highlight detection — they are a positive
    signal that something happened.

    Returns:
        np.ndarray of shape (N,) normalised to [0, 1].
    """
    n = len(frames)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        gray_prev = _bgr_to_gray(frames[i - 1])
        gray_curr = _bgr_to_gray(frames[i])

        hist_prev = cv2.calcHist([gray_prev], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([gray_curr], [0], None, [64], [0, 256])

        # Bhattacharyya distance: 0 = identical, 1 = totally different
        dist = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
        scores[i] = float(dist)

    return _normalize_01(scores)


# ── Stage 4: motion score ─────────────────────────────────────────────────────


def compute_motion_scores(frames: list[np.ndarray]) -> np.ndarray:
    """
    Frame-differencing motion score on grayscale sampled frames.

    For each consecutive pair:
        diff = |gray(t) − gray(t-1)|
        score = mean(diff) / 255   (already in [0, 1] before normalisation)

    Returns:
        np.ndarray of shape (N,) normalised to [0, 1].
    """
    n = len(frames)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        gray_prev = _bgr_to_gray(frames[i - 1]).astype(np.float32)
        gray_curr = _bgr_to_gray(frames[i]).astype(np.float32)

        diff = np.abs(gray_curr - gray_prev)
        scores[i] = float(diff.mean()) / 255.0

    return _normalize_01(scores)


# ── Stage 5: optical flow (sparse Lucas-Kanade) ───────────────────────────────


def compute_optical_flow_scores(
    frames: list[np.ndarray],
    motion_scores: np.ndarray,
    motion_threshold: float = 0.30,
) -> np.ndarray:
    """
    Compute sparse Lucas-Kanade optical flow ONLY on high-motion windows
    (where motion_score > motion_threshold).  This avoids wasting CPU on
    quiet passages.

    Failure cases handled:
    - No good points found        → score = 0
    - Too few tracked points (<4) → score = 0
    - Scene change too abrupt     → score = 0 (Bhattacharyya > 0.8 means
                                    the pair is unreliable for flow)

    Returns:
        np.ndarray of shape (N,) normalised to [0, 1].
    """
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    feature_params = dict(
        maxCorners=80,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    n = len(frames)
    scores = np.zeros(n, dtype=np.float32)
    motion_scores = np.asarray(motion_scores, dtype=np.float32)

    for i in range(1, n):
        # Skip low-motion frames — save CPU
        if motion_scores[i] < motion_threshold:
            continue

        gray_prev = _bgr_to_gray(frames[i - 1])
        gray_curr = _bgr_to_gray(frames[i])

        # Skip frames whose scene change is too abrupt for reliable flow
        hist_prev = cv2.calcHist([gray_prev], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([gray_curr], [0], None, [64], [0, 256])
        bhatt = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
        if bhatt > 0.80:
            continue

        pts_prev = cv2.goodFeaturesToTrack(gray_prev, **feature_params)
        if pts_prev is None or len(pts_prev) < 4:
            continue

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, pts_prev, None, **lk_params
        )

        if pts_curr is None or status is None:
            continue

        good_prev = pts_prev[status.ravel() == 1]
        good_curr = pts_curr[status.ravel() == 1]

        if len(good_prev) < 4:
            continue

        displacements = np.linalg.norm(good_curr - good_prev, axis=2).ravel()
        avg_disp = float(displacements.mean())

        # Cap at 50 pixels to avoid outliers from abrupt cuts dominating
        scores[i] = min(avg_disp / 50.0, 1.0)

    return _normalize_01(scores)


# ── Stage 6: non-green activity density ──────────────────────────────────────

# HSV range for the football pitch grass colour.
# Tuned for broadcast footage.  Widen H range slightly for synthetic pitches.
_GREEN_H_LOW = 35
_GREEN_H_HIGH = 85
_GREEN_S_LOW = 40
_GREEN_V_LOW = 40


def compute_foreground_density_scores(
    frames: list[np.ndarray],
    roi_top_frac: float = 0.10,
    roi_bottom_frac: float = 0.90,
) -> np.ndarray:
    """
    Estimate activity density by measuring the fraction of NON-green pixels
    inside the central region of each frame.

    This is NOT player detection.  It counts any non-field pixel:
    players, referees, shadows, ad boards, lines.  Useful as a weak
    activity signal.

    roi_top_frac / roi_bottom_frac : vertical crop to ignore scoreboard
    overlays (top ~10%) and bottom ad banners (bottom ~10%).

    Returns:
        np.ndarray of shape (N,) normalised to [0, 1].
    """
    n = len(frames)
    scores = np.zeros(n, dtype=np.float32)

    green_lower = np.array([_GREEN_H_LOW, _GREEN_S_LOW, _GREEN_V_LOW], dtype=np.uint8)
    green_upper = np.array([_GREEN_H_HIGH, 255, 255], dtype=np.uint8)

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        y0 = int(h * roi_top_frac)
        y1 = int(h * roi_bottom_frac)
        roi = frame[y0:y1, :]

        hsv = _bgr_to_hsv(roi)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        total_pixels = green_mask.size
        green_pixels = int(green_mask.sum() // 255)
        non_green = total_pixels - green_pixels

        scores[i] = non_green / max(total_pixels, 1)

    return _normalize_01(scores)


# ── Public entry point ────────────────────────────────────────────────────────


def analyze_video_frames(
    video_path: Path,
    sample_fps: float = 2.0,
    motion_threshold_for_flow: float = 0.30,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict:
    """
    Run the full image-processing pipeline on a video file.

    Pipeline:
        1. Sample frames at sample_fps
        2. Compute scene-change scores
        3. Compute motion scores
        4. Compute optical-flow scores (only on high-motion windows)
        5. Compute foreground-density scores

    Returns a dict with parallel lists (one entry per sampled frame):
        {
            "timestamps":               list[float],
            "motion_scores":            list[float],
            "scene_change_scores":      list[float],
            "optical_flow_scores":      list[float],
            "foreground_density_scores": list[float],
            "sample_fps":               float,
            "total_sampled_frames":     int,
        }

    Memory note:
        At 2 fps a 90-min match yields ~10 800 frames.  Full-res 1080p
        frames are ~6 MB each → ~65 GB if all kept.  We keep them in
        memory only while processing, then discard.  If RAM is tight,
        lower sample_fps to 1.
    """

    def _prog(msg: str, pct: int) -> None:
        if progress_callback:
            progress_callback(msg, pct)

    _prog("Sampling frames", 0)

    def _frame_prog(idx: int, total: int) -> None:
        pct = min(40, int(40 * idx / max(total, 1)))
        _prog(f"Sampling frames ({idx}/{total})", pct)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval: int = max(1, round(native_fps / sample_fps))

    timestamps: list[float] = []
    motion_scores_list: list[float] = []
    scene_scores_list: list[float] = []
    flow_scores_list: list[float] = []
    density_scores_list: list[float] = []

    green_lower = np.array([_GREEN_H_LOW, _GREEN_S_LOW, _GREEN_V_LOW], dtype=np.uint8)
    green_upper = np.array([_GREEN_H_HIGH, 255, 255], dtype=np.uint8)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    feature_params = dict(maxCorners=80, qualityLevel=0.3, minDistance=7, blockSize=7)

    prev_gray: np.ndarray | None = None
    frame_idx = 0
    sampled = 0

    _prog("Sampling and analysing frames", 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / native_fps
            timestamps.append(timestamp)

            gray = _bgr_to_gray(frame)

            # Motion score
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray).astype(np.float32)
                motion_val = float(diff.mean()) / 255.0
            else:
                motion_val = 0.0
            motion_scores_list.append(motion_val)

            # Scene-change score
            if prev_gray is not None:
                hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
                hist_curr = cv2.calcHist([gray], [0], None, [64], [0, 256])
                scene_val = float(
                    cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
                )
            else:
                scene_val = 0.0
            scene_scores_list.append(scene_val)

            # Optical flow score (only on high-motion frames)
            flow_val = 0.0
            if prev_gray is not None and motion_val >= motion_threshold_for_flow:
                bhatt = scene_val  # already computed
                if bhatt <= 0.80:
                    pts_prev = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
                    if pts_prev is not None and len(pts_prev) >= 4:
                        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, gray, pts_prev, None, **lk_params
                        )
                        if pts_curr is not None and status is not None:
                            good_prev = pts_prev[status.ravel() == 1]
                            good_curr = pts_curr[status.ravel() == 1]
                            if len(good_prev) >= 4:
                                displacements = np.linalg.norm(
                                    good_curr - good_prev, axis=2
                                ).ravel()
                                flow_val = min(float(displacements.mean()) / 50.0, 1.0)
            flow_scores_list.append(flow_val)

            # Foreground density score
            h, w = frame.shape[:2]
            y0 = int(h * 0.10)
            y1 = int(h * 0.90)
            roi = frame[y0:y1, :]
            hsv = _bgr_to_hsv(roi)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            total_pixels = green_mask.size
            green_pixels = int(green_mask.sum() // 255)
            density_scores_list.append(
                (total_pixels - green_pixels) / max(total_pixels, 1)
            )

            prev_gray = gray
            sampled += 1
            if total_frames > 0:
                pct = min(90, int(90 * frame_idx / total_frames))
                _prog(f"Analysing frames ({sampled} sampled)", pct)

        frame_idx += 1

    cap.release()

    if not timestamps:
        raise RuntimeError("No frames were extracted from the video.")

    n = len(timestamps)
    scene_scores = _normalize_01(np.array(scene_scores_list, dtype=np.float32))
    motion_scores = _normalize_01(np.array(motion_scores_list, dtype=np.float32))
    flow_scores = _normalize_01(np.array(flow_scores_list, dtype=np.float32))
    density_scores = _normalize_01(np.array(density_scores_list, dtype=np.float32))

    _prog("Video feature extraction complete", 100)

    return {
        "timestamps": timestamps,
        "motion_scores": motion_scores.tolist(),
        "scene_change_scores": scene_scores.tolist(),
        "optical_flow_scores": flow_scores.tolist(),
        "foreground_density_scores": density_scores.tolist(),
        "sample_fps": sample_fps,
        "total_sampled_frames": n,
    }

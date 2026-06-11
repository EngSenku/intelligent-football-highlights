"""
audio.py
--------
Audio analysis for football highlight detection — v6.

The core insight for football:
- GOALS produce a SUSTAINED high-energy crowd reaction lasting 20-60 seconds.
- FOULS produce a sharp spike + quick decay (whistle, brief protest).
- CORNERS/FREE-KICKS produce moderate sustained energy during delivery.
- MIDFIELD PLAY produces low, variable energy.

So we compute THREE separate signals:
1. rms_sustained  — smoothed RMS over a ~15s window (catches sustained crowd cheering)
2. spike_signal   — sudden sharp onset that decays quickly (whistle, tackle)
3. base_energy    — raw RMS (used for fusion)

We also compute spectral features:
- spectral_centroid: crowd cheering shifts to higher frequencies than commentary
- crowd_ratio: ratio of energy in high-frequency bands (1k-8kHz) vs total
  This is the most reliable differentiator: crowd cheer is broadband high-freq,
  commentary is narrowband low-freq.
"""

from pathlib import Path

import librosa
import numpy as np


def normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = float(np.min(values)) if values.size else 0.0
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def _smooth_window(arr: np.ndarray, window_frames: int) -> np.ndarray:
    """Box-car (moving average) smoothing."""
    if window_frames <= 1 or len(arr) == 0:
        return arr
    w = min(window_frames, len(arr))
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def analyze_audio(
    wav_path: Path,
    sample_rate: int,
    hop_length: int,
) -> dict:
    """
    Analyze audio and return multiple timeline signals for football highlight detection.

    Key signals returned:
    - audio_scores      : primary fused score (sustained excitement)
    - sustained_scores  : long-window smoothed RMS (best for goals)
    - spike_scores      : sharp transient signal (good for fouls/whistles)
    - crowd_ratio_scores: high-freq energy ratio (crowd vs commentary)
    - rms_scores        : raw RMS energy
    """
    y, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)

    if y.size == 0:
        raise ValueError("Audio extraction succeeded, but the WAV file is empty.")

    # ── Basic features ────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    length = min(len(rms), len(onset))
    rms = rms[:length]
    onset = onset[:length]

    rms_norm = normalize_01(rms)
    onset_norm = normalize_01(onset)

    # ── Sustained excitement signal (key for goals) ───────────────────────────
    # Football crowd cheering after a goal sustains for 20-60 seconds.
    # Smooth RMS over ~15 second window to capture this sustained energy.
    frames_per_second = sr / hop_length
    window_15s = max(1, int(15 * frames_per_second))
    window_5s = max(1, int(5 * frames_per_second))
    window_30s = max(1, int(30 * frames_per_second))

    sustained_15s = _smooth_window(rms_norm, window_15s)
    sustained_30s = _smooth_window(rms_norm, window_30s)
    # FIX: was 0.6*15s + 0.4*30s — weighting too much on the short window.
    # Goals sustain 30-60s; the 30s window is the more reliable indicator.
    sustained = normalize_01(0.45 * sustained_15s + 0.55 * sustained_30s)

    # ── Spike signal (sharp events: whistle, tackle, crowd gasp) ─────────────
    short_smooth = _smooth_window(rms_norm, window_5s)
    spike = np.maximum(rms_norm - short_smooth, 0.0)
    spike_norm = normalize_01(spike)

    # ── Crowd ratio signal ────────────────────────────────────────────────────
    # Crowd cheering has strong energy in 1kHz-8kHz band.
    # Commentary/stadium noise is concentrated in 200Hz-1kHz.
    # This ratio is the most reliable signal for distinguishing crowd cheering
    # from normal broadcast commentary.
    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    # High frequency band: 1000-6000 Hz (crowd, whistles)
    # FIX: was 1000-8000 Hz. Broadcast streams (128kbps AAC) roll off above ~6kHz.
    # The 6k-8k range adds noise to the numerator without adding crowd signal,
    # diluting the ratio and making it less reliable for goal detection.
    high_mask = (freqs >= 1000) & (freqs <= 6000)
    # Total energy band: 200-10000 Hz
    total_mask = (freqs >= 200) & (freqs <= 10000)

    high_energy = (
        stft[high_mask, :].mean(axis=0) if high_mask.any() else np.zeros(stft.shape[1])
    )
    total_energy = (
        stft[total_mask, :].mean(axis=0) if total_mask.any() else np.ones(stft.shape[1])
    )

    # Align length
    min_len = min(length, len(high_energy), len(total_energy))
    high_energy = high_energy[:min_len]
    total_energy = total_energy[:min_len]
    high_energy = np.pad(high_energy, (0, length - min_len), constant_values=0)
    total_energy = np.pad(total_energy, (0, length - min_len), constant_values=1e-8)

    crowd_ratio = (high_energy / (total_energy + 1e-8)).astype(np.float32)
    crowd_ratio_smooth = _smooth_window(normalize_01(crowd_ratio), window_5s)
    crowd_ratio_norm = normalize_01(crowd_ratio_smooth)

    # ── Positive RMS delta (sudden rise in energy) ────────────────────────────
    positive_delta = np.maximum(np.diff(rms_norm, prepend=rms_norm[0]), 0)
    delta_norm = normalize_01(positive_delta)

    # ── Primary fused audio score ─────────────────────────────────────────────
    # For football highlight detection, sustained crowd cheering is the #1 signal.
    # Crowd ratio helps separate genuine celebration from background commentary.
    # Onset picks up sudden events (fouls, whistles).
    audio_score = (
        0.40 * sustained  # sustained cheering — best for goals
        + 0.25 * crowd_ratio_norm  # crowd vs commentary ratio
        + 0.20 * rms_norm  # raw loudness
        + 0.10 * onset_norm  # sudden events
        + 0.05 * delta_norm  # energy spikes
    )
    audio_score = normalize_01(audio_score)

    times = librosa.frames_to_time(
        np.arange(length),
        sr=sr,
        hop_length=hop_length,
    )

    return {
        "times": times.tolist(),
        "audio_scores": audio_score.tolist(),
        "sustained_scores": sustained.tolist(),
        "spike_scores": spike_norm.tolist(),
        "crowd_ratio_scores": crowd_ratio_norm.tolist(),
        "rms_scores": rms_norm.tolist(),
        "onset_scores": onset_norm.tolist(),
        "energy_delta_scores": delta_norm.tolist(),
        "sample_rate": sr,
        "hop_length": hop_length,
    }

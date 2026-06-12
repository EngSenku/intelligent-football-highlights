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

    y, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)

    if y.size == 0:
        raise ValueError("Audio extraction succeeded, but the WAV file is empty.")

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    length = min(len(rms), len(onset))
    rms = rms[:length]
    onset = onset[:length]

    rms_norm = normalize_01(rms)
    onset_norm = normalize_01(onset)

    frames_per_second = sr / hop_length
    window_15s = max(1, int(15 * frames_per_second))
    window_5s = max(1, int(5 * frames_per_second))
    window_30s = max(1, int(30 * frames_per_second))

    sustained_15s = _smooth_window(rms_norm, window_15s)
    sustained_30s = _smooth_window(rms_norm, window_30s)

    sustained = normalize_01(0.45 * sustained_15s + 0.55 * sustained_30s)

    short_smooth = _smooth_window(rms_norm, window_5s)
    spike = np.maximum(rms_norm - short_smooth, 0.0)
    spike_norm = normalize_01(spike)

    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    high_mask = (freqs >= 1000) & (freqs <= 6000)

    total_mask = (freqs >= 200) & (freqs <= 10000)

    high_energy = (
        stft[high_mask, :].mean(axis=0) if high_mask.any() else np.zeros(stft.shape[1])
    )
    total_energy = (
        stft[total_mask, :].mean(axis=0) if total_mask.any() else np.ones(stft.shape[1])
    )

    min_len = min(length, len(high_energy), len(total_energy))
    high_energy = high_energy[:min_len]
    total_energy = total_energy[:min_len]
    high_energy = np.pad(high_energy, (0, length - min_len), constant_values=0)
    total_energy = np.pad(total_energy, (0, length - min_len), constant_values=1e-8)

    crowd_ratio = (high_energy / (total_energy + 1e-8)).astype(np.float32)
    crowd_ratio_smooth = _smooth_window(normalize_01(crowd_ratio), window_5s)
    crowd_ratio_norm = normalize_01(crowd_ratio_smooth)

    positive_delta = np.maximum(np.diff(rms_norm, prepend=rms_norm[0]), 0)
    delta_norm = normalize_01(positive_delta)

    audio_score = (
        0.40 * sustained
        + 0.25 * crowd_ratio_norm
        + 0.20 * rms_norm
        + 0.10 * onset_norm
        + 0.05 * delta_norm
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

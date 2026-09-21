"""Audio -> weights: turn sound into a conditioning vector for the NCA.

Feature vector (8 dims, all roughly in [0,1] or [-1,1]):
  0 energy    — RMS loudness. Scales growth rate + writes AUDIO channel.
  1 centroid  — spectral brightness. Biases pigment toward blue/white.
  2 flux      — onset/change. Triggers symmetry-breaking bursts.
  3 beat      — phase of a tracked pulse in [0,1). Pulses regeneration waves.
  4 low band  — bass energy. Thickens the girdle / rim.
  5 mid band  — vocal-range energy. Drives rib/striae contrast.
  6 high band — air/sizzle. Seeds pores (areolae).
  7 contrast  — tonal vs noisy. Picks gold (tonal) vs glass (noisy) palette.

No hard dependency on librosa: features come from numpy/scipy so training
and the browser demo stay light. A WAV loader is included for real audio.
"""
from __future__ import annotations

import numpy as np

N_AUDIO_DIMS = 8
FEATURE_NAMES = ("energy", "centroid", "flux", "beat", "low", "mid", "high", "contrast")


def frame_features(prev_mag: np.ndarray | None, mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Single-frame 8-dim features from a magnitude spectrum."""
    eps = 1e-8
    energy = float(np.sqrt(np.mean(mag**2)))
    total = float(mag.sum()) + eps
    centroid = float((freqs * mag).sum() / total)
    nyq = float(freqs.max()) + eps
    centroid_n = np.clip(centroid / nyq, 0, 1)
    if prev_mag is not None:
        flux = float(np.mean(np.maximum(mag - prev_mag, 0.0)) / (total / len(mag) + eps))
        flux = float(np.clip(flux, 0, 1))
    else:
        flux = 0.0
    bands = np.array_split(mag, 3)
    low = float(np.clip(bands[0].mean() / (mag.mean() + eps) / 3, 0, 1))
    mid = float(np.clip(bands[1].mean() / (mag.mean() + eps) / 3, 0, 1))
    high = float(np.clip(bands[2].mean() / (mag.mean() + eps) / 3, 0, 1))
    # Spectral contrast proxy: peak-to-mean ratio, squashed.
    contrast = float(np.clip((mag.max() / (mag.mean() + eps) - 1) / 20.0, 0, 1))
    beat = 0.0  # filled by the tracker below
    return np.array([energy, centroid_n, flux, beat, low, mid, high, contrast], dtype=np.float32)


def stft_features(
    waveform: np.ndarray, sr: int = 22050, frame: int = 1024, hop: int = 512
) -> np.ndarray:
    """Waveform (mono float) -> (T,8) conditioning sequence."""
    from scipy.signal import get_window

    x = np.asarray(waveform, dtype=np.float32).ravel()
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    window = get_window("hann", frame, fftbins=True).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr).astype(np.float32)
    feats, prev = [], None
    # Simple beat tracker: smoothed energy peaks -> phase.
    energies: list[float] = []
    for start in range(0, len(x) - frame + 1, hop):
        seg = x[start : start + frame] * window
        mag = np.abs(np.fft.rfft(seg)).astype(np.float32)
        f = frame_features(prev, mag, freqs)
        feats.append(f)
        energies.append(float(f[0]))
        prev = mag
    feats = np.stack(feats).astype(np.float32)
    # Normalise energy to [0,1] across the clip; derive beat phase from peaks.
    e = np.array(energies, dtype=np.float32)
    e = (e - e.min()) / (e.max() - e.min() + 1e-8)
    feats[:, 0] = e
    peaks = (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:])
    last_peak = 0
    period = 8
    for t in range(len(feats)):
        if t - 1 >= 0 and t - 1 < len(peaks) and peaks[t - 1]:
            if t - last_peak > 2:
                period = t - last_peak
            last_peak = t
        feats[t, 3] = ((t - last_peak) / max(period, 1)) % 1.0
    return feats


def load_wav(path: str, target_sr: int = 22050, max_seconds: float = 8.0) -> tuple[np.ndarray, int]:
    """Load mono float waveform with scipy (no librosa needed)."""
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    x = np.asarray(data)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / max(1, np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
    if sr != target_sr:
        from scipy.signal import resample

        x = resample(x, int(len(x) * target_sr / sr)).astype(np.float32)
        sr = target_sr
    x = x[: int(max_seconds * sr)]
    peak = np.abs(x).max()
    if peak > 0:
        x = (x / peak * 0.9).astype(np.float32)
    return x, sr


def synth_demo_signal(kind: str = "beat", seconds: float = 4.0, sr: int = 22050) -> np.ndarray:
    """Synthetic audio for demos without audio files: beat / sweep / bloom."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    if kind == "beat":
        kick = np.maximum(0, np.sin(2 * np.pi * 2.0 * t)) ** 8
        bass = np.sin(2 * np.pi * 55 * t) * kick
        hats = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32) * 0.05
        x = bass * 0.8 + hats
    elif kind == "sweep":
        f = 200 + 1800 * (t / seconds)
        x = np.sin(2 * np.pi * f * t).astype(np.float32) * 0.6
    else:  # bloom: swelling chord
        env = np.sin(np.pi * t / seconds) ** 2
        x = env * sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330, 440)).astype(np.float32) / 4
    peak = np.abs(x).max()
    return (x / (peak + 1e-8) * 0.9).astype(np.float32)


def random_condition(batch: int, energy_bias: float = 0.5, seed: int | None = None):
    """Random audio vectors for training augmentation (robustness to sound)."""
    import torch

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    a = torch.rand(batch, N_AUDIO_DIMS, generator=g) * 0.6
    a[:, 0] = torch.clamp(torch.randn(batch, generator=g) * 0.25 + energy_bias, 0, 1)
    return a


def smooth(feats: np.ndarray, width: int = 5) -> np.ndarray:
    """Temporal smoothing so morphology bends instead of jittering."""
    kernel = np.ones(width, dtype=np.float32) / width
    out = feats.copy()
    for i in range(feats.shape[1]):
        out[:, i] = np.convolve(feats[:, i], kernel, mode="same")
    return out.astype(np.float32)

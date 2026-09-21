"""The creature talks: sonify NCA growth into song.

The loop this completes: microphone → audio conditioning → morphogenesis →
this module → speaker. The organism hears, grows, and sings back; play its
song into its own conditioning and it listens to itself.

Mapping (state → sound), all from first principles:
  alive mass      → pitch (quantised to A-minor pentatonic, so it can't sing ugly)
  R/G/B pigment   → root / fifth / octave shimmer chord
  growth rate     → loudness (still tissue whispers, growing tissue sings)
  centre of mass  → stereo pan (it leans toward where it grows)
  symmetry        → vibrato depth (broken symmetry wobbles the voice)
"""
from __future__ import annotations

import numpy as np

PENTA = np.array([0, 3, 5, 7, 10, 12, 15], dtype=np.float32)  # A-minor pentatonic
A1 = 55.0


def frame_descriptor(state) -> dict:
    """One growth frame -> interpretable features. state: (C,H,W) tensor/array."""
    try:
        import torch

        if isinstance(state, torch.Tensor):
            state = state.detach().float().cpu().numpy()
    except ImportError:
        pass
    s = np.asarray(state, dtype=np.float32)
    c, h, w = s.shape
    alpha = np.clip(s[3], 0, 1)
    mass = float(alpha.sum() / (h * w))
    rgb = [float(np.clip(s[i], 0, 1).mean()) for i in range(3)]
    alive = (alpha > 0.1).astype(np.float32)
    spread = float(alive.mean())
    if alive.sum() > 0:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        cx = float((xx * alive).sum() / alive.sum() / w)  # 0..1
    else:
        cx = 0.5
    left, right = alpha[:, : w // 2], alpha[:, w // 2 :]
    mw = min(left.shape[1], right.shape[1])
    symmetry = float(1.0 - np.abs(left[:, :mw] - right[:, ::-1][:, :mw]).mean())
    return {"mass": mass, "rgb": rgb, "spread": spread, "pan": cx, "symmetry": symmetry}


def describe_trajectory(traj) -> list[dict]:
    """traj: (T,C,H,W) tensor/array -> per-frame descriptors."""
    try:
        import torch

        if isinstance(traj, torch.Tensor):
            traj = traj.detach().float().cpu()
            return [frame_descriptor(traj[t]) for t in range(traj.shape[0])]
    except ImportError:
        pass
    return [frame_descriptor(np.asarray(traj[t])) for t in range(len(traj))]


def sing(
    traj,
    sr: int = 22050,
    frame_dur: float = 0.15,
    base_octave: float = 2.0,
) -> np.ndarray:
    """Render a trajectory as stereo song. Returns (N,2) float32 in [-1,1]."""
    descs = describe_trajectory(traj)
    t = len(descs)
    n = int(t * frame_dur * sr)
    if n < sr // 10:
        raise ValueError("trajectory too short to sing")
    masses = np.array([d["mass"] for d in descs], dtype=np.float32)
    growth = np.zeros_like(masses)
    growth[1:] = np.diff(masses)
    # Normalise against the organism's own range (every creature finds its voice).
    span = masses.max() - masses.min() + 1e-6
    mass_n = (masses - masses.min()) / span
    grow_n = np.clip(growth / (np.abs(growth).max() + 1e-6), -1, 1)

    # Per-frame pitch: pentatonic degree from mass, chord weights from pigment.
    degree = np.clip((mass_n * (len(PENTA) - 1)).astype(int), 0, len(PENTA) - 1)
    semis = PENTA[degree]
    f_root = A1 * (2.0 ** (base_octave + semis / 12.0))
    rgb = np.array([d["rgb"] for d in descs], dtype=np.float32) + 0.15
    rgb = rgb / rgb.sum(axis=1, keepdims=True)
    pans = np.array([d["pan"] for d in descs], dtype=np.float32)
    syms = np.array([d["symmetry"] for d in descs], dtype=np.float32)

    # Upsample frame tracks to audio rate.
    tt = np.arange(n, dtype=np.float32) / sr
    ft = np.clip((tt / frame_dur).astype(int), 0, t - 1)
    f0 = f_root[ft]
    fifth = f0 * 1.5
    octv = f0 * 2.0 * (1.0 + 0.002 * np.sin(2 * np.pi * 5 * tt))  # shimmer
    w0, w1, w2 = rgb[ft, 0], rgb[ft, 1], rgb[ft, 2]
    amp = (0.15 + 0.85 * np.clip(grow_n[ft] * 2.0, 0, 1))
    amp = np.convolve(amp, np.ones(sr // 20) / (sr // 20), mode="same")  # smooth breath
    vib = 1.0 + (1.0 - syms[ft]) * 0.01 * np.sin(2 * np.pi * 6 * tt)

    # Phase-continuous additive synthesis.
    ph = lambda f: np.cumsum(2 * np.pi * f * vib / sr)
    voice = w0 * np.sin(ph(f0)) + w1 * 0.7 * np.sin(ph(fifth)) + w2 * 0.4 * np.sin(ph(octv))
    voice = voice * amp
    voice = np.tanh(voice * 1.5) * 0.85

    pan = pans[ft]
    left = voice * np.sqrt(1 - pan)
    right = voice * np.sqrt(pan)
    return np.stack([left, right], axis=1).astype(np.float32)


def save_wav(path: str, wave: np.ndarray, sr: int = 22050) -> None:
    """Write mono/stereo float song to 16-bit WAV."""
    from scipy.io import wavfile

    x = np.asarray(wave, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))

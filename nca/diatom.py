"""Procedural diatom targets — the "Diatom road".

Real diatoms build glass (silica) frustules with radial (centric) or
bilateral (pennate) symmetry, pores (areolae) in rows, ribs and a central
nodule. Here we synthesise differentiable-free RGBA targets with the same
design language so the NCA has something morphogenetic to grow toward.

Two families:
  centric — round, n-fold rotational symmetry, concentric rings + striae rays
  pennate — elongated boat/needle, mirror symmetry, transverse striae + raphe

All outputs are float32 HxWx4 in [0,1]. Alpha is the frustule mask.
"""
from __future__ import annotations

import numpy as np


def _soft_mask(d: np.ndarray, radius: float, feather: float = 1.5) -> np.ndarray:
    return np.clip((radius - d) / feather + 0.5, 0.0, 1.0)


def centric_diatom(
    size: int = 48,
    n_fold: int = 8,
    n_rings: int = 3,
    seed: int = 0,
    palette: str = "gold-glass",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    R = size * 0.42

    # Girdle (outer rim) + valve face.
    rim = _soft_mask(np.abs(r - R), 1.6, 1.6)
    face = _soft_mask(r, R - 1.0, 2.0)

    # Radial ribs with n-fold symmetry + slight wobble (hand of the cell).
    wobble = 0.15 * np.sin(3 * theta + rng.uniform(0, 2 * np.pi))
    ribs = 0.5 + 0.5 * np.cos(n_fold * theta + 2.0 * wobble)
    ribs = np.power(ribs, 1.5) * face

    # Concentric growth rings.
    rings = 0.5 + 0.5 * np.cos(2 * np.pi * n_rings * r / R + rng.uniform(0, 2 * np.pi))
    rings = np.power(rings, 2.0) * face

    # Areolae pores: hex-ish dotted rows along rays.
    pore_field = np.sin(n_fold * theta * 1.0) ** 2 * np.sin(np.pi * n_rings * 2 * r / R) ** 2
    pore_noise = rng.random((size, size))
    pores = ((pore_field > 0.55) & (pore_noise > 0.45)).astype(np.float32) * face

    # Central rosette / nodule.
    centre = _soft_mask(r, size * 0.08, 1.5)

    structure = np.clip(0.35 * ribs + 0.35 * rings + 0.5 * pores + 0.6 * centre, 0, 1)

    rgb = _paint(structure, face, rim, palette, rng)
    alpha = np.clip(face + rim * 0.9, 0, 1)
    return np.dstack([rgb, alpha]).astype(np.float32)


def pennate_diatom(
    size: int = 48,
    aspect: float = 2.2,
    striae_freq: float = 9.0,
    seed: int = 0,
    palette: str = "gold-glass",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    # Boat-shaped valve: superellipse-ish distance.
    ex, ey = dx * 1.0, dy * aspect
    d = np.sqrt(ex**2 + ey**2)
    R = size * 0.42
    face = _soft_mask(d, R, 2.0)
    rim = _soft_mask(np.abs(d - R), 1.6, 1.6)

    # Raphe: central slit along the long axis.
    raphe = _soft_mask(np.abs(dx), 1.1, 1.2) * _soft_mask(np.abs(dy), R * 0.8, 2.0)
    # Transverse striae (comb teeth), gently curved toward apices.
    curve = 0.02 * dy**2 / size
    striae = 0.5 + 0.5 * np.cos(2 * np.pi * striae_freq * (dy + curve) / size)
    striae = np.power(striae, 3.0) * face
    # Pores along striae + polar nodules at the tips.
    pores = ((striae > 0.6) & (rng.random((size, size)) > 0.5)).astype(np.float32) * face
    tips = _soft_mask(np.abs(np.abs(dy) - R * 0.82), 2.0, 1.8) * _soft_mask(np.abs(dx), 3.0, 2.0)
    centre = _soft_mask(np.sqrt(dx**2 + dy**2), 2.5, 1.5)

    structure = np.clip(0.5 * striae + 0.5 * pores + 0.7 * raphe + 0.6 * tips + 0.5 * centre, 0, 1)

    rgb = _paint(structure, face, rim, palette, rng)
    alpha = np.clip(face + rim * 0.9, 0, 1)
    return np.dstack([rgb, alpha]).astype(np.float32)


def _paint(structure, face, rim, palette, rng) -> np.ndarray:
    """Map structure height -> pigment. Gold (fucoxanthin) inside, glass blue rim."""
    h, w = face.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if palette == "gold-glass":
        deep = np.array([0.45, 0.28, 0.08])  # umber
        gold = np.array([0.95, 0.72, 0.25])  # fucoxanthin gold
        glass = np.array([0.55, 0.85, 0.95])  # silica blue
    elif palette == "abyss":
        deep = np.array([0.05, 0.12, 0.25])
        gold = np.array([0.20, 0.65, 0.80])
        glass = np.array([0.75, 0.95, 1.0])
    else:  # bloom
        deep = np.array([0.10, 0.30, 0.12])
        gold = np.array([0.55, 0.90, 0.35])
        glass = np.array([0.85, 0.95, 0.70])
    tint = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    rgb += face[..., None] * (deep + (gold - deep) * structure[..., None]) * tint
    rgb += rim[..., None] * glass[None, None, :] * 0.9
    return np.clip(rgb, 0, 1)


DIATOM_FAMILIES = ("centric", "pennate")


def sample_target(
    size: int = 48, seed: int = 0, family: str | None = None, palette: str = "gold-glass"
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    family = family or str(rng.choice(list(DIATOM_FAMILIES)))
    if family == "centric":
        return centric_diatom(
            size=size,
            n_fold=int(rng.integers(5, 13)),
            n_rings=int(rng.integers(2, 5)),
            seed=seed,
            palette=palette,
        )
    return pennate_diatom(
        size=size,
        aspect=float(rng.uniform(1.8, 3.0)),
        striae_freq=float(rng.uniform(7, 12)),
        seed=seed,
        palette=palette,
    )


def batch_targets(
    batch: int, size: int = 48, seed: int = 0, num_channels: int = 16
) -> "torch.Tensor":
    """Procedural target batch as NCA state (B,C,H,W); extra channels zero."""
    import torch

    states = torch.zeros(batch, num_channels, size, size)
    for i in range(batch):
        t = sample_target(size=size, seed=seed + i * 977)
        t = torch.from_numpy(t).permute(2, 0, 1)  # (4,H,W)
        states[i, :4] = t
    return states

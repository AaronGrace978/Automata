"""Morph pose: the numbers the 3D frustule is built from.

``electron/renderer/morph.js`` is the same map. ``tests/test_morph_parity.py``
fails if the two diverge. Speech changes the body through the same keyword
scan the talking NCA already used for fire rate.
"""
from __future__ import annotations


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def modulation_from_text(text: str) -> tuple[float, float]:
    """Words back into growth: (fire_rate multiplier, audio-energy bias)."""
    t = (text or "").lower()
    fire, energy = 1.0, 0.0
    for w in ("bloom", "spread", "more", "now", "motion", "ecstatic", "quickens"):
        if w in t:
            fire += 0.15
    for w in ("rest", "still", "quiet", "sleep", "waiting", "quiescent"):
        if w in t:
            fire -= 0.15
    for w in ("tore", "torn", "pain", "damage", "wounded"):
        if w in t:
            fire -= 0.2
    for w in ("pulse", "beat", "drum", "sing", "tide"):
        if w in t:
            energy += 0.2
    return max(0.3, min(2.0, fire)), max(0.0, min(1.0, energy))


def pose_from_state(
    rid: list[float],
    utterance: str = "",
    user_text: str = "",
    genome_fold: int = 8,
    damage_pulse: float = 0.0,
) -> dict:
    """Centric-frustule controls in a fixed, unitless range.

    ``folds`` is the genome (5–16). Everything else moves when the creature
    feels or when someone speaks to it.
    """
    vals = list(rid) + [0.0] * 6
    a, v, p, h, r, c = (float(vals[i]) for i in range(6))
    fire, energy = modulation_from_text(f"{utterance} {user_text}")
    folds = int(genome_fold)
    if folds < 5:
        folds = 5
    if folds > 16:
        folds = 16
    girdle = _clip(0.22 + 0.20 * h + 0.12 * (fire - 1.0) + 0.10 * a, 0.08, 0.72)
    dome = _clip(0.16 + 0.22 * v + 0.08 * c, 0.04, 0.55)
    pores = _clip(0.40 + 0.35 * r + 0.25 * energy, 0.08, 1.0)
    ribs = _clip(0.45 + 0.30 * max(a, 0.0) + 0.25 * (fire - 1.0), 0.08, 1.0)
    asymmetry = _clip(0.65 * max(p, 0.0) + 0.35 * max(-v, 0.0), 0.0, 1.0)
    spin = _clip(0.20 + 0.90 * max(r, 0.0) + 0.45 * max(a, 0.0), 0.0, 2.2)
    damage = _clip(0.85 * max(float(damage_pulse), 0.0) + 0.55 * max(p, 0.0), 0.0, 1.0)
    hue = _clip(0.11 + 0.06 * v - 0.08 * max(p, 0.0) + 0.02 * c, 0.02, 0.20)
    bloom = _clip(0.35 + 0.40 * max(v, 0.0) + 0.25 * energy, 0.1, 1.0)
    return {
        "folds": folds,
        "girdle": girdle,
        "dome": dome,
        "pores": pores,
        "ribs": ribs,
        "asymmetry": asymmetry,
        "spin": spin,
        "damage": damage,
        "hue": hue,
        "bloom": bloom,
        "fire": fire,
        "energy": energy,
    }

"""The felt layer: RID — Reactive Instinct Drive.

Not a mood ring: six scalar drives in [-1,1] with momentum, computed from
the body's own signals every step. This is what the creature "feels," and
it is the only thing the language backbone is allowed to talk about.
Grounding by construction: no drive, no sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _clip(v: float) -> float:
    return max(-1.0, min(1.0, v))


@dataclass
class InstinctState:
    arousal: float = 0.0   # growth + sound energy: how much is happening
    valence: float = 0.0   # symmetry + brightness: is it going well
    pain: float = 0.0      # sudden mass loss: damage
    hunger: float = 0.0    # small + still: drive to expand
    rhythm: float = 0.0    # beat-phase alignment: entrainment to sound
    calm: float = 0.0      # stillness + wholeness: rest
    momentum: float = field(default=0.6, repr=False)

    @property
    def rid(self) -> list[float]:
        """The RID vector: the creature's felt state, six numbers."""
        return [self.arousal, self.valence, self.pain, self.hunger, self.rhythm, self.calm]

    def update(self, desc: dict, growth: float, audio_energy: float = 0.0, beat: float = 0.0, rel_drop: float = 0.0) -> "InstinctState":
        m = self.momentum
        blend = lambda old, new: _clip(old * m + new * (1 - m))
        self.arousal = blend(self.arousal, growth * 30.0 + audio_energy * 1.5)
        self.valence = blend(self.valence, desc["symmetry"] * 1.2 - 0.4 + sum(desc["rgb"]) / 3)
        self.pain = blend(self.pain, 1.0 if (growth < -0.02 or rel_drop > 0.3) else 0.0)
        self.hunger = blend(self.hunger, 0.8 if (desc["mass"] < 0.15 and abs(growth) < 0.003) else -0.5)
        self.rhythm = blend(self.rhythm, audio_energy * (0.5 + 0.5 * (1 - abs(beat - 0.5) * 2)))
        self.calm = blend(self.calm, 0.8 if (abs(growth) < 0.002 and growth > -0.005) else -0.6)
        return self

    def to_text(self) -> str:
        bits = []
        bits.append("I am still." if self.calm > 0.3 else "I am restless." if self.arousal > 0.3 else "I drift.")
        if self.pain > 0.2:
            bits.append("Something tore me.")
        elif self.valence > 0.3:
            bits.append("My glass holds.")
        if self.hunger > 0.3:
            bits.append("I want to spread.")
        if self.rhythm > 0.4:
            bits.append("I feel the pulse.")
        return " ".join(bits)

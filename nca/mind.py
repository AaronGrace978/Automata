"""DiatomMind: the talking NCA as one component.

Stack, bottom to top:
  body (DiatomNCA) → visual module (encoder + predicates) → signal layer
  (instinct / RID) → language backbone (narrator or small LM) → persona.

And the loop closes: the creature's own words modulate its next growth
(excited speech quickens the fire rate; pain slows it), while its song
still sings in parallel. It sees itself, feels itself, speaks, and its
speech moves its body.
"""
from __future__ import annotations

import torch

from .backbone import GroundedNarrator
from .instinct import InstinctState
from .model import DiatomNCA
from .persona import get_persona
from .vision import VisualEncoder, predicates
from .voice import frame_descriptor


def modulation_from_text(text: str) -> tuple[float, float]:
    """Words back into growth: (fire_rate multiplier, audio-energy bias).

    Deterministic keyword scan — the smallest possible language-to-body map.
    """
    t = text.lower()
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


class DiatomMind:
    def __init__(
        self,
        model: DiatomNCA | None = None,
        persona: str = "diatom_elder",
        backbone=None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = model or DiatomNCA()
        self.model.to(device)
        self.encoder = VisualEncoder().to(device)
        self.instinct = InstinctState()
        self.persona = get_persona(persona)
        self.backbone = backbone or GroundedNarrator(seed=seed)
        self._prev_mass: float | None = None
        self.memory: list[str] = []  # past utterances; the creature remembers

    def observe(self, x: torch.Tensor, audio_energy: float = 0.0, beat: float = 0.0) -> dict:
        with torch.no_grad():
            desc = frame_descriptor(x[0])
            emb = self.encoder(x).detach()
        mass = desc["mass"]
        growth = 0.0 if self._prev_mass is None else mass - self._prev_mass
        rel_drop = 0.0
        if self._prev_mass is not None and self._prev_mass > 1e-6 and growth < 0:
            rel_drop = -growth / self._prev_mass
        self._prev_mass = mass
        self.instinct.update(desc, growth, audio_energy, beat, rel_drop)
        return {
            "descriptor": desc,
            "embedding": emb,
            "growth": growth,
            "predicates": predicates(desc, growth, rel_drop),
            "instinct_text": self.instinct.to_text(),
            "rid": self.instinct.rid,
        }

    def utter(self, obs: dict) -> str:
        context = {
            "instinct_text": obs["instinct_text"],
            "predicates": obs["predicates"],
            "rid": obs["rid"],
            "persona_preamble": self.persona.preamble,
            "memory": self.memory[-3:],
        }
        raw = self.backbone.speak(context)
        said = self.persona.voice(raw)
        self.memory.append(said)
        return said

    def run(self, steps: int = 96, size: int = 48, speak_every: int = 24, audio=None) -> dict:
        """Grow, feel, and speak. Returns transcript + trajectory + final state."""
        self.model.eval()
        x = self.model.seed(1, size, device=self.device)
        traj, transcript = [x.clone()], []
        fire_mult, energy_bias = 1.0, 0.0
        import torch as _t

        with _t.no_grad():
            for t in range(steps):
                a = audio[t] if audio is not None else None
                eng = float(a[0, 0]) if a is not None else energy_bias
                x = self.model.update(x, audio=a, fire_rate=0.5 * fire_mult)
                obs = self.observe(x, audio_energy=eng)
                if (t + 1) % speak_every == 0:
                    said = self.utter(obs)
                    transcript.append({"step": t + 1, "said": said, "rid": obs["rid"]})
                    fire_mult, energy_bias = modulation_from_text(said)
                traj.append(x.clone())
        return {
            "transcript": transcript,
            "trajectory": _t.stack(traj, dim=1),
            "final": x,
            "rid": self.instinct.rid,
        }

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
from .glyph import render_text, words_of
from .instinct import InstinctState
from .model import DiatomNCA
from .morph import modulation_from_text, pose_from_state
from .persona import get_persona
from .vision import VisualEncoder, predicates
from .voice import frame_descriptor


class DiatomMind:
    def __init__(
        self,
        model: DiatomNCA | None = None,
        persona: str = "diatom_elder",
        backbone=None,
        seed: int = 0,
        device: str = "cpu",
        fold: int = 8,
    ) -> None:
        self.device = device
        self.model = model or DiatomNCA()
        self.model.to(device)
        self.encoder = VisualEncoder().to(device)
        self.instinct = InstinctState()
        self.persona = get_persona(persona)
        self.backbone = backbone or GroundedNarrator(seed=seed)
        self.fold = fold
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
            "predicates": predicates(desc, growth, rel_drop, pain=self.instinct.pain),
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

    def reply(self, user_text: str, obs: dict | None = None, damage_pulse: float = 0.0) -> dict:
        """One turn of conversation. The pose is what the 3D frustule builds."""
        if obs is None:
            obs = {
                "instinct_text": self.instinct.to_text(),
                "predicates": [],
                "rid": list(self.instinct.rid),
            }
        shape = None
        try:
            from .shapes import default_index, find_shape, label

            _, cp, _ = find_shape(user_text, default_index())
            if cp is not None:
                shape = {"codepoint": cp, "emoji": chr(cp), "label": label(cp)}
        except ImportError:
            pass
        predicates = list(obs.get("predicates") or [])
        if shape:
            predicates.insert(0, f"becoming {shape['label'].lower()}")
        context = {
            "instinct_text": obs["instinct_text"],
            "predicates": predicates,
            "rid": obs["rid"],
            "persona_preamble": self.persona.preamble,
            "memory": self.memory[-3:],
            "user_text": user_text,
        }
        raw = self.backbone.speak(context)
        said = self.persona.voice(raw)
        self.memory.append(said)
        pose = pose_from_state(obs["rid"], said, user_text, self.fold, damage_pulse)
        return {
            "said": said,
            # A named shape is held; otherwise the body spells the reply.
            "shape": shape,
            "words": [] if shape else words_of(said),
            "pose": pose,
            "fire": pose["fire"],
            "energy": pose["energy"],
        }

    def become(self, x: torch.Tensor, codepoint: int, steps: int = 80, fire_rate: float = 0.5,
               size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Grow the body into an emoji silhouette. Returns (trajectory, final)."""
        from .shapes import render_shape

        size = size or x.shape[-1]
        tpl = torch.from_numpy(render_shape(codepoint, size=size)).unsqueeze(0).to(x.device)
        traj = [x.clone()]
        with torch.no_grad():
            for _ in range(steps):
                x = self.model.update(x, fire_rate=fire_rate, template=tpl)
                traj.append(x.clone())
        return torch.stack(traj, dim=1), x

    def speak_with_body(self, x: torch.Tensor, said: str, size: int | None = None,
                        morph_steps: int = 40, hold_steps: int = 24, rest_steps: int = 30,
                        fire_rate: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """Grow the body through the words of ``said`` and back to glass.

        Returns (trajectory (1,T,C,H,W), final state). This is the same
        schedule the desktop app runs: template on for morph+hold steps per
        word, then cleared so the frustule returns.
        """
        size = size or x.shape[-1]
        traj = [x.clone()]
        with torch.no_grad():
            for word in words_of(said):
                tpl = torch.from_numpy(render_text(word, size=size)).unsqueeze(0).to(x.device)
                for _ in range(morph_steps + hold_steps):
                    x = self.model.update(x, fire_rate=fire_rate, template=tpl)
                    traj.append(x.clone())
            clear = torch.zeros(1, size, size, device=x.device)
            for _ in range(rest_steps):
                x = self.model.update(x, fire_rate=fire_rate, template=clear)
                traj.append(x.clone())
        return torch.stack(traj, dim=1), x

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
                    pose = pose_from_state(obs["rid"], said, "", self.fold, 0.0)
                    transcript.append({"step": t + 1, "said": said, "rid": obs["rid"], "pose": pose})
                    fire_mult, energy_bias = pose["fire"], pose["energy"]
                traj.append(x.clone())
        return {
            "transcript": transcript,
            "trajectory": _t.stack(traj, dim=1),
            "final": x,
            "rid": self.instinct.rid,
        }

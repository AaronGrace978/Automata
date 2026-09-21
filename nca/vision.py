"""Eyes: a tiny visual module that reads the body into vectors and words.

Two readouts, no training required:
  embed(x)      — a small CNN maps the 16-channel grid to a compact vector.
                  This is what the language backbone "sees".
  predicates(d) — symbolic words (growing, wounded, symmetric, dim, …) from
                  frame descriptors. This is what grounds the narrator so it
                  cannot hallucinate a body it doesn't have.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """16-channel grid -> dense body embedding."""

    def __init__(self, in_channels: int = 16, embed_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def predicates(desc: dict, growth: float, rel_drop: float = 0.0, pain: float = 0.0) -> list[str]:
    """Grounding words for the narrator. desc: frame_descriptor output.

    `pain` is the lingering instinct drive, so the body reports "healing"
    while the wound still aches — body words and felt words stay consistent.
    """
    words: list[str] = []
    m = desc["mass"]
    words.append("dormant" if m < 0.05 else "small" if m < 0.2 else "grown" if m < 0.5 else "vast")
    words.append("growing" if growth > 0.002 else "shrinking" if growth < -0.005 else "still")
    wounded = growth < -0.02 or rel_drop > 0.3
    if wounded:
        words.append("wounded")
    elif pain > 0.2 or (0.0 < growth <= 0.002 and m > 0.1):
        words.append("healing")
    else:
        words.append("whole")
    words.append("symmetric" if desc["symmetry"] > 0.75 else "asymmetric")
    bright = sum(desc["rgb"]) / 3
    words.append("luminous" if bright > 0.45 else "dim" if bright < 0.2 else "glowing")
    words.append("wide" if desc["spread"] > 0.4 else "compact")
    words.append("leaning-left" if desc["pan"] < 0.4 else "leaning-right" if desc["pan"] > 0.6 else "centred")
    return words

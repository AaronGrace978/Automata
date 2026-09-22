"""One exchange with the creature.

Uses the local allow-listed model when Ollama is up. Otherwise the grounded
narrator answers, and either way you get the morph pose the 3D frustule wears.
"""
from __future__ import annotations

import argparse
import json

import torch

from nca.backbone import OllamaBackbone
from nca.mind import DiatomMind
from nca.model import DiatomNCA
from nca.persona import PERSONAS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="Who are you?")
    ap.add_argument("--persona", default="diatom_elder", choices=sorted(PERSONAS))
    args = ap.parse_args()

    mind = DiatomMind(DiatomNCA(), persona=args.persona, backbone=OllamaBackbone())
    x = mind.model.seed(1, 32)
    with torch.no_grad():
        x = mind.model(x, steps=16)
    out = mind.reply(args.text, mind.observe(x))
    print(out["said"])
    print(json.dumps(out["pose"], indent=2))


if __name__ == "__main__":
    main()

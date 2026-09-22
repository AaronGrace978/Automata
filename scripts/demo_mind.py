"""A conversation with the creature: grow, wound, and listen to what it says."""
from __future__ import annotations

import argparse
import os

import torch

from nca.mind import DiatomMind
from nca.model import DiatomNCA
from nca.utils import half_damage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--persona", type=str, default="diatom_elder",
                    choices=["diatom_elder", "lab_assistant", "feral_bloom"])
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--ollama", action="store_true",
                    help="Speak with the local allow-listed model. Falls back to the body if Ollama is down.")
    args = ap.parse_args()

    model = DiatomNCA()
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    backbone = None
    if args.ollama:
        from nca.backbone import OllamaBackbone
        backbone = OllamaBackbone()
    mind = DiatomMind(model, persona=args.persona, backbone=backbone)

    print(f"── {args.persona} wakes ──")
    out = mind.run(args.steps // 2, size=args.size, speak_every=args.steps // 4)
    x = out["final"]
    for line in out["transcript"]:
        print(f"[t={line['step']}] {line['said']}")

    print("── ✂️  surgery ──")
    with torch.no_grad():
        x = half_damage(x)
        obs = mind.observe(x)
        print(f"[wounded] {mind.utter(obs)}")
        for _ in range(args.steps // 2):
            x = model.update(x)
        print(f"[healed?] {mind.utter(mind.observe(x))}")


if __name__ == "__main__":
    main()

"""Paper-grade healing score. Needs the trained checkpoint from Colab.

    python scripts/score_regeneration.py --checkpoint diatom_nca.pt --n 10
"""
from __future__ import annotations

import argparse

import torch

from nca.eval import regeneration_iou
from nca.model import DiatomNCA
from nca.utils import circle_damage, half_damage


def trial(model, size, steps, damage, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        twin = model(model.seed(1, size), steps=steps)
        healed = model(damage(twin.clone()), steps=steps)
    return regeneration_iou(healed, twin)


def summarize(name, scores):
    t = torch.tensor(scores)
    print(f"{name:22s}  n={len(scores):2d}  mean={t.mean():.3f}  std={t.std(unbiased=False):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    trained = DiatomNCA()
    trained.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    trained.eval()
    fresh = DiatomNCA()
    fresh.eval()

    for label, model in (("trained", trained), ("untrained", fresh)):
        half = [trial(model, args.size, args.steps, half_damage, i) for i in range(args.n)]
        disc = [trial(model, args.size, args.steps, circle_damage, 100 + i) for i in range(args.n)]
        summarize(f"{label} half-cut", half)
        summarize(f"{label} disc", disc)


if __name__ == "__main__":
    main()

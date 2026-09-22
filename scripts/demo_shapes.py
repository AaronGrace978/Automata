"""The body becomes shapes.

    python scripts/demo_shapes.py --say "cloud" "dino" "become a dragon" "cat"

Grows the frustule, then for each thing said finds its silhouette
(nca/shapes.py) and lets the same cells grow into it, then back to glass.
Writes assets/shapes.gif and assets/demo/shapes_strip.png.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import torch
from PIL import Image

from nca.mind import DiatomMind
from nca.model import DiatomNCA
from nca.shapes import default_index, find_shape, label
from nca.utils import save_gif, state_to_rgb, trajectory_to_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="assets/checkpoint_words.pt")
    ap.add_argument("--say", nargs="+", default=["cloud", "dino", "become a dragon", "cat"])
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--grow", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--rest", type=int, default=60)
    ap.add_argument("--gif", default="assets/shapes.gif")
    ap.add_argument("--strip", default="assets/demo/shapes_strip.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = DiatomNCA()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    mind = DiatomMind(model)
    index = default_index()
    empty = torch.zeros(1, args.size, args.size)

    x = model.seed(1, args.size)
    parts, picks, names = [], [], []
    with torch.no_grad():
        grown = [x.clone()]
        for _ in range(args.grow):
            x = model.update(x, template=empty)
            grown.append(x.clone())
    parts.append(torch.stack(grown, dim=1))
    picks.append(args.grow)
    t = args.grow
    for phrase in args.say:
        _, cp, _ = find_shape(phrase, index)
        if cp is None:
            print(f"no shape for {phrase!r}; skipped")
            continue
        traj, x = mind.become(x, cp, steps=args.steps, size=args.size)
        parts.append(traj[:, 1:])
        t += args.steps
        picks.append(t)
        names.append(f"{phrase} -> {label(cp)}")
    with torch.no_grad():
        rest = []
        for _ in range(args.rest):
            x = model.update(x, template=empty)
            rest.append(x.clone())
    parts.append(torch.stack(rest, dim=1))
    picks.append(t + args.rest)
    full = torch.cat(parts, dim=1)

    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
    save_gif(trajectory_to_frames(full), args.gif, fps=16)
    tiles = [np.kron(state_to_rgb(full[0, i]), np.ones((4, 4, 1), dtype=np.uint8)) for i in picks]
    os.makedirs(os.path.dirname(args.strip) or ".", exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(args.strip)
    print(f"saved {args.gif} and {args.strip}: frustule -> {' -> '.join(names)} -> frustule")


if __name__ == "__main__":
    main()

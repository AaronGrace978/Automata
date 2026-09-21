"""Grow + visualise: python scripts/demo_grow.py [--checkpoint assets/checkpoint.pt]."""
from __future__ import annotations

import argparse
import os

import torch

from nca.model import DiatomNCA
from nca.utils import save_gif, save_grid, trajectory_to_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--out", type=str, default="assets/demo_grow.gif")
    args = ap.parse_args()

    model = DiatomNCA()
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"loaded {args.checkpoint}")
    else:
        print("no checkpoint — growing from random weights (primordial soup)")

    with torch.no_grad():
        traj = model.grow(args.steps, size=args.size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_gif(trajectory_to_frames(traj), args.out)
    save_grid(traj[0, -8:], "assets/demo_grow_grid.png")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

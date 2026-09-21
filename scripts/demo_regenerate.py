"""Regeneration probe: grow -> cut in half -> regrow. Saves assets/demo_regenerate.gif."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from nca.model import DiatomNCA
from nca.utils import half_damage, save_gif, state_to_rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--grow-steps", type=int, default=64)
    ap.add_argument("--heal-steps", type=int, default=64)
    ap.add_argument("--size", type=int, default=48)
    args = ap.parse_args()

    model = DiatomNCA()
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
        print(f"loaded {args.checkpoint}")
    else:
        print("no checkpoint — probing the untrained creature")

    with torch.no_grad():
        x = model.seed(1, args.size)
        frames = []
        for _ in range(args.grow_steps):
            x = model.update(x)
            frames.append(state_to_rgb(x)[0])
        x = half_damage(x)  # the cut
        for _ in range(8):
            frames.append(state_to_rgb(x)[0])
        for _ in range(args.heal_steps):
            x = model.update(x)
            frames.append(state_to_rgb(x)[0])
    os.makedirs("assets", exist_ok=True)
    save_gif(np.stack(frames), "assets/demo_regenerate.gif")
    print("saved assets/demo_regenerate.gif")


if __name__ == "__main__":
    main()

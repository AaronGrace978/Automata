"""The body becomes words.

    python scripts/demo_words.py --checkpoint assets/checkpoint_words.pt --text "glass holds"

Grows the frustule, then writes each word into the silica template and lets
the same cells rebuild themselves as the letters, then return to glass.
Writes assets/words.gif and assets/demo/words_strip.png.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os

import numpy as np
import torch
from PIL import Image

from nca.glyph import words_of
from nca.mind import DiatomMind
from nca.model import DiatomNCA
from nca.utils import save_gif, state_to_rgb, trajectory_to_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="assets/checkpoint_words.pt")
    ap.add_argument("--text", default="glass holds")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--grow", type=int, default=64)
    ap.add_argument("--morph", type=int, default=40)
    ap.add_argument("--hold", type=int, default=24)
    ap.add_argument("--rest", type=int, default=40)
    ap.add_argument("--gif", default="assets/words.gif")
    ap.add_argument("--strip", default="assets/demo/words_strip.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = DiatomNCA()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    mind = DiatomMind(model)
    x = model.seed(1, args.size)
    with torch.no_grad():
        grown = [x.clone()]
        for _ in range(args.grow):
            x = model.update(x, template=torch.zeros(1, args.size, args.size))
            grown.append(x.clone())
    traj, x = mind.speak_with_body(
        x, args.text, size=args.size, morph_steps=args.morph, hold_steps=args.hold, rest_steps=args.rest,
    )
    full = torch.cat([torch.stack(grown, dim=1), traj[:, 1:]], dim=1)
    frames = trajectory_to_frames(full)
    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
    save_gif(frames, args.gif, fps=16)
    print(f"saved {args.gif} ({full.shape[1]} frames)")

    # Strip: grown frustule, then the end of each word's hold, then the return.
    picks = [args.grow]
    per_word = args.morph + args.hold
    for i, _word in enumerate(words_of(args.text)):
        picks.append(args.grow + (i + 1) * per_word - 1)
    picks.append(full.shape[1] - 1)
    tiles = [state_to_rgb(full[0, t]) for t in picks]
    scale = 4
    strip = np.concatenate([np.kron(t, np.ones((scale, scale, 1), dtype=np.uint8)) for t in tiles], axis=1)
    os.makedirs(os.path.dirname(args.strip) or ".", exist_ok=True)
    Image.fromarray(strip).save(args.strip)
    print(f"saved {args.strip}: frustule -> {' -> '.join(words_of(args.text))} -> frustule")


if __name__ == "__main__":
    main()

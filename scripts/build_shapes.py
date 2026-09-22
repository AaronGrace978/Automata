"""Bake emoji silhouettes for the desktop app.

    python scripts/build_shapes.py

Writes electron/renderer/shapes.data.js: an atlas of 48x48 masks (grey =
alpha), keyword -> tile, and the request grammar, so the app grows into the
same silhouettes the rule was trained on and matches words the same way,
with no emoji font needed at run time.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import base64
import io
import json

import numpy as np
from PIL import Image

from nca.shapes import ASK, STOP, build_index, emoji_font, font_codepoints, label, render_shape


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--cols", type=int, default=48)
    ap.add_argument("--out", default="electron/renderer/shapes")
    args = ap.parse_args()
    font = emoji_font()
    if not font:
        raise SystemExit("no emoji font: apt install fonts-noto-color-emoji")
    cps = font_codepoints(font)
    masks, kept = [], []
    for cp in cps:
        m = render_shape(cp, size=args.size, font_path=font)
        if (m > 0.5).sum() >= 20:
            masks.append(m)
            kept.append(cp)
    index = build_index(kept)
    tile = {cp: i for i, cp in enumerate(kept)}
    rows = (len(kept) + args.cols - 1) // args.cols
    atlas = np.zeros((rows * args.size, args.cols * args.size), dtype=np.uint8)
    for i, m in enumerate(masks):
        r, c = divmod(i, args.cols)
        atlas[r * args.size : (r + 1) * args.size, c * args.size : (c + 1) * args.size] = np.round(m * 255)
    buf = io.BytesIO()
    Image.fromarray(atlas, "L").save(buf, format="PNG", optimize=True)
    payload = {
        "size": args.size,
        "cols": args.cols,
        "font": "Noto Color Emoji (SIL OFL-1.1)",
        "ask": ASK.pattern,
        "stop": sorted(STOP),
        "tiles": [[cp, label(cp)] for cp in kept],
        "index": {k: tile[cp] for k, cp in sorted(index.items()) if cp in tile},
        # A data: URL, because a file:// image would taint the canvas it is read from.
        "atlas": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
    }
    out = args.out + ".data.js"
    with open(out, "w") as f:
        f.write("// Built by scripts/build_shapes.py. Do not edit.\nwindow.DIATOM_SHAPES = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"{len(kept)} shapes, {len(payload['index'])} keywords -> {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

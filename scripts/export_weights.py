"""Export torch weights -> JSON for the browser demo (web/demo.html)."""
from __future__ import annotations

import argparse
import json
import os

import torch

from nca.model import DiatomNCA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="assets/checkpoint.pt")
    ap.add_argument("--out", type=str, default="web/weights.json")
    args = ap.parse_args()
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"missing {args.checkpoint} — train first: python scripts/train_diatom.py")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt["state_dict"]
    payload = {
        "num_channels": 16,
        "hidden_dim": sd["fc1.weight"].shape[0],
        "audio_dim": 8,
        "fc1_weight": sd["fc1.weight"].tolist(),
        "fc1_bias": sd["fc1.bias"].tolist(),
        "film_weight": sd["audio_encoder.weight"].tolist(),
        "film_bias": sd["audio_encoder.bias"].tolist(),
        "fc2_weight": sd["fc2.weight"].tolist(),
        "fc2_bias": sd["fc2.bias"].tolist(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"exported {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()

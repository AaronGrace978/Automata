"""Fine-tune the diatom so its body morphs into words.

    python scripts/train_words.py --init assets/checkpoint.pt --steps 3000 --device cuda

Then export for the desktop creature:

    python scripts/export_weights.py --checkpoint assets/checkpoint_words.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from nca.model import DiatomNCA
from nca.train import TrainConfig, train_morph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="assets/checkpoint.pt")
    ap.add_argument("--out", type=str, default="assets/checkpoint_words.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout-min", type=int, default=24)
    ap.add_argument("--rollout-max", type=int, default=48)
    ap.add_argument("--word-prob", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    model = DiatomNCA()
    if args.init and os.path.exists(args.init):
        model.load_state_dict(torch.load(args.init, map_location="cpu")["state_dict"])
        print(f"starting from {args.init}")
    cfg = TrainConfig(
        size=args.size, steps=args.steps, batch=args.batch, lr=args.lr,
        rollout_min=args.rollout_min, rollout_max=args.rollout_max,
        word_prob=args.word_prob, device=args.device, log_every=args.log_every,
    )

    def on_log(it: int, loss: float, _pool) -> None:
        print(f"[{it}/{cfg.steps}] loss={loss:.5f}", flush=True)
        if it % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "config": vars(cfg)}, args.out)

    train_morph(model, cfg, on_log=on_log)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

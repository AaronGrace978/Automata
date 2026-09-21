"""Train a diatom-growing NCA: python scripts/train_diatom.py --steps 3000."""
from __future__ import annotations

import argparse
import os

import torch

from nca.model import DiatomNCA
from nca.train import TrainConfig, train
from nca.utils import save_gif, trajectory_to_frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Grow diatoms from local rules.")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--damage-prob", type=float, default=0.5)
    ap.add_argument("--audio-prob", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="assets/checkpoint.pt")
    args = ap.parse_args()

    cfg = TrainConfig(
        size=args.size,
        hidden_dim=args.hidden,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        damage_prob=args.damage_prob,
        audio_prob=args.audio_prob,
    )
    model = DiatomNCA(num_channels=cfg.num_channels, hidden_dim=cfg.hidden_dim)

    def on_log(it: int, loss: float, sample: torch.Tensor) -> None:
        print(f"[{it}/{cfg.steps}] loss={loss:.5f}", flush=True)

    train(model, cfg, on_log=on_log)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)}, args.out)
    print(f"saved {args.out}")

    with torch.no_grad():
        traj = model.grow(96, size=args.size)
    save_gif(trajectory_to_frames(traj), "assets/growth.gif")
    print("saved assets/growth.gif")


if __name__ == "__main__":
    main()

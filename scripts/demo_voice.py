"""Let the creature sing: grow -> sonify -> creature listens to itself."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from nca import audio as audio_lib
from nca.model import DiatomNCA
from nca.utils import save_gif, trajectory_to_frames
from nca.voice import save_wav, sing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--size", type=int, default=48)
    args = ap.parse_args()

    model = DiatomNCA()
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])

    with torch.no_grad():
        traj = model.grow(args.steps, size=args.size)  # (B,T,C,H,W)
    song = sing(traj[0])
    os.makedirs("assets", exist_ok=True)
    save_wav("assets/demo_voice.wav", song)
    print(f"saved assets/demo_voice.wav ({len(song)/22050:.1f}s stereo)")

    # Call and response: the creature hears its own song and grows again.
    feats = audio_lib.smooth(audio_lib.stft_features(song.mean(axis=1).astype(np.float32)))
    idx = (np.linspace(0, len(feats) - 1, args.steps)).astype(int)
    with torch.no_grad():
        x = model.seed(1, args.size)
        frames = []
        from nca.utils import state_to_rgb

        for tstep in range(args.steps):
            a = torch.from_numpy(feats[idx[tstep]]).unsqueeze(0)
            x = model.update(x, audio=a)
            frames.append(state_to_rgb(x)[0])
    save_gif(np.stack(frames), "assets/demo_self_listening.gif")
    print("saved assets/demo_self_listening.gif (grown under its own song)")


if __name__ == "__main__":
    main()

"""Audio-reactive growth: synth sound bends the local rule. Saves assets/demo_audio.gif."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from nca import audio as audio_lib
from nca.model import DiatomNCA, N_AUDIO_DIMS
from nca.utils import save_gif, state_to_rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--kind", type=str, default="beat", choices=["beat", "sweep", "bloom"])
    ap.add_argument("--wav", type=str, default=None, help="optional real audio file")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--steps", type=int, default=128)
    args = ap.parse_args()

    model = DiatomNCA()
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])

    if args.wav and os.path.exists(args.wav):
        wave, sr = audio_lib.load_wav(args.wav)
        feats = audio_lib.stft_features(wave, sr)
        print(f"conditioning on {args.wav}: {feats.shape[0]} audio frames")
    else:
        wave = audio_lib.synth_demo_signal(args.kind)
        feats = audio_lib.stft_features(wave)
        print(f"conditioning on synthetic '{args.kind}' signal")

    feats = audio_lib.smooth(feats)
    # Stretch/compress audio time onto growth time; energy also scales fire rate.
    idx = (np.linspace(0, len(feats) - 1, args.steps)).astype(int)
    with torch.no_grad():
        x = model.seed(1, args.size)
        frames = []
        for t in range(args.steps):
            a = torch.from_numpy(feats[idx[t]]).unsqueeze(0)
            energy = float(a[0, 0])
            x = model.update(x, audio=a, fire_rate=0.3 + 0.6 * min(1.0, energy * 3))
            x[:, 11] = (x[:, 11] * 0.9 + energy * 0.1)  # AUDIO sense channel remembers loudness
            frames.append(state_to_rgb(x)[0])
    os.makedirs("assets", exist_ok=True)
    save_gif(np.stack(frames), "assets/demo_audio.gif")
    print("saved assets/demo_audio.gif")


if __name__ == "__main__":
    main()

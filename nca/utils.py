"""Visualisation + damage + pool utilities shared by training and demos."""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F


def state_to_rgb(state: torch.Tensor) -> np.ndarray:
    """NCA state (B,C,H,W) or (C,H,W) -> uint8 RGB (B,H,W,3) or (H,W,3)."""
    squeeze = False
    if state.dim() == 3:
        state = state.unsqueeze(0)
        squeeze = True
    rgb = state[:, :3].detach().float().cpu().clamp(0, 1)
    alpha = state[:, 3:4].detach().float().cpu().clamp(0, 1)
    # Composite over a dark lab background so alpha reads as growth.
    bg = torch.tensor([0.04, 0.05, 0.08]).view(1, 3, 1, 1)
    img = rgb * alpha + bg * (1 - alpha)
    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return img[0] if squeeze else img


def save_grid(states: torch.Tensor, path: str, nrow: int = 8) -> None:
    from PIL import Image

    imgs = state_to_rgb(states)
    b, h, w, _ = imgs.shape
    ncol = min(nrow, b)
    nrows = (b + ncol - 1) // ncol
    canvas = np.zeros((nrows * h, ncol * w, 3), dtype=np.uint8) + 10
    for i, im in enumerate(imgs):
        canvas[(i // ncol) * h : (i // ncol + 1) * h, (i % ncol) * w : (i % ncol + 1) * w] = im
    Image.fromarray(canvas).save(path)


def save_gif(frames: np.ndarray, path: str, fps: int = 12) -> None:
    """frames: (T,H,W,3) uint8 -> animated GIF."""
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def trajectory_to_frames(traj: torch.Tensor) -> np.ndarray:
    """(B,T,C,H,W) or (T,C,H,W) -> (T,H,W,3) uint8 of batch 0."""
    if traj.dim() == 5:
        traj = traj[0]
    return np.stack([state_to_rgb(s) for s in traj])


# -- damage (regeneration training) ---------------------------------------
def circle_damage(x: torch.Tensor, radius_ratio: float = 0.25, seed: int | None = None) -> torch.Tensor:
    """Knock out a disc — the classic regeneration probe."""
    g = torch.Generator(device=x.device)
    if seed is not None:
        g.manual_seed(seed)
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
    )
    y = torch.randint(0, h, (b,), generator=g, device=x.device)
    xx = xx.unsqueeze(0).expand(b, -1, -1)
    yy = yy.unsqueeze(0).expand(b, -1, -1)
    dist = torch.sqrt(((xx - w // 2) ** 2 + (yy - y[:, None, None]) ** 2).float())
    mask = (dist > max(2, int(radius_ratio * min(h, w)))).float().unsqueeze(1)
    return x * mask


def half_damage(x: torch.Tensor) -> torch.Tensor:
    """Remove the right half — can it re-grow bilateral symmetry?"""
    y = x.clone()
    y[:, :, :, y.shape[3] // 2 :] = 0
    return y


def noise_damage(x: torch.Tensor, p: float = 0.3, seed: int | None = None) -> torch.Tensor:
    g = torch.Generator(device=x.device)
    if seed is not None:
        g.manual_seed(seed)
    mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], generator=g, device=x.device) > p)
    return x * mask.float()

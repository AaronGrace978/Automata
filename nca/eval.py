"""Scores for the two next measurements: regeneration overlap, latent prediction.

regeneration_iou — alive-mask overlap in the cut half, healed vs. an intact twin.
jepa_step — one gradient step predicting the full-body embedding from a masked body.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def regeneration_iou(healed: torch.Tensor, intact: torch.Tensor, alpha: int = 3, thresh: float = 0.1) -> float:
    """IoU of alive cells in the right half. Both tensors are (B,C,H,W) or (C,H,W)."""
    if healed.dim() == 3:
        healed, intact = healed.unsqueeze(0), intact.unsqueeze(0)
    h = healed[:, alpha] > thresh
    i = intact[:, alpha] > thresh
    half = h.shape[-1] // 2
    h, i = h[..., half:], i[..., half:]
    inter = (h & i).sum().float()
    union = (h | i).sum().float().clamp(min=1)
    return float(inter / union)


class LatentPredictor(torch.nn.Module):
    """Predict the full-body embedding from the embedding of a masked body."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))

    def forward(self, z_masked: torch.Tensor) -> torch.Tensor:
        return self.net(z_masked)


def mask_right_half(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., y.shape[-1] // 2 :] = 0
    return y


def jepa_step(encoder, predictor, opt, x: torch.Tensor) -> float:
    """One step: masked embedding -> predicted full embedding. Returns cosine similarity."""
    with torch.no_grad():
        target = encoder(x)
        masked = encoder(mask_right_half(x))
    pred = predictor(masked)
    loss = 1 - F.cosine_similarity(pred, target.detach(), dim=-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(1 - loss.detach())

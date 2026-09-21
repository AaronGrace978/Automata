"""Pool-based NCA training with regeneration + audio conditioning.

Follows Mordvintsev et al. "Growing Neural Cellular Automata" (2020):
a replay pool of organisms, iterated short rollouts, sample replacement of
the worst, periodic damage so regeneration is learned, not luck.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .audio import random_condition
from .diatom import batch_targets
from .model import N_CHANNELS
from .utils import circle_damage, half_damage


@dataclass
class TrainConfig:
    size: int = 48
    num_channels: int = N_CHANNELS
    hidden_dim: int = 128
    steps: int = 3000
    batch: int = 8
    pool_size: int = 64
    rollout_min: int = 32
    rollout_max: int = 64
    lr: float = 2e-3
    damage_prob: float = 0.5
    audio_prob: float = 0.5  # fraction of batches grown under random sound
    seed_pool_every: int = 8  # reseed oldest organism every N iters (diversity)
    log_every: int = 200
    device: str = "cpu"


def organism_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on RGBA + small overflow penalty on latent channels."""
    rgba = F.mse_loss(pred[:, :4], target[:, :4])
    overflow = (pred[:, 4:].abs() - 4.0).clamp(min=0).pow(2).mean()
    return rgba + 0.02 * overflow


def train_step(
    model, opt: torch.optim.Optimizer, pool: torch.Tensor, target: torch.Tensor, cfg: TrainConfig, it: int
) -> tuple[float, torch.Tensor]:
    b = cfg.batch
    idx = torch.randperm(len(pool))[:b]
    x = pool[idx].to(cfg.device)
    target_b = target[idx % len(target)].to(cfg.device) if len(target) != b else target.to(cfg.device)

    # Reseed the worst organism with a fresh zygote so the pool keeps exploring.
    with torch.no_grad():
        losses = F.mse_loss(x[:, :4], target_b[:, :4], reduction="none").mean(dim=(1, 2, 3))
        worst = idx[losses.argmax()]
        pool[worst] = model.seed(1, cfg.size, device="cpu")[0].detach()
        x[losses.argmax()] = pool[worst].to(cfg.device)

    # Damage half the batch: the regeneration creature is made, not born.
    if torch.rand(1).item() < cfg.damage_prob:
        with torch.no_grad():
            n = b // 2
            if torch.rand(1).item() < 0.5:
                x[:n] = circle_damage(x[:n])
            else:
                x[:n] = half_damage(x[:n])

    audio = None
    if torch.rand(1).item() < cfg.audio_prob:
        audio = random_condition(b).to(cfg.device)

    rollout = int(torch.randint(cfg.rollout_min, cfg.rollout_max + 1, (1,)).item())
    x = model(x, audio=audio, steps=rollout)
    loss = organism_loss(x, target_b)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    with torch.no_grad():
        pool[idx] = x.detach().cpu()
    return float(loss.item()), x.detach()


def train(model, cfg: TrainConfig, target_seed: int = 0, on_log=None):
    """Run training; returns pool. on_log(it, loss, sample) for previews."""
    from .model import DiatomNCA  # noqa: F401  (model passed in; import keeps API tidy)

    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pool = model.seed(cfg.pool_size, cfg.size, device="cpu").detach()
    target_pool = batch_targets(cfg.pool_size, size=cfg.size, seed=target_seed).detach()

    for it in range(1, cfg.steps + 1):
        loss, sample = train_step(model, opt, pool, target_pool, cfg, it)
        if it % cfg.log_every == 0 and on_log is not None:
            on_log(it, loss, sample)
    return pool

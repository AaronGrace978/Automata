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
    # Word morphing (train_morph): share of tasks that are words, and how
    # often a sampled organism is handed a new task while keeping its body.
    word_prob: float = 0.5
    retarget_prob: float = 0.35


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


# -- words: the body morphs into text and back ------------------------------
def _new_tasks(n: int, cfg: TrainConfig, rng, target_seed: int):
    """n fresh tasks: (targets (n,C,H,W), templates (n,H,W)). Word or diatom."""
    from .glyph import batch_word_targets, random_word

    targets = torch.zeros(n, cfg.num_channels, cfg.size, cfg.size)
    templates = torch.zeros(n, cfg.size, cfg.size)
    is_word = [rng.random() < cfg.word_prob for _ in range(n)]
    words = [random_word(rng) for w in is_word if w]
    if words:
        w_states, w_masks = batch_word_targets(words, size=cfg.size, num_channels=cfg.num_channels, rng=rng)
    wi = 0
    for i, w in enumerate(is_word):
        if w:
            targets[i], templates[i] = w_states[wi], w_masks[wi]
            wi += 1
        else:
            targets[i] = batch_targets(1, size=cfg.size, seed=target_seed + rng.randrange(1 << 20),
                                       num_channels=cfg.num_channels)[0]
    return targets, templates


def morph_step(model, opt, pool, targets, templates, cfg: TrainConfig, rng) -> float:
    b = cfg.batch
    idx = torch.randperm(len(pool))[:b]
    with torch.no_grad():
        # Hand some organisms a new task without touching their bodies, so the
        # rule learns frustule -> word, word -> word, and word -> frustule.
        swap = [i for i in range(b) if rng.random() < cfg.retarget_prob]
        if swap:
            t, m = _new_tasks(len(swap), cfg, rng, target_seed=0)
            for j, i in enumerate(swap):
                targets[idx[i]], templates[idx[i]] = t[j], m[j]
        # Reseed the worst so the pool keeps a supply of zygotes.
        losses = F.mse_loss(pool[idx][:, :4], targets[idx][:, :4], reduction="none").mean(dim=(1, 2, 3))
        worst = idx[losses.argmax()]
        pool[worst] = model.seed(1, cfg.size, device="cpu")[0].detach()

    x = pool[idx].to(cfg.device)
    target_b = targets[idx].to(cfg.device)
    template_b = templates[idx].to(cfg.device)

    if rng.random() < cfg.damage_prob:
        with torch.no_grad():
            n = b // 2
            x[:n] = circle_damage(x[:n]) if rng.random() < 0.5 else half_damage(x[:n])

    audio = random_condition(b).to(cfg.device) if rng.random() < cfg.audio_prob else None
    rollout = rng.randint(cfg.rollout_min, cfg.rollout_max)
    x = model(x, audio=audio, steps=rollout, template=template_b)
    loss = organism_loss(x, target_b)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        pool[idx] = x.detach().cpu()
    return float(loss.item())


def train_morph(model, cfg: TrainConfig, seed: int = 0, on_log=None):
    """Fine-tune a diatom rule so its body becomes words along a template.

    Start from a grown frustule (``assets/checkpoint.pt``). Half the tasks
    are frustules with an empty template, so the old behaviour is kept.
    """
    import random

    rng = random.Random(seed)
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pool = model.seed(cfg.pool_size, cfg.size, device="cpu").detach()
    targets, templates = _new_tasks(cfg.pool_size, cfg, rng, target_seed=seed)
    for it in range(1, cfg.steps + 1):
        loss = morph_step(model, opt, pool, targets, templates, cfg, rng)
        if it % cfg.log_every == 0 and on_log is not None:
            on_log(it, loss, pool)
    return pool

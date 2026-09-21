"""Smoke tests: local rules hold, diatoms are valid, audio bends the rule."""
import numpy as np
import torch

from nca import audio as audio_lib
from nca.audio import N_AUDIO_DIMS
from nca.diatom import batch_targets, centric_diatom, pennate_diatom
from nca.model import AUDIO_CH, DiatomNCA
from nca.train import TrainConfig, train_step
from nca.utils import circle_damage, half_damage, state_to_rgb


def test_forward_shape_and_alive_mask():
    m = DiatomNCA(num_channels=16, hidden_dim=32)
    x = m.seed(2, 24)
    y = m(x, steps=4)
    assert y.shape == (2, 16, 24, 24)
    assert torch.isfinite(y).all()


def test_silence_vs_sound_diverge():
    m = DiatomNCA(num_channels=16, hidden_dim=32)
    torch.manual_seed(0)
    x = m.seed(1, 24)
    loud = torch.zeros(1, N_AUDIO_DIMS)
    loud[0, 0] = 1.0  # full energy
    silent = torch.zeros(1, N_AUDIO_DIMS)
    torch.manual_seed(1)
    a = m(x.clone(), audio=silent, steps=8)
    torch.manual_seed(1)
    b = m(x.clone(), audio=loud, steps=8)
    assert not torch.allclose(a, b), "audio conditioning must change growth"


def test_diatom_targets_valid():
    for fn in (centric_diatom, pennate_diatom):
        t = fn(size=48, seed=3)
        assert t.shape == (48, 48, 4)
        assert t.min() >= 0 and t.max() <= 1
        assert t[..., 3].max() > 0.5  # has a body
    batch = batch_targets(4, size=32)
    assert batch.shape == (4, 16, 32, 32)


def test_damage_and_regrow_runs():
    m = DiatomNCA(num_channels=16, hidden_dim=32)
    x = m.seed(1, 32)
    x = m(x, steps=16)
    cut = half_damage(x)
    assert cut[:, :, :, 16:].abs().sum() == 0
    healed = m(cut, steps=8)
    assert torch.isfinite(healed).all()
    circ = circle_damage(x)
    assert circ.shape == x.shape


def test_train_step_runs_and_keeps_pool():
    m = DiatomNCA(num_channels=16, hidden_dim=32)
    cfg = TrainConfig(size=32, steps=2, batch=4, pool_size=8, rollout_min=4, rollout_max=6)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    pool = m.seed(cfg.pool_size, cfg.size).detach()
    target = batch_targets(cfg.pool_size, size=cfg.size).detach()
    loss, _ = train_step(m, opt, pool, target, cfg, it=1)
    assert np.isfinite(loss)


def test_audio_features_from_synth():
    wave = audio_lib.synth_demo_signal("beat", seconds=1.0)
    feats = audio_lib.stft_features(wave)
    assert feats.shape[1] == N_AUDIO_DIMS
    assert np.isfinite(feats).all()
    assert feats[:, 0].max() > 0.5  # beat has dynamics


def test_visualisation():
    m = DiatomNCA(num_channels=16, hidden_dim=16)
    x = m.seed(1, 24)
    img = state_to_rgb(x)
    assert img.shape == (1, 24, 24, 3)
    assert img.dtype == np.uint8

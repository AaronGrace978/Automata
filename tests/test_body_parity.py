"""The JavaScript body in the desktop app is the PyTorch rule.

With every cell firing, the two engines are deterministic and must agree on
the same weights, genome, template, and audio.
"""
import json
import os
import subprocess

import torch

from nca.glyph import render_text, word_target, words_of, fit_lines
from nca.model import DiatomNCA, TEMPLATE_CH
from nca.voice import frame_descriptor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _export(model):
    sd = model.state_dict()
    return {
        "hidden_dim": sd["fc1.weight"].shape[0],
        "fc1_weight": sd["fc1.weight"].tolist(),
        "fc1_bias": sd["fc1.bias"].tolist(),
        "film_weight": sd["audio_encoder.weight"].tolist(),
        "film_bias": sd["audio_encoder.bias"].tolist(),
        "fc2_weight": sd["fc2.weight"].tolist(),
        "fc2_bias": sd["fc2.bias"].tolist(),
    }


def test_js_body_matches_torch_rule():
    torch.manual_seed(3)
    model = DiatomNCA(hidden_dim=32)
    size, steps = 24, 20
    genome = [0.1, -0.2, 0.05, 0.3]
    template = render_text("GO", size=size)
    audio = [0.4, 0.2, 0.0, 0.5, 0.1, 0.1, 0.1, 0.3]

    x = model.seed(1, size, genome=torch.tensor([genome]))
    with torch.no_grad():
        for _ in range(steps):
            x = model.update(
                x, audio=torch.tensor([audio]), fire_rate=1.0,
                template=torch.from_numpy(template).unsqueeze(0),
            )
    desc = frame_descriptor(x[0])
    assert torch.allclose(x[0, TEMPLATE_CH], torch.from_numpy(template))

    case = {
        "weights": _export(model), "size": size, "genome": genome, "steps": steps,
        "fire": 1.0, "template": template.flatten().tolist(), "audio": audio,
    }
    proc = subprocess.run(
        ["node", "electron/renderer/nca.js"], cwd=ROOT, input=json.dumps(case),
        capture_output=True, text=True, check=True,
    )
    got = json.loads(proc.stdout)
    assert abs(got["alive"] - round(desc["spread"] * size * size)) <= 2
    assert abs(got["mass"] - desc["mass"]) < 2e-3
    for a, b in zip(got["rgb"], desc["rgb"]):
        assert abs(a - b) < 2e-3


def test_glyph_layout_rules():
    assert fit_lines("hello") == ["HELLO"]
    assert fit_lines("frustule") == ["FRUS", "TULE"]
    assert fit_lines("...") == []
    assert words_of("I heard you. Body: grown, whole-ringed.") == ["I", "HEARD", "YOU", "BODY", "GROWN", "WHOLERINGED"]
    rgba, mask = word_target("GLASS", size=48)
    assert rgba.shape == (48, 48, 4) and mask.shape == (48, 48)
    assert 0.05 < (mask > 0.5).mean() < 0.4
    assert rgba[..., 3].max() == 1.0
    # Letters stay inside the grid with a margin on every side.
    ys, xs = (mask > 0.5).nonzero()
    assert ys.min() >= 2 and xs.min() >= 1 and ys.max() <= 45 and xs.max() <= 46


def test_morph_training_step_runs():
    import random
    from nca.train import TrainConfig, morph_step, _new_tasks

    model = DiatomNCA(hidden_dim=16)
    cfg = TrainConfig(size=24, batch=4, pool_size=8, rollout_min=4, rollout_max=6, word_prob=0.5)
    rng = random.Random(0)
    pool = model.seed(cfg.pool_size, cfg.size).detach()
    targets, templates = _new_tasks(cfg.pool_size, cfg, rng, 0)
    assert templates.max() > 0.5  # at least one word task
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = morph_step(model, opt, pool, targets, templates, cfg, rng)
    assert loss > 0 and loss < 10

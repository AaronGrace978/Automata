"""Say a thing, the body takes its outline. Python and the desktop app agree."""
import base64
import io
import json
import os
import random
import subprocess

import numpy as np
import pytest

from nca.shapes import default_index, emoji_font, find_shape, render_shape, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "electron", "renderer", "shapes.data.js")
needs_font = pytest.mark.skipif(emoji_font() is None, reason="no emoji font (fonts-noto-color-emoji)")

PHRASES = [
    "cloud", "Dino", "become a dragon", "turn into a t-rex", "I want a cloud", "morph into a tree",
    "make yourself a rocket", "cats", "octopus", "show me a volcano", "be a spaceship", "unicorn",
    "hello", "Can you hear me", "A blade just took half of you.", "Grow a new frustule.",
    "Sing with the tide.", "become a zorbleflux", "Give me a cloud please",
]


def _data():
    text = open(DATA).read()
    return json.loads(text[text.index("=") + 1 : text.rstrip().rindex(";")])


def test_request_grammar():
    assert request("become a cloud") == (["cloud"], True)
    assert request("Dino") == (["dino"], False)
    assert request("A blade just took half of you.")[0] == []
    cands, explicit = request("turn into a t-rex")
    assert explicit and "trex" in cands


@needs_font
def test_names_map_to_shapes():
    idx = default_index()
    want = {"cloud": 0x2601, "Dino": 0x1F996, "become a dragon": 0x1F409, "I want a cloud": 0x2601,
            "be a spaceship": 0x1F680, "cats": 0x1F431}
    for phrase, cp in want.items():
        assert find_shape(phrase, idx)[1] == cp, phrase
    for chat in ("hello", "Can you hear me", "Grow a new frustule.", "Sing with the tide."):
        assert find_shape(chat, idx)[1] is None, chat


@needs_font
def test_app_matches_python():
    data = _data()
    script = (
        "global.window = {}; require('./electron/renderer/shapes.data.js');"
        "const s = require('./electron/renderer/shapes.js');"
        f"const out = {json.dumps(PHRASES)}.map(p => {{ const r = s.find(p);"
        " return r.tile == null ? null : window.DIATOM_SHAPES.tiles[r.tile][0]; });"
        "process.stdout.write(JSON.stringify(out));"
    )
    got = json.loads(subprocess.run(["node", "-e", script], cwd=ROOT, capture_output=True, text=True,
                                    check=True).stdout)
    idx = default_index()
    for phrase, js_cp in zip(PHRASES, got):
        assert js_cp == find_shape(phrase, idx)[1], phrase
    assert len(data["tiles"]) > 1000


@needs_font
def test_atlas_is_the_python_silhouette():
    from PIL import Image

    data = _data()
    atlas = np.asarray(Image.open(io.BytesIO(base64.b64decode(data["atlas"].split(",", 1)[1]))), dtype=np.float32) / 255
    S, cols = data["size"], data["cols"]
    for cp in (0x2601, 0x1F996, 0x1F409):
        i = [row[0] for row in data["tiles"]].index(cp)
        r, c = divmod(i, cols)
        tile = atlas[r * S : (r + 1) * S, c * S : (c + 1) * S]
        assert np.abs(tile - render_shape(cp, size=S)).max() < 1.5 / 255


@needs_font
def test_shape_training_step_runs():
    import torch

    from nca.model import DiatomNCA
    from nca.train import TrainConfig, _new_tasks, morph_step

    model = DiatomNCA(hidden_dim=16)
    cfg = TrainConfig(size=24, batch=4, pool_size=8, rollout_min=4, rollout_max=6, word_prob=0.0, shape_prob=1.0)
    rng = random.Random(1)
    pool = model.seed(cfg.pool_size, cfg.size).detach()
    targets, templates = _new_tasks(cfg.pool_size, cfg, rng, 0)
    assert (templates.flatten(1).max(1).values > 0.5).sum() >= 6
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert 0 < morph_step(model, opt, pool, targets, templates, cfg, rng) < 10

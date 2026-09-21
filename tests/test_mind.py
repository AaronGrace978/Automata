"""Tests for the talking NCA: eyes, instinct, backbone, persona, mind."""
import torch

from nca.backbone import GroundedNarrator, HuggingFaceBackbone
from nca.instinct import InstinctState
from nca.mind import DiatomMind, modulation_from_text
from nca.model import DiatomNCA
from nca.persona import PERSONAS, get_persona
from nca.utils import half_damage
from nca.vision import VisualEncoder, predicates
from nca.voice import frame_descriptor


def test_visual_encoder_shape():
    enc = VisualEncoder(embed_dim=32)
    x = torch.zeros(2, 16, 24, 24)
    x[:, 3, 12, 12] = 1.0
    assert enc(x).shape == (2, 32)


def test_predicates_ground_wound():
    m = DiatomNCA(num_channels=16, hidden_dim=16)
    x = m.seed(1, 24)
    x = m(x, steps=8)
    cut = half_damage(x)
    desc = frame_descriptor(x[0])
    assert "wounded" in predicates(frame_descriptor(cut[0]), growth=-0.05)
    assert "wounded" not in predicates(desc, growth=0.0)


def test_instinct_feels_pain_then_calms():
    desc = {"mass": 0.3, "rgb": [0.5, 0.5, 0.5], "spread": 0.5, "pan": 0.5, "symmetry": 0.9}
    s = InstinctState(momentum=0.0)
    s.update(desc, growth=-0.05)
    assert s.pain > 0.2
    assert "tore" in s.to_text()
    s2 = InstinctState()  # default momentum: pain must decay gradually
    s2.update(desc, growth=-0.05)
    p0 = s2.pain
    for _ in range(20):
        s2.update(desc, growth=0.0)
    assert s2.pain < p0
    assert len(s2.rid) == 6


def test_wound_detected_by_relative_drop():
    # Halving a tiny creature is a small absolute drop — must still count.
    assert "wounded" in predicates({"mass": 0.01, "rgb": [0.4] * 3, "spread": 0.1,
                                    "pan": 0.5, "symmetry": 0.9}, growth=-0.005, rel_drop=0.5)


def test_regeneration_iou_and_jepa_step():
    from nca.eval import LatentPredictor, jepa_step, regeneration_iou
    from nca.vision import VisualEncoder

    intact = torch.zeros(1, 16, 16, 16)
    intact[:, 3, :, 8:] = 1
    healed = intact.clone()
    healed[:, 3, :, 12:] = 0
    score = regeneration_iou(healed, intact)
    assert 0 < score < 1
    assert regeneration_iou(intact, intact) == 1.0

    enc, pred = VisualEncoder(), LatentPredictor()
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    x = torch.rand(4, 16, 16, 16)
    a = jepa_step(enc, pred, opt, x)
    b = jepa_step(enc, pred, opt, x)
    assert b >= a - 0.05


def test_body_and_feeling_agree_while_healing():
    # Lingering pain must read as "healing", not "whole" — layers stay consistent.
    desc = {"mass": 0.3, "rgb": [0.4] * 3, "spread": 0.3, "pan": 0.5, "symmetry": 0.9}
    assert "healing" in predicates(desc, growth=0.0, pain=0.5)
    assert "whole" in predicates(desc, growth=0.0, pain=0.0)


def test_persona_does_not_rewrite_inside_words():
    said = get_persona("diatom_elder").voice("Body: small, shrinking, wounded, asymmetric.")
    assert "awhole" not in said
    assert "asymmetric" in said or "unringed" in said


def test_narrator_grounded_and_persona_styled():
    n = GroundedNarrator(seed=1)
    ctx = {"instinct_text": "I am still.", "predicates": ["small", "still", "whole"]}
    raw = n.speak(ctx)
    assert "still" in raw  # grounded in context
    styled = get_persona("diatom_elder").voice(raw)
    assert styled != raw  # persona changes diction
    assert len(PERSONAS) >= 3


def test_hf_backbone_falls_back_offline():
    b = HuggingFaceBackbone(model_id="nonexistent-model-xyz-123")
    text = b.speak({"instinct_text": "I drift.", "predicates": ["vast"]})
    assert isinstance(text, str) and len(text) > 0


def test_modulation_from_text():
    fire, _ = modulation_from_text("More! Bloom now, ecstatic!")
    assert fire > 1.0
    fire2, _ = modulation_from_text("Quiet water. Rest, sleep.")
    assert fire2 < 1.0


def test_mind_run_transcript_and_memory():
    mind = DiatomMind(DiatomNCA(num_channels=16, hidden_dim=16), persona="lab_assistant", seed=0)
    out = mind.run(steps=24, size=24, speak_every=12)
    assert len(out["transcript"]) == 2
    assert len(mind.memory) == 2
    assert all("step" in t and "said" in t and "rid" in t for t in out["transcript"])
    assert out["trajectory"].shape[1] == 25

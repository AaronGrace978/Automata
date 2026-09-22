"""Diatom NCA — Neural Cellular Automata for morphogenesis."""
from .model import DiatomNCA
from . import (
    diatom, audio, train, utils, voice, vision, instinct, persona, backbone, mind, morph, runtime, glyph, shapes,
)

__all__ = ["DiatomNCA", "diatom", "audio", "train", "utils", "voice",
           "vision", "instinct", "persona", "backbone", "mind", "morph", "runtime", "glyph", "shapes"]
__version__ = "1.8.1"

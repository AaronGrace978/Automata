"""Diatom NCA — Neural Cellular Automata for morphogenesis."""
from .model import DiatomNCA
from . import (
    diatom, audio, train, utils, voice, vision, instinct, persona, backbone, mind, morph, runtime, glyph,
)

__all__ = ["DiatomNCA", "diatom", "audio", "train", "utils", "voice",
           "vision", "instinct", "persona", "backbone", "mind", "morph", "runtime", "glyph"]
__version__ = "1.7.1"

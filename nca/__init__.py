"""Diatom NCA — Neural Cellular Automata for morphogenesis."""
from .model import DiatomNCA
from . import diatom, audio, train, utils, voice, vision, instinct, persona, backbone, mind

__all__ = ["DiatomNCA", "diatom", "audio", "train", "utils", "voice",
           "vision", "instinct", "persona", "backbone", "mind"]
__version__ = "0.2.0"

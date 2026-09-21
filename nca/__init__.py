"""Diatom NCA — Neural Cellular Automata for morphogenesis."""
from .model import DiatomNCA
from . import diatom, audio, train, utils, voice

__all__ = ["DiatomNCA", "diatom", "audio", "train", "utils", "voice"]
__version__ = "0.1.0"

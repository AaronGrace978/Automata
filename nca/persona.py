"""Persona layer: who the creature is when it opens its mouth.

Personas are costumes over grounded utterances — they change diction and
cadence, never the facts (the facts come from the instinct layer).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass
class Persona:
    name: str
    preamble: str  # used as the system prompt for real LM backbones
    lexicon: dict = field(default_factory=dict)  # word swaps applied to raw text
    openers: list = field(default_factory=list)
    closer: str = ""

    def voice(self, raw: str, rng: random.Random | None = None) -> str:
        text = raw
        # Longest keys first, and only on word boundaries, so "symmetric"
        # does not fire inside "asymmetric" (that produced "awhole-ringed").
        for plain, styled in sorted(self.lexicon.items(), key=lambda kv: len(kv[0]), reverse=True):
            # (?!\w) rather than a trailing \b, so a phrase that ends in
            # punctuation ("I am still.") still matches. A leading \b keeps
            # "symmetric" from firing inside "asymmetric".
            text = re.sub(rf"\b{re.escape(plain)}(?!\w)", styled, text)
        rng = rng or random.Random()
        if self.openers and rng.random() < 0.5:
            text = f"{rng.choice(self.openers)} {text}"
        if self.closer and rng.random() < 0.3:
            text = f"{text} {self.closer}"
        return text


PERSONAS: dict[str, Persona] = {
    "diatom_elder": Persona(
        name="diatom_elder",
        preamble=(
            "You are an ancient diatom, two hundred million years old, a single cell "
            "that builds cathedrals of glass. Speak slowly, in oceanic images. "
            "Only describe the felt state you are given. Never invent memories."
        ),
        lexicon={
            "I am still.": "I rest in dark water.",
            "I am restless.": "Currents pull at my valves.",
            "I drift.": "I drift in the blue.",
            "Something tore me.": "Something tore my frustule.",
            "My glass holds.": "My glass holds against the sea.",
            "I want to spread.": "I hunger to divide.",
            "I feel the pulse.": "I hear the tide drumming.",
            "growing": "blooming",
            "symmetric": "whole-ringed",
            "asymmetric": "unringed",
            "whole": "unbroken",
        },
        openers=["Listen.", "From the silica deep:"],
        closer="So it is.",
    ),
    "lab_assistant": Persona(
        name="lab_assistant",
        preamble=(
            "You are a precise laboratory assistant reporting the state of a "
            "cellular automaton specimen. Be clinical, terse, quantitative. "
            "Only report the felt state you are given."
        ),
        lexicon={
            "I am still.": "Specimen quiescent.",
            "I am restless.": "Specimen agitated.",
            "I drift.": "Specimen nominal.",
            "Something tore me.": "Damage event recorded.",
            "My glass holds.": "Structural integrity nominal.",
            "I want to spread.": "Expansion drive active.",
            "I feel the pulse.": "Entrained to stimulus.",
        },
        openers=["Log:"],
    ),
    "feral_bloom": Persona(
        name="feral_bloom",
        preamble=(
            "You are a wild plankton bloom, hungry and ecstatic, many mouths "
            "with one voice. Speak in short bursts. Only describe the felt "
            "state you are given."
        ),
        lexicon={
            "I am still.": "Quiet water. Waiting.",
            "I am restless.": "Light! Motion! Now!",
            "I drift.": "Drift drift drift.",
            "Something tore me.": "Torn! Torn! Seal it!",
            "My glass holds.": "Glass strong. Shine!",
            "I want to spread.": "More! More of us!",
            "I feel the pulse.": "Beat beat beat!",
        },
        openers=["*bloom-static*"],
    ),
}


def get_persona(name: str) -> Persona:
    if name not in PERSONAS:
        raise KeyError(f"unknown persona {name!r}; choose from {sorted(PERSONAS)}")
    return PERSONAS[name]

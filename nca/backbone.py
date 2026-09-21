"""Language backbone: from felt state to sentences.

Two backends, one interface:
  GroundedNarrator   — offline, deterministic, always available. Composes the
                       instinct text with body predicates. Cannot hallucinate
                       beyond what the body reports. (Default.)
  HuggingFaceBackbone — optional small LM (default SmolLM2-135M) prompted with
                       the persona preamble + felt state. Falls back to the
                       grounded narrator if transformers/weights are missing.

The backbone never sees pixels — only the RID vector, predicates, and persona.
That constraint is the whole point: the words are *about* the body.
"""
from __future__ import annotations

import random


class GroundedNarrator:
    """Deterministic template engine grounded on instinct + predicates."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def speak(self, context: dict) -> str:
        instinct_text: str = context.get("instinct_text", "")
        preds: list = context.get("predicates", [])
        body = ", ".join(preds[:4]) if preds else "unformed"
        bridges = [
            f"{instinct_text} Body: {body}.",
            f"{instinct_text} I am {body}.",
            f"Body: {body}. {instinct_text}",
        ]
        return self.rng.choice(bridges)


class HuggingFaceBackbone:
    """Small causal LM backend. Lazy; falls back to GroundedNarrator offline."""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-135M", max_new_tokens: int = 60) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipe = None
        self._fallback = GroundedNarrator()

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("text-generation", model=self.model_id)
        return self._pipe

    def speak(self, context: dict) -> str:
        try:
            pipe = self._load()
            prompt = (
                f"{context.get('persona_preamble', '')}\n"
                f"Felt state: {context.get('instinct_text', '')}\n"
                f"Body: {', '.join(context.get('predicates', []))}\n"
                "Speak in two sentences:"
            )
            out = pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.8)
            text = out[0]["generated_text"][len(prompt):].strip().split("\n")[0]
            return text or self._fallback.speak(context)
        except Exception:
            return self._fallback.speak(context)

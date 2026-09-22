"""Language backbone: from felt state to sentences.

Three backends, one interface:
  GroundedNarrator    — offline, deterministic, always available. Composes the
                        instinct text with body predicates. Cannot hallucinate
                        beyond what the body reports. (Default.)
  OllamaBackbone      — local commercial model (see nca/runtime.json). Falls
                        back to the grounded narrator if Ollama is down.
  HuggingFaceBackbone — optional small LM (default SmolLM2-135M). Falls back
                        the same way if transformers/weights are missing.

The backbone never sees pixels — only the RID vector, predicates, and persona.
That constraint is the whole point: the words are *about* the body.
"""
from __future__ import annotations

import json
import random
import urllib.error
import urllib.request

from .runtime import lock, resolve_host, resolve_model


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
        line = self.rng.choice(bridges)
        if (context.get("user_text") or "").strip():
            line = f"I heard you. {line}"
        return line


def messages_for(context: dict) -> list[dict]:
    """Chat messages for a local instruct model. No pixels, no cloud."""
    preamble = context.get("persona_preamble") or "You are a diatom."
    system = (
        preamble
        + " You are the glass body in front of the human. Answer them in character. "
        "Use the felt state for anything you claim about your own body. "
        "Two or three sentences. No lists. No markdown."
    )
    preds = ", ".join(context.get("predicates") or []) or "unformed"
    memory = [m for m in (context.get("memory") or []) if m][-3:]
    user = (context.get("user_text") or "").strip() or "(silence — speak your state)"
    content = (
        f"Felt state: {context.get('instinct_text', '')}\n"
        f"Body: {preds}\n"
    )
    if memory:
        content += "You already said: " + " ".join(memory) + "\n"
    content += f"Human says: {user}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def complete(host: str, model: str, messages: list[dict], timeout: float = 90.0) -> str:
    """One non-streaming turn against a local Ollama server."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": lock["temperature"],
            "num_predict": lock["num_predict"],
            "num_ctx": lock["num_ctx"],
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Ollama {exc.code}: {detail}") from exc
    return ((data.get("message") or {}).get("content") or "").strip()


class OllamaBackbone:
    """Local allow-listed model. The grounded narrator speaks if it cannot."""

    def __init__(self, model_id: str | None = None, host: str | None = None, timeout: float = 90.0) -> None:
        self.model_id = resolve_model(model_id)
        self.host = resolve_host(host)
        self.timeout = timeout
        self._fallback = GroundedNarrator()

    def speak(self, context: dict) -> str:
        try:
            text = complete(self.host, self.model_id, messages_for(context), self.timeout)
        except Exception:
            return self._fallback.speak(context)
        return text or self._fallback.speak(context)


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

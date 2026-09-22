"""Commercial runtime lock.

The desktop creature talks to a local model. The allow-list, the context
cap, and the refusal of Ollama Cloud all live in ``runtime.json`` so the
Electron app and this package cannot drift apart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

_LOCK_PATH = Path(__file__).resolve().parent / "runtime.json"
with _LOCK_PATH.open(encoding="utf-8") as _fh:
    lock: dict = json.load(_fh)

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def resolve_model(model_id: str | None = None) -> str:
    """Return an allow-listed local model id, or raise."""
    chosen = (model_id or os.environ.get("DIATOM_MODEL") or lock["default_model"]).strip()
    if "cloud" in chosen.lower():
        raise ValueError(
            f"Refusing model {chosen!r}. Cloud tags are not part of this product."
        )
    if chosen not in lock["models"]:
        allowed = ", ".join(sorted(lock["models"]))
        raise ValueError(f"Model {chosen!r} is not on the commercial allow-list: {allowed}")
    return chosen


def resolve_host(host: str | None = None) -> str:
    """Return a loopback Ollama host, or raise.

    ``DIATOM_ALLOW_REMOTE=1`` permits a host the operator runs themselves.
    It does not turn Ollama Cloud into a supported dependency.
    """
    raw = (host or os.environ.get("OLLAMA_HOST") or lock["default_host"]).strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Bad model host {raw!r}")
    name = parsed.hostname.lower()
    allow = os.environ.get("DIATOM_ALLOW_REMOTE", "").strip().lower() in {"1", "true", "yes"}
    if name not in LOOPBACK and not allow:
        raise ValueError(
            "Refusing a non-local model host. Ollama Cloud is not part of this product. "
            "Set DIATOM_ALLOW_REMOTE=1 only for a server you operate."
        )
    return raw


def model_spec(model_id: str | None = None) -> dict:
    mid = resolve_model(model_id)
    spec = dict(lock["models"][mid])
    spec["id"] = mid
    return spec

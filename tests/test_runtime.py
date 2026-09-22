"""The commercial runtime stays local, allow-listed, and offline-capable."""
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nca.backbone import OllamaBackbone, complete, messages_for
from nca.runtime import lock, resolve_host, resolve_model


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_lock_file_is_commercial_and_fits_16gb():
    assert lock["ollama_cloud"] is False
    assert lock["default_model"] == "qwen2.5:14b"
    assert lock["num_ctx"] == 4096
    spec = lock["models"][lock["default_model"]]
    assert spec["license"] == "Apache-2.0"
    assert spec["quant"] == "Q4_K_M"
    assert spec["non_embedding_b"] == 13.1
    assert spec["weights_gb"] + 4 < lock["vram_gb"]
    for name, row in lock["models"].items():
        assert row["license"] in {"Apache-2.0", "MIT"}
        assert "cloud" not in name
        assert row["weights_gb"] <= 12
    assert "qwen2.5:72b" not in lock["models"]
    assert "qwen2.5:3b" not in lock["models"]


def test_cloud_and_unlicensed_models_are_refused(monkeypatch):
    monkeypatch.delenv("DIATOM_MODEL", raising=False)
    monkeypatch.delenv("DIATOM_ALLOW_REMOTE", raising=False)
    assert resolve_model() == "qwen2.5:14b"
    with pytest.raises(ValueError):
        resolve_model("llama3.1:8b")
    with pytest.raises(ValueError):
        resolve_model("qwen2.5:72b")
    with pytest.raises(ValueError):
        resolve_model("gpt-oss:120b-cloud")
    with pytest.raises(ValueError):
        resolve_host("https://ollama.com")
    assert resolve_host("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"
    monkeypatch.setenv("DIATOM_ALLOW_REMOTE", "1")
    assert resolve_host("http://10.1.1.1:11434") == "http://10.1.1.1:11434"
    monkeypatch.setenv("DIATOM_MODEL", "phi4")
    assert resolve_model() == "phi4"


def test_node_runtime_selfcheck():
    env = os.environ.copy()
    env.pop("DIATOM_MODEL", None)
    env.pop("OLLAMA_HOST", None)
    env.pop("DIATOM_ALLOW_REMOTE", None)
    proc = subprocess.run(
        ["node", "electron/runtime.js"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "runtime ok" in proc.stdout


def test_messages_stay_about_the_body():
    messages = messages_for({
        "persona_preamble": "You are a diatom.",
        "instinct_text": "I drift.",
        "predicates": ["small", "whole"],
        "user_text": "who are you?",
        "memory": ["I drift in the blue."],
    })
    assert messages[0]["role"] == "system"
    blob = messages[1]["content"]
    assert "I drift." in blob
    assert "who are you?" in blob
    assert "I drift in the blue." in blob
    assert "ollama.com" not in json.dumps(messages)


def test_ollama_complete_roundtrip():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            assert body["model"] == "qwen2.5:14b"
            assert body["options"]["num_ctx"] == 4096
            assert body["stream"] is False
            raw = json.dumps({"message": {"role": "assistant", "content": "The tide answers."}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        text = complete(host, "qwen2.5:14b", [{"role": "user", "content": "hi"}], timeout=5)
        assert text == "The tide answers."
        said = OllamaBackbone(model_id="qwen2.5:14b", host=host, timeout=5).speak({
            "instinct_text": "I drift.",
            "predicates": ["small"],
            "persona_preamble": "You are a diatom.",
            "user_text": "hello",
        })
        assert said == "The tide answers."
    finally:
        server.shutdown()
        server.server_close()


def test_ollama_falls_back_when_nothing_is_listening():
    backbone = OllamaBackbone(model_id="qwen2.5:14b", host="http://127.0.0.1:1", timeout=0.4)
    said = backbone.speak({"instinct_text": "I drift.", "predicates": ["vast"], "user_text": "hello"})
    assert "heard" in said.lower()
    assert "drift" in said.lower()

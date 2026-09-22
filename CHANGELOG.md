# Changelog

## 1.7.0 — 2026-09-22

The body speaks.

- Channel 10 is the silica template. `train_morph` fine-tunes the diatom rule so the frustule grows into a word written there, holds it, and returns to glass when it clears (`scripts/train_words.py`, `scripts/demo_words.py`).
- The desktop window runs the real automaton (`electron/renderer/nca.js`, cell-for-cell with PyTorch, checked by `tests/test_body_parity.py`). Its grid is the 3D glass; there is no procedural frustule.
- Replies from the local model are spelled by the cells, one word at a time.
- Camera looks at the body. The felt-state meters sit in the dock.

## 1.6.0 — 2026-09-22

Desktop creature.

- Electron app with a 3D centric frustule (valves, girdle, ribs, areolae) that morphs from the felt state and from what you say to it.
- Locked local model: Qwen2.5-14B-Instruct at Q4_K_M (Apache-2.0), sized for an RTX 5060 Ti 16 GB. Context capped at 4096.
- Allow-list also includes Qwen2.5-7B-Instruct (Apache-2.0) and Phi-4 / Phi-4-mini (MIT). Llama and Ollama Cloud are refused.
- The grounded narrator still speaks, and the frustule still moves, when Ollama is not running.

## 1.5.0 — 2026-09-21

Working paper v1.5. Ten-seed regeneration scores, citation metadata.

# Changelog

## 1.8.0 — 2026-09-22

The body takes shape.

- Say *cloud*, *dino*, *become a dragon*: the same cells grow into that outline and hold it until you speak again (`nca/shapes.py`, `electron/renderer/shapes.js`). 1,379 Noto Color Emoji silhouettes (OFL-1.1) and 2,248 names, baked by `scripts/build_shapes.py`. Words the table doesn't know go to the local model, which picks the emoji.
- `train_morph --shape-prob` adds silhouette tasks; the Colab notebook trains words and shapes together.
- `DiatomMind.reply` returns the shape; `DiatomMind.become` grows it. `scripts/demo_shapes.py` writes the proof strip.
- App: Cloud / Dino / Dragon / Let go buttons. Seeded a flaky voice test.

## 1.7.1 — 2026-09-22

- `notebooks/train_words_colab.ipynb`: fine-tune the word rule on a Colab A100 in two phases, proof strip, stability check, export and download `web/weights.js`.
- `scripts/*.py` find `nca/` from any working directory; `train_words.py --pool` sets the replay pool size.

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

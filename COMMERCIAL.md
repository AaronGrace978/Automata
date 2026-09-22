# What can be sold

The application code in this repository is MIT. That license allows commercial use, modification, and resale, provided the copyright and permission notice travel with it.

The creature's language model is a separate work. This release locks it so a product built on the desktop app does not pick up a non-commercial weight license or a hosted-service dependency.

## Locked default

| | |
|---|---|
| Model | Qwen2.5-14B-Instruct (`ollama pull qwen2.5:14b`) |
| Quantization | Q4_K_M, about 9 GB |
| License of the weights | Apache-2.0 |
| Parameters | 14.7B total, 13.1B non-embedding |
| Context sent to the runtime | 4096 tokens |
| Target GPU | NVIDIA RTX 5060 Ti, 16 GB |

A 13B-class model in FP16 is about 26 GB. It does not fit. The 4-bit 14B above is the same class of network, it fits with room for the cache and the frustule, and its weights can be used in a commercial product. Apache-2.0 also allows bundling the GGUF later, if the Apache notice goes with it. This release does not vendor the weights.

## Allow-list

`DIATOM_MODEL` may be one of:

| Tag | Weights | License | Pull |
|---|---|---|---|
| `qwen2.5:14b` | Qwen2.5-14B-Instruct, ~9 GB | Apache-2.0 | `ollama pull qwen2.5:14b` |
| `qwen2.5:7b` | Qwen2.5-7B-Instruct, ~4.7 GB | Apache-2.0 | `ollama pull qwen2.5:7b` |
| `phi4` | Phi-4, ~9.1 GB | MIT | `ollama pull phi4` |
| `phi4-mini` | Phi-4-mini, ~2.5 GB | MIT | `ollama pull phi4-mini` |

Not on the list, and refused by the app:

- **Llama** family. The Llama community license allows many commercial uses and also caps them by monthly active users.
- **Qwen2.5-3B and Qwen2.5-72B.** Those two sizes are under the Qwen license, not Apache-2.0. The 7B and 14B sizes are Apache-2.0.
- Any tag containing `cloud`.

## Ollama Cloud

No. Ollama Cloud is a hosted API with its own terms, its own bill, and a catalog of models this product is not allowed to assume are commercial. Conversations would leave the machine.

Local Ollama is MIT software. The customer installs it and pulls an allow-listed model. The desktop app talks only to `127.0.0.1` (or `localhost` / `::1`). `DIATOM_ALLOW_REMOTE=1` exists for an operator who runs their own server. It is not a switch for Ollama Cloud, and the commercial configuration leaves it unset.

## When the model is not installed

The body still grows, the grounded narrator still answers from the felt state, and the cells still spell that answer. A sale does not depend on a weight download completing first. The 14B model is what speaks once `ollama pull qwen2.5:14b` has finished.

## The diatom's own weights

The cellular automaton (`assets/checkpoint.pt`, `assets/checkpoint_words.pt`, `web/weights.js`) is trained in this repository on procedural targets and rendered words. Those weights are yours under the MIT license, with no third-party model license attached. The fonts used to render training words (DejaVu Sans Bold) are under the DejaVu license, which permits this use; the weights do not embed the font.

## Notices a shipped product still has to carry

- The MIT notice in `LICENSE` (this repository).
- The Apache-2.0 notice for Qwen2.5 if those weights are bundled, or a pointer to it if the customer pulls them.
- The MIT notice for Phi-4 if that model is the one shipped.
- The MIT notice for Ollama if the Ollama binary is bundled. This release expects the customer to install Ollama rather than bundling it.

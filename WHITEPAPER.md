# Diatom NCA: Audio-Conditioned Neural Cellular Automata That Grow, Heal, Sing, and Speak

**Aaron Alexander Grace, M.Ed.**
*Independent Researcher. M.Ed., Higher Education Administration, University of Massachusetts Lowell.*
Correspondence: [github.com/AaronGrace978/Automata](https://github.com/AaronGrace978/Automata)

*Working paper v1.5, September 2026. All reported numbers come from one
3000-step GPU rule at 48 px; regeneration is scored over 10 seeds against an
untrained baseline. All code, targets, and demos are open source (MIT).*

**Keywords:** neural cellular automata, morphogenesis, diatoms, audio conditioning,
FiLM, regeneration, sonification, grounded language, artificial life

---

## Abstract

We present **Diatom NCA**, a Neural Cellular Automaton in which a single shared
local rule grows diatom-like silica frustules from one cell, regenerates after
damage, and reports its own state in sound and language. Each cell of a
16-channel grid runs the same small network over its 3×3 neighbourhood; centric
and pennate frustule morphologies emerge from iterated local computation. Three
extensions distinguish the system. **(1) Audio conditioning:** an 8-dimensional
sound vector modulates the update network via FiLM every step, so loudness bends
growth rate, brightness bends pigment, and onsets break symmetry, while the rule
stays local. **(2) Closed sensorimotor loops:** a deterministic sonification maps
growth to stereo song, and the song re-enters the conditioner. **(3) A talking
NCA:** a visual module, a six-drive instinct layer (RID), a grounded language
backbone, and a persona layer let the organism narrate its bodily state; its
words feed back into its growth rate. Grounding is enforced by construction:
the narrator may only speak predicates computed from the body. To our knowledge
this is the first NCA that reports its own morphology in language, and the first
in which morphogenesis is conditioned on live audio. Procedural diatom generators
provide an infinite training distribution. Code, tests, and a Colab notebook
reproduce every result.

---

## 1. Introduction

A diatom is a single cell that fabricates a frustule: a silica pillbox with
radial or bilateral symmetry, rows of pores (areolae), ribs, slits, and polar
nodules, assembled at sub-micron precision by local chemistry with no blueprint
(Round, Crawford & Mann, 1990). If local chemistry can do that, local
computation should be able to learn it.

Neural Cellular Automata (Mordvintsev et al., 2020) showed that a tiny shared
rule iterated over a grid can grow and regenerate a target from a single cell.
Two directions remain open: conditioning the rule on a live external signal
without breaking locality, and giving the organism a readout of its own state
richer than pixels. We address both, using the diatom as model organism.

**Contributions.**

1. **Audio-conditioned local rule.** An 8-dim sound vector modulates the update
   MLP through FiLM gains and biases and an energy-driven fire rate. The rule
   remains local and shared; only its tuning varies with sound (§3.3).
2. **Procedural diatom targets.** Parametric centric and pennate generators with
   girdle, ribs, striae, areolae, raphe, and rosette structure (§3.4).
3. **Regeneration by construction.** Pool training with damage on half of every
   batch yields organisms that re-grow excised halves: IoU 0.89 ± 0.02 on
   half-cuts and 0.79 ± 0.02 on disc wounds over 10 seeds, against 0.33 and
   0.00 for an untrained rule (§3.5, Table 3).
4. **Sonification loop.** A fixed growth-to-song map whose output re-enters the
   conditioner, closing a sensorimotor loop with no extra training (§3.6).
5. **The talking NCA.** Visual module, instinct layer, grounded language
   backbone, and persona, with speech modulating growth (§3.7).
6. **Reproducibility.** Test suite, demo scripts, browser creature with
   microphone conditioning, and a Colab notebook covering every experiment.

---

## 2. Related Work

**Growing NCA** (Mordvintsev et al., 2020) supplies the recipe we build on:
fixed Sobel perception, small update MLP, stochastic updates, alive masking,
sample-pool training, damage-driven regeneration. We keep every element and add
conditioning, readouts, and a diatom target family. **Continuous CA** (Lenia,
Chan 2019; Flow Lenia, Plantec et al., 2022) show rich creature dynamics from
hand-tuned rules; ours are learned and externally conditioned. **FiLM** (Perez
et al., 2018) conditions networks by affine modulation of activations; we
repurpose it so that sound retunes the physics of growth rather than the pixels
of an output. Audio-conditioned image and video generation is established;
audio-conditioned *morphogenesis* is, to our knowledge, new. **Diatom
morphology** (Round et al., 1990) provides our design vocabulary. **Grounded
language** research typically grounds text in images or environments; we ground
it in an organism's own internal state variables, a stricter contract.

---

## 3. Method

### 3.1 The cell

State is $x \in \mathbb{R}^{B \times 16 \times H \times W}$ with the channel
roles in Table 1. Only RGBA is supervised. DNA channels are per-organism
constants; morphogen and hidden channels are free memory; the AUDIO channel is a
leaky integrator of loudness, so tissue remembers what it heard.

| Channels | Role |
|---|---|
| 0–2 | RGB pigment |
| 3 | Alpha / maturity; `alive = maxpool(alpha) > 0.1` |
| 4–7 | DNA genome: constant per organism, steers fate |
| 8–10 | Morphogen: slow auxiliary memory |
| 11 | AUDIO sense: leaky integrator of loudness |
| 12–15 | Hidden / silica-deposition state |

### 3.2 The local rule

Perception is fixed and depthwise: identity plus Sobel-x/y per channel (48-dim).
The update is `Linear(48→128) → ReLU → FiLM(audio) → Linear(128→16)`, applied
residually with per-cell probability `fire_rate` and gated by the alive mask.
Small non-zero initialisation keeps the untrained rule alive at birth and stable
over hundred-step rollouts.

### 3.3 Audio into the weights

An 8-dim vector $a$ (Table 2) is encoded to $(\gamma, \beta)$ and applied as
$h \leftarrow h \odot (1+\gamma) + \beta$. Energy also sets the fire rate,
$0.3 + 0.6 \cdot \mathrm{energy}$, and writes the AUDIO channel. Random audio
vectors condition half of all training batches, so the organism is steerable
yet stable under arbitrary sound.

| Dim | Feature | Morphological role |
|---|---|---|
| 0 | RMS energy | growth rate, AUDIO memory |
| 1 | spectral centroid | pigment shift toward glass blue |
| 2 | spectral flux | symmetry-breaking bursts |
| 3 | beat phase | regeneration wave timing |
| 4–6 | low / mid / high band energy | rim thickness, rib contrast, pore seeding |
| 7 | tonal-vs-noise contrast | gold vs. glass palette |

### 3.4 Procedural diatom targets

The centric generator composes a girdle rim, $n$-fold ribs with wobble,
concentric growth rings, ray-aligned areolae, and a central rosette. The pennate
generator composes a superellipse valve, raphe slit, curved transverse striae,
striae pores, and polar and central nodules. Three palettes, infinite seeds, no
downloads.

### 3.5 Training

A pool of 64 organisms; rollouts of 32–64 steps; worst-organism reseeding;
circle, half, or noise damage on ~50% of batches; MSE on RGBA plus a latent
overflow penalty; Adam with gradient clipping. Regeneration is in the training
distribution, not a post-hoc property.

### 3.6 Voice: sonification loop

Per-frame descriptors (alive mass, mean RGB, spread, centre of mass, bilateral
symmetry) drive phase-continuous additive synthesis: mass sets pitch on an
A-minor pentatonic scale, RGB sets root/fifth/octave weights, growth rate sets
the amplitude envelope, centre of mass sets stereo pan, asymmetry sets vibrato.
Output is stereo 22.05 kHz. Re-analysing the song with the audio front-end and
feeding it back as conditioning yields self-listening growth.

### 3.7 The talking NCA

`DiatomMind` stacks four layers above the body. Each is small; the design
constraint is that language can only be *about* the body.

**Visual module.** A CNN encodes the 16-channel grid to a 32-dim body embedding.
In parallel, symbolic predicates are computed from frame descriptors: mass
(dormant / small / grown / vast), motion (growing / shrinking / still),
integrity (wounded / healing / whole), symmetry, brightness, spread, and lean.

**Instinct layer (RID, Reactive Instinct Drive).** Six drives in $[-1,1]$ with
momentum: arousal, valence, pain, hunger, rhythm, calm. They integrate mass,
growth, symmetry, pigment, and audio energy each step. Wounds register on
absolute or relative mass loss, so halving a small organism still hurts, and
pain decays rather than snapping off. Integrity predicates read the pain drive,
so body words and felt words stay consistent during recovery.

**Language backbone.** The default `GroundedNarrator` is a deterministic
composer over instinct sentences and predicates; it cannot state anything the
body did not report. An optional `HuggingFaceBackbone` (SmolLM2-135M) is
prompted with persona and felt state and falls back to the grounded narrator
when unavailable.

**Persona layer.** Three diction styles, `diatom_elder`, `lab_assistant`, and
`feral_bloom`, restyle utterances without changing their propositional content.

**Closing the loop.** Keywords in the organism's own speech modulate its fire
rate: expansion vocabulary quickens growth, pain vocabulary slows it. The
organism perceives itself, feels, speaks, and its speech moves its body.

---

## 4. Experiments

All protocols ship as scripts and as cells in `notebooks/diatom_nca_colab.ipynb`.
Unit tests (20) cover the rule, targets, audio features, training step, voice,
vision, instinct, backbone, and mind.

| ID | Protocol | Result |
|---|---|---|
| E1 Growth | Seed one cell, 96 steps | GPU run, 3000 steps, 48 px: training loss falls from 0.072 to the 0.02–0.05 range and the organism forms a coherent pigmented body. |
| E2 Regeneration | Grow 64, damage, grow 64; alive-cell IoU in the damaged region against an intact twin; 10 seeds | See Table 3. Trained rule: 0.892 ± 0.017 (half-cut), 0.790 ± 0.019 (disc). Untrained rule: 0.331 ± 0.076 and 0.000 ± 0.000. |
| E3 Sound | Same genome under silence vs. full-energy audio | Trajectories diverge (unit-tested). Synthetic beat / sweep / bloom and real WAVs all condition growth. |
| E4 Speciation | Six genomes, one rule | Six distinct morphologies with no rule change. |
| E5 Voice | Render 96-step trajectory to song, feed back | ~15 s stereo song; self-listening growth runs to completion. |
| E6 Speech | Narrate every 24 steps, three personas, then wound | Post-surgery utterances contain wound predicates; recovery utterances read "healing" while the pain drive persists, then "whole". |
| E7 Latent prediction | Frozen body encoder; a predictor maps the masked-half embedding to the full-body embedding; fresh grown bodies every step | Train cosine 0.676, held-out cosine 0.672 on bodies never trained on, including damaged ones. Gap 0.004: the ceiling is the untrained encoder, not the predictor. |

**Table 3. Regeneration IoU, 3000-step rule, 48 px, GPU, 10 seeds per cell.**
Higher is better; 1.0 means the healed region matches the intact twin exactly.

| Rule | Half-cut | Disc |
|---|---|---|
| Trained | 0.892 ± 0.017 | 0.790 ± 0.019 |
| Untrained | 0.331 ± 0.076 | 0.000 ± 0.000 |

Two earlier single-seed rollouts of the same rule scored 0.923 and 0.908 on
the half-cut, consistent with the 10-seed mean. Disc damage is harder than a
half-cut: the wound is interior, so no intact edge borders it on one side. The
untrained rule scores 0 on disc damage because it never re-enters the hole.
Spread between seeds is small (std ≤ 0.02), so healing is a property of the
trained rule rather than of a lucky rollout. Script: `scripts/score_regeneration.py`.

---

## 5. Limitations

Locality is a constraint as well as a thesis: global coordination such as exact
$n$-fold symmetry must be negotiated through diffusion-like channels, so
symmetry is approximate. Audio conditioning is correlational, not semantic; the
organism responds to rhythm and brightness, not melody or meaning. The grounded
narrator is deliberately narrow; the LM backbone trades that guarantee for
fluency and is not yet evaluated for grounding failures. Procedural targets
trade biological fidelity for infinite data. All results are small-grid and
preliminary.

## 6. Future Work

Joint-embedding training objectives (predict the organism's future latent under
masking, not its pixels); song-to-species (a track's summary features as
genome); 3D frustules; genome evolution against symmetry and pore-regularity
fitness; fine-tuning on SEM imagery; grounding evaluation for the LM backbone.

## 7. Author Statement

This is the independent research contribution of Aaron Alexander Grace, M.Ed.
(Higher Education Administration, University of Massachusetts Lowell), who also
holds several professional certificates. The author's background is in how
people develop in learning environments; this work turns the same question
toward how cells develop in computational ones. Ideation, direction, and
research framing are the author's. Implementation was produced with AI
assistance under the author's direction.

## References

- Chan, B. W.-C. (2019). Lenia: Biology of Artificial Life. *Complex Systems*, 28(3).
- Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020). Growing Neural Cellular Automata. *Distill*. https://distill.pub/2020/growing-ca/
- Niklasson, E., Mordvintsev, A., Randazzo, E., & Levin, M. (2021). Self-Organising Textures. *Distill*.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
- Plantec, E., Hamon, G., Etcheverry, M., Oudeyer, P.-Y., Moulin-Frier, C., & Chan, B. W.-C. (2022). Flow Lenia. *ALIFE*.
- Round, F. E., Crawford, R. M., & Mann, D. G. (1990). *The Diatoms: Biology and Morphology of the Genera*. Cambridge University Press.

## Appendix A: Reproducibility

```bash
pip install -r requirements.txt
pytest -q                                              # 20 tests
python scripts/train_diatom.py --steps 3000 --size 48  # E1
python scripts/demo_regenerate.py --checkpoint assets/checkpoint.pt   # E2
python scripts/demo_audio.py --checkpoint assets/checkpoint.pt --wav your.wav  # E3
python scripts/demo_voice.py --checkpoint assets/checkpoint.pt        # E5
python scripts/demo_mind.py --checkpoint assets/checkpoint.pt --persona diatom_elder  # E6
```

Or run `notebooks/diatom_nca_colab.ipynb` top to bottom.

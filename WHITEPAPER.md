# Diatom NCA: Audio-Conditioned Neural Cellular Automata for Diatom Morphogenesis

**Aaron Alexander Grace, M.Ed.**
*Independent Researcher — Higher Education Administration, University of Massachusetts Lowell*

> Grace holds a Master's degree in Education (Higher Education Administration) from
> the University of Massachusetts Lowell, along with several professional certificates,
> and conducts independent research in artificial intelligence and artificial life.
> Correspondence: via the [Automata repository](https://github.com/AaronGrace978/Automata).

*Working paper v1.1 — September 2026. Preliminary results from CPU smoke runs;
full-scale evaluation is ongoing. All code, targets, and demos are open source
in the companion repository.*

**Keywords:** neural cellular automata, morphogenesis, diatoms, audio conditioning,
FiLM, regeneration, sonification, artificial life

---

## Abstract

We present **Diatom NCA**, a Neural Cellular Automaton (NCA) framework that grows
diatom-inspired silica frustules from a single cell using strictly local rules.
Each cell of a 16-channel grid executes the same small neural network over its
3×3 neighbourhood; global form — radial centric discs and bilateral pennate
valves with ribs, striae, pores, and raphes — emerges from iterated local
computation. The local rule is **conditioned on audio** via Feature-wise Linear
Modulation (FiLM): an 8-dimensional sound vector retunes the update network's
hidden layer every step, so loudness bends growth rate, brightness bends pigment,
and onsets break symmetry. Pool-based training with systematic damage (disc
knockouts, hemi-sections) produces a **regeneration creature** that re-grows
excised halves. A companion **sonification** module maps growth trajectories
back to stereo song (pentatonic pitch from alive mass, chord from pigment,
loudness from growth rate, pan from centre of mass), closing the loop:
*microphone → morphogenesis → speaker*. Procedural diatom targets provide an
infinite, download-free training distribution. We argue the diatom — a single
cell that builds a cathedral of glass — is the ideal model organism for
studying order from local rules, and for asking what else local rules can be
conditioned on.

---

## 1. Introduction

A diatom is a single eukaryotic cell that fabricates a frustule: a pillbox of
amorphous silica ornamented with radial or bilateral symmetry, rows of pores
(areolae), ribs, slits, and polar nodules — structures with sub-micron
precision, assembled without a blueprint, by local chemistry (Round, Crawford &
Mann, 1990). If local chemistry can do that, local computation should be able
to learn it.

Neural Cellular Automata (Mordvintsev et al., 2020) showed that a tiny shared
neural rule, iterated over a grid, can grow and regenerate a target image from
a single cell. Subsequent work extended NCA to textures, 3D growing creatures,
and differentiable self-organisation. Yet two frontiers remain comparatively
unexplored: **(i)** conditioning the local rule on a live external signal
without breaking locality, and **(ii)** the diatom as target morphology — an
organism whose entire body plan is, by definition, a cellular-automaton-scale
phenomenon.

Our contributions:

1. **Audio-conditioned local rule.** An 8-dim sound vector (energy, spectral
   centroid, flux, beat phase, three band energies, spectral contrast) modulates
   the NCA update MLP through FiLM gains and biases, plus an energy-driven
   fire rate. The rule stays local and shared; only its *tuning* varies with
   sound (Section 3.3).
2. **Procedural diatom target distribution.** Parametric centric and pennate
   frustule generators (Section 3.4) yielding infinite RGBA supervision with
   girdle, ribs, striae, areolae, raphe, and rosette structures.
3. **Regeneration by construction.** Pool training that damages half of every
   batch produces organisms that re-grow hemi-sections (Section 3.5).
4. **Sonification / voice.** A deterministic state→song renderer closes the
   perception-action loop and gives the organism an audible phenotype
   (Section 3.6).
5. **The talking NCA.** A full cognitive stack — visual module, felt instinct
   layer (RID), language backbone, persona — with words feeding back into
   growth. To our knowledge, the first NCA that reports its own bodily state
   in language (Section 3.7).
5. **Open interactive science.** A pure-JavaScript in-browser creature with
   microphone conditioning, surgery tools, genome controls, and a Google Colab
   notebook reproducing every result end to end.

---

## 2. Background and Related Work

**Growing Neural Cellular Automata** (Mordvintsev et al., 2020) introduced the
recipe we build on: fixed Sobel perception, a small update MLP, stochastic
updates, alive masking via pooled alpha, sample-pool training, and regeneration
through damage. We preserve every element and add conditioning plus a diatom
target family.

**Self-organising systems** (Lenia, Chan 2019; Flow Lenia, Plantec et al.,
2022) demonstrate rich creature-like dynamics from continuous CA. Our work is
complementary: learned rather than hand-tuned rules, with external conditioning.

**Conditioned generation.** FiLM (Perez et al., 2018) conditions networks by
affine modulation of activations — originally for visual reasoning, here
repurposed as "sound retunes physics." Audio-conditioned image/video generation
is well established; to our knowledge, audio-conditioned *morphogenesis* (the
signal altering growth dynamics rather than pixels) is novel.

**Diatom biology.** The frustule literature (Round et al., 1990) supplies our
design language: centric vs. pennate symmetry groups, areolae patterning,
raphe slits, sternum ribs. Diatoms are also a climate-relevant organism
(responsible for ~20% of global primary production), giving the model organism
stakes beyond aesthetics.

**Sonification.** Mapping data to sound for monitoring and art has a long
history; our contribution is a fixed, interpretable growth→song map with a
musicality constraint (pentatonic quantisation), used as a *readout of a
learning system* rather than of raw data.

---

## 3. Method

### 3.1 The cell

State is a grid of 16-dimensional cells, $x \in \mathbb{R}^{B \times 16 \times H \times W}$,
with the channel semantics in Table 1. Only RGBA is supervised; DNA channels
are per-organism constants (the genome), morphogen and hidden channels are free
latent memory, and the AUDIO channel accumulates recent loudness — the tissue
remembers what it heard.

| Channels | Role |
|---|---|
| 0–2 | RGB pigment (fucoxanthin golds, silica blues) |
| 3 | Alpha / maturity; `alive = maxpool(alpha) > 0.1` |
| 4–7 | DNA genome: constant per organism, steers pattern fate |
| 8–10 | Morphogen: slow auxiliary memory |
| 11 | AUDIO sense: leaky integrator of loudness |
| 12–15 | Hidden / silica-deposition state |

### 3.2 The local rule

Perception is fixed and depthwise: identity plus Sobel-x/y per channel
(48-dim perception). The update is
`Linear(48→128) → ReLU → FiLM(audio) → Linear(128→16)`, added residually with
per-cell probability `fire_rate`, gated by the alive mask. Small (non-zero)
initialisation keeps the untrained creature alive-at-birth ("primordial soup")
while remaining stable over hundred-step rollouts.

### 3.3 Audio into the weights

An 8-dim vector $a$ (Table 2) is encoded to $(\gamma, \beta)$ that modulate the
hidden layer: $h \leftarrow h \odot (1+\gamma) + \beta$. Energy additionally
scales the fire rate ($0.3 + 0.6\cdot\mathrm{energy}$) and writes the AUDIO
channel. During training, random audio vectors condition half of all batches,
so the creature is steerable yet stable under arbitrary sound.

| Dim | Feature | Morphological role |
|---|---|---|
| 0 | energy (RMS) | growth rate, AUDIO memory |
| 1 | spectral centroid | pigment toward glass blue |
| 2 | spectral flux | symmetry-breaking bursts |
| 3 | beat phase | regeneration wave timing |
| 4–6 | low / mid / high bands | rim thickness, rib contrast, pore seeding |
| 7 | tonal-vs-noise contrast | gold vs. glass palette |

### 3.4 Procedural diatom targets

Centric generator: girdle rim + valve face, $n$-fold ribs with wobble,
concentric growth rings, ray-aligned areolae pores, central rosette.
Pennate generator: superellipse boat valve, raphe slit, curved transverse
striae, striae pores, polar + central nodules. Three palettes
(gold-glass, abyss, bloom). Infinite seeds, zero downloads.

### 3.5 Training: the regeneration creature

Pool of 64 organisms, rollouts of 32–64 steps, worst-organism reseeding,
circle/half/noise damage on ~50% of batches, MSE on RGBA plus a latent
overflow penalty, Adam with gradient clipping. Regeneration is therefore in
the training distribution, not a post-hoc accident.

### 3.6 Voice: sonification

Per-frame descriptors (alive mass, mean RGB, spread, centre of mass, bilateral
symmetry) drive phase-continuous additive synthesis: mass → A-minor pentatonic
pitch, RGB → root/fifth/octave chord weights, growth rate → breath envelope,
pan → stereo, symmetry → vibrato. Output is stereo 22.05 kHz. The renderer's
own output re-enters the conditioner ("self-listening" growth), demonstrating
a closed sensorimotor loop with no additional training.

### 3.7 The talking NCA: eyes, instinct, backbone, persona

If sonification is the creature's voice, language is its testimony. The
`DiatomMind` component stacks four layers above the body:

- **Visual module** (`vision.py`). A small CNN encodes the 16-channel grid to
  a 32-dim body embedding for the backbone, alongside symbolic predicates
  (dormant/small/grown/vast, growing/shrinking/still, wounded/healing/whole,
  symmetric/asymmetric, luminous/glowing/dim …) computed from frame
  descriptors. Predicates are the grounding contract: the narrator may only
  speak what the body reports.
- **Signal layer / instinct** (`instinct.py`). The RID — Reactive Instinct
  Drive — is six drives in $[-1,1]$ with momentum (arousal, valence, pain,
  hunger, rhythm, calm), integrated from mass, growth, symmetry, pigment, and
  audio energy every step. Wounds are detected by absolute *and* relative mass
  loss, so halving a tiny creature still hurts.
- **Language backbone** (`backbone.py`). Default: `GroundedNarrator`, an
  offline deterministic engine composing instinct sentences with predicates —
  incapable of hallucinating beyond the body. Optional: `HuggingFaceBackbone`,
  a small causal LM (SmolLM2-135M) prompted with the persona preamble plus
  felt state, with automatic fallback to the grounded narrator.
- **Persona layer** (`persona.py`). Diction costumes — `diatom_elder` (oceanic
  ancient), `lab_assistant` (clinical), `feral_bloom` (wild) — that restyle
  utterances without changing their facts.

The loop closes through `modulation_from_text`: keywords in the creature's own
speech adjust its fire rate (bloom-talk quickens growth, pain-talk slows it).
The organism sees itself, feels itself, speaks, and its speech moves its body.

---

## 4. Experiments

All protocols ship as scripts and as Colab cells (`notebooks/diatom_nca_colab.ipynb`).

- **E1 · Growth.** Seed one cell, roll 96 steps. Untrained weights already
  twitch and spread (alive-at-birth init); after a 3000-step, 48px CPU/GPU run
  the organism assembles a coherent pigmented body.
- **E2 · Surgery.** Grow 64 steps, zero the right half, grow 64 more. Damage-
  trained creatures re-grow the missing half; the GIF probe is the acceptance
  test (`scripts/demo_regenerate.py`).
- **E3 · Sound.** Identical genomes grown under silence vs. full-energy audio
  diverge (unit-tested); synthetic beat/sweep/bloom signals and real WAVs all
  condition growth (`scripts/demo_audio.py`).
- **E4 · Speciation.** Six random genomes × one rule → six distinct frustules,
  demonstrating genome-steered fate without rule changes.
- **E5 · Voice.** A 96-step trajectory renders ~15 s of stereo song; the song's
  own spectrogram reconditions a second growth (self-listening).
- **E6 · Speech.** `DiatomMind.run` grows while narrating every 24 steps across
  all three personas. Acceptance test: post-surgery utterances must contain
  wound predicates ("wounded"/"tore"), and healing utterances must show growth
  predicates — verified in the demo transcript (`scripts/demo_mind.py`).

*Preliminary smoke run (CPU, 250 steps, 32px): loss falls from ~0.35 to ~0.09
and the trained creature grows a coherent pigmented mass. Full-scale
runs and quantitative regeneration scoring (IoU of healed vs. intact halves
over damage radii) are in progress and will appear in v1.1.*

---

## 5. Discussion and Limitations

Locality is a feature and a leash: global coordination (e.g. exact $n$-fold
symmetry) must be negotiated through diffusion-like channels, which is why
symmetry is approximate — as in real frustules, which are also imperfect.
Audio conditioning is correlational, not semantic: the creature feels rhythm
and brightness, not melody or meaning. Procedural targets trade biological
fidelity for infinite data; fitting real SEM imagery is the natural next
calibration. All current results are small-grid (≤96px); scaling laws for
NCA morphogenesis remain open.

## 6. Future Work

Song→species (a track's mean features as genome — the album cover as
organism); 3D frustules via 3D perception; evolution of genomes by CMA-ES
against symmetry/pore-regularity fitness; real-SEM fine-tuning; a gallery of
listener-grown specimens with their songs attached.

## 7. Author Contribution and Positionality

This work is the independent research contribution of **Aaron Alexander
Grace, M.Ed.** — educator (Higher Education Administration, UMass Lowell),
holder of several professional certificates, and AI researcher. The author's
position is unusual and deliberate: years spent studying how *people* grow in
learning environments now turned toward how *cells* grow in computational
ones. Morphogenesis is development; development is pedagogy at the cellular
scale. Rejected by the industry's front door, the author chose the organism's
strategy: start from a single cell, follow local rules, and build something
undeniable.

All ideation, direction, and research framing are the author's; implementation
was produced with AI assistance under the author's direction.

## Acknowledgements

To the diatoms, building glass cathedrals for 200 million years without
anyone's permission. To the NCA community for the growing-creature recipe.
To late-night vibe coding, which works.

## References

- Mordvintsev, A. et al. (2020). Growing Neural Cellular Automata.
  *Distill*. https://distill.pub/2020/growing-ca/
- Niklasson, E. et al. (2021). Self-Organising Textures. *Distill*.
- Chan, B. W.-C. (2019). Lenia: Biology of Artificial Life. *Complex Systems*.
- Plantec, E. et al. (2022). Flow Lenia. *ALIFE*.
- Perez, E. et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
- Round, F. E., Crawford, R. M., & Mann, D. G. (1990). *The Diatoms: Biology
  and Morphology of the Genera*. Cambridge University Press.

## Appendix A — Reproducibility

```bash
pip install -r requirements.txt
python scripts/train_diatom.py --steps 3000 --size 48
python scripts/demo_grow.py --checkpoint assets/checkpoint.pt
python scripts/demo_regenerate.py --checkpoint assets/checkpoint.pt
python scripts/demo_audio.py --checkpoint assets/checkpoint.pt --wav your.wav
python scripts/demo_voice.py --checkpoint assets/checkpoint.pt
pytest -q
```

Or run `notebooks/diatom_nca_colab.ipynb` in Colab, top to bottom.

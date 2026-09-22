"use strict";

// Same map as nca/morph.py. tests/test_morph_parity.py checks the two.
// Wrapped so a classic <script> does not leak names into the page.
(function () {

function clip(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function modulationFromText(text) {
  const t = String(text || "").toLowerCase();
  let fire = 1;
  let energy = 0;
  for (const w of ["bloom", "spread", "more", "now", "motion", "ecstatic", "quickens"]) {
    if (t.includes(w)) fire += 0.15;
  }
  for (const w of ["rest", "still", "quiet", "sleep", "waiting", "quiescent"]) {
    if (t.includes(w)) fire -= 0.15;
  }
  for (const w of ["tore", "torn", "pain", "damage", "wounded"]) {
    if (t.includes(w)) fire -= 0.2;
  }
  for (const w of ["pulse", "beat", "drum", "sing", "tide"]) {
    if (t.includes(w)) energy += 0.2;
  }
  return [clip(fire, 0.3, 2), clip(energy, 0, 1)];
}

function poseFromState(rid, utterance, userText, genomeFold, damagePulse) {
  const vals = (rid || []).concat([0, 0, 0, 0, 0, 0]);
  const a = +vals[0];
  const v = +vals[1];
  const p = +vals[2];
  const h = +vals[3];
  const r = +vals[4];
  const c = +vals[5];
  const [fire, energy] = modulationFromText(`${utterance || ""} ${userText || ""}`);
  let folds = genomeFold | 0;
  if (folds < 5) folds = 5;
  if (folds > 16) folds = 16;
  return {
    folds: folds,
    girdle: clip(0.22 + 0.2 * h + 0.12 * (fire - 1) + 0.1 * a, 0.08, 0.72),
    dome: clip(0.16 + 0.22 * v + 0.08 * c, 0.04, 0.55),
    pores: clip(0.4 + 0.35 * r + 0.25 * energy, 0.08, 1),
    ribs: clip(0.45 + 0.3 * Math.max(a, 0) + 0.25 * (fire - 1), 0.08, 1),
    asymmetry: clip(0.65 * Math.max(p, 0) + 0.35 * Math.max(-v, 0), 0, 1),
    spin: clip(0.2 + 0.9 * Math.max(r, 0) + 0.45 * Math.max(a, 0), 0, 2.2),
    damage: clip(0.85 * Math.max(+damagePulse || 0, 0) + 0.55 * Math.max(p, 0), 0, 1),
    hue: clip(0.11 + 0.06 * v - 0.08 * Math.max(p, 0) + 0.02 * c, 0.02, 0.2),
    bloom: clip(0.35 + 0.4 * Math.max(v, 0) + 0.25 * energy, 0.1, 1),
    fire: fire,
    energy: energy,
  };
}

function stepDrives(rid, desc, growth, audioEnergy, beat, relDrop, momentum) {
  const m = momentum == null ? 0.6 : momentum;
  const blend = (old, neu) => clip(old * m + neu * (1 - m), -1, 1);
  const rgb = desc.rgb || [0, 0, 0];
  const sumRgb = +rgb[0] + +rgb[1] + +rgb[2];
  const ae = audioEnergy || 0;
  const bt = beat || 0;
  const rd = relDrop || 0;
  return [
    blend(rid[0], growth * 30 + ae * 1.5),
    blend(rid[1], desc.symmetry * 1.2 - 0.4 + sumRgb / 3),
    blend(rid[2], growth < -0.02 || rd > 0.3 ? 1 : 0),
    blend(rid[3], desc.mass < 0.15 && Math.abs(growth) < 0.003 ? 0.8 : -0.5),
    blend(rid[4], ae * (0.5 + 0.5 * (1 - Math.abs(bt - 0.5) * 2))),
    blend(rid[5], Math.abs(growth) < 0.002 && growth > -0.005 ? 0.8 : -0.6),
  ];
}

function evalCase(c) {
  if (c.op === "drives") {
    return stepDrives(c.rid, c.desc, c.growth, c.audio_energy || 0, c.beat || 0, c.rel_drop || 0, c.momentum);
  }
  return poseFromState(c.rid, c.utterance || "", c.user_text || "", c.genome_fold == null ? 8 : c.genome_fold, c.damage_pulse || 0);
}

const api = { clip, modulationFromText, poseFromState, stepDrives, evalCase };

if (typeof module !== "undefined" && module.exports) {
  module.exports = api;
}
if (typeof window !== "undefined") {
  window.DiatomMorph = api;
}

if (typeof require !== "undefined" && require.main === module) {
  const fs = require("fs");
  const raw = fs.readFileSync(0, "utf8").trim();
  if (!raw) {
    process.stderr.write("expected JSON cases on stdin\n");
    process.exit(1);
  }
  const out = JSON.parse(raw).map(evalCase);
  process.stdout.write(JSON.stringify(out));
}
})();

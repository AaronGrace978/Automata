"use strict";

// The body. A line-for-line port of nca/model.py DiatomNCA.update, with
// TEMPLATE_CH (10) written each step from the current glyph, AUDIO (11)
// integrating loudness, and FiLM from the audio vector. Weights come from
// web/weights.js (scripts/export_weights.py).
(function () {
  const C = 16;
  const AUDIO = 8;
  const ALPHA = 3;
  const TEMPLATE_CH = 10;
  const AUDIO_CH = 11;

  function randn() {
    let u = 0;
    let v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  class DiatomBody {
    constructor(size, weights) {
      this.size = size;
      this.n = size * size;
      this.state = new Float32Array(C * this.n);
      this.next = new Float32Array(C * this.n);
      this.template = new Float32Array(this.n);
      this.audio = new Float32Array(AUDIO);
      this.fireRate = 0.5;
      this.genome = new Float32Array(4);
      this.trained = false;
      this.hidden = 128;
      this.setWeights(weights);
      this.newGenome(0.2);
    }

    setWeights(w) {
      if (!w) {
        this.hidden = 64;
        const rnd = (n, s) => Float32Array.from({ length: n }, () => randn() * s);
        this.W1 = Array.from({ length: this.hidden }, () => rnd(C * 3, 0.12));
        this.b1 = new Float32Array(this.hidden);
        this.Wf = Array.from({ length: this.hidden * 2 }, () => rnd(AUDIO, 0.07));
        this.bf = new Float32Array(this.hidden * 2);
        this.W2 = Array.from({ length: C }, () => rnd(this.hidden, 0.02));
        this.b2 = new Float32Array(C);
        return;
      }
      this.hidden = w.hidden_dim;
      this.W1 = w.fc1_weight.map((r) => Float32Array.from(r));
      this.b1 = Float32Array.from(w.fc1_bias);
      this.Wf = w.film_weight.map((r) => Float32Array.from(r));
      this.bf = Float32Array.from(w.film_bias);
      this.W2 = w.fc2_weight.map((r) => Float32Array.from(r));
      this.b2 = Float32Array.from(w.fc2_bias);
      this.trained = true;
    }

    newGenome(scale) {
      for (let k = 0; k < 4; k++) this.genome[k] = randn() * (scale == null ? 0.2 : scale);
      this.reseed();
    }

    reseed() {
      this.state.fill(0);
      const c = this.size >> 1;
      const i = c * this.size + c;
      this.state[ALPHA * this.n + i] = 1;
      for (let k = 0; k < 4; k++) this.state[(4 + k) * this.n + i] = this.genome[k];
    }

    setTemplate(mask) {
      if (!mask) this.template.fill(0);
      else this.template.set(mask);
    }

    cutHalf() {
      const S = this.size;
      for (let ch = 0; ch < C; ch++) {
        for (let y = 0; y < S; y++) {
          for (let x = S >> 1; x < S; x++) this.state[ch * this.n + y * S + x] = 0;
        }
      }
    }

    cutDisc(ratio) {
      const S = this.size;
      const r = Math.max(2, Math.floor((ratio || 0.25) * S));
      const cy = Math.floor(Math.random() * S);
      const cx = S >> 1;
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          if ((x - cx) * (x - cx) + (y - cy) * (y - cy) > r * r) continue;
          for (let ch = 0; ch < C; ch++) this.state[ch * this.n + y * S + x] = 0;
        }
      }
    }

    step() {
      const S = this.size;
      const n = this.n;
      const H = this.hidden;
      const state = this.state;
      const next = this.next;
      const tpl = this.template;
      const audio = this.audio;

      // The template is an input, written before perception (model.update).
      for (let i = 0; i < n; i++) state[TEMPLATE_CH * n + i] = tpl[i];
      next.set(state);

      const film = new Float32Array(H * 2);
      for (let i = 0; i < H * 2; i++) {
        let s = this.bf[i];
        const w = this.Wf[i];
        for (let k = 0; k < AUDIO; k++) s += w[k] * audio[k];
        film[i] = s;
      }
      const fire = this.fireRate;
      const p = new Float32Array(C * 3);
      const h = new Float32Array(H);
      const g = (ch, yy, xx) =>
        yy < 0 || xx < 0 || yy >= S || xx >= S ? 0 : state[ch * n + yy * S + xx];

      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          if (Math.random() > fire) continue;
          let alive = false;
          for (let dy = -1; dy <= 1 && !alive; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (g(ALPHA, y + dy, x + dx) > 0.1) {
                alive = true;
                break;
              }
            }
          }
          if (!alive) continue;
          for (let ch = 0; ch < C; ch++) {
            const c00 = g(ch, y - 1, x - 1);
            const c01 = g(ch, y - 1, x);
            const c02 = g(ch, y - 1, x + 1);
            const c10 = g(ch, y, x - 1);
            const cc = state[ch * n + y * S + x];
            const c12 = g(ch, y, x + 1);
            const c20 = g(ch, y + 1, x - 1);
            const c21 = g(ch, y + 1, x);
            const c22 = g(ch, y + 1, x + 1);
            p[3 * ch] = cc;
            p[3 * ch + 1] = (c02 + c12 * 2 + c22 - c00 - c10 * 2 - c20) / 8;
            p[3 * ch + 2] = (c20 + c21 * 2 + c22 - c00 - c01 * 2 - c02) / 8;
          }
          for (let i = 0; i < H; i++) {
            let s = this.b1[i];
            const w = this.W1[i];
            for (let k = 0; k < C * 3; k++) s += w[k] * p[k];
            s = s > 0 ? s : 0;
            h[i] = s * (1 + film[i]) + film[H + i];
          }
          for (let ch = 0; ch < C; ch++) {
            let s = this.b2[ch];
            const w = this.W2[ch];
            for (let i = 0; i < H; i++) s += w[i] * h[i];
            next[ch * n + y * S + x] = state[ch * n + y * S + x] + s;
          }
        }
      }
      this.next = state;
      this.state = next;
    }

    // Same features as nca/voice.py frame_descriptor.
    describe() {
      const S = this.size;
      const n = this.n;
      const st = this.state;
      let mass = 0;
      const rgb = [0, 0, 0];
      let aliveCount = 0;
      let cxSum = 0;
      for (let i = 0; i < n; i++) {
        const a = Math.max(0, Math.min(1, st[ALPHA * n + i]));
        mass += a;
        for (let c = 0; c < 3; c++) rgb[c] += Math.max(0, Math.min(1, st[c * n + i]));
        if (a > 0.1) {
          aliveCount += 1;
          cxSum += i % S;
        }
      }
      let sym = 0;
      const half = S >> 1;
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < half; x++) {
          const l = Math.max(0, Math.min(1, st[ALPHA * n + y * S + x]));
          const r = Math.max(0, Math.min(1, st[ALPHA * n + y * S + (S - 1 - x)]));
          sym += Math.abs(l - r);
        }
      }
      return {
        mass: mass / n,
        rgb: rgb.map((v) => v / n),
        spread: aliveCount / n,
        pan: aliveCount ? cxSum / aliveCount / S : 0.5,
        symmetry: 1 - sym / (S * half),
      };
    }
  }

  const api = { DiatomBody, C, ALPHA, TEMPLATE_CH, AUDIO_CH };
  if (typeof window !== "undefined") window.DiatomNCA = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;

  // Parity harness: node nca.js < case.json
  // {weights, size, genome, template, steps, fire} -> {alive, rgb, mass}
  if (typeof require !== "undefined" && require.main === module) {
    const fs = require("fs");
    const c = JSON.parse(fs.readFileSync(0, "utf8"));
    const b = new DiatomBody(c.size, c.weights);
    b.genome.set(c.genome);
    b.reseed();
    b.fireRate = c.fire == null ? 1 : c.fire;
    if (c.template) b.setTemplate(Float32Array.from(c.template));
    if (c.audio) b.audio.set(c.audio);
    for (let i = 0; i < c.steps; i++) b.step();
    const d = b.describe();
    process.stdout.write(JSON.stringify({ alive: Math.round(d.spread * b.n), rgb: d.rgb, mass: d.mass, symmetry: d.symmetry }));
  }
})();

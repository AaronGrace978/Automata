"use strict";

// Concept -> silhouette. Same grammar and table as nca/shapes.py, baked into
// shapes.data.js by scripts/build_shapes.py.
(function () {
  const data = (typeof window !== "undefined" && window.DIATOM_SHAPES) || null;
  const STOP = new Set(data ? data.stop : []);
  const ASK = data ? new RegExp(data.ask) : null;
  let atlas = null;

  const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");

  function singular(w) {
    if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
    if (w.length > 3 && w.endsWith("es") && "sxz".includes(w[w.length - 3])) return w.slice(0, -2);
    if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
    return w;
  }

  // Candidate nouns, and whether the person asked outright ("become a cloud").
  function request(text) {
    let t = String(text || "").toLowerCase().replace(/[^a-z\- ]/g, " ").replace(/\s+/g, " ").trim();
    const m = ASK ? ASK.exec(t) : null;
    let words;
    let explicit;
    if (m) {
      words = m[1].split(" ").filter((w) => w && !STOP.has(w)).slice(0, 3);
      explicit = true;
    } else {
      words = t.split(" ").filter((w) => w && !STOP.has(w));
      explicit = false;
      if (t.split(" ").length > 4) return { cands: [], explicit: false };
    }
    const cands = [];
    if (words.length > 1) cands.push(words.join(""));
    for (const w of words) cands.push(w, singular(w));
    const seen = [];
    for (const c of cands.map(norm)) if (c && !seen.includes(c)) seen.push(c);
    return { cands: seen, explicit };
  }

  function find(text) {
    const { cands, explicit } = request(text);
    for (const c of cands) {
      if (data && Object.prototype.hasOwnProperty.call(data.index, c)) {
        return { key: c, tile: data.index[c], explicit };
      }
    }
    return { key: cands[0] || null, tile: null, explicit };
  }

  function tileForEmoji(text) {
    if (!data) return null;
    for (const ch of Array.from(String(text || ""))) {
      const cp = ch.codePointAt(0);
      const i = data.tiles.findIndex((row) => row[0] === cp);
      if (i >= 0) return i;
    }
    return null;
  }

  function load() {
    if (!data) return Promise.resolve(false);
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        atlas = ctx.getImageData(0, 0, img.width, img.height);
        resolve(true);
      };
      img.onerror = () => resolve(false);
      img.src = data.atlas;
    });
  }

  function mask(tile, size) {
    const S = data.size;
    const out = new Float32Array(size * size);
    if (!atlas || tile == null) return out;
    const r = Math.floor(tile / data.cols);
    const c = tile % data.cols;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const sy = Math.floor((y * S) / size);
        const sx = Math.floor((x * S) / size);
        out[y * size + x] = atlas.data[((r * S + sy) * atlas.width + (c * S + sx)) * 4] / 255;
      }
    }
    return out;
  }

  function info(tile) {
    const row = data.tiles[tile];
    return { emoji: String.fromCodePoint(row[0]), label: row[1] };
  }

  const api = { request, find, tileForEmoji, load, mask, info, count: data ? data.tiles.length : 0 };
  if (typeof window !== "undefined") window.DiatomShapes = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();

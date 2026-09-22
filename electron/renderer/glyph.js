"use strict";

// Word -> silica template. Same layout rule as nca/glyph.py: upper-case,
// letters only, one line up to 7 characters, otherwise split in half.
(function () {
  const MAX_LINE = 7;

  function fitLines(text) {
    const word = String(text || "").replace(/[^A-Za-z0-9]/g, "").toUpperCase();
    if (!word) return [];
    if (word.length <= MAX_LINE) return [word];
    const cut = Math.ceil(word.length / 2);
    return [word.slice(0, cut), word.slice(cut, cut + MAX_LINE)];
  }

  function wordsOf(text, limit) {
    const out = [];
    for (const raw of String(text || "").split(/\s+/)) {
      const lines = fitLines(raw);
      if (lines.length) out.push(lines.join(""));
      if (out.length >= (limit || 12)) break;
    }
    return out;
  }

  let scratch = null;

  function renderText(text, size) {
    const lines = fitLines(text);
    const mask = new Float32Array(size * size);
    if (!lines.length) return mask;
    if (!scratch || scratch.width !== size) {
      scratch = document.createElement("canvas");
      scratch.width = size;
      scratch.height = size;
    }
    const ctx = scratch.getContext("2d", { willReadFrequently: true });
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);
    const family = '"DejaVu Sans", Arial, Helvetica, sans-serif';
    const boxW = size * 0.88;
    const boxH = size * 0.8;
    let px = 6;
    for (let candidate = 6; candidate < size; candidate++) {
      ctx.font = `bold ${candidate}px ${family}`;
      const widths = lines.map((line) => ctx.measureText(line).width + 2);
      const gap = Math.max(1, Math.floor(candidate / 5));
      const totalH = lines.length * candidate * 0.78 + (lines.length - 1) * gap;
      if (Math.max(...widths) > boxW || totalH > boxH) break;
      px = candidate;
    }
    ctx.font = `bold ${px}px ${family}`;
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const gap = Math.max(1, Math.floor(px / 5));
    const lineH = px * 0.78;
    const totalH = lines.length * lineH + (lines.length - 1) * gap;
    let y = size / 2 - totalH / 2 + lineH / 2;
    for (const line of lines) {
      ctx.strokeText(line, size / 2, y);
      ctx.fillText(line, size / 2, y);
      y += lineH + gap;
    }
    const data = ctx.getImageData(0, 0, size, size).data;
    for (let i = 0; i < size * size; i++) mask[i] = data[i * 4] / 255;
    return mask;
  }

  window.DiatomGlyph = { fitLines, wordsOf, renderText, MAX_LINE };
})();

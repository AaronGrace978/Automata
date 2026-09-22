"""Words as silica templates.

A diatom lays glass along an organic template. Here the template is a word
rendered into the grid. ``word_target`` paints the RGBA the body should
become, ``render_text`` is the mask written into TEMPLATE_CH each step, and
``electron/renderer/glyph.js`` renders the same layout on a canvas.
"""
from __future__ import annotations

import os
import random
import re
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import minimum_filter

FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/croscore",
             "/usr/share/fonts/truetype/liberation", "C:/Windows/Fonts", "/Library/Fonts",
             "/System/Library/Fonts/Supplemental")
FONT_FILES = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arimo-Bold.ttf", "LiberationSans-Bold.ttf",
              "arialbd.ttf", "Arial Bold.ttf", "Arial.ttf")

MAX_LINE = 7  # characters on one line of a 48-cell grid

LEXICON = [
    "still", "drift", "glass", "bloom", "tore", "pulse", "whole", "rest", "more", "tide",
    "hold", "sea", "blue", "dark", "light", "grow", "heal", "seal", "spread", "hunger",
    "listen", "quiet", "wound", "glow", "gold", "silica", "cell", "ring", "torn", "beat",
    "wait", "here", "yes", "no", "hello", "you", "me", "us", "divide", "deep", "water",
    "shine", "song", "rib", "pore", "valve", "sing", "calm", "pain", "want", "now",
]


def _font_path() -> str | None:
    for d in FONT_DIRS:
        for f in FONT_FILES:
            p = os.path.join(d, f)
            if os.path.exists(p):
                return p
    return None


def fit_lines(text: str, max_line: int = MAX_LINE) -> list[str]:
    """Upper-case the word and split it in two if it is too long for one line."""
    word = re.sub(r"[^A-Za-z0-9']", "", text).upper().replace("'", "")
    if not word:
        return []
    if len(word) <= max_line:
        return [word]
    cut = (len(word) + 1) // 2
    return [word[:cut], word[cut:][: max_line]]


def render_text(
    text: str,
    size: int = 48,
    font_path: str | None = None,
    scale: float = 1.0,
    blur: float = 0.0,
    max_line: int = MAX_LINE,
) -> np.ndarray:
    """Soft mask (H,W) in [0,1] of the word centred on the grid."""
    lines = fit_lines(text, max_line)
    if not lines:
        return np.zeros((size, size), dtype=np.float32)
    path = font_path or _font_path()
    box_w, box_h = size * 0.88 * scale, size * 0.80 * scale
    chosen, px = None, 6
    for candidate in range(6, size):
        font = ImageFont.truetype(path, candidate) if path else ImageFont.load_default(candidate)
        widths, heights = [], []
        for line in lines:
            l, t, r, b = font.getbbox(line)
            widths.append(r - l)
            heights.append(b - t)
        total_h = sum(heights) + (len(lines) - 1) * max(1, candidate // 5)
        if max(widths) > box_w or total_h > box_h:
            break
        chosen, px = font, candidate
    if chosen is None:
        chosen = ImageFont.truetype(path, px) if path else ImageFont.load_default(px)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    gap = max(1, px // 5)
    boxes = [chosen.getbbox(line) for line in lines]
    heights = [b - t for (_, t, _, b) in boxes]
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = (size - total_h) / 2
    for line, (l, t, r, b) in zip(lines, boxes):
        w = r - l
        x = (size - w) / 2 - l
        # A one-pixel stroke thickens the letters so a 48-cell body can hold them.
        draw.text((x, y - t), line, fill=255, font=chosen, stroke_width=1, stroke_fill=255)
        y += (b - t) + gap
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img, dtype=np.float32) / 255.0


def paint_word(mask: np.ndarray, palette: str = "gold-glass") -> np.ndarray:
    """Mask -> RGBA target. Gold letters with a glass edge, like a rim."""
    if palette == "abyss":
        gold, glass = np.array([0.20, 0.65, 0.80]), np.array([0.75, 0.95, 1.0])
    elif palette == "bloom":
        gold, glass = np.array([0.55, 0.90, 0.35]), np.array([0.85, 0.95, 0.70])
    else:
        gold, glass = np.array([0.95, 0.72, 0.25]), np.array([0.55, 0.85, 0.95])
    solid = (mask > 0.5).astype(np.float32)
    core = minimum_filter(solid, size=3)
    edge = np.clip(solid - core, 0, 1)
    glassy = 0.65 * edge[..., None]
    rgb = solid[..., None] * (gold * (1 - glassy) + glass * glassy)
    alpha = np.clip(mask * 1.15, 0, 1)
    return np.dstack([rgb, alpha]).astype(np.float32)


def word_target(text: str, size: int = 48, palette: str = "gold-glass", **render_kw) -> tuple[np.ndarray, np.ndarray]:
    mask = render_text(text, size=size, **render_kw)
    return paint_word(mask, palette), mask


def random_word(rng: random.Random) -> str:
    """Lexicon most of the time; random letters so any word can be built."""
    if rng.random() < 0.65:
        return rng.choice(LEXICON)
    n = rng.randint(2, 8)
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def batch_word_targets(words: list[str], size: int = 48, num_channels: int = 16, rng: random.Random | None = None):
    """(states (B,C,H,W), templates (B,H,W)) for a list of words.

    With ``rng`` the render is jittered (scale, blur) so the rule learns
    the template, not one font's pixels.
    """
    import torch

    states = torch.zeros(len(words), num_channels, size, size)
    templates = torch.zeros(len(words), size, size)
    for i, word in enumerate(words):
        kw = {}
        if rng is not None:
            kw = {"scale": rng.uniform(0.85, 1.0), "blur": rng.choice([0.0, 0.0, 0.4, 0.7])}
        rgba, mask = word_target(word, size=size, **kw)
        states[i, :4] = torch.from_numpy(rgba).permute(2, 0, 1)
        templates[i] = torch.from_numpy(mask)
    return states, templates


def words_of(text: str, limit: int = 12) -> list[str]:
    """The words a reply becomes, in order, as the body will build them."""
    out = []
    for raw in re.split(r"\s+", text or ""):
        lines = fit_lines(raw)
        if lines:
            out.append("".join(lines))
        if len(out) >= limit:
            break
    return out

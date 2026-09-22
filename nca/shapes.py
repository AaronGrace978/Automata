"""Shapes as silica templates.

Any concept that has an emoji can be a body. Silhouettes come from Noto
Color Emoji (SIL OFL-1.1): each glyph's alpha, fitted to the grid, is the
template the rule grows into, exactly like a word (``nca/glyph.py``).

``build_index`` names every emoji the font draws (Unicode names, plus
aliases like "dino" -> T-REX). ``scripts/build_shapes.py`` bakes the masks
and the lookup table into ``electron/renderer/shapes.{png,json}`` so the
desktop app uses the same silhouettes and the same matching.
"""
from __future__ import annotations

import os
import re
import unicodedata

import numpy as np

EMOJI_FONTS = (
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "C:/Windows/Fonts/seguiemj.ttf",
)
FIT = 42  # longest side of a silhouette on a 48-cell grid

ALIASES = {
    "dino": "T-REX", "dinosaur": "T-REX", "trex": "T-REX", "tyrannosaurus": "T-REX", "raptor": "T-REX",
    "brontosaurus": "SAUROPOD", "longneck": "SAUROPOD", "diplodocus": "SAUROPOD",
    "kitty": "CAT FACE", "kitten": "CAT FACE", "cat": "CAT FACE", "puppy": "DOG FACE", "dog": "DOG FACE",
    "doggo": "DOG FACE", "heart": "HEAVY BLACK HEART", "love": "HEAVY BLACK HEART",
    "star": "WHITE MEDIUM STAR", "moon": "CRESCENT MOON", "sun": "BLACK SUN WITH RAYS",
    "house": "HOUSE BUILDING", "home": "HOUSE BUILDING", "tree": "DECIDUOUS TREE",
    "flower": "HIBISCUS", "rain": "CLOUD WITH RAIN", "storm": "THUNDER CLOUD AND RAIN",
    "lightning": "HIGH VOLTAGE SIGN", "bolt": "HIGH VOLTAGE SIGN", "fire": "FIRE", "flame": "FIRE",
    "car": "AUTOMOBILE", "plane": "AIRPLANE", "boat": "SAILBOAT", "ship": "SHIP", "bird": "BIRD",
    "fish": "FISH", "whale": "SPOUTING WHALE", "horse": "HORSE", "bunny": "RABBIT FACE",
    "rabbit": "RABBIT FACE", "bear": "BEAR FACE", "skull": "SKULL", "ghost": "GHOST", "alien": "EXTRATERRESTRIAL ALIEN",
    "robot": "ROBOT FACE", "smile": "SMILING FACE WITH OPEN MOUTH", "face": "SMILING FACE WITH OPEN MOUTH",
    "mountain": "MOUNTAIN", "wave": "WATER WAVE", "planet": "RINGED PLANET", "saturn": "RINGED PLANET",
    "earth": "EARTH GLOBE AMERICAS", "world": "EARTH GLOBE AMERICAS", "snowflake": "SNOWFLAKE",
    "snow": "SNOWFLAKE", "leaf": "LEAF FLUTTERING IN WIND", "mushroom": "MUSHROOM",
    "butterfly": "BUTTERFLY", "spider": "SPIDER", "snake": "SNAKE", "turtle": "TURTLE",
    "shark": "SHARK", "crab": "CRAB", "jellyfish": "JELLYFISH", "spaceship": "ROCKET", "ufo": "FLYING SAUCER",
    "diatom": None,
}
# Words that never name a shape on their own ("i want a ...", "the", ...).
STOP = {
    "a", "an", "the", "some", "me", "my", "you", "your", "i", "it", "into", "to", "be", "like", "please",
    "now", "and", "of", "with", "for", "can", "could", "would", "want", "make", "become", "turn", "show",
    "give", "morph", "shape", "form", "yourself", "say", "is", "are", "am", "that", "this", "so", "just",
    "hello", "hi", "hey", "yes", "no", "ok", "okay", "what", "how", "why", "who", "hear", "heard",
    "black", "white", "heavy", "sign", "symbol", "with", "face", "open", "mouth", "light", "dark", "medium",
    "small", "large", "big", "little", "grow", "glass", "listen", "rest", "sing", "tide", "blade", "half",
    "new", "frustule", "again", "back", "normal", "yourself", "thing", "something", "there", "here",
}
# Boxed letters, keycaps and signs are text, not silhouettes.
TEXTY = ("SQUARED", "CIRCLED", "NEGATIVE", "INPUT SYMBOL", "KEYCAP", "BUTTON", "SIGN", "MARK", "ARROW",
         "PARENTHESIZED", "DOUBLE", "IDEOGRAPH")
ASK = re.compile(
    r"\b(?:become|be|turn(?:\s+yourself)?\s+into|morph(?:\s+yourself)?\s+into|transform\s+into|change\s+into|"
    r"shape(?:\s+yourself)?\s+into|make(?:\s+yourself)?(?:\s+into)?|show\s+me|i\s+want|give\s+me|draw|form)"
    r"\s+(?:(?:a|an|the|some|me\s+a|like\s+a)\s+)?([a-z][a-z\- ]*)"
)


def emoji_font() -> str | None:
    for p in EMOJI_FONTS:
        if os.path.exists(p):
            return p
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _skip(cp: int) -> bool:
    return (
        cp < 0x2000
        or 0x1F1E6 <= cp <= 0x1F1FF  # regional indicators
        or 0x1F3FB <= cp <= 0x1F3FF  # skin tones
        or cp in (0x200D, 0x20E3, 0xFE0F, 0xFE0E)
        or cp >= 0xE0000
    )


def font_codepoints(path: str) -> list[int]:
    from fontTools.ttLib import TTFont

    cmap = TTFont(path, fontNumber=0, lazy=True).getBestCmap()
    return sorted(cp for cp in cmap if not _skip(cp) and _name(cp))


def _name(cp: int) -> str:
    try:
        return unicodedata.name(chr(cp))
    except ValueError:
        return ""


def build_index(codepoints: list[int]) -> dict[str, int]:
    """keyword -> codepoint. Aliases first, then full names, then single words
    of names (the shortest name containing that word wins)."""
    by_name = {_name(cp): cp for cp in codepoints}
    index: dict[str, int] = {}
    word_best: dict[str, tuple[int, int]] = {}
    for name, cp in by_name.items():
        index.setdefault(_norm(name), cp)
        if any(t in name for t in TEXTY):
            continue
        for w in re.split(r"[\s\-]+", name.lower()):
            if len(w) < 3 or w in STOP:
                continue
            best = word_best.get(w)
            if best is None or len(name) < best[0]:
                word_best[w] = (len(name), cp)
    for w, (_, cp) in word_best.items():
        index.setdefault(_norm(w), cp)
    for alias, name in ALIASES.items():
        if name is None:
            index.pop(_norm(alias), None)
        elif name in by_name:
            index[_norm(alias)] = by_name[name]
    return index


def request(text: str) -> tuple[list[str], bool]:
    """Candidate nouns in what the person said, and whether they asked for a
    shape outright ("become a cloud") rather than just naming one ("dino")."""
    t = re.sub(r"[^a-z\- ]", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    m = ASK.search(t)
    if m:
        words = [w for w in m.group(1).split() if w not in STOP][:3]
        explicit = True
    else:
        words = [w for w in t.split() if w not in STOP]
        explicit = False
        if len(t.split()) > 4:
            return [], False
    cands = []
    if len(words) > 1:
        cands.append("".join(words))
    for w in words:
        cands += [w, _singular(w)]
    seen = []
    for c in cands:
        c = _norm(c)
        if c and c not in seen:
            seen.append(c)
    return seen, explicit


def find_shape(text: str, index: dict[str, int]) -> tuple[str | None, int | None, bool]:
    """(keyword, codepoint, explicit). codepoint is None when nothing matched."""
    cands, explicit = request(text)
    for c in cands:
        if c in index:
            return c, index[c], explicit
    return (cands[0] if cands else None), None, explicit


def render_shape(cp: int, size: int = 48, font_path: str | None = None, fit: int | None = None) -> np.ndarray:
    """Soft silhouette mask (H,W) in [0,1], centred, longest side ``fit``."""
    from PIL import Image, ImageDraw, ImageFont

    path = font_path or emoji_font()
    if path is None:
        raise FileNotFoundError("no emoji font: install fonts-noto-color-emoji")
    font = ImageFont.truetype(path, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((80, 80), chr(cp), font=font, embedded_color=True, anchor="mm")
    box = im.getbbox()
    out = np.zeros((size, size), dtype=np.float32)
    if box is None:
        return out
    im = im.crop(box)
    fit = fit or round(size * FIT / 48)
    s = fit / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)[..., 3] / 255.0
    y0, x0 = (size - a.shape[0]) // 2, (size - a.shape[1]) // 2
    out[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = a
    return out


def shape_target(cp: int, size: int = 48, palette: str = "gold-glass", **kw) -> tuple[np.ndarray, np.ndarray]:
    from .glyph import paint_word

    mask = render_shape(cp, size=size, **kw)
    return paint_word(mask, palette), mask


_CACHE: dict = {}


def default_index() -> dict[str, int]:
    if "index" not in _CACHE:
        path = emoji_font()
        _CACHE["index"] = build_index(font_codepoints(path)) if path else {}
    return _CACHE["index"]


def label(cp: int) -> str:
    return _name(cp).title()

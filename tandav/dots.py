from __future__ import annotations

import cv2
import numpy as np

# 5x7 dot-matrix glyphs: the same fixed shapes an LCD pages through.
FONT: dict[str, list[str]] = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
}

GLYPH_COLS = 5
GLYPH_ROWS = 7
CHAR_ADVANCE = 6  # five dot columns plus one blank column


def text_width(text: str, pitch: int) -> int:
    return max(0, len(text) * CHAR_ADVANCE * pitch - pitch)


def draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    pitch: int,
    value: int = 255,
    dot_scale: float = 0.34,
) -> int:
    """Draw `text` as round dots. Returns the x cursor after the last glyph."""
    radius = max(1, int(round(pitch * dot_scale)))
    color = int(max(0, min(255, value)))
    cx = int(x)
    for ch in text:
        pattern = FONT.get(ch)
        if pattern is not None:
            for row, bits in enumerate(pattern):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        cv2.circle(
                            img,
                            (cx + col * pitch, int(y) + row * pitch),
                            radius,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )
        cx += CHAR_ADVANCE * pitch
    return cx


def render_glyph(ch: str, pitch: int, value: int = 255) -> np.ndarray | None:
    """Render one glyph and crop to its ink, as a template source."""
    pattern = FONT.get(ch)
    if pattern is None:
        return None
    img = np.zeros((GLYPH_ROWS * pitch + 4, GLYPH_COLS * pitch + 4), np.uint8)
    draw_text(img, ch, 2, 2, pitch, value)
    ys, xs = np.nonzero(img)
    if xs.size == 0 or ys.size == 0:
        return None
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()

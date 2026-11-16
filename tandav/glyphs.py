from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_W = 24
TEMPLATE_H = 32
_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")
_DIGITS = set("0123456789.")


@dataclass(slots=True)
class Reading:
    valid: bool
    value: float | None = None
    text: str = ""
    conf: float = 0.0
    reason: str = ""


def prepare_cell(img: np.ndarray) -> np.ndarray | None:
    resized = cv2.resize(img, (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    arr -= float(arr.mean())
    std = float(arr.std())
    if std < 1e-3:
        return None
    return arr / std


def _prep(img: np.ndarray) -> np.ndarray | None:
    return prepare_cell(img)


def segment_cells(roi: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find digit cells in a field crop despite dot gaps and vibration.

    Dots within one glyph are bridged with a closing kernel scaled to the
    field height; the blank column between glyphs is wider than that kernel,
    so digits stay separate.
    """
    if roi.size == 0:
        return []
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if float(bw.mean()) > 127.0:
        bw = cv2.bitwise_not(bw)

    h = roi.shape[0]
    k = max(3, int(round(h * 0.09)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    cols = np.count_nonzero(closed, axis=0)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x, count in enumerate(cols):
        if count > 0 and start is None:
            start = x
        elif count == 0 and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(cols)))

    # A dot column can drop out for a frame or two under PWM/defocus. Bridge
    # gaps that are small relative to the glyph height, but never the wider
    # blank column between two digits.
    merge_gap = max(2, int(round(h * 0.06)))
    merged: list[tuple[int, int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)

    boxes: list[tuple[int, int, int, int]] = []
    for x0, x1 in merged:
        band = closed[:, x0:x1]
        rows = np.flatnonzero(np.any(band > 0, axis=1))
        if rows.size == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        bw_ = x1 - x0
        bh_ = y1 - y0
        if bw_ < 2 or bh_ < 3:
            continue
        if bw_ > roi.shape[1] * 0.85:
            continue
        boxes.append((x0, y0, bw_, bh_))

    if not boxes:
        return []
    # A droplet or a stray highlight is a small blob; a glyph is a tall cell.
    # Keeping only near-full-height cells removes those distractors before
    # recognition instead of letting one veto the whole field.
    max_h = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= 0.5 * max_h]
    return sorted(boxes, key=lambda b: b[0])


class GlyphDictionary:
    def __init__(
        self,
        templates: dict[str, list[np.ndarray]],
        *,
        min_score: float = 0.55,
        min_margin: float = 0.04,
    ):
        self.templates = templates
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)

    @classmethod
    def load(
        cls, directory: str | Path, **kwargs
    ) -> "GlyphDictionary":
        templates: dict[str, list[np.ndarray]] = {}
        d = Path(directory)
        if d.is_dir():
            for path in sorted(d.glob("*.png")):
                label = path.stem.split("_")[0]
                if label not in _DIGITS:
                    continue
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                prepared = _prep(img)
                if prepared is not None:
                    templates.setdefault(label, []).append(prepared)
        return cls(templates, **kwargs)

    @property
    def empty(self) -> bool:
        return not self.templates

    def recognize(self, cell: np.ndarray) -> tuple[str | None, float, float]:
        prepared = _prep(cell)
        if prepared is None or not self.templates:
            return None, 0.0, 0.0
        best_by_label: dict[str, float] = {}
        for label, samples in self.templates.items():
            best = -1.0
            for tmpl in samples:
                if tmpl.shape != prepared.shape:
                    continue
                best = max(best, float(np.mean(prepared * tmpl)))
            if best > -1.0:
                best_by_label[label] = best
        if not best_by_label:
            return None, 0.0, 0.0
        ranked = sorted(best_by_label.items(), key=lambda kv: kv[1], reverse=True)
        label, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best - second if len(ranked) > 1 else best
        return label, best, margin

    def read_number(
        self,
        roi: np.ndarray,
        *,
        decimals: int = 0,
        lo: float = 0.0,
        hi: float = 100000.0,
    ) -> Reading:
        if self.empty:
            return Reading(False, reason="glyph dictionary empty")
        boxes = segment_cells(roi)
        if not boxes:
            return Reading(False, reason="no glyph cells")
        if len(boxes) > 8:
            return Reading(False, reason="too many glyph cells")

        text_parts: list[str] = []
        scores: list[float] = []
        for x, y, w, h in boxes:
            cell = roi[y : y + h, x : x + w]
            label, score, margin = self.recognize(cell)
            if label is None:
                return Reading(False, reason="unrecognized cell")
            if score < self.min_score:
                return Reading(False, reason=f"low glyph score {score:.2f}")
            if margin < self.min_margin:
                return Reading(False, reason=f"ambiguous glyph {margin:.2f}")
            text_parts.append(label)
            scores.append(score)

        text = "".join(text_parts)
        if not _NUMBER_RE.match(text):
            return Reading(False, reason=f"not a number: {text!r}")
        if text.count(".") > 1 or (text.count(".") == 1 and decimals <= 0):
            return Reading(False, reason="unexpected decimal point")
        if decimals > 0 and "." in text:
            frac = len(text.split(".", 1)[1])
            if frac != decimals:
                return Reading(False, reason="decimal width mismatch")
        try:
            value = float(text)
        except ValueError:
            return Reading(False, reason="parse failure")
        if not lo <= value <= hi:
            return Reading(False, reason=f"value {value} outside [{lo}, {hi}]")
        return Reading(True, value=value, text=text, conf=min(scores))

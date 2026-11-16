from __future__ import annotations

import cv2
import numpy as np

from .calibration import Calibration, PageSpec


def masked_zncc(
    image: np.ndarray, template: np.ndarray, mask: np.ndarray
) -> float:
    """Zero-mean normalized correlation over the mask."""
    m = mask.astype(bool)
    if int(m.sum()) < 64:
        return 0.0
    a = image[m].astype(np.float64)
    b = template[m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-9:
        return 0.0
    return float((a * b).sum() / denom)


def _patch_score(
    image: np.ndarray, template: np.ndarray, rect: tuple[int, int, int, int], pad: int
) -> float:
    x, y, w, h = rect
    if w < 4 or h < 4:
        return 0.0
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    tpl = template[y : y + h, x : x + w]
    roi = image[y0:y1, x0:x1]
    if (
        tpl.size == 0
        or roi.shape[0] < tpl.shape[0]
        or roi.shape[1] < tpl.shape[1]
    ):
        return 0.0
    result = cv2.matchTemplate(roi.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    if result.size == 0:
        return 0.0
    return float(result.max())


class PageGate:
    """Prove which page is displayed before trusting any number.

    Each page carries a few tight static-label checks. The gate score is the
    second-best check, so one obscured label cannot invalidate the page while
    a different page (which fails both discriminating labels) must still fail.
    Older calibrations without checks fall back to the masked whole-page score.
    """

    def __init__(
        self,
        calibration: Calibration,
        *,
        width: int = 200,
        height: int = 100,
        min_score: float = 0.45,
        margin: float = 0.05,
        search_pad: int = 6,
    ):
        self.width = int(width)
        self.height = int(height)
        self.min_score = float(min_score)
        self.margin = float(margin)
        self.search_pad = int(search_pad)
        self.pages: list[tuple[PageSpec, np.ndarray, np.ndarray | None]] = []
        sx = self.width / float(calibration.width)
        sy = self.height / float(calibration.height)

        for page in calibration.pages.values():
            tpl = cv2.resize(
                page.template, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            mask = None
            if not page.checks:
                mask = np.ones((self.height, self.width), np.uint8)
                for x, y, w, h in page.mask_rects:
                    x0 = max(0, min(self.width, int(round(x * sx))))
                    y0 = max(0, min(self.height, int(round(y * sy))))
                    x1 = max(0, min(self.width, int(round((x + w) * sx))))
                    y1 = max(0, min(self.height, int(round((y + h) * sy))))
                    if x1 > x0 and y1 > y0:
                        mask[y0:y1, x0:x1] = 0
            self.pages.append((page, tpl, mask))
        self.last_scores: dict[str, float] = {}

    def _page_score(self, aligned: np.ndarray, page: PageSpec) -> float:
        if page.checks:
            scores = sorted(
                (
                    _patch_score(aligned, page.template, rect, self.search_pad)
                    for rect in page.checks
                ),
                reverse=True,
            )
            if not scores:
                return 0.0
            # One check is enough only when the calibration provides one.
            return scores[1] if len(scores) > 1 else scores[0]

        for candidate, tpl, mask in self.pages:
            if candidate.name == page.name and mask is not None:
                thumb = cv2.resize(
                    aligned,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float32)
                return masked_zncc(thumb, tpl, mask)
        return 0.0

    def match(self, aligned: np.ndarray) -> tuple[str | None, float]:
        ranked: list[tuple[str, float]] = []
        self.last_scores = {}
        for page, _tpl, _mask in self.pages:
            score = self._page_score(aligned, page)
            self.last_scores[page.name] = score
            ranked.append((page.name, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return None, 0.0
        best_name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if best < self.min_score:
            return None, max(0.0, best)
        if len(ranked) > 1 and best - second < self.margin:
            return None, max(0.0, best)
        return best_name, float(modest(best))

    def match_debug(self, aligned: np.ndarray) -> dict[str, float]:
        self.match(aligned)
        return dict(self.last_scores)


def modest(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

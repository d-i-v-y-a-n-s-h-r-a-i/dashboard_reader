from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .calibration import Calibration


class Aligner:
    """Track the dashboard with ECC against a static anchor patch.

    ECC returns the warp that samples the current frame in nominal dashboard
    coordinates, so one full-frame warp puts every calibrated crop back where
    calibration said it was. Validating the warp keeps a glare or a hand from
    teleporting the crop boxes.
    """

    def __init__(
        self,
        calibration: Calibration,
        *,
        enabled: bool = True,
        scale: float = 0.5,
        max_iters: int = 25,
        eps: float = 1e-4,
        min_score: float = 0.55,
        max_shift_px: float = 40.0,
        max_rot_deg: float = 5.0,
        max_scale_dev: float = 0.08,
        fallback_s: float = 0.75,
    ):
        self.enabled = bool(enabled)
        self.scale = max(0.2, min(1.0, float(scale)))
        self.max_iters = int(max_iters)
        self.eps = float(eps)
        self.min_score = float(min_score)
        self.max_shift = float(max_shift_px)
        self.max_rot = math.radians(float(max_rot_deg))
        self.max_scale_dev = float(max_scale_dev)
        self.fallback_s = float(fallback_s)

        self.rect = calibration.anchor_rect
        self.template = calibration.anchor_template
        self.width = calibration.width
        self.height = calibration.height

        x, y, w, h = self.rect
        sw = max(16, int(round(w * self.scale)))
        sh = max(16, int(round(h * self.scale)))
        self._small_size = (sw, sh)
        self._template_small = cv2.resize(
            self.template, (sw, sh), interpolation=cv2.INTER_AREA
        ).astype(np.float32)

        self._last_warp: np.ndarray | None = None
        self._last_quality_t = 0.0
        self.last_score = 0.0
        self.rejected = 0

    def _identity(self) -> np.ndarray:
        return np.eye(2, 3, dtype=np.float32)

    def _valid(self, warp_full: np.ndarray) -> bool:
        a = warp_full[:, :2]
        scale = math.hypot(float(a[0, 0]), float(a[1, 0]))
        rot = math.atan2(float(a[1, 0]), float(a[0, 0]))
        shift = math.hypot(float(warp_full[0, 2]), float(warp_full[1, 2]))
        return (
            abs(scale - 1.0) <= self.max_scale_dev
            and abs(rot) <= self.max_rot
            and shift <= self.max_shift
        )

    def align(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.enabled:
            return frame, 1.0

        x, y, w, h = self.rect
        roi = frame[y : y + h, x : x + w]
        if roi.shape != self.template.shape:
            return frame, 0.0

        roi_small = cv2.resize(
            roi, self._small_size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        warp = self._identity()
        score = 0.0
        try:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.max_iters,
                self.eps,
            )
            score, warp = cv2.findTransformECC(
                self._template_small,
                roi_small,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )
            warp_full = warp.copy()
            warp_full[0, 2] *= self.scale
            warp_full[1, 2] *= self.scale
        except cv2.error:
            score = 0.0
            warp_full = self._identity()

        now = time.monotonic()
        good = float(score) >= self.min_score and self._valid(warp_full)
        if good:
            self._last_warp = warp_full
            self._last_quality_t = now
        else:
            self.rejected += 1
            if (
                self._last_warp is not None
                and now - self._last_quality_t <= self.fallback_s
            ):
                warp_full = self._last_warp
            else:
                warp_full = self._identity()

        aligned = cv2.warpAffine(
            frame,
            warp_full,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        self.last_score = float(max(0.0, min(1.0, score)))
        return aligned, self.last_score

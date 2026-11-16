from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

from .dots import draw_text, render_glyph, text_width
from .types import Frame

# Geometry is shared with the calibration builder so the bench rig and the
# real rig use exactly the same extraction code.
W = 400
H = 200
ANCHOR_RECT = (8, 6, 120, 30)
SPEED_RECT = (90, 38, 270, 84)
RPM_RECT = (38, 148, 150, 32)
SPEED_PITCH = 10
RPM_PITCH = 4

PAGE_MAIN = "MAIN_TELEMETRY"
PAGE_FUEL = "FUEL"

# Tight static patches used to prove which page is on screen. Two labels per
# page plus the shared header: the second-best score tolerates one being hit by
# a droplet or glare.
PAGE_CHECKS = {
    PAGE_MAIN: [
        (14, 10, 72, 24),    # TANDAV header (shared)
        (310, 121, 46, 17),  # km/h
        (35, 131, 32, 14),   # RPM
    ],
    PAGE_FUEL: [
        (14, 10, 72, 24),    # TANDAV header (shared)
        (49, 58, 92, 30),    # FUEL
        (49, 155, 112, 15),  # RANGE 120 KM
    ],
}

MAIN_FIELDS = {
    "speed": {
        "rect": SPEED_RECT,
        "glyph_set": "speed",
        "decimals": 0,
        "min": 0,
        "max": 99,
    },
    "rpm": {
        "rect": RPM_RECT,
        "glyph_set": "rpm",
        "decimals": 0,
        "min": 0,
        "max": 9990,
    },
}


class DashboardRenderer:
    """A clean, nominal rendering of the paged LCD (no camera effects)."""

    W = W
    H = H
    anchor_rect = ANCHOR_RECT
    speed_rect = SPEED_RECT
    rpm_rect = RPM_RECT
    pages = (PAGE_MAIN, PAGE_FUEL)

    def render(
        self,
        page: str = PAGE_MAIN,
        speed: int = 0,
        rpm: int = 0,
        brightness: float = 1.0,
    ) -> np.ndarray:
        b = max(0.0, min(1.0, brightness))
        img = np.full((H, W), 5, np.uint8)

        # Static bezel and header: these are the alignment anchor and must be
        # present on every page, independent of what the paged display shows.
        cv2.rectangle(img, (3, 3), (W - 5, H - 5), int(38 * b), 1)
        cv2.putText(
            img,
            "TANDAV",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            int(185 * b),
            1,
            cv2.LINE_AA,
        )
        cv2.line(img, (8, 34), (W - 9, 34), int(70 * b), 1)

        if page == PAGE_MAIN:
            spd_text = str(int(max(0, min(99, speed))))
            spd_w = text_width(spd_text, SPEED_PITCH)
            spd_x = SPEED_RECT[0] + (SPEED_RECT[2] - spd_w) // 2
            spd_y = SPEED_RECT[1] + (SPEED_RECT[3] - 7 * SPEED_PITCH) // 2
            draw_text(img, spd_text, spd_x, spd_y, SPEED_PITCH, int(245 * b))

            cv2.putText(
                img,
                "km/h",
                (314, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                int(170 * b),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "RPM",
                (38, 142),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                int(160 * b),
                1,
                cv2.LINE_AA,
            )
            rpm_text = str(int(max(0, min(9999, rpm))))
            draw_text(img, rpm_text, 44, 152, RPM_PITCH, int(225 * b))
            cv2.putText(
                img,
                "MODE",
                (300, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                int(120 * b),
                1,
                cv2.LINE_AA,
            )
        elif page == PAGE_FUEL:
            cv2.putText(
                img,
                "FUEL",
                (52, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                int(210 * b),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "84%",
                (52, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                int(240 * b),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "RANGE 120 KM",
                (52, 166),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                int(160 * b),
                1,
                cv2.LINE_AA,
            )
        else:
            raise ValueError(f"unknown page {page!r}")
        return img

    def glyph_set(self, name: str) -> dict[str, np.ndarray]:
        pitch = SPEED_PITCH if name == "speed" else RPM_PITCH
        out: dict[str, np.ndarray] = {}
        for ch in "0123456789":
            glyph = render_glyph(ch, pitch)
            if glyph is not None:
                out[ch] = glyph
        return out


class SyntheticSource:
    """Bench camera: PWM, vibration, glare, droplets, page changes, truth."""

    def __init__(
        self,
        *,
        fps: int = 60,
        realtime: bool = True,
        page_cycle_s: float = 0.0,
        seed: int = 7,
        dashboard: DashboardRenderer | None = None,
    ):
        self.dashboard = dashboard or DashboardRenderer()
        self.fps = max(1, int(fps))
        self.realtime = bool(realtime)
        self.page_cycle_s = max(0.0, float(page_cycle_s))
        self.rng = random.Random(seed)

        self.n = 0
        self.seq = 0
        self.dx = 0.0
        self.dy = 0.0
        self.angle = 0.0
        self.speed = 0.0
        self.rpm = 3000.0
        self.disp_speed = 0
        self.disp_rpm = 3000
        self.last_slot = -1
        self.next_t = time.monotonic()
        # Frames carry a simulated monotonic clock, so a non-real-time bench
        # run still advances the filter at the true 60 fps rate.
        self.epoch_us = time.monotonic_ns() // 1000

        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        self.vignette = np.clip(1.0 - 0.22 * r**2, 0.65, 1.0).astype(np.float32)

    def _page(self, t: float) -> str:
        if self.page_cycle_s <= 0:
            return PAGE_MAIN
        phase = t % self.page_cycle_s
        return PAGE_MAIN if phase < self.page_cycle_s * 0.75 else PAGE_FUEL

    def _motion(self) -> np.ndarray:
        self.dx = 0.92 * self.dx + self.rng.gauss(0.0, 0.22)
        self.dy = 0.92 * self.dy + self.rng.gauss(0.0, 0.18)
        self.angle = 0.90 * self.angle + self.rng.gauss(0.0, 0.05)
        self.dx = max(-3.5, min(3.5, self.dx))
        self.dy = max(-3.0, min(3.0, self.dy))
        self.angle = max(-0.55, min(0.55, self.angle))
        return self.dx, self.dy, self.angle

    def _compose(self, clean: np.ndarray, t: float) -> np.ndarray:
        dx, dy, angle = self._motion()
        # A wave slam every ~7 s: short, strong, decaying translation.
        slam_phase = t % 7.0
        if slam_phase < 0.32:
            env = math.sin(math.pi * slam_phase / 0.32)
            dx += 5.5 * env * math.sin(slam_phase * 95.0)
            dy += 3.0 * env * math.cos(slam_phase * 70.0)
            angle += 1.0 * env * math.sin(slam_phase * 55.0)

        c = W / 2.0
        a = math.radians(angle)
        cos_a, sin_a = math.cos(a), math.sin(a)
        M = np.array(
            [
                [cos_a, -sin_a, c - cos_a * c + sin_a * (H / 2.0) + dx],
                [sin_a, cos_a, H / 2.0 - sin_a * c - cos_a * (H / 2.0) + dy],
            ],
            dtype=np.float32,
        )
        img = cv2.warpAffine(
            clean,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Lens glare: broad and slow, strong enough to wash a digit.
        self.glare_active = False
        glare_phase = t % 11.0
        if 3.0 < glare_phase < 6.0:
            self.glare_active = True
            alpha = 0.26 * math.sin(math.pi * (glare_phase - 3.0) / 3.0)
            overlay = np.zeros((H, W), np.uint8)
            gx = int(W * (0.55 + 0.12 * math.sin(t * 0.7)))
            gy = int(H * (0.28 + 0.10 * math.cos(t * 0.5)))
            cv2.ellipse(overlay, (gx, gy), (70, 28), 20, 0, 360, 255, -1)
            overlay = cv2.GaussianBlur(overlay, (0, 0), 22)
            img = np.clip(
                img.astype(np.float32) + overlay.astype(np.float32) * alpha, 0, 255
            ).astype(np.uint8)

        # Droplets sit on the lens, after scene motion.
        self.droplet_active = False
        if self.rng.random() < 0.16:
            self.droplet_active = True
            overlay = np.zeros((H, W), np.uint8)
            for _ in range(self.rng.randint(1, 2)):
                cx = self.rng.randint(30, W - 30)
                cy = self.rng.randint(30, H - 30)
                radius = self.rng.randint(2, 6)
                cv2.circle(overlay, (cx, cy), radius, 200, -1, cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), radius + 1, 90, 1, cv2.LINE_AA)
            overlay = cv2.GaussianBlur(overlay, (0, 0), 2.0)
            img = np.clip(
                img.astype(np.float32) + overlay.astype(np.float32) * 0.55, 0, 255
            ).astype(np.uint8)

        img = np.clip(
            img.astype(np.float32) * self.vignette
            + self.rng.gauss(0.0, 2.2)
            + np.zeros((H, W), np.float32),
            0,
            255,
        ).astype(np.uint8)
        return img

    def read(self, timeout: float | None = None) -> Frame | None:
        if self.realtime:
            now = time.monotonic()
            if self.next_t > now:
                time.sleep(min(0.25, self.next_t - now))
            elif now - self.next_t > 0.5:
                self.next_t = now  # do not spiral after a stall
            self.next_t += 1.0 / self.fps
        t = self.n / float(self.fps)

        self.speed = 34.0 + 11.0 * math.sin(t * 0.23) + 2.5 * math.sin(t * 1.7 + 1.1)
        self.speed = max(0.0, min(99.0, self.speed))
        self.rpm = 3000.0 + self.speed * 58.0 + 180.0 * math.sin(t * 2.3)
        self.rpm = max(900.0, min(9990.0, self.rpm))

        slot = int(t * 4.0)
        changed = slot != self.last_slot
        if changed:
            self.last_slot = slot
            self.disp_speed = int(round(self.speed))
            self.disp_rpm = int(round(self.rpm / 10.0) * 10)

        page = self._page(t)
        brightness = 0.90 + 0.10 * math.sin(2.0 * math.pi * 173.0 * t)
        clean = self.dashboard.render(page, self.disp_speed, self.disp_rpm, brightness)
        self.glare_active = False
        self.droplet_active = False
        img = self._compose(clean, t)

        frame = Frame(
            image=img,
            t_mono_us=self.epoch_us + int(round(t * 1_000_000.0)),
            seq=self.seq,
            truth={
                "page": page,
                "speed": self.disp_speed,
                "rpm": self.disp_rpm,
                "speed_visible": page == PAGE_MAIN,
                "lcd_slot": slot,
                "t": t,
                "changed": changed,
                "glare": self.glare_active,
                "droplet": self.droplet_active,
            },
        )
        self.seq += 1
        self.n += 1
        return frame

    def close(self) -> None:
        return None

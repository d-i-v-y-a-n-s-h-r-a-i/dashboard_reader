"""Direct Video Test: Run TANDAV OCR pipeline on screen.mp4 with real-time UI & UDP broadcast."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tandav.sender import UdpSender
from tandav.types import Telemetry

DEFAULT_VIDEO = ROOT / "data" / "screen.mp4"
VIDEO_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
TEMPLATES_DIR = ROOT / "calibration" / "real_digits"


def load_templates() -> dict[int, np.ndarray]:
    templates = {}
    for i in range(10):
        img = cv2.imread(str(TEMPLATES_DIR / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        templates[i] = (img > 90)
    return templates


def match_slot(roi: np.ndarray, templates: dict[int, np.ndarray], min_ink: int = 500) -> tuple[int | None, float]:
    if np.count_nonzero(roi) < min_ink:
        return None, 0.0

    coords = cv2.findNonZero(roi)
    if coords is None:
        return None, 0.0
    x, y, w, h = cv2.boundingRect(coords)
    if w < 12 or h < 45:
        return None, 0.0

    crop = roi[y : y + h, x : x + w]
    norm = cv2.resize(crop, (40, 60), interpolation=cv2.INTER_AREA) > 90

    best_d = None
    best_iou = -1.0
    for d, tmpl in templates.items():
        inter = np.logical_and(norm, tmpl).sum()
        union = np.logical_or(norm, tmpl).sum()
        iou = float(inter / max(1, union))
        if iou > best_iou:
            best_iou = iou
            best_d = d
    return best_d, best_iou


def main() -> int:
    templates = load_templates()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[error] Cannot open video: {VIDEO_PATH}")
        return 1

    sender = UdpSender("127.0.0.1", 47001, hz=20.0)
    print(f"[video-test] Playing {VIDEO_PATH.name} (100% Ground Truth)")
    print("[video-test] Broadcasting UDP telemetry to 127.0.0.1:47001")
    print("[video-test] Press 'Q' or ESC to exit, SPACE to pause/resume.\n")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_delay = 1.0 / fps

    cv2.namedWindow("TANDAV Direct Dashboard Test - screen.mp4", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t0 = time.time()
            display = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if otsu.mean() > 127:
                otsu = 255 - otsu

            # Read calibrated slots
            t_roi = otsu[120:255, 330:410]
            o_roi = otsu[120:255, 410:485]

            dt, st = match_slot(t_roi, templates, min_ink=500)
            do, so = match_slot(o_roi, templates, min_ink=500)

            speed: int | None = None
            conf = 0.0
            if dt is None and do is not None:
                speed = do
                conf = so
            elif dt is not None and do is not None:
                speed = dt * 10 + do
                conf = min(st, so)

            # Draw visual slot bounding boxes
            cv2.rectangle(display, (330, 120), (410, 255), (0, 255, 0) if dt is not None else (80, 80, 80), 1)
            cv2.rectangle(display, (410, 120), (485, 255), (0, 255, 0) if do is not None else (80, 80, 80), 1)

            # Draw HUD
            dh = display.shape[0]
            cv2.rectangle(display, (10, dh - 75), (420, dh - 10), (0, 0, 0), -1)
            cv2.rectangle(display, (10, dh - 75), (420, dh - 10), (0, 255, 0) if speed is not None else (0, 140, 255), 2)

            spd_text = f"SPEED: {speed if speed is not None else '--'} km/h"
            status_text = f"CONF: {conf:.2f} | MODE: SPORT MODE | FPS: {fps:.0f}"
            cv2.putText(display, spd_text, (20, dh - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0) if speed is not None else (180, 180, 180), 2)
            cv2.putText(display, status_text, (20, dh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            # Send Telemetry over UDP
            tel = Telemetry(
                page="MAIN_TELEMETRY",
                speed=float(speed) if speed is not None else None,
                rpm=None,
                conf=conf,
                valid=speed is not None,
                t_mono_us=time.monotonic_ns() // 1000,
            )
            sender.maybe_send(tel)

            # Print to terminal
            if speed is not None:
                out = {"event": "telemetry", "speed_kmh": speed, "mode": "SPORT MODE", "conf": round(conf, 2), "valid": True}
                print(json.dumps(out), flush=True)

            cv2.imshow("TANDAV Direct Dashboard Test - screen.mp4", display)
            elapsed = time.time() - t0
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                cv2.waitKey(0)

    finally:
        cap.release()
        sender.close()
        cv2.destroyAllWindows()
        print("\n[video-test] Video closed. Telemetry stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


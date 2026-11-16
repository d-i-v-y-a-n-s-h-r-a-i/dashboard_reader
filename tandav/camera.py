from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .synthetic import DashboardRenderer, SyntheticSource
from .types import Frame


class OpenCVSource:
    """Camera/file reader that never builds a latency queue.

    The reader thread overwrites a single slot; the consumer always gets the
    newest frame and older frames are dropped. This is what keeps a slow OCR
    frame from turning into a growing delay.
    """

    def __init__(self, cfg: Config, kind: str):
        self.cfg = cfg
        self.kind = kind
        self.width = int(cfg.get("source.width", 400))
        self.height = int(cfg.get("source.height", 200))
        self.fps = int(cfg.get("source.fps", 60))
        self.realtime = bool(cfg.get("source.realtime", True))
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._stop = threading.Event()
        self._seq = 0
        self._pace_period = 0.0
        self.reconnects = 0
        self.read_errors = 0

    def _gst_pipeline(self) -> str | None:
        override = self.cfg.get("source.gst_pipeline")
        if override:
            return str(override).format(
                url=self.cfg.get("source.rtsp_url", ""),
                width=self.width,
                height=self.height,
                latency=int(self.cfg.get("source.gst_latency_ms", 0)),
            )
        url = str(self.cfg.get("source.rtsp_url", ""))
        if not url:
            return None
        latency = int(self.cfg.get("source.gst_latency_ms", 0))
        # H.264 is the common recorder substream. Replace this template in
        # config if the NVR emits H.265 or MJPEG.
        return (
            f"rtspsrc location={url} protocols=tcp latency={latency} "
            "drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! "
            f"video/x-raw,format=GRAY8,width={self.width},height={self.height} ! "
            "appsink max-buffers=1 drop=true sync=false"
        )

    def _apply_v4l2_controls(self) -> None:
        controls = self.cfg.section("source.v4l2_controls")
        if not controls:
            return
        exe = shutil.which("v4l2-ctl")
        device = str(self.cfg.get("source.device", "/dev/video0"))
        if not exe:
            print("[camera] v4l2-ctl not found; controls not applied", flush=True)
            return
        for name, value in controls.items():
            cmd = [exe, "-d", device, f"--set-ctrl={name}={value}"]
            try:
                subprocess.run(cmd, timeout=2, capture_output=True, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"[camera] v4l2 control {name}: {exc}", flush=True)

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "v4l2":
            self._apply_v4l2_controls()
            cap = cv2.VideoCapture(
                str(self.cfg.get("source.device", "/dev/video0")), cv2.CAP_V4L2
            )
            fourcc = str(self.cfg.get("source.fourcc", "GREY"))
            if len(fourcc) == 4:
                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*fourcc),
                )
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        elif self.kind == "rtsp":
            backend = str(self.cfg.get("source.backend", "auto")).lower()
            cap = None
            if backend in ("auto", "gst"):
                pipeline = self._gst_pipeline()
                if pipeline and re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if cap is not None and cap.isOpened():
                        print("[camera] RTSP via GStreamer", flush=True)
                    else:
                        if cap is not None:
                            cap.release()
                        cap = None
            if cap is None:
                os.environ.setdefault(
                    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
                    "max_delay;0|reorder_queue_size;0",
                )
                cap = cv2.VideoCapture(
                    str(self.cfg.get("source.rtsp_url", "")), cv2.CAP_FFMPEG
                )
                print("[camera] RTSP via FFmpeg (TCP, low delay)", flush=True)
        elif self.kind == "gst":
            pipeline = str(self.cfg.get("source.gst_pipeline", ""))
            if not pipeline:
                raise ValueError("source.kind=gst requires source.gst_pipeline")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap is not None and cap.isOpened():
                print("[camera] custom GStreamer tap", flush=True)
        elif self.kind == "file":
            path = str(self.cfg.get("source.file", ""))
            if not path or not Path(path).exists():
                raise FileNotFoundError(f"video file missing: {path}")
            cap = cv2.VideoCapture(path)
            if self.realtime:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps)
                self._pace_period = 1.0 / max(1.0, fps)
        else:
            raise ValueError(f"unknown source kind {self.kind}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.kind in ("v4l2", "rtsp"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def start(self) -> None:
        self._cap = self._open()
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                f"cannot open {self.kind} source "
                f"({self.cfg.get('source.device') or self.cfg.get('source.rtsp_url')})"
            )
        ok, probe = self._cap.read()
        if not ok or probe is None:
            raise RuntimeError("source opened but produced no frame")
        h, w = probe.shape[:2]
        if (w, h) != (self.width, self.height):
            self._cap.release()
            raise RuntimeError(
                f"source produced {w}x{h}, expected {self.width}x{self.height}. "
                "Hardware windowing/calibration must match; refusing to rescale."
            )
        self._thread = threading.Thread(
            target=self._reader, name="camera-reader", daemon=True
        )
        self._thread.start()

    def _reader(self) -> None:
        failures = 0
        last_emit = 0.0
        while not self._stop.is_set():
            if self._cap is None:
                break
            ok, image = self._cap.read()
            if not ok or image is None:
                failures += 1
                self.read_errors += 1
                if failures >= 30:
                    self._reopen()
                    failures = 0
                time.sleep(0.01)
                continue
            failures = 0
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self._pace_period:
                now = time.monotonic()
                wait = self._pace_period - (now - last_emit)
                if wait > 0:
                    time.sleep(wait)
                last_emit = time.monotonic()

            frame = Frame(
                image=image,
                t_mono_us=time.monotonic_ns() // 1000,
                seq=self._seq,
            )
            self._seq += 1
            with self._cond:
                self._latest = frame
                self._cond.notify_all()

    def _reopen(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except OSError:
            pass
        self._cap = None
        time.sleep(0.5)
        try:
            self._cap = self._open()
            self.reconnects += 1
            print("[camera] reopened source", flush=True)
        except Exception as exc:  # keep the watchdog alive
            print(f"[camera] reopen failed: {exc}", flush=True)
            self._cap = None

    def read(self, timeout: float | None = 0.5) -> Frame | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._stop.is_set():
                if self._latest is not None:
                    frame = self._latest
                    self._latest = None
                    return frame
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    return None
                self._cond.wait(remaining)
        return None

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def make_source(cfg: Config, *, realtime: bool | None = None):
    kind = str(cfg.get("source.kind", "synthetic")).lower()
    if kind == "synthetic":
        return SyntheticSource(
            fps=int(cfg.get("source.fps", 60)),
            realtime=bool(cfg.get("source.realtime", True)) if realtime is None else realtime,
            page_cycle_s=float(cfg.get("source.page_cycle_s", 0.0)),
            dashboard=DashboardRenderer(),
        )
    if kind in ("v4l2", "rtsp", "file", "gst"):
        source = OpenCVSource(cfg, kind)
        source.start()
        return source
    raise ValueError(f"unknown source.kind {kind!r}")

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .align import Aligner
from .calibration import Calibration, FieldSpec
from .config import Config
from .glyphs import GlyphDictionary, Reading
from .kalman import ConstantVelocityFilter
from .pages import PageGate
from .synthetic import PAGE_MAIN
from .types import Frame, Telemetry
from .voting import MajorityVote, NVote


@dataclass
class _FieldRuntime:
    spec: FieldSpec
    glyphs: GlyphDictionary
    voter: NVote
    last_conf: float = 0.0
    conf_ema: float = 0.0


class OpticalPipeline:
    def __init__(self, cfg: Config, calibration: Calibration):
        self.cfg = cfg
        self.calibration = calibration
        acfg = cfg.section("alignment")
        self.aligner = Aligner(
            calibration,
            enabled=bool(acfg.get("enabled", True)),
            scale=float(acfg.get("scale", 0.5)),
            max_iters=int(acfg.get("max_iters", 25)),
            eps=float(acfg.get("eps", 1e-4)),
            min_score=float(acfg.get("min_score", 0.55)),
        )
        ocfg = cfg.section("ocr")
        self.page_gate = PageGate(
            calibration,
            width=int(ocfg.get("page_width", 200)),
            height=int(ocfg.get("page_height", 100)),
            min_score=float(ocfg.get("page_min_score", 0.45)),
            margin=float(ocfg.get("page_margin", 0.05)),
            search_pad=int(ocfg.get("page_search_pad", 6)),
        )
        self.vote_need = int(ocfg.get("vote_need", 3))
        self.vote_window = int(ocfg.get("vote_frames", 5))
        self.page_vote = MajorityVote(window=3, need=2)

        min_score = float(ocfg.get("template_min_score", 0.55))
        min_margin = float(ocfg.get("template_margin", 0.04))
        dictionaries: dict[str, GlyphDictionary] = {}
        for page in calibration.pages.values():
            for spec in page.fields.values():
                if spec.glyph_set not in dictionaries:
                    dictionaries[spec.glyph_set] = GlyphDictionary.load(
                        calibration.glyph_dir(spec.glyph_set),
                        min_score=min_score,
                        min_margin=min_margin,
                    )
        self.dictionaries = dictionaries

        self.runtimes: dict[str, dict[str, _FieldRuntime]] = {}
        for page in calibration.pages.values():
            self.runtimes[page.name] = {
                name: _FieldRuntime(
                    spec=spec,
                    glyphs=dictionaries[spec.glyph_set],
                    voter=NVote(self.vote_need, self.vote_window),
                )
                for name, spec in page.fields.items()
            }

        # Filters are keyed by field name and persist across page changes;
        # their dt gate re-seeds naturally after a gap.
        self.filters: dict[str, ConstantVelocityFilter] = {}
        for page in calibration.pages.values():
            for name, spec in page.fields.items():
                if name in self.filters:
                    continue
                fcfg = cfg.section(f"filter.{name}")
                self.filters[name] = ConstantVelocityFilter(
                    process_noise=float(fcfg.get("process_noise", 1.2)),
                    measure_noise=float(fcfg.get("measure_noise", 0.8)),
                    max_accel_per_s=float(
                        fcfg.get(
                            "max_accel",
                            17.0 if name == "speed" else 6000.0,
                        )
                    ),
                    gate_sigmas=float(fcfg.get("gate_sigmas", 4.0)),
                    reject_to_reset=int(fcfg.get("reject_to_reset", 5)),
                    resolution=float(
                        fcfg.get("resolution", 1.0 if name == "speed" else 10.0)
                    ),
                    lo=float(fcfg.get("min", spec.lo)),
                    hi=float(fcfg.get("max", spec.hi)),
                    alpha=float(fcfg.get("alpha", 1.0)),
                )

        self.page = "UNKNOWN"
        self.page_score = 0.0
        self.align_score = 0.0
        self.proc_us = deque(maxlen=1200)
        self._proc_total = 0

    def _read_field(self, aligned: np.ndarray, runtime: _FieldRuntime) -> Reading:
        x, y, w, h = runtime.spec.rect
        if x < 0 or y < 0 or x + w > aligned.shape[1] or y + h > aligned.shape[0]:
            return Reading(False, reason="field outside frame")
        roi = aligned[y : y + h, x : x + w]
        return runtime.glyphs.read_number(
            roi,
            decimals=runtime.spec.decimals,
            lo=runtime.spec.lo,
            hi=runtime.spec.hi,
        )

    def process(self, frame: Frame) -> Telemetry:
        started = time.perf_counter_ns()
        aligned, align_score = self.aligner.align(frame.image)
        page_name, page_score = self.page_gate.match(aligned)
        effective_page = self.page_vote.push(page_name)
        now = frame.t_mono_us / 1_000_000.0

        if effective_page != self.page and effective_page is not None:
            self.page = effective_page
            for page_runtimes in self.runtimes.values():
                for runtime in page_runtimes.values():
                    runtime.voter.reset()
                    runtime.last_conf = 0.0

        if effective_page is None:
            self.page = "UNKNOWN"
        self.page_score = float(max(0.0, min(1.0, page_score)))
        self.align_score = align_score

        active = self.runtimes.get(effective_page or "", {})
        field_values: dict[str, float | None] = {}
        field_valid: dict[str, bool] = {}
        field_conf: dict[str, float] = {}

        # Fields that are not on the current page must decay out of stability.
        for page_name, page_runtimes in self.runtimes.items():
            if page_name == effective_page:
                continue
            for runtime in page_runtimes.values():
                runtime.voter.push(None)

        for name, runtime in active.items():
            reading = self._read_field(aligned, runtime)
            if reading.valid:
                runtime.conf_ema = (
                    reading.conf
                    if runtime.conf_ema <= 0.0
                    else 0.75 * runtime.conf_ema + 0.25 * reading.conf
                )
            else:
                # Hold confidence through a one-frame dropout; the N-of-M vote
                # is still carrying the last valid reading.
                runtime.conf_ema *= 0.90
            runtime.last_conf = runtime.conf_ema
            token = (
                round(reading.value, runtime.spec.decimals)
                if reading.valid and reading.value is not None
                else None
            )
            stable, updated = runtime.voter.push(token)
            if stable is not None and updated:
                filt = self.filters.get(name)
                if filt is not None:
                    filt.update(float(stable), now)
            filt = self.filters.get(name)
            field_values[name] = None if filt is None else filt.value
            field_valid[name] = stable is not None
            field_conf[name] = runtime.last_conf if stable is not None else 0.0

        page_ok = effective_page is not None and bool(active)
        valid = page_ok and all(field_valid.values()) and bool(field_values)
        conf = 0.0
        if valid:
            conf = self.page_score * min(field_conf.values())
            conf = float(max(0.0, min(1.0, conf)))

        speed = field_values.get("speed")
        rpm = field_values.get("rpm")
        tel = Telemetry(
            page=effective_page or "UNKNOWN",
            speed=None if speed is None else round(float(speed), 1),
            rpm=None if rpm is None else int(round(rpm)),
            conf=conf,
            valid=bool(valid),
            t_mono_us=frame.t_mono_us,
            align_score=align_score,
            page_score=self.page_score,
            proc_us=0,
        )
        elapsed = (time.perf_counter_ns() - started) // 1000
        tel.proc_us = int(elapsed)
        self.proc_us.append(elapsed)
        self._proc_total += 1
        return tel

    def stats(self) -> dict[str, float]:
        if not self.proc_us:
            return {"frames": 0}
        arr = np.asarray(self.proc_us, dtype=np.float64) / 1000.0
        return {
            "frames": float(self._proc_total),
            "proc_ms_p50": float(np.percentile(arr, 50)),
            "proc_ms_p95": float(np.percentile(arr, 95)),
            "align": float(self.align_score),
            "page": float(self.page_score),
        }

    def reset(self) -> None:
        self.page_vote.reset()
        for runtimes in self.runtimes.values():
            for runtime in runtimes.values():
                runtime.voter.reset()
        for filt in self.filters.values():
            filt.x = None
            filt.t = None

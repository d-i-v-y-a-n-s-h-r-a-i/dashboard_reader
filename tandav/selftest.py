from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field

from .calibration import build_synthetic_calibration
from .config import load_config
from .pipeline import OpticalPipeline
from .synthetic import PAGE_MAIN, SyntheticSource


@dataclass
class Result:
    frames: int = 0
    steady: int = 0
    correct: int = 0
    wrong: int = 0
    invalid: int = 0
    fuel_frames: int = 0
    fuel_blocked: int = 0
    main_after: int = 0
    main_valid: int = 0
    artifact_steady: int = 0
    artifact_invalid: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    proc_ms_p50: float = 0.0
    proc_ms_p95: float = 0.0
    wall_s: float = 0.0
    fps: float = 0.0
    fp_events: int = 0

    @property
    def coverage(self) -> float:
        return self.correct / max(1, self.steady)

    @property
    def wrong_rate(self) -> float:
        return self.wrong / max(1, self.steady)


def run_bench(*, frames: int = 900, fps: int = 60, gate: bool = False) -> Result:
    cfg = load_config()
    cfg.data.setdefault("source", {})
    cfg.data["source"]["fps"] = fps
    cfg.data["source"]["realtime"] = False
    cycle = 8.0 if gate else 0.0

    with tempfile.TemporaryDirectory(prefix="tandav-selftest-") as tmp:
        calibration = build_synthetic_calibration(tmp)
        pipeline = OpticalPipeline(cfg, calibration)
        source = SyntheticSource(
            fps=fps,
            realtime=False,
            page_cycle_s=cycle,
            dashboard=None,
        )
        result = Result()
        last_lcd_change = -1.0
        pending_change: tuple[float, int] | None = None
        last_page = PAGE_MAIN
        page_change_t = 0.0
        started = time.perf_counter()

        for _ in range(frames):
            frame = source.read()
            if frame is None:
                break
            truth = frame.truth or {}
            t = float(truth.get("t", 0.0))
            tel = pipeline.process(frame)
            result.frames += 1

            if truth.get("changed"):
                last_lcd_change = t
                if truth.get("page") == PAGE_MAIN:
                    pending_change = (t, int(truth["speed"]))
            if truth.get("page") != last_page:
                last_page = truth.get("page")
                page_change_t = t
                pending_change = None
                pipeline.reset()

            if pending_change is not None and tel.valid and tel.speed is not None:
                change_t, target = pending_change
                if abs(float(tel.speed) - target) <= 0.51:
                    result.latencies_ms.append((t - change_t) * 1000.0)
                    pending_change = None

            truth_page = truth.get("page")
            artifact = bool(truth.get("glare") or truth.get("droplet"))
            if truth_page == PAGE_MAIN:
                steady = t - last_lcd_change > 0.20 and (not gate or t - page_change_t > 0.6)
                if steady:
                    if artifact:
                        # Deliberate blinding glare/droplets: invalid is the
                        # correct fail-loud result; nominal accuracy is
                        # measured on the clear frames only.
                        result.artifact_steady += 1
                        if not tel.valid:
                            result.artifact_invalid += 1
                    else:
                        result.steady += 1
                        if tel.valid and tel.speed is not None:
                            if abs(float(tel.speed) - float(truth["speed"])) <= 0.51:
                                result.correct += 1
                            else:
                                result.wrong += 1
                                result.fp_events += 1
                        else:
                            result.invalid += 1
                        if gate and t - page_change_t > 0.6:
                            result.main_after += 1
                            if tel.valid:
                                result.main_valid += 1
            elif truth_page == "FUEL" and gate and t - page_change_t > 0.25 and not artifact:
                result.fuel_frames += 1
                if not tel.valid:
                    result.fuel_blocked += 1

        result.wall_s = time.perf_counter() - started
        stats = pipeline.stats()
        result.proc_ms_p50 = float(stats.get("proc_ms_p50", 0.0))
        result.proc_ms_p95 = float(stats.get("proc_ms_p95", 0.0))
        result.fps = result.frames / max(1e-9, result.wall_s)
    return result


def format_result(result: Result) -> tuple[str, dict]:
    lat = sorted(result.latencies_ms)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
    gate_acc = (
        result.fuel_blocked / result.fuel_frames if result.fuel_frames else 1.0
    )
    main_acc = result.main_valid / result.main_after if result.main_after else 1.0
    summary = {
        "frames": result.frames,
        "fps": round(result.fps, 1),
        "proc_ms_p50": round(result.proc_ms_p50, 3),
        "proc_ms_p95": round(result.proc_ms_p95, 3),
        "steady_frames": result.steady,
        "coverage": round(result.coverage, 4),
        "wrong_rate": round(result.wrong_rate, 4),
        "invalid_steady": result.invalid,
        "update_latency_ms_p50": round(p50, 1),
        "update_latency_ms_p95": round(p95, 1),
        "gate_fuel_blocked": round(gate_acc, 4),
        "gate_main_valid": round(main_acc, 4),
        "artifact_steady": result.artifact_steady,
        "artifact_invalid": result.artifact_invalid,
        "updates": len(result.latencies_ms),
    }
    lines = [
        f"frames            {result.frames}  ({result.fps:.1f} fps end-to-end)",
        f"ocr proc          p50 {result.proc_ms_p50:.2f} ms  p95 {result.proc_ms_p95:.2f} ms/frame",
        f"steady coverage   {result.coverage * 100:.2f}%  ({result.correct}/{result.steady})",
        f"wrong steady      {result.wrong} ({result.wrong_rate * 100:.2f}%)",
        f"update latency    p50 {p50:.0f} ms  p95 {p95:.0f} ms  ({len(lat)} updates)",
        f"page gate         fuel blocked {gate_acc * 100:.2f}%  main valid {main_acc * 100:.2f}%",
        f"artifact stress   {result.artifact_invalid}/{result.artifact_steady} "
        "glare/droplet frames rejected (fail loud)",
    ]
    return "\n".join(lines), summary

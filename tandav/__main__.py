from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .calibration import (
    build_synthetic_calibration,
    init_from_frame,
    load_calibration,
    parse_rect,
)
from .camera import make_source
from .config import ROOT, load_config
from .discover import build_glyph_dictionary, discover_glyphs
from .pipeline import OpticalPipeline
from .selftest import format_result, run_bench
from .sender import UdpSender


def _pairs(values: list[str] | None, cast=str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = cast(value.strip())
    return out


def _ensure_calibration(cfg, path: Path):
    if not (path / "calibration.json").exists():
        if str(cfg.get("source.kind", "synthetic")).lower() == "synthetic":
            print(f"[calib] creating synthetic calibration in {path}", flush=True)
            return build_synthetic_calibration(path)
        raise SystemExit(
            f"no calibration at {path}. Capture a reference frame and run "
            "`python -m tandav init ...` (see README)."
        )
    return load_calibration(path)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cal_path = Path(args.calibration) if args.calibration else cfg.calibration_path
    calibration = _ensure_calibration(cfg, cal_path)
    pipeline = OpticalPipeline(cfg, calibration)
    source = make_source(cfg)
    host = args.host or str(cfg.get("telemetry.host", "255.255.255.255"))
    port = int(args.port or cfg.get("telemetry.port", 47001))
    sender = UdpSender(
        host,
        port,
        hz=float(cfg.get("telemetry.hz", 20)),
        dscp=int(cfg.get("telemetry.dscp", 0)),
    )
    interval = float(cfg.get("runtime.stats_interval_s", 5))
    started = time.monotonic()
    last_stats = started
    last_frame_t = started
    last_seq = -1
    frames = 0

    print(
        json.dumps(
            {
                "event": "start",
                "source": cfg.get("source.kind"),
                "frame": [calibration.width, calibration.height],
                "anchor": list(calibration.anchor_rect),
                "pages": list(calibration.pages),
                "udp": f"{host}:{port}",
                "boot_id": sender.boot_id,
            }
        ),
        flush=True,
    )
    try:
        while True:
            frame = source.read(timeout=0.5)
            if frame is None:
                if time.monotonic() - last_frame_t > 5.0:
                    print(
                        json.dumps({"event": "no_frames", "seconds": 5}),
                        flush=True,
                    )
                    last_frame_t = time.monotonic()
                continue
            last_frame_t = time.monotonic()
            if frame.seq == last_seq:
                continue
            last_seq = frame.seq
            frames += 1
            tel = pipeline.process(frame)
            sender.maybe_send(tel)

            now = time.monotonic()
            if now - last_stats >= interval:
                stats = pipeline.stats()
                line = {
                    "event": "stats",
                    "uptime_s": round(now - started, 1),
                    "frames": frames,
                    "fps": round(frames / max(1e-9, now - started), 1),
                    "proc_ms_p50": round(stats.get("proc_ms_p50", 0), 2),
                    "proc_ms_p95": round(stats.get("proc_ms_p95", 0), 2),
                    "align": round(stats.get("align", 0), 3),
                    "page_score": round(stats.get("page", 0), 3),
                    "page": tel.page,
                    "speed_kmh": tel.speed,
                    "rpm": tel.rpm,
                    "conf": round(tel.conf, 3),
                    "valid": tel.valid,
                    "sent": sender.sent,
                    "tx_errors": sender.errors,
                }
                print(json.dumps(line), flush=True)
                last_stats = now

            if args.duration and now - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        sender.close()
        print(
            json.dumps(
                {
                    "event": "stop",
                    "frames": frames,
                    "sent": sender.sent,
                    "errors": sender.errors,
                }
            ),
            flush=True,
        )
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    result = run_bench(frames=args.frames, fps=args.fps, gate=args.gate)
    text, summary = format_result(result)
    print(text)
    if args.json:
        print(json.dumps(summary, indent=2))

    ok = result.coverage >= 0.95 and result.wrong_rate <= 0.02
    gate_ok = True
    if args.gate:
        gate_acc = (
            result.fuel_blocked / result.fuel_frames if result.fuel_frames else 0.0
        )
        gate_ok = gate_acc >= 0.95
    if not ok or not gate_ok:
        print("SELFTEST FAILED")
        return 1
    print("SELFTEST PASS")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    calibration = build_synthetic_calibration(out)
    print(
        json.dumps(
            {
                "event": "calibrated",
                "out": str(out),
                "pages": list(calibration.pages),
                "glyphs": {
                    name: len(list(calibration.glyph_dir(name).glob("*.png")))
                    for name in ("speed", "rpm")
                },
            }
        )
    )
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    source = make_source(cfg)
    try:
        frame = None
        for _ in range(max(1, args.warmup)):
            frame = source.read(timeout=2.0)
        if frame is None:
            raise SystemExit("no frame received")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame.image)
        print(f"saved {out} ({frame.image.shape[1]}x{frame.image.shape[0]})")
    finally:
        source.close()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    fields: list[tuple[str, tuple[int, int, int, int]]] = []
    checks = [parse_rect(item) for item in args.check]
    for spec in args.field:
        if "=" not in spec:
            raise SystemExit(f"--field expects name=x,y,w,h, got {spec!r}")
        name, rect = spec.split("=", 1)
        fields.append((name.strip(), parse_rect(rect)))
    decimals = {k: int(v) for k, v in _pairs(args.decimals).items()}
    glyph_sets = _pairs(args.glyph_set)
    ranges: dict[str, tuple[float, float]] = {}
    for name, value in _pairs(args.range).items():
        lo, hi = value.split(",", 1)
        ranges[name] = (float(lo), float(hi))

    calibration = init_from_frame(
        frame_path=args.frame,
        page=args.page,
        anchor=parse_rect(args.anchor),
        fields=fields,
        out_dir=args.out,
        decimals=decimals,
        ranges=ranges,
        glyph_sets=glyph_sets,
        checks=checks,
    )
    print(
        json.dumps(
            {
                "event": "page_initialised",
                "page": args.page,
                "pages": list(calibration.pages),
                "fields": [name for name, _ in fields],
                "next": "capture live frames, run `discover`, label, then `build-glyphs`",
            }
        )
    )
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    calibration = _ensure_calibration(cfg, cfg.calibration_path)
    source = make_source(cfg)
    try:
        out = discover_glyphs(
            cfg=cfg,
            calibration=calibration,
            source=source,
            page_name=args.page,
            seconds=args.seconds,
            out_dir=args.out,
        )
    finally:
        source.close()
    print(f"discovery written to {out}")
    print(f"fill labels in {out / 'labels.json'}, then run `build-glyphs`")
    return 0


def cmd_build_glyphs(args: argparse.Namespace) -> int:
    calibration = load_calibration(args.calibration)
    counts = build_glyph_dictionary(
        calibration=calibration,
        labels_path=args.labels,
        discover_dir=args.discover,
    )
    print(json.dumps({"event": "glyphs_built", "templates": counts}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tandav",
        description="TANDAV optical telemetry pipeline",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the live optical pipeline")
    run.add_argument("--config", default=str(ROOT / "config" / "tandav.yaml"))
    run.add_argument("--calibration", default=None)
    run.add_argument("--host", default=None, help="UDP destination override")
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--duration", type=float, default=0.0)
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("selftest", help="ground-truth bench run")
    bench.add_argument("--frames", type=int, default=900)
    bench.add_argument("--fps", type=int, default=60)
    bench.add_argument("--gate", action="store_true", help="exercise page changes")
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=cmd_selftest)

    cal = sub.add_parser("calibrate", help="write the synthetic bench calibration")
    cal.add_argument("--out", default=str(ROOT / "calibration"))
    cal.set_defaults(func=cmd_calibrate)

    cap = sub.add_parser("capture", help="save one frame from the configured source")
    cap.add_argument("--config", default=str(ROOT / "config" / "tandav.yaml"))
    cap.add_argument("--out", default=str(ROOT / "calibration" / "reference.png"))
    cap.add_argument("--warmup", type=int, default=12)
    cap.set_defaults(func=cmd_capture)

    init = sub.add_parser("init", help="initialise a page from a reference frame")
    init.add_argument("--frame", required=True)
    init.add_argument("--page", default="MAIN_TELEMETRY")
    init.add_argument("--anchor", required=True, help="x,y,w,h")
    init.add_argument("--field", action="append", default=[], help="name=x,y,w,h")
    init.add_argument("--decimals", action="append", default=[])
    init.add_argument("--range", action="append", default=[])
    init.add_argument("--glyph-set", action="append", default=[])
    init.add_argument("--check", action="append", default=[], help="static label x,y,w,h")
    init.add_argument("--out", default=str(ROOT / "calibration"))
    init.set_defaults(func=cmd_init)

    disc = sub.add_parser("discover", help="cluster glyph cells from live frames")
    disc.add_argument("--config", default=str(ROOT / "config" / "tandav.yaml"))
    disc.add_argument("--page", default="MAIN_TELEMETRY")
    disc.add_argument("--seconds", type=float, default=20.0)
    disc.add_argument("--out", default=str(ROOT / "calibration" / "discover"))
    disc.set_defaults(func=cmd_discover)

    build = sub.add_parser("build-glyphs", help="turn labelled clusters into templates")
    build.add_argument("--calibration", default=str(ROOT / "calibration"))
    build.add_argument("--discover", default=str(ROOT / "calibration" / "discover"))
    build.add_argument("--labels", default=None)
    build.set_defaults(func=cmd_build_glyphs)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "build-glyphs" and args.labels is None:
        args.labels = str(Path(args.discover) / "labels.json")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

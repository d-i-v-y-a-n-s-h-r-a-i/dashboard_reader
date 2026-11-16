from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .synthetic import (
    ANCHOR_RECT,
    MAIN_FIELDS,
    PAGE_CHECKS,
    PAGE_FUEL,
    PAGE_MAIN,
    W,
    H,
    DashboardRenderer,
)

CAL_VERSION = 1


@dataclass(frozen=True)
class FieldSpec:
    name: str
    rect: tuple[int, int, int, int]
    glyph_set: str
    decimals: int
    lo: float
    hi: float


@dataclass
class PageSpec:
    name: str
    template: np.ndarray
    mask_rects: list[tuple[int, int, int, int]]
    checks: list[tuple[int, int, int, int]]
    fields: dict[str, FieldSpec]


class Calibration:
    def __init__(
        self,
        *,
        base_dir: Path,
        width: int,
        height: int,
        anchor_rect: tuple[int, int, int, int],
        anchor_template: np.ndarray,
        pages: dict[str, PageSpec],
    ):
        self.base_dir = base_dir
        self.width = width
        self.height = height
        self.anchor_rect = anchor_rect
        self.anchor_template = anchor_template
        self.pages = pages

    def glyph_dir(self, name: str) -> Path:
        return self.base_dir / "glyphs" / name


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read calibration image: {path}")
    return img


def load_calibration(path: str | Path) -> Calibration:
    p = Path(path)
    if p.suffix.lower() == ".json":
        json_path = p
    else:
        json_path = p / "calibration.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} missing. Run `python -m tandav calibrate synthetic` "
            "for the bench rig, or `python -m tandav init` against a captured frame."
        )
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if raw.get("version") != CAL_VERSION:
        raise ValueError("unsupported calibration version")
    base = json_path.parent

    width = int(raw["frame"]["width"])
    height = int(raw["frame"]["height"])
    anchor_rect = tuple(int(v) for v in raw["anchor"]["rect"])
    anchor_template = _load_gray(base / raw["anchor"]["file"])
    if anchor_template.shape != (anchor_rect[3], anchor_rect[2]):
        raise ValueError("anchor template does not match its rect")

    pages: dict[str, PageSpec] = {}
    for page_raw in raw.get("pages", []):
        template = _load_gray(base / page_raw["file"])
        fields = {
            name: FieldSpec(
                name=name,
                rect=tuple(int(v) for v in f["rect"]),
                glyph_set=str(f.get("glyph_set", name)),
                decimals=int(f.get("decimals", 0)),
                lo=float(f.get("min", 0)),
                hi=float(f.get("max", 10000)),
            )
            for name, f in page_raw.get("fields", {}).items()
        }
        pages[page_raw["name"]] = PageSpec(
            name=page_raw["name"],
            template=template,
            mask_rects=[tuple(int(v) for v in r) for r in page_raw.get("mask", [])],
            checks=[tuple(int(v) for v in r) for r in page_raw.get("checks", [])],
            fields=fields,
        )
    if not pages:
        raise ValueError("calibration has no pages")

    return Calibration(
        base_dir=base,
        width=width,
        height=height,
        anchor_rect=anchor_rect,
        anchor_template=anchor_template,
        pages=pages,
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_synthetic_calibration(
    out_dir: str | Path,
    renderer: DashboardRenderer | None = None,
) -> Calibration:
    """Create a complete calibration (pages, anchor, glyph dictionaries)."""
    dash = renderer or DashboardRenderer()
    out = Path(out_dir)
    (out / "glyphs" / "speed").mkdir(parents=True, exist_ok=True)
    (out / "glyphs" / "rpm").mkdir(parents=True, exist_ok=True)

    main_clean = dash.render(PAGE_MAIN, speed=0, rpm=0)
    fuel_clean = dash.render(PAGE_FUEL)
    cv2.imwrite(str(out / "page_MAIN_TELEMETRY.png"), main_clean)
    cv2.imwrite(str(out / "page_FUEL.png"), fuel_clean)

    ax, ay, aw, ah = dash.anchor_rect
    anchor = main_clean[ay : ay + ah, ax : ax + aw]
    cv2.imwrite(str(out / "anchor.png"), anchor)

    for set_name in ("speed", "rpm"):
        for ch, glyph in dash.glyph_set(set_name).items():
            cv2.imwrite(str(out / "glyphs" / set_name / f"{ch}.png"), glyph)

    pages = [
        {
            "name": PAGE_MAIN,
            "file": "page_MAIN_TELEMETRY.png",
            "mask": [list(f["rect"]) for f in MAIN_FIELDS.values()],
            "checks": [list(r) for r in PAGE_CHECKS[PAGE_MAIN]],
            "fields": {
                name: {
                    "rect": list(spec["rect"]),
                    "glyph_set": spec["glyph_set"],
                    "decimals": spec["decimals"],
                    "min": spec["min"],
                    "max": spec["max"],
                }
                for name, spec in MAIN_FIELDS.items()
            },
        },
        {
            "name": PAGE_FUEL,
            "file": "page_FUEL.png",
            "mask": [],
            "checks": [list(r) for r in PAGE_CHECKS[PAGE_FUEL]],
            "fields": {},
        },
    ]
    data = {
        "version": CAL_VERSION,
        "frame": {"width": W, "height": H},
        "anchor": {"rect": list(ANCHOR_RECT), "file": "anchor.png"},
        "pages": pages,
    }
    _write_json(out / "calibration.json", data)
    return load_calibration(out)


def parse_rect(text: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"rect must be x,y,w,h: {text!r}")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("rect width/height must be positive")
    return x, y, w, h


def init_from_frame(
    *,
    frame_path: str | Path,
    page: str,
    anchor: tuple[int, int, int, int],
    fields: Iterable[tuple[str, tuple[int, int, int, int]]],
    out_dir: str | Path,
    decimals: dict[str, int] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    glyph_sets: dict[str, str] | None = None,
    checks: Iterable[tuple[int, int, int, int]] = (),
) -> Calibration:
    """Bootstrap a page calibration from one clean reference frame."""
    frame_img = _load_gray(Path(frame_path))
    h, w = frame_img.shape[:2]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ax, ay, aw, ah = anchor
    if ax < 0 or ay < 0 or ax + aw > w or ay + ah > h:
        raise ValueError("anchor rect outside frame")
    anchor_img = frame_img[ay : ay + ah, ax : ax + aw]
    cv2.imwrite(str(out / "anchor.png"), anchor_img)

    json_path = out / "calibration.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        data = {"version": CAL_VERSION, "frame": {"width": w, "height": h}, "pages": []}
        data["anchor"] = {"rect": list(anchor), "file": "anchor.png"}
    if data["frame"]["width"] != w or data["frame"]["height"] != h:
        raise ValueError(
            f"reference frame is {w}x{h}, calibration is "
            f"{data['frame']['width']}x{data['frame']['height']}"
        )

    page_file = f"page_{page}.png"
    cv2.imwrite(str(out / page_file), frame_img)
    fields_raw: dict[str, dict] = {}
    mask: list[list[int]] = []
    decimals = decimals or {}
    ranges = ranges or {}
    glyph_sets = glyph_sets or {}
    for name, rect in fields:
        x, y, fw, fh = rect
        if x < 0 or y < 0 or x + fw > w or y + fh > h:
            raise ValueError(f"field {name} outside frame")
        lo, hi = ranges.get(name, (0.0, 99999.0))
        fields_raw[name] = {
            "rect": list(rect),
            "glyph_set": glyph_sets.get(name, name),
            "decimals": int(decimals.get(name, 0)),
            "min": lo,
            "max": hi,
        }
        mask.append(list(rect))

    page_raw = {
        "name": page,
        "file": page_file,
        "mask": mask,
        "checks": [list(r) for r in checks],
        "fields": fields_raw,
    }
    data["pages"] = [p for p in data["pages"] if p.get("name") != page]
    data["pages"].append(page_raw)
    _write_json(json_path, data)
    return load_calibration(out)

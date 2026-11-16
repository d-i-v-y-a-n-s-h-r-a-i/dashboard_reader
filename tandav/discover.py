from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .align import Aligner
from .calibration import Calibration
from .config import Config
from .glyphs import TEMPLATE_H, TEMPLATE_W, prepare_cell, segment_cells
from .pages import PageGate


def _cluster(
    samples: np.ndarray, k: int, seed: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(k, len(samples)))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 0.1)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(
        samples.astype(np.float32),
        k,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers, labels.ravel()


def _save_center(center: np.ndarray, path: Path) -> None:
    img = center.reshape(TEMPLATE_H, TEMPLATE_W)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        img = np.zeros_like(img)
    else:
        img = (img - lo) / (hi - lo) * 255.0
    cv2.imwrite(str(path), img.astype(np.uint8))


def _contact_sheet(centers: np.ndarray, names: list[str], path: Path) -> None:
    cell_w, cell_h = 64, 82
    cols = min(6, len(centers))
    rows = int(np.ceil(len(centers) / cols))
    sheet = np.full((rows * cell_h, cols * cell_w), 18, np.uint8)
    for i, center in enumerate(centers):
        row, col = divmod(i, cols)
        img = center.reshape(TEMPLATE_H, TEMPLATE_W)
        lo, hi = float(img.min()), float(img.max())
        norm = (img - lo) / max(1e-6, hi - lo) * 255.0
        thumb = cv2.resize(norm.astype(np.uint8), (48, 64), interpolation=cv2.INTER_NEAREST)
        x0 = col * cell_w + 8
        y0 = row * cell_h + 4
        sheet[y0 : y0 + 64, x0 : x0 + 48] = thumb
        cv2.putText(
            sheet,
            names[i],
            (x0, y0 + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            210,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), sheet)


def discover_glyphs(
    *,
    cfg: Config,
    calibration: Calibration,
    source,
    page_name: str,
    seconds: float,
    out_dir: str | Path,
    max_samples: int = 4000,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    aligner = Aligner(calibration, enabled=bool(cfg.get("alignment.enabled", True)))
    gate = PageGate(
        calibration,
        min_score=float(cfg.get("ocr.page_min_score", 0.55)),
    )
    page = calibration.pages.get(page_name)
    if page is None:
        raise KeyError(f"page {page_name!r} not in calibration")
    if not page.fields:
        raise ValueError(f"page {page_name!r} has no fields to learn")

    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    fps = float(cfg.get("source.fps", 60))
    total = max(10, int(seconds * fps))
    seen = 0
    while seen < total:
        frame = source.read(timeout=1.0)
        if frame is None:
            break
        seen += 1
        aligned, _ = aligner.align(frame.image)
        got_page, _ = gate.match(aligned)
        if got_page != page_name:
            continue
        for name, spec in page.fields.items():
            if len(buckets[name]) >= max_samples:
                continue
            x, y, w, h = spec.rect
            roi = aligned[y : y + h, x : x + w]
            for cx, cy, cw, ch in segment_cells(roi):
                cell = roi[cy : cy + ch, cx : cx + cw]
                prepared = prepare_cell(cell)
                if prepared is not None:
                    buckets[name].append(prepared.ravel())

    labels_template: dict[str, dict[str, str]] = {}
    for name, spec in page.fields.items():
        samples = buckets.get(name, [])
        if len(samples) < 10:
            print(f"[discover] {name}: only {len(samples)} cells, skipped")
            continue
        data = np.stack(samples)
        k = 11 if spec.decimals > 0 else 10
        centers, _ = _cluster(data, k)
        names = [f"c{i}" for i in range(len(centers))]
        for i, center in enumerate(centers):
            _save_center(center, out / f"{name}_c{i}.png")
        _contact_sheet(centers, names, out / f"{name}_clusters.png")
        labels_template[name] = {name_i: "" for name_i in names}
        print(
            f"[discover] {name}: {len(samples)} cells -> {len(centers)} clusters; "
            f"see {out / f'{name}_clusters.png'}"
        )

    (out / "labels.json").write_text(
        json.dumps(labels_template, indent=2) + "\n", encoding="utf-8"
    )
    return out


def build_glyph_dictionary(
    *,
    calibration: Calibration,
    labels_path: str | Path,
    discover_dir: str | Path,
) -> dict[str, int]:
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    source_dir = Path(discover_dir)
    field_to_set: dict[str, str] = {}
    for page in calibration.pages.values():
        for name, spec in page.fields.items():
            field_to_set[name] = spec.glyph_set

    written: dict[str, int] = {}
    for field, mapping in labels.items():
        glyph_set = field_to_set.get(field, field)
        target = calibration.glyph_dir(glyph_set)
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob("*.png"):
            old.unlink()
        counts: dict[str, int] = defaultdict(int)
        for cluster, label in mapping.items():
            label = str(label).strip()
            if label not in set("0123456789."):
                continue
            src = source_dir / f"{field}_{cluster}.png"
            if not src.exists():
                continue
            counts[label] += 1
            suffix = "" if counts[label] == 1 else f"_{counts[label]}"
            cv2.imwrite(
                str(target / f"{label}{suffix}.png"),
                cv2.imread(str(src), cv2.IMREAD_GRAYSCALE),
            )
        written[glyph_set] = sum(counts.values())
        missing = sorted(set("0123456789") - set(counts))
        if missing:
            print(f"[glyphs] {glyph_set}: missing {''.join(missing)}")
        print(f"[glyphs] {glyph_set}: {sum(counts.values())} templates")
    return written

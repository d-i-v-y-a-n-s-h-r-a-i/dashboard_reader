from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "tandav.yaml"


class Config:
    """Thin dotted-key view over the YAML document."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self.data = data or {}
        self.path = path

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, key: str) -> dict[str, Any]:
        value = self.get(key, {})
        return value if isinstance(value, dict) else {}

    @property
    def calibration_path(self) -> Path:
        raw = str(self.get("calibration.path", "calibration"))
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config(data, p)

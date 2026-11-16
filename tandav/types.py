from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(slots=True)
class Frame:
    """One captured (or synthesized) camera frame, timestamped at arrival."""

    image: np.ndarray
    t_mono_us: int
    seq: int
    truth: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class Telemetry:
    """Decoded optical state, ready for the UDP payload."""

    page: str
    speed: Optional[float]
    rpm: Optional[int]
    conf: float
    valid: bool
    t_mono_us: int
    align_score: float = 0.0
    page_score: float = 0.0
    proc_us: int = 0

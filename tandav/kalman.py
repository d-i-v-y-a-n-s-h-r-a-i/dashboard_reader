from __future__ import annotations

import math
from typing import Optional

import numpy as np


class ConstantVelocityFilter:
    """1-D constant-velocity track with a physical plausibility gate.

    Units are whatever the caller supplies (km/h and seconds here). The gate
    rejects a measurement that would require more acceleration than the hull
    can physically produce; a run of rejected samples re-seeds the filter so a
    genuine step change (new page, reconnect) cannot wedge it forever.
    """

    def __init__(
        self,
        *,
        process_noise: float = 2.0,
        measure_noise: float = 0.8,
        max_accel_per_s: float = 17.0,
        gate_sigmas: float = 4.0,
        reject_to_reset: int = 5,
        resolution: float = 1.0,
        lo: float = 0.0,
        hi: float = 120.0,
        alpha: float = 1.0,
    ):
        self.q = float(process_noise)
        self.r = float(measure_noise)
        self.max_accel = float(max_accel_per_s)
        self.gate_sigmas = float(gate_sigmas)
        self.reject_to_reset = max(1, int(reject_to_reset))
        self.resolution = float(resolution)
        self.lo = float(lo)
        self.hi = float(hi)
        # 1.0 means accepted voted readings pass through immediately. Voting
        # has already removed frame noise; this filter's job is the physical
        # plausibility gate, not extra lag on every 4 Hz LCD step.
        self.alpha = max(0.05, min(1.0, float(alpha)))

        self.x: Optional[np.ndarray] = None
        self.P = np.eye(2, dtype=np.float64)
        self.t: Optional[float] = None
        self.rejects = 0
        self.resets = 0
        self.last_gate = math.inf
        self.last_innovation = 0.0

    def reset(self, value: float, now: float) -> float:
        z = self._clamp(value)
        self.x = np.array([z, 0.0], dtype=np.float64)
        self.P = np.diag([self.resolution**2, (self.max_accel * 0.5) ** 2])
        self.t = float(now)
        self.rejects = 0
        return z

    def _clamp(self, value: float) -> float:
        return min(self.hi, max(self.lo, float(value)))

    def predict(self, now: float) -> float:
        if self.x is None or self.t is None:
            return math.nan
        dt = min(2.0, max(1e-3, float(now) - self.t))
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = self.q
        Q = q * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=np.float64,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(now)
        return float(self.x[0])

    def update(self, value: float, now: float) -> float:
        z = self._clamp(value)
        if self.x is None or self.t is None:
            return self.reset(z, now)

        dt = min(2.0, max(1e-3, float(now) - self.t))
        pred = self.predict(now)
        innovation = z - pred
        # Four sigma of the predicted track, plus the change the hull could
        # physically make in this interval. The resolution term keeps an
        # integer LCD from being rejected at rest.
        sigma = math.sqrt(max(1e-9, float(self.P[0, 0]) + self.r))
        self.last_gate = max(
            self.gate_sigmas * sigma,
            self.max_accel * dt * 1.25 + 2.0 * self.resolution,
        )
        self.last_innovation = innovation

        if abs(innovation) > self.last_gate:
            self.rejects += 1
            if self.rejects >= self.reject_to_reset:
                self.resets += 1
                return self.reset(z, now)
            return self._clamp(pred)

        self.rejects = 0
        S = float(self.P[0, 0] + self.r)
        K = self.P[:, 0] / S
        previous = float(self.x[0])
        self.x[0] = pred + self.alpha * innovation
        rate = (self.x[0] - previous) / max(1e-3, dt)
        self.x[1] = max(-self.max_accel, min(self.max_accel, rate))
        I = np.eye(2, dtype=np.float64)
        self.P = (I - np.outer(K, np.array([1.0, 0.0]))) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return self._clamp(float(self.x[0]))

    @property
    def value(self) -> Optional[float]:
        return None if self.x is None else self._clamp(float(self.x[0]))

    @property
    def rate(self) -> float:
        return 0.0 if self.x is None else float(self.x[1])

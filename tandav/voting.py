from __future__ import annotations

from collections import deque
from typing import Any, Optional


class NVote:
    """Require the same token in the last `need` of `window` frames.

    The LCD refreshes around 4 Hz while the camera runs at 30-60 fps, so an
    unchanged token across consecutive frames is strong evidence that the
    reading is real rather than a droplet, glare or PWM artifact.
    """

    def __init__(self, need: int = 3, window: int = 5):
        self.need = max(1, int(need))
        self.window = max(self.need, int(window))
        self.buf: deque[Any] = deque(maxlen=self.window)
        self.last: Optional[Any] = None

    def push(self, token: Any) -> tuple[Optional[Any], bool]:
        """Return (stable_token, changed). `changed` is true once per new token."""
        self.buf.append(token)
        if len(self.buf) < self.need:
            return self.last, False
        tail = list(self.buf)[-self.need :]
        if all(item == tail[0] for item in tail):
            if tail[0] != self.last:
                self.last = tail[0]
                return self.last, True
        return self.last, False

    def reset(self) -> None:
        self.buf.clear()
        self.last = None


class MajorityVote:
    """Majority over a short window; a single bad frame cannot flip the page."""

    def __init__(self, window: int = 3, need: int = 2):
        self.window = max(1, int(window))
        self.need = max(1, int(need))
        self.buf: deque[Any] = deque(maxlen=self.window)

    def push(self, token: Any) -> Optional[Any]:
        self.buf.append(token)
        if len(self.buf) < self.need:
            return None
        counts: dict[Any, int] = {}
        for item in self.buf:
            counts[item] = counts.get(item, 0) + 1
        winner, count = max(counts.items(), key=lambda kv: kv[1])
        return winner if count >= self.need else None

    def reset(self) -> None:
        self.buf.clear()

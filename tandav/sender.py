from __future__ import annotations

import socket
import time

from .protocol import encode_payload, new_boot_id
from .types import Telemetry


class UdpSender:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        hz: float = 20.0,
        dscp: int = 0,
        boot_id: str | None = None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.period = 1.0 / max(0.5, float(hz))
        self.boot_id = boot_id or new_boot_id()
        self.seq = 0
        self.sent = 0
        self.errors = 0
        self.last_error = ""
        self.last_sent_t = 0.0
        self.last_signature: tuple | None = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if dscp:
            try:
                # DSCP occupies the top six bits of the TOS byte.
                self.sock.setsockopt(
                    socket.IPPROTO_IP, socket.IP_TOS, int(dscp) << 2
                )
            except OSError:
                pass
        self.sock.setblocking(False)

    def _payload(self, tel: Telemetry) -> bytes:
        payload = encode_payload(
            seq=self.seq,
            boot_id=self.boot_id,
            t_mono_us=tel.t_mono_us,
            page=tel.page,
            speed=tel.speed,
            rpm=tel.rpm,
            conf=tel.conf,
            valid=tel.valid,
        )
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        return payload

    def maybe_send(self, tel: Telemetry, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        signature = (
            tel.valid,
            tel.page,
            None if tel.speed is None else round(tel.speed, 1),
            tel.rpm,
        )
        changed = signature != self.last_signature
        due = (now - self.last_sent_t) >= self.period
        if not (changed or due):
            return False

        payload = self._payload(tel)
        try:
            self.sock.sendto(payload, (self.host, self.port))
            self.sent += 1
            self.last_sent_t = now
            self.last_signature = signature
            return True
        except OSError as exc:
            self.errors += 1
            self.last_error = str(exc)
            return False

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

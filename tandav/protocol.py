from __future__ import annotations

import json
import os
import zlib
from typing import Any

PROTO_VERSION = 1
MAX_PACKET = 512

# Values outside these bounds are rejected by the ground station rather than
# rendered, so a corrupted packet can never drive the instrument cluster.
_LIMITS = {
    "spd": (0.0, 500.0),
    "rpm": (0, 30000),
    "conf": (0.0, 1.0),
}


class ProtocolError(ValueError):
    pass


def new_boot_id() -> str:
    return os.urandom(2).hex()


def _body_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(
        body, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def encode_payload(
    *,
    seq: int,
    boot_id: str,
    t_mono_us: int,
    page: str,
    speed: float | None,
    rpm: int | None,
    conf: float,
    valid: bool,
) -> bytes:
    if speed is not None and not _LIMITS["spd"][0] <= float(speed) <= _LIMITS["spd"][1]:
        raise ValueError("speed out of range")
    if rpm is not None and not _LIMITS["rpm"][0] <= int(rpm) <= _LIMITS["rpm"][1]:
        raise ValueError("rpm out of range")
    if not _LIMITS["conf"][0] <= float(conf) <= _LIMITS["conf"][1]:
        raise ValueError("confidence out of range")

    body: dict[str, Any] = {
        "v": PROTO_VERSION,
        "seq": int(seq) & 0xFFFFFFFF,
        "boot_id": str(boot_id)[:8],
        "t_mono_us": int(t_mono_us),
        "page": str(page)[:32],
        "spd": None if speed is None else round(float(speed), 1),
        "rpm": None if rpm is None else int(round(rpm)),
        "conf": round(float(conf), 3),
        "valid": bool(valid),
    }
    raw = _body_bytes(body)
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    # The CRC covers the exact bytes before it, so C++/Rust/JS receivers can
    # verify the datagram without re-serializing JSON.
    return raw[:-1] + b',"crc":' + str(crc).encode("ascii") + b"}"


def decode_payload(data: bytes) -> dict[str, Any]:
    if not data or len(data) > MAX_PACKET:
        raise ProtocolError("bad packet length")
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid json: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("payload is not an object")

    crc = obj.get("crc")
    if not isinstance(crc, int) or not 0 <= crc <= 0xFFFFFFFF:
        raise ProtocolError("missing crc")
    marker = b',"crc":'
    idx = data.rfind(marker)
    if idx < 0 or not data.endswith(b"}"):
        raise ProtocolError("malformed crc trailer")
    crc_text = data[idx + len(marker) : -1]
    if not crc_text.isdigit():
        raise ProtocolError("bad crc trailer")
    if int(crc_text) != crc:
        raise ProtocolError("crc trailer mismatch")
    body_bytes = data[:idx] + b"}"
    if (zlib.crc32(body_bytes) & 0xFFFFFFFF) != crc:
        raise ProtocolError("crc mismatch")

    if obj.get("v") != PROTO_VERSION:
        raise ProtocolError("unsupported version")
    if not isinstance(obj.get("seq"), int) or not 0 <= obj["seq"] <= 0xFFFFFFFF:
        raise ProtocolError("bad seq")
    if not isinstance(obj.get("boot_id"), str) or len(obj["boot_id"]) > 8:
        raise ProtocolError("bad boot_id")
    if not isinstance(obj.get("t_mono_us"), int) or obj["t_mono_us"] < 0:
        raise ProtocolError("bad t_mono_us")
    if not isinstance(obj.get("page"), str) or not obj["page"]:
        raise ProtocolError("bad page")
    if not isinstance(obj.get("valid"), bool):
        raise ProtocolError("bad valid flag")

    for key, (lo, hi) in _LIMITS.items():
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"bad {key}")
        if not lo <= float(value) <= hi:
            raise ProtocolError(f"{key} out of range")

    if obj["valid"] and (obj.get("spd") is None or obj.get("rpm") is None):
        raise ProtocolError("valid packet without readings")
    return obj

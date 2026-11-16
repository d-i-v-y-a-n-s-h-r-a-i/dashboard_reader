#!/usr/bin/env python3
"""Print validated TANDAV telemetry packets as they arrive."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tandav.protocol import ProtocolError, decode_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=47001)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    print(f"listening on {args.host}:{args.port}")
    last = 0.0
    bad = 0
    while True:
        data, addr = sock.recvfrom(2048)
        now = time.monotonic()
        try:
            obj = decode_payload(data)
        except ProtocolError as exc:
            bad += 1
            print(json.dumps({"error": str(exc), "from": f"{addr[0]}:{addr[1]}", "bad": bad}))
            continue
        obj["_from"] = f"{addr[0]}:{addr[1]}"
        obj["_interval_ms"] = round((now - last) * 1000.0, 1) if last else None
        last = now
        print(json.dumps(obj))


if __name__ == "__main__":
    raise SystemExit(main())

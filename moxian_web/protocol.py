"""Adapt the supplied V1 parser to the supplied QnBot SDK without editing either.

Only CRC-valid V1 frames enter the SDK. A pressure-only report must never
manufacture valid pose tracking (the vendor helper supplies a dummy pose).
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "MoxianEMF-Windows-20260917" / "MoxianEMF"
sys.path.insert(0, str(VENDOR / "sdk"))
sys.path.insert(0, str(VENDOR / "server" / "qnbot_hooks"))

import emf_v1  # noqa: E402
from qnbot_sdk.glove.qntp import GloveStreamFrame, GloveStreamFramer  # noqa: E402


class V1Framer:
    def __init__(self):
        self.buffer = bytearray()
        self.extra = {}
        self.frames = 0
        self.crc_errors = 0
        self.samples = deque(maxlen=1024)

    def feed(self, chunk: bytes):
        self.buffer.extend(chunk)
        frames = []
        while len(self.buffer) >= emf_v1.HEADER_LEN:
            if not emf_v1.looks_like_v1(self.buffer):
                del self.buffer[0]
                continue
            size = int.from_bytes(self.buffer[5:7], "little")
            if not emf_v1.HEADER_LEN <= size <= 4096:
                del self.buffer[0]
                continue
            if len(self.buffer) < size + 2:
                break
            if emf_v1.crc16_modbus(self.buffer[:size]) != int.from_bytes(self.buffer[size:size + 2], "little"):
                self.crc_errors += 1
                del self.buffer[0]
                continue
            taken = emf_v1.take_v1_frame(self.buffer)
            if taken is None:
                break
            kind, payload, raw, extra = taken
            if not kind:
                continue
            self.extra = extra
            self.frames += 1
            self.samples.append((time.monotonic(), tuple(payload[:460])))
            if not extra.get("emf80"):
                payload = payload[:460] + bytes(80)
            frames.append(GloveStreamFrame(kind, payload, raw))
        return frames


def install_v1_adapter():
    """Install once before constructing the serial SDK driver; return diagnostics."""
    parser = V1Framer()

    def feed(_self, chunk):
        return parser.feed(chunk)

    GloveStreamFramer.feed = feed
    return parser

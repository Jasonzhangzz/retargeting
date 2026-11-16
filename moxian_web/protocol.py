"""Adapt the supplied V1 parser to the supplied QnBot SDK without editing either.

Only CRC-valid V1 frames enter the SDK. A pressure-only report must never
manufacture valid pose tracking (the vendor helper supplies a dummy pose).
"""
from __future__ import annotations

import sys
import time
import struct
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "moxian_web" / "vendor" / "MoxianEMF-Windows-20260917" / "MoxianEMF"
sys.path.insert(0, str(VENDOR / "sdk"))
sys.path.insert(0, str(VENDOR / "server" / "qnbot_hooks"))

import emf_v1  # noqa: E402
from qnbot_sdk.glove.qntp import GloveStreamFrame, GloveStreamFramer, qntp_snapshot_message  # noqa: E402


def bound_live_sdk_history(buffer, capacity=16):
    """Limit the read-only console's SDK history before connecting its device.

    This SDK exposes constructor capacities but its system builder does not
    forward them. All console consumers use latest(); raw recording has its
    own stream. Avoid retaining 1024 nested EMF/ADC packets per SDK channel,
    which makes cyclic GC pause serial dispatch for hundreds of milliseconds.
    Keep the existing buffer object and its subscriptions/sequence semantics.
    """
    if not isinstance(capacity, int) or capacity < 2:
        raise ValueError('Live SDK history capacity must be at least two')
    from qnbot_sdk.core.buffer import TimeSeriesBuffer
    if not isinstance(buffer, TimeSeriesBuffer) or buffer.keys():
        raise ValueError('Configure the supported SDK buffer before connecting')
    buffer._default_capacity = capacity
    buffer._capacities = {key: min(value, capacity) for key, value in buffer._capacities.items()}


class V1Framer:
    def __init__(self):
        self.buffer = bytearray()
        self.extra = {}
        self.frames = 0
        self.crc_errors = 0
        self.samples = deque(maxlen=1024)
        self.pose_received_at = None

    def pose_diagnostics(self):
        """Read the CRC-accepted wire snapshot, before any SDK pose processing.

        Manual signed-i16 decoding is compared with the vendor decoder on the
        exact same 80 bytes. Channel names follow the documented SDK order;
        this cannot establish which physical receiver was assigned a channel.
        """
        extra = self.extra
        snapshot = extra.get('emf80')
        if not snapshot or len(snapshot) != 80:
            return {'ready': False, 'reason': 'no_pose_snapshot'}
        message = qntp_snapshot_message(snapshot)
        channels = []
        for i, name in enumerate(('thumb', 'index', 'middle', 'ring', 'pinky')):
            fields = list(struct.unpack_from('<7h', snapshot, 10 + i * 14))
            manual = [v * .00001 for v in fields[:3]]
            vendor = [v / 1_000_000 for v in message['fsq'][i][1:4]]
            channels.append({'channel_index': i, 'sdk_name': name,
                             'valid': bool(snapshot[8] & (1 << i)), 'fields_i16': fields,
                             'wire_position_m': manual, 'vendor_position_m': vendor,
                             'decoder_difference_m': max(abs(a-b) for a,b in zip(manual,vendor))})
        return {'ready': True, 'source': 'crc_valid_v1_emf_snapshot_before_sdk',
                'age_ms': (time.monotonic()-self.pose_received_at)*1000 if self.pose_received_at is not None else None,
                'coordinate_frame': 'device_reported_no_registration',
                'hand_side': 'left' if snapshot[9] else 'right',
                'stream_sequence': int.from_bytes(snapshot[:4], 'little'),
                'frame_sequence': int.from_bytes(snapshot[4:8], 'little'),
                'emf_block_bytes': len(extra.get('emf_raw', b'')),
                'snapshot_offset': extra.get('emf_raw', b'').find(snapshot),
                'snapshot_hex': snapshot.hex(), 'channels': channels}

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
            self.pose_received_at = time.monotonic() if extra.get('emf80') else None
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

import pytest
pytest.importorskip("serial", reason="Moxian service requires requirements-moxian.txt")
"""Run using .venv-moxian/bin/python -m unittest discover -s tests -p test_moxian_protocol.py."""
import struct
import unittest
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from moxian_web.protocol import V1Framer, emf_v1
from qnbot_sdk.glove.decoder import GloveFrameDecoder
from qnbot_sdk.glove.qntp import qntp_snapshot_message
from moxian_web.server import ReadOnlyGloveModule, Runtime
from qnbot_sdk.core import GloveHapticsCommand


def make_frame(with_pose=True, side=1):
    pressure = bytes(i % 256 for i in range(460))
    bitmap = emf_v1.SUB_PRESSURE
    blocks = struct.pack("<H", 460) + pressure
    if with_pose:
        bitmap |= emf_v1.SUB_EMF
        pose = struct.pack("<IIBB", 17, 23, 31, side)
        pose += b"".join(struct.pack("<7h", 1000 + i, -2000, 3000, 32767, 0, 0, 0) for i in range(5))
        blocks += struct.pack("<H", len(pose)) + pose
    return emf_v1.build_frame(domain=emf_v1.DOMAIN_DATA, cmd=emf_v1.CMD_STREAM,
                             payload=bytes([0]) + struct.pack("<I", bitmap) + blocks,
                             ftype=emf_v1.TYPE_RPT)


class V1ProtocolTests(unittest.TestCase):
    def test_every_fragment_size_keeps_pose_and_all_tactile_cells(self):
        raw = make_frame()
        for step in (1, 2, 7, 12, 13, 80, 548, 8192):
            with self.subTest(step=step):
                parser = V1Framer()
                result = []
                stream = b"noise" + raw + raw
                for start in range(0, len(stream), step):
                    result.extend(parser.feed(stream[start:start + step]))
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0].payload[:460], bytes(i % 256 for i in range(460)))
                self.assertEqual(result[0].payload[468], 31)

    def test_bad_crc_is_rejected_and_next_frame_resynchronizes(self):
        raw = make_frame()
        bad = bytearray(raw)
        bad[40] ^= 1
        parser = V1Framer()
        frames = parser.feed(bytes(bad) + raw)
        self.assertEqual(parser.crc_errors, 1)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].raw, raw)

    def test_pressure_only_report_never_invents_valid_pose(self):
        frame = V1Framer().feed(make_frame(with_pose=False))[0]
        self.assertEqual(frame.payload[460:], bytes(80))
        self.assertEqual(len(frame.payload[:460]), 460)

    def test_sdk_preserves_firmware_side_position_and_orientation(self):
        frame = V1Framer().feed(make_frame(side=1))[0]
        message = qntp_snapshot_message(frame.payload[460:])
        payload = GloveFrameDecoder(side="right").decode_message(message)
        self.assertEqual(payload.hand_side, "left")
        self.assertTrue(all(payload.raw_valid_mask.values()))
        self.assertEqual(payload.fingertip_local["thumb"].position, (.01, -.02, .03))
        self.assertEqual(payload.fingertip_local["thumb"].quaternion_xyzw, (0., 0., 0., 1.))

    def test_usb_loss_does_not_kill_cleanup_or_leave_port_owned(self):
        runtime = Runtime.__new__(Runtime)
        runtime.sdk = Mock()
        runtime.sdk.disconnect.side_effect = OSError("USB disconnected")
        endpoint = runtime.endpoint = Mock()
        runtime.lock = threading.Lock()
        runtime.last_frame = 1
        runtime.state = {"status": "streaming"}
        runtime.samples = deque()
        runtime.parser = SimpleNamespace(buffer=bytearray(b"partial frame"))
        runtime.disconnect()
        endpoint.close.assert_called_once()
        self.assertIsNone(runtime.sdk)
        self.assertIsNone(runtime.endpoint)
        self.assertEqual(runtime.state["status"], "disconnected")
        self.assertEqual(runtime.last_frame, 0)
        self.assertFalse(runtime.parser.buffer)

    def test_console_never_writes_haptics_even_during_sdk_cleanup(self):
        spec, driver = Mock(), Mock()
        device = ReadOnlyGloveModule()._create_device(spec, driver)
        writer = device.command_writers[0]
        writer.submit(GloveHapticsCommand(source=spec.source, strengths={"thumb": 0}, timestamp=0))
        writer.clear()
        with self.assertRaises(ValueError):
            writer.submit(GloveHapticsCommand(source=spec.source, strengths={"thumb": 10}, timestamp=0))
        driver.write_serial_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()

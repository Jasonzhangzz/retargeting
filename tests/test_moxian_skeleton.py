import copy
import tempfile
import time
import signal
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from moxian_web.skeleton import JOINT_CHAINS, JOINT_NAMES, SkeletonBridge, raw_input, validate_pose


class SkeletonTests(unittest.TestCase):
    def pose(self):
        return {"hand_side": "left", "coordinate_frame": "wrist_local", "timestamp_ms": 123450.,
                "names": JOINT_NAMES.copy(), "positions_local": [[0., 0., 0.] for _ in JOINT_NAMES],
                "orientations_local_xyzw": [[0., 0., 0., 1.] for _ in JOINT_NAMES]}

    def test_official_topology_and_named_arrays(self):
        self.assertEqual(len(JOINT_NAMES), 25)
        self.assertEqual(sum(len(c)-1 for c in JOINT_CHAINS), 24)
        pose = self.pose()
        for k in ("names", "positions_local", "orientations_local_xyzw"):
            pose[k].reverse()
        self.assertEqual(validate_pose(pose), pose)

    def test_reject_bad_side_frame_nan_orientation_or_missing_joint(self):
        for kind in ("side", "frame", "nan", "orientation", "missing", "wrist"):
            pose = self.pose()
            if kind == "side": pose["hand_side"] = "right"
            if kind == "frame": pose["coordinate_frame"] = "world"
            if kind == "nan": pose["positions_local"][1][0] = float("nan")
            if kind == "orientation": pose["orientations_local_xyzw"][1] = [0., 0., 0., 0.]
            if kind == "missing": pose["names"][1] = "unknown"
            if kind == "wrist": pose["positions_local"][0][0] = .1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_pose(pose)

    def test_pre_filter_input_units_quaternion_and_invalid_mask(self):
        nodes = {name: {"position": [.01, .02, .03], "quaternion_wxyz": [1., 0., 0., 0.], "valid": True}
                 for name in ("thumb", "index", "middle", "ring", "pinky")}
        telemetry = SimpleNamespace(meta=SimpleNamespace(timestamp=123.45), quality={"input_6dof": {"fingertip_local": nodes}})
        frame = raw_input(telemetry)
        self.assertEqual(frame["timestamp_ms"], 123450.)
        self.assertEqual(frame["fingers"]["thumb"], {"position": [.01, .02, .03], "quaternion_xyzw": [0., 0., 0., 1.]})
        nodes["thumb"]["valid"] = False
        self.assertIsNone(raw_input(telemetry))

    def test_missing_worker_runtime_isolated_from_sensor_service(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = SkeletonBridge(Path(folder), Path(folder))
            bridge.start("test-device")
            self.assertEqual(bridge.snapshot()["status"], "unavailable")
            self.assertFalse(bridge.snapshot()["ready"])
            for _ in range(10000):
                bridge.send({"type": "frame"})
            bridge.stop()

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / ".venv-moxian-ik/bin/python").is_file(), "Official SDK environment not installed")
    def test_official_worker_progress_controls_backpressure_and_failure(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            bridge = SkeletonBridge(root, Path(folder))
            self.addCleanup(bridge.stop)
            bridge.start("automated-test-no-serial-device")
            deadline = time.monotonic() + 8
            while not bridge.snapshot().get("capture") and time.monotonic() < deadline:
                time.sleep(.02)
            state = bridge.snapshot()
            self.assertEqual(state["status"], "needs_calibration", state)
            stage = state["capture"]["stage"]
            self.assertEqual(stage["stage_id"], "close")
            with self.assertRaises(ValueError):
                bridge.control("confirm", {"request_id": "expired-request"})
            frame = {"type": "frame", "timestamp_ms": 1., "fingers": {
                name: {"position": [.03, .02, .01], "quaternion_xyzw": [0., 0., 0., 1.]}
                for name in ("thumb", "index", "middle", "ring", "pinky")}}
            os.kill(bridge.proc.pid, signal.SIGSTOP)
            try:
                started = time.monotonic()
                for i in range(10000):
                    bridge.send({**frame, "timestamp_ms": i + 1.})
                self.assertLess(time.monotonic() - started, 1.)
                self.assertLessEqual(bridge.frames.qsize(), 1)
            finally:
                os.kill(bridge.proc.pid, signal.SIGCONT)
            bridge.control("confirm", {"request_id": stage["request_id"]})
            for i in range(5):
                bridge.send({**frame, "timestamp_ms": 10001. + i})
                time.sleep(.015)
            stage = bridge.snapshot()["capture"]["stage"]
            self.assertEqual(stage["state"], "collecting")
            bridge.control("cancel", {"request_id": stage["request_id"], "stage_id": stage["stage_id"]})
            deadline = time.monotonic() + 2
            while bridge.snapshot()["status"] != "cancelled" and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(bridge.snapshot()["status"], "cancelled")
            bridge.proc.kill()
            bridge.proc.wait(timeout=2)
            self.assertFalse(bridge.snapshot()["ready"])
            self.assertEqual(bridge.snapshot()["status"], "error")
            bridge.stop()
            bridge.start("automated-test-no-serial-device")
            deadline = time.monotonic() + 8
            while not bridge.snapshot().get("capture") and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(bridge.snapshot()["capture"]["stage"]["state"], "awaiting_confirmation")


if __name__ == "__main__":
    unittest.main()

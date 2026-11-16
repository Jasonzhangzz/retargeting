"""Timeline edge cases that would silently change selected hand angles."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from emf_retarget.episode import align_camera, fingerprint


class AlignmentTests(unittest.TestCase):
    def native(self):
        native = {}
        for side in ("left", "right"):
            native[f"{side}_action_timestamp_ns"] = np.asarray([100, 200, 300])
            native[f"{side}_source_timestamp_ns"] = np.asarray([1001, 2002, 3003])
            native[f"{side}_hand_q"] = np.repeat(np.arange(3)[:, None], 20, axis=1).astype(np.float32)
            native[f"{side}_skeleton_position"] = np.zeros((3, 21, 3))
            native[f"{side}_skeleton_orientation_xyzw"] = np.tile([0, 0, 0, 1], (3, 21, 1))
            native[f"{side}_skeleton_confidence"] = np.ones((3, 21))
            native[f"{side}_hand_q_names"] = np.asarray([f"{side}_{i}" for i in range(20)])
        return native

    def run_alignment(self, frames, lookups=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.json"
            path.write_text(json.dumps({"data": frames}), encoding="utf8")
            return align_camera(path, self.native(), lookups or {"left": {}, "right": {}})

    def test_nearest_tie_and_period_boundary(self):
        camera, info = self.run_alignment([{"timestamp": t} for t in (150, 400, 401)])
        np.testing.assert_array_equal(camera["left_native_frame_indices"], [0, 2])
        np.testing.assert_array_equal(camera["camera_timestamp_ns"], [150, 400])
        np.testing.assert_array_equal(camera["left_source_timestamp_ns"], [1001, 3003])
        self.assertEqual(info["dropped_frame_indices"], [2])
        np.testing.assert_array_equal(camera["wuji_hand_q"][:, :20], camera["right_hand_q"])

    def test_duplicate_exact_payload_uses_closest_action_clock(self):
        p, q, c = np.zeros((5, 3)), np.tile([0, 0, 0, 1], (5, 1)), np.ones(5)
        message = {"poses": [{"pose": {"position": a.tolist(), "orientation": b.tolist()}, "confidence": float(d)}
                              for a, b, d in zip(p, q, c)]}
        lookup = {s: {fingerprint(p, q, c): [0, 2]} for s in ("left", "right")}
        camera, info = self.run_alignment([{"timestamp": 210, "left_glove_emf_poses": message,
                                            "right_glove_emf_poses": message}], lookup)
        np.testing.assert_array_equal(camera["left_native_frame_indices"], [2])
        self.assertEqual(info["nearest_action_frame_indices"]["left"], [])

    def test_fingerprint_retains_float64_before_solver_boundary(self):
        p = np.zeros((5, 3)); q = np.ones((5, 4)); c = np.ones(5)
        first = fingerprint(p, q, c)
        q[0, 0] += 1e-12
        self.assertNotEqual(first, fingerprint(p, q, c))

    def test_no_pair_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError, "No paired"):
            self.run_alignment([{"timestamp": 1000}])


if __name__ == "__main__":
    unittest.main()

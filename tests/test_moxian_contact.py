import json
import tempfile
import unittest
from pathlib import Path

from moxian_web.contact import ContactEngine, PATCH_IDS, default_mapping, validate_mapping


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = ContactEngine(Path(self.tmp.name))
        self.engine.bind("device", "left")

    def tare(self, base=10):
        self.engine.start_tare(now=0)
        for i in range(361):
            self.engine.feed([base] * 460, i / 120)

    def test_default_mapping_covers_all_460_cells_and_twenty_regions(self):
        for side in ("left", "right"):
            assignments = default_mapping(side)
            self.assertEqual(len(assignments), 460)
            self.assertEqual(set(assignments), set(range(20)))
        left, right = default_mapping("left"), default_mapping("right")
        for row in range(23):
            self.assertEqual(left[row*20:(row+1)*20], right[row*20:(row+1)*20][::-1])
        self.assertEqual(PATCH_IDS[left[380]], "thumb_distal")
        self.assertEqual(PATCH_IDS[left[344]], "index_intermediate")
        self.assertEqual(PATCH_IDS[left[304]], "index_proximal")

    def test_tare_median_mad_persistence_and_normalization(self):
        self.engine.start_tare(now=0)
        for i in range(361):
            self.engine.feed([10 + (i % 5 - 2)] * 460, i / 120)
        profile = self.engine.profile["tare"]
        self.assertEqual(profile["baseline"], [10.] * 460)
        self.assertEqual(profile["sigma"], [1.4826] * 460)
        self.engine.feed([16] * 460, 3.01)
        first = self.engine.snapshot(now=3.01)["normalized"][0]
        self.assertGreater(first, 0)
        self.assertLess(first, .5)
        for i in range(1, 61):
            self.engine.feed([16] * 460, 3.01 + i/120)
        self.assertAlmostEqual(self.engine.snapshot(now=3.51)["normalized"][0], .5, places=3)
        restored = ContactEngine(Path(self.tmp.name))
        restored.bind("device", "left")
        self.assertEqual(restored.profile["tare"], profile)
        restored.bind("other-device", "left")
        self.assertEqual(restored.snapshot(now=3)["status"], "needs_tare")

    def test_single_cell_spike_never_creates_patch_contact(self):
        self.tare()
        raw = [10] * 460
        raw[380] = 255
        for i in range(10):
            self.engine.feed(raw, 3.01 + i*.01)
        self.assertEqual(self.engine.snapshot(now=3.1)["count"], 0)
        self.assertFalse(any(self.engine.snapshot(now=3.1)["cell_contact"]))

    def test_neighbor_support_and_enter_exit_hysteresis(self):
        self.tare()
        raw = [10] * 460
        # A sustained 2 ADC light touch was below the previous 6 ADC floor.
        raw[380] = raw[381] = 12
        self.engine.feed(raw, 3.01)
        self.assertEqual(self.engine.snapshot(now=3.01)["count"], 0)
        for i in range(1, 12):
            self.engine.feed(raw, 3.01 + i/100)
        self.assertEqual(self.engine.snapshot(now=3.12)["active_patches"], ["thumb_distal"])
        cells=self.engine.snapshot(now=3.12)['cell_contact']
        self.assertEqual([i for i,on in enumerate(cells) if on],[380,381])
        # Signal below entry threshold stays active above the lower exit threshold.
        raw[380] = raw[381] = 11
        for i in range(1, 21):
            self.engine.feed(raw, 3.12 + i/100)
        self.assertEqual(self.engine.snapshot(now=3.32)["count"], 1)
        raw[380] = raw[381] = 10
        for i in range(1, 9):
            self.engine.feed(raw, 3.32 + i/100)
        self.assertEqual(self.engine.snapshot(now=3.40)["count"], 1)
        for i in range(9, 16):
            self.engine.feed(raw, 3.32 + i/100)
        self.assertEqual(self.engine.snapshot(now=3.47)["count"], 0)

    def test_ema_rejects_brief_adjacent_spike_and_signed_noise(self):
        self.tare()
        raw = [10] * 460
        raw[380] = raw[381] = 22
        self.engine.feed(raw, 3.01)
        for i in range(1, 30):
            self.engine.feed([10] * 460, 3.01 + i/120)
        self.assertEqual(self.engine.snapshot(now=3.25)["count"], 0)
        for i in range(1, 121):
            self.engine.feed([10 + (2 if i % 2 else -2)] * 460, 3.25 + i/120)
        snapshot = self.engine.snapshot(now=4.25)
        self.assertEqual(snapshot["count"], 0)
        self.assertLess(max(snapshot["normalized"]), .03)

    def test_ema_uses_elapsed_time_and_resets_after_data_gap(self):
        # Same physical duration at different sample rates yields equal response.
        responses = []
        for rate in (60, 120):
            self.tare()
            for i in range(1, int(rate*.2)+1):
                self.engine.feed([14] * 460, 3 + i/rate)
            responses.append(self.engine.snapshot(now=3.2)["normalized"][0])
        self.assertAlmostEqual(*responses, places=3)
        self.assertEqual(self.engine.snapshot(now=3.2)["count"], 20)
        self.engine.feed([10] * 460, 3.6)
        after_gap = self.engine.snapshot(now=3.6)
        self.assertEqual(after_gap["count"], 0)
        self.assertEqual(max(after_gap["normalized"]), 0)

    def test_unknown_before_tare_and_after_loss(self):
        self.assertIsNone(self.engine.snapshot(now=0)["count"])
        self.tare()
        self.assertTrue(self.engine.snapshot(now=3)["ready"])
        self.assertFalse(self.engine.snapshot(now=3.3)["ready"])
        self.assertIsNone(self.engine.snapshot(live=False, now=3)["count"])
        self.assertTrue(all(p["contact"] is None for p in self.engine.snapshot(live=False, now=3)["patches"]))
        self.assertTrue(all(c is None for c in self.engine.snapshot(live=False,now=3)['cell_contact']))

    def test_interrupted_or_insufficient_tare_keeps_old_profile(self):
        self.tare()
        original = json.loads(json.dumps(self.engine.profile["tare"]))
        self.engine.start_tare(now=4)
        for i in range(31):
            self.engine.feed([50] * 460, 4 + i/10)
        self.assertEqual(self.engine.profile["tare"], original)
        self.assertEqual(self.engine.tare_status["state"], "failed")
        self.engine.start_tare(now=8)
        self.engine.cancel_tare("offline")
        self.assertEqual(self.engine.profile["tare"], original)

    def test_edit_mapping_validate_persist_restore_and_clear_tare(self):
        self.tare()
        assignments = default_mapping("left")
        assignments[380] = PATCH_IDS.index("index_distal")
        self.engine.set_mapping(assignments, ["index_distal"])
        self.assertEqual(self.engine.mapping()["assignments"][380], PATCH_IDS.index("index_distal"))
        self.assertEqual(self.engine.mapping()["confirmed"], ["index_distal"])
        self.engine.restore_mapping()
        self.assertEqual(self.engine.mapping()["assignments"], default_mapping("left"))
        self.assertEqual(self.engine.mapping()["confirmed"], [])
        with self.assertRaises(ValueError):
            validate_mapping([0] * 460, [])
        with self.assertRaises(ValueError):
            validate_mapping(default_mapping("left"), ["unknown"])
        self.engine.clear_tare()
        self.assertEqual(self.engine.snapshot(now=4)["status"], "needs_tare")


if __name__ == "__main__":
    unittest.main()

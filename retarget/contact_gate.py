"""Advance the SQP demonstration only after executed grasp contact is stable."""
from dataclasses import dataclass
import numpy as np


def prelift_frame(sequence, start_frame, end_frame):
    """Find the last reference frame before the demonstrated grasp rises."""
    start, end = int(np.floor(start_frame)), int(end_frame)
    active = np.flatnonzero(np.any(sequence.contact.mask[start:end+1] > 0, axis=1))
    if not len(active):
        return float(end)
    touch = start + int(active[0])
    frames = np.arange(touch, end+1)
    # Object input is xyzw,xyz. Also inspect the hand: it can start lifting
    # before the demonstrated object leaves its support.
    object_z = sequence.object_pose_xyzw[:, 6]
    hand_z = np.interp(np.arange(len(object_z)) * sequence.hand_index_per_frame,
                       np.arange(len(sequence.hand)), sequence.hand[:, 2])
    lifting = ((hand_z[frames] - hand_z[touch] >= .004)
               | (object_z[frames] - object_z[touch] >= .005))
    candidates = frames[lifting]
    return float(max(touch, int(candidates[0])-1)) if len(candidates) else float(end)


@dataclass
class ContactReferenceGate:
    frame: float
    hold_frame: float
    end_frame: float
    dt: float
    min_fingers: int = 2
    stable_seconds: float = .09
    stable_elapsed: float = 0.
    released: bool = False
    contacting_fingers: tuple = ()

    @property
    def ready(self):
        return self.stable_elapsed + 1e-9 >= self.stable_seconds

    @property
    def frame_limit(self):
        if self.ready:
            return self.end_frame
        return self.frame if self.released else max(self.frame, self.hold_frame)

    def make_targets(self, sequence, cfg):
        from forceaware.targets import make_window
        return make_window(
            sequence, executed_steps=0, horizon=cfg.optimizer.horizon,
            time_grid=cfg.simulator.time_grid, start_frame=self.frame,
            end_frame=self.frame_limit, hold_frame=-1)

    def observe(self, fingers_per_step):
        for fingers in fingers_per_step:
            self.contacting_fingers = tuple(sorted(set(fingers)))
            self.stable_elapsed = (self.stable_elapsed + self.dt
                                   if len(self.contacting_fingers) >= self.min_fingers else 0.)

    def advance(self, target_frames):
        # These were the references actually issued, not elapsed wall time.
        self.frame = float(target_frames[-1])
        self.released |= self.frame > self.hold_frame + 1e-6

    def diagnostic(self):
        return dict(reference_frame=self.frame, hold_frame=self.hold_frame,
                    frame_limit=self.frame_limit, ready=self.ready,
                    waiting_for_contact=not self.ready and self.frame >= self.hold_frame - 1e-6,
                    released=self.released,
                    contacting_fingers=list(self.contacting_fingers),
                    stable_contact_seconds=self.stable_elapsed,
                    required_fingers=self.min_fingers,
                    required_stable_seconds=self.stable_seconds)

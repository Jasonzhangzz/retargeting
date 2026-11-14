"""Retarget one ParaHome clip onto G1 with Inspire hands; example 11 batches the dataset."""

import time
from pathlib import Path

import numpy as np
import warp as wp

from robokit.assets.datasets import parahome
from robokit.helpers.full_body_retargeting import FullBodyRetargeter, solve_grasp_refine
from robokit.helpers.full_body_retargeting.parahome_utils import find_parahome_intervals, load_parahome_clip
from robokit.utils.visualize_utils import RetargetingPlayback, view_retargeting


SEQUENCE = "s1"
INTERVAL = 2  # "Put pot on gas stove": both hands grasp the pot


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- clip ---
    data_root = str(parahome.data_root())
    interval = find_parahome_intervals(data_root, SEQUENCE)[INTERVAL]
    clip = load_parahome_clip(data_root, interval, device=device)
    print(f"retargeting {SEQUENCE}[{INTERVAL}] {interval.label} ({clip.num_frames} frames)")

    # --- retargeting: one held grasp per sustained contact run ---
    retargeter = FullBodyRetargeter(device=device, load_meshes=True)
    start = time.time()
    result = solve_grasp_refine(clip, retargeter)
    qpos = result.qpos
    print(f"solved {qpos.shape[0]} frames in {time.time() - start:.1f}s")

    # --- viewer: shared with the batch example ---
    meshes = {hand.object_name: hand.object_mesh for hand in clip.hands.values() if hand.object_mesh is not None}

    def load_clip(path: Path) -> RetargetingPlayback:
        return RetargetingPlayback(
            qpos=qpos,
            fps=clip.fps,
            object_meshes=[meshes[name] for name in clip.object_names],
            T_world_object=qpos[:, 7 + retargeter.body_robot.num_actuated_joints :].reshape(len(qpos), -1, 7),
            targets_world=np.concatenate(list(result.targets_world.values()), axis=1),
            contacts_world=np.concatenate(list(result.contacts_world.values()), axis=1),
            contact_mask=np.concatenate(list(result.contact_masks.values()), axis=1),
        )

    view_retargeting(retargeter.body_robot, {interval.label: Path("clip")}, load_clip)


if __name__ == "__main__":
    main()

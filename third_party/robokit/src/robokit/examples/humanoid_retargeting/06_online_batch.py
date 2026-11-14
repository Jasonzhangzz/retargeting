"""Batch online retargeting example: one solve_numpy() call advances many clips in lockstep."""

import time

import numpy as np
import viser
import warp as wp

from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.loaders import fetch_smplx_clip
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned
from robokit.utils.visualize_utils import ViserBatchUrdf


CLIPS = [
    "datasets/sfu/0005/0005_2FeetJump001_poses.npz",
    "datasets/sfu/0017/0017_JumpAndRoll001_poses.npz",
    "datasets/sfu/0017/0017_WushuKicks001_poses.npz",
    "datasets/sfu/0018/0018_Moonwalk001_poses.npz",
]
GRID_SPACING = 2.0
FPS = 30.0


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    batch = len(CLIPS)

    # --- retargeting setup: one helper with batch_size=len(CLIPS) ---
    robot = g1_aligned.load_robot(load_meshes=True)
    helper = HumanoidRetargeterOnline(g1_aligned, device=device, robot=robot)

    # --- motions: raw clips plus per-clip heights; step() applies the correct scale per clip ---
    clips, heights = [], []
    for path in CLIPS:
        smplx, motion, _ = fetch_smplx_clip(path, helper.human_joint_names, tgt_fps=int(FPS))
        clips.append(motion)
        heights.append(smplx.human_height)
    human_heights = np.asarray(heights, dtype=np.float32)
    lengths = [len(c) for c in clips]
    max_len = max(lengths)
    helper.warmup(batch)
    helper.solve_numpy(np.stack([clip[0] for clip in clips]), human_heights=human_heights)
    helper.reset()

    # --- viewer: one robot per clip, spread on a line (shorter clips freeze on their last frame) ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10, height=10)
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=batch, root_node_name="/robots")
    offsets = (np.arange(batch, dtype=np.float32) - (batch - 1) / 2) * GRID_SPACING
    timing = server.gui.add_number("Batch solve (ms)", 0.001, disabled=True)

    helper.reset()
    f = 0
    while True:
        # --- retargeting: one solve advances every clip by one frame ---
        t0 = time.time()
        frame_f = np.stack([clips[b][min(f, lengths[b] - 1)] for b in range(batch)])
        qpos = helper.solve_numpy(frame_f, human_heights=human_heights)
        timing.value = 0.99 * timing.value + 0.01 * (time.time() - t0) * 1000

        # --- viewer: update all robots ---
        T_world_base = qpos[:, :7].copy()
        T_world_base[:, 1] += offsets
        batch_urdf.update_cfg(qpos[:, 7:], T_world_base=T_world_base)
        f = (f + 1) % max_len
        if f == 0:
            helper.reset()
        time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))


if __name__ == "__main__":
    main()

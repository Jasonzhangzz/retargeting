"""Batch offline retargeting example: solve several clips in one batched LM."""

import time

import numpy as np
import viser
import warp as wp

from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOffline
from robokit.helpers.humanoid_retargeting.loaders import fetch_smplx_clip
from robokit.helpers.humanoid_retargeting.presets.g1_offline import g1_offline
from robokit.helpers.humanoid_retargeting.presets.g1_offline_mapping import g1_offline_mapping
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

    # --- motions: SMPL-X clips -> ordered transform arrays ---
    robot = g1_offline_mapping.load_robot(load_meshes=True)
    retargeter = HumanoidRetargeterOffline(g1_offline_mapping, g1_offline, robot=robot, device=device)
    motions, heights = [], []
    for path in CLIPS:
        smplx, motion, _ = fetch_smplx_clip(path, retargeter.human_joint_names, tgt_fps=int(FPS))
        motions.append(motion)
        heights.append(smplx.human_height)

    # --- retargeting: clips padded to one length, one batched LM (padded frames masked out) ---
    lengths = np.array([len(motion) for motion in motions], dtype=np.int32)
    length = int(lengths.max())
    padded = np.stack(
        [np.concatenate([motion, np.repeat(motion[-1:], length - len(motion), axis=0)]) for motion in motions]
    )
    t0 = time.time()
    qpos = retargeter.solve_numpy(padded, valid_lengths=lengths, human_heights=np.asarray(heights, dtype=np.float32))
    print(f"clips {lengths.tolist()} frames, solved in {time.time() - t0:.1f}s")
    batch = len(motions)

    # --- viewer: one robot per clip; shorter clips freeze on their last frame ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10, height=10)
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=batch, root_node_name="/robots")
    offsets = (np.arange(batch, dtype=np.float32) - (batch - 1) / 2) * GRID_SPACING
    frame_slider = server.gui.add_slider("frame", min=0, max=length - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", True)
    while True:
        if playing.value:
            frame_slider.value = (frame_slider.value + 1) % length
        idx = np.minimum(frame_slider.value, lengths - 1)
        q = qpos[np.arange(batch), idx]
        T_world_base = q[:, :7].copy()
        T_world_base[:, 1] += offsets
        batch_urdf.update_cfg(q[:, 7:], T_world_base=T_world_base)
        time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()

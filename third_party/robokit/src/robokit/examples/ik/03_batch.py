"""Batch IK example: one solve() call tracks many targets."""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKSolver, presets
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf


GRID_SIZE = 4
BATCH_SIZE = GRID_SIZE**2


def main():
    # --- setup ---
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

    xy = np.stack(np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE)), axis=-1).reshape(-1, 2).astype(np.float32)
    T_world_base = np.zeros((BATCH_SIZE, 7), dtype=np.float32)
    T_world_base[:, :2] = (xy - 1.5) * 1.2
    T_world_base[:, 3] = 1.0
    target_pos = np.column_stack([0.45 + 0.04 * xy[:, 0], -0.18 + 0.12 * xy[:, 1], 0.35 + 0.08 * xy[:, 0]])
    target_wxyz = np.tile(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), (BATCH_SIZE, 1))

    # --- IK ---
    robot = Robot.load(urdf, load_meshes=True)
    ik = IKSolver(presets.smooth, robot=robot, target_links="panda_hand", device=device)
    ik.warmup(batch_size=BATCH_SIZE)

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6, height=6)
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=BATCH_SIZE, root_node_name="/robots")
    target_axes = server.scene.add_batched_axes(
        "/targets",
        batched_wxyzs=target_wxyz,
        batched_positions=target_pos + T_world_base[:, :3],
        axes_length=0.1,
        axes_radius=0.005,
    )
    control = server.scene.add_transform_controls("/offset", scale=0.2, disable_rotations=True)

    prev_state = None
    while True:
        # --- IK: one solve for all targets ---
        shifted_target_pos = target_pos + np.asarray(control.position, dtype=np.float32)
        state = ik.solve_numpy(
            np.concatenate([shifted_target_pos, target_wxyz], axis=-1),
            prev_state=prev_state,
            init_state=prev_state,
        )
        prev_state = state

        # --- viewer: update ---
        batch_urdf.update_cfg(state.q.numpy(), T_world_base=T_world_base)
        target_axes.batched_positions = shifted_target_pos + T_world_base[:, :3]
        time.sleep(0.001)


if __name__ == "__main__":
    main()

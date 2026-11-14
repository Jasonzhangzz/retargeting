"""Smooth IK example: warm-chain each solve into the next for jitter-free tracking."""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKSolver, presets
from robokit.robo import Robot


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # --- IK ---
    robot = Robot.load(urdf)
    ik = IKSolver(presets.smooth, robot=robot, target_links=target_link_name, device=device)
    ik.warmup(batch_size=1)

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.midrange_q)
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.2, 0.5), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    prev_state = None
    while True:
        # --- IK: solve ---
        start_time = time.time()
        state = ik.solve_numpy(
            np.concatenate([ik_target.position, ik_target.wxyz]), prev_state=prev_state, init_state=prev_state
        )
        wp.synchronize()
        elapsed_time = (time.time() - start_time) * 1000.0
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_time

        # --- viewer: render ---
        urdf_vis.update_cfg(state.q.numpy()[0])
        prev_state = state


if __name__ == "__main__":
    main()

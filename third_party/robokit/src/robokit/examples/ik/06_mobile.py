"""Mobile-base IK example: Fetch manipulator with a floating base."""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    urdf = load_robot_description("fetch_description")
    target_link_name = "gripper_link"

    # --- IK ---
    robot = Robot.load(urdf)
    config = (
        IKSolverConfig(
            enable_T_world_base=True,
            base_init_sample_range=(1.0, 1.0, 0.0),
            base_active_mask=[1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            sampler=RobertsSamplerConfig(keep_first=True, scale=1.0),
        )
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=5.0))
        .add(PositionLimit(weight=50.0))
        .add(
            SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.1, base_weight=0.5)
        )  # base_weight damps base motion vs prev frame
    )
    ik = IKSolver(config, robot=robot, target_links=target_link_name, device=device)
    ik.warmup(batch_size=1)

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(robot.spec.midrange_q)
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.6, 0.0, 0.55), wxyz=(0, 0.707, 0, -0.707)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    state = None
    while True:
        # --- IK: solve ---
        start_time = time.time()
        # warm-start from last solution: enables SmoothnessTask + prev-state hysteresis
        state = ik.solve_numpy(np.concatenate([ik_target.position, ik_target.wxyz]), prev_state=state, init_state=state)
        elapsed_time = (time.time() - start_time) * 1000.0
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_time

        # --- viewer: render ---
        base_np = state.T_world_base.numpy()[0]
        with server.atomic():
            urdf_vis.update_cfg(state.q.numpy()[0])
            base_frame.position = base_np[:3]
            base_frame.wxyz = base_np[3:]
        time.sleep(0.001)


if __name__ == "__main__":
    main()

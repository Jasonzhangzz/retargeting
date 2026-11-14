"""Bimanual self-collision aware IK example."""

import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import yumi
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


YUMI_SPHERES = yumi.COLLISION_SPHERE_PATH
TARGET_LINKS = ["yumi_link_7_r", "yumi_link_7_l"]


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("yumi_description")

    # --- IK ---
    robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=YUMI_SPHERES)
    config = (
        IKSolverConfig(solver=presets.smooth.solver, sampler=RobertsSamplerConfig(keep_first=True, scale=1.0))
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
        .add(SelfCollisionTask(robot, representation="sphere", weight=20.0, margin=0.02))
    )
    ik = IKSolver(config, robot=robot, target_links=TARGET_LINKS, device=device)
    ik.warmup(batch_size=1)  # optional

    init_q = np.array(
        [-1.2, -0.7, 0.3, 0.6, 0.0, 0.3, 0.0] + [1.2, -0.7, 0.3, 0.6, 0.0, 0.3, 0.0] + [0.0, 0.0],
        dtype=np.float32,
    )
    init_state = robot.forward_kinematics(
        robot.state(q=wp.from_numpy(init_q.reshape(1, -1), dtype=wp.float32, device=device))
    )

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(init_q)
    ik_targets = []
    for name in TARGET_LINKS:
        T_world_link = init_state.get_T_world_link(robot.spec.link_names.index(name))
        ik_targets.append(
            server.scene.add_transform_controls(
                f"/ik_targets/{name}",
                scale=0.2,
                position=T_world_link.numpy()[0, :3],
                wxyz=T_world_link.numpy()[0, 3:],
            )
        )
    timing = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    prev_state = init_state
    while True:
        # --- IK: one solve per frame ---
        t0 = time.time()
        positions = [t.position for t in ik_targets]
        wxyzs = [t.wxyz for t in ik_targets]
        state = ik.solve_numpy(
            np.stack([np.concatenate([p, q]) for p, q in zip(positions, wxyzs)], axis=-2),
            prev_state=prev_state,
            init_state=prev_state,
        )
        timing.value = 0.99 * timing.value + 0.01 * (time.time() - t0) * 1000

        # --- viewer: visualize the result ---
        urdf_vis.update_cfg(state.q.numpy()[0])
        prev_state = state
        time.sleep(0.001)


if __name__ == "__main__":
    main()

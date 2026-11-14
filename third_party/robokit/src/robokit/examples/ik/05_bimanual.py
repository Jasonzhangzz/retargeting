"""Bimanual IK example: two end effectors."""

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
    urdf = load_robot_description("yumi_description")
    target_link_names = ["yumi_link_7_r", "yumi_link_7_l"]

    # --- IK ---
    robot = Robot.load(urdf)
    ik = IKSolver(presets.single_candidate, robot=robot, target_links=target_link_names, device=device)
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
    for name in target_link_names:
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

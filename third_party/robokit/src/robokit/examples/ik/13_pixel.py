"""Pixel IK example: the hand follows a path drawn in a fixed camera's image. The red target pixel moves once along the gray path on the camera's image plane ("Replay" runs it again). `PixelTask` puts the hand origin (green) on the red ray from the camera through that pixel while `RotationTask` keeps the hand pointing down; smoothness picks the depth along the ray. Untick "Use pixels" and the hand ignores the path."""

import time

import numpy as np
import viser
import viser.transforms as vtf
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms import PixelTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.xform.numpy import project_points, rot_tl_to_tf_mat, transform_points


# camera at x = 1.6 m looking back at the robot (OpenCV axes: x right, y down, z forward)
K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
R_camera_world, C = np.array([[0.0, 1, 0], [0, 0, -1], [-1, 0, 0]]), np.array([1.6, 0.0, 0.5])
T_camera_world = rot_tl_to_tf_mat(R_camera_world, -R_camera_world @ C)


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

    # --- IK: the hand's pixel and orientation; nothing fixes its depth along the ray ---
    robot = Robot.load(urdf)
    hand = robot.link_names.index("panda_hand")
    pixels = PixelTask(frame_index=IKContext.TARGET_LINKS, uv=np.zeros((1, 1, 2)), intrinsic=K,
                       T_camera_world=T_camera_world, weight=0.05, device=device)  # fmt: skip
    config = (
        IKSolverConfig(solver=presets.smooth.solver, sampler=RobertsSamplerConfig(keep_first=True, scale=1.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(pixels)
        .add(PositionLimit(weight=50.0))
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
    )
    ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device)
    T_world_target = np.array([0.5, 0.0, 0.4, 0.0, 0.0, 1.0, 0.0])  # only its rotation is used: pointing down

    # --- target: a pixel path, 4 s at 30 fps, and the same path on the camera's image plane (depth 0.3 m) ---
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 120))
    path = np.stack([200 + 240 * s, 280 - 80 * np.sin(np.pi * s)], 1)
    path_on_plane = C + 0.3 * np.c_[path, np.ones(len(path))] @ np.linalg.inv(K).T @ R_camera_world

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    server.scene.add_camera_frustum("/camera", fov=2 * np.arctan(240 / 600), aspect=640 / 480, scale=0.3,
                                    wxyz=vtf.SO3.from_matrix(R_camera_world.T).wxyz, position=C)  # fmt: skip
    server.scene.add_line_segments(
        "/path", np.stack([path_on_plane[:-1], path_on_plane[1:]], 1), colors=(150, 150, 150)
    )
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    use_pixels = server.gui.add_checkbox("Use pixels", True)
    error = server.gui.add_number("Pixel error (px)", 0.0, disabled=True)
    restart = [True]
    server.gui.add_button("Replay").on_click(lambda _: restart.__setitem__(0, True))

    while True:
        while not restart[0]:
            time.sleep(0.05)
        restart[0], state = False, None
        for uv, point in zip(path, path_on_plane):
            # --- IK: refill the target pixel in place, then track from the previous answer ---
            pixels.set_weight(0.05 if use_pixels.value else 0.0)
            pixels.set_target(uv[None, None])
            state = ik.solve_numpy(T_world_target, init_state=state, prev_state=state)
            robot.forward_kinematics(state)
            p_hand = state.T_world_link.numpy()[0, hand, :3]
            error.value = float(np.abs(project_points(transform_points(p_hand[None], T_camera_world), K) - uv).mean())

            # --- viewer: the target pixel, the ray through it, and the hand origin that should sit on it ---
            ray_end = C + 2.5 * (point - C) / np.linalg.norm(point - C)
            server.scene.add_icosphere("/target", radius=0.008, color=(220, 40, 40), position=point)
            server.scene.add_line_segments("/ray", np.array([[C, ray_end]]), colors=(220, 40, 40))
            server.scene.add_icosphere("/hand", radius=0.02, color=(40, 160, 40), position=p_hand)
            urdf_vis.update_cfg(state.q.numpy()[0])
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()

# pyright: reportArgumentType=false
"""Online L-BFGS motion planning with fresh trajectory optimization."""

import time

import numpy as np
import trimesh
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

    # --- scene ---
    T_world_box = np.eye(4, dtype=np.float32)[None]
    T_world_box[0, :3, 3] = (0.45, 0.3, 0.45)
    box = BoxGeom(
        np.full((1, 3), 0.06, np.float32),
        np.array([0, 1], np.int32),
        poses=wp.from_numpy(T_world_box, dtype=wp.mat44, device=device),
    )
    scene = WarpScene(1, device).add(box)

    # --- planner ---
    robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH)
    cur_q = robot.spec.midrange_q.astype(np.float32)
    planner = MotionPlanner(
        presets.online_pose_lbfgs,
        robot,
        "panda_hand",
        scene=scene,
        device=device,
    )

    # --- viewer ---
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(cur_q)
    target = server.scene.add_transform_controls("/target", scale=0.2, position=(0.45, 0.0, 0.35), wxyz=(0, 1, 0, 0))
    obstacle = server.scene.add_transform_controls("/obstacle", scale=0.2, position=(0.45, 0.3, 0.45))
    server.scene.add_mesh_trimesh("/obstacle/mesh", trimesh.creation.box((0.12,) * 3))
    timing = server.gui.add_number("Tick (ms)", 0.001, disabled=True)

    online_state = None

    while True:
        t0 = time.perf_counter()

        # --- update scene and target ---
        box_center = np.asarray(obstacle.position, np.float32)
        T_world_box[0, :3, 3] = box_center
        box.update(poses=wp.from_numpy(T_world_box, dtype=wp.mat44, device=device))
        T_world_target = np.asarray((*target.position, *target.wxyz), dtype=np.float32)

        # --- solve ---
        result = planner.solve_online_numpy(cur_q, T_world_target=T_world_target, online_state=online_state)
        online_state = result.online_state
        cur_q = result.q_traj[0, 1].copy()

        # --- viewer ---
        with server.atomic():
            timing.value = 0.99 * timing.value + 0.01 * (time.perf_counter() - t0) * 1e3
            urdf_vis.update_cfg(cur_q)
        time.sleep(max(0.0, planner.config.dt - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()

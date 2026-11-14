"""Offline joint-space goal planning around a box."""

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
    robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH)

    # --- scene ---
    T_world_box = np.eye(4, dtype=np.float32)[None]
    T_world_box[0, :3, 3] = (0.45, 0.0, 0.2)
    scene = WarpScene(1, device).add(
        BoxGeom(
            np.array([[0.04, 0.25, 0.2]], np.float32),
            np.array([0, 1], np.int32),
            poses=wp.from_numpy(T_world_box, dtype=wp.mat44, device=device),
        )
    )

    # --- planner ---
    planner = MotionPlanner(
        presets.offline_joint_lbfgs,
        robot,
        "panda_hand",
        scene=scene,
        device=device,
    )

    # --- IK both ends: start and goal sit on opposite sides of the box, so the arm has to go around ---
    start_q, target_q = (
        planner.ik.solve_numpy(np.array([0.45, y, 0.25, 0.0, 1.0, 0.0, 0.0], np.float32)).q.numpy()[0]
        for y in (-0.35, 0.35)
    )

    # --- solve ---
    q_traj = planner.solve_offline_numpy([start_q, target_q]).q_traj[0]

    # --- viewer ---
    server = viser.ViserServer()
    box_mesh = trimesh.creation.box(extents=(0.08, 0.5, 0.4)).apply_translation((0.45, 0.0, 0.2))
    server.scene.add_mesh_trimesh("/box", box_mesh)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    while True:
        for q in q_traj:
            with server.atomic():
                urdf_vis.update_cfg(q)
            time.sleep(0.05)
        time.sleep(0.3)


if __name__ == "__main__":
    main()

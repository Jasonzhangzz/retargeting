"""Offline joint-space planning, interpolated to a control rate at a chosen speed and played in real time."""

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

    # --- solve: the preset retimes the plan to the fastest speed its limits allow ---
    planner = MotionPlanner(presets.offline_joint_lbfgs, robot, "panda_hand", scene=scene, device=device)
    start_q, target_q = (
        planner.ik.solve_numpy(np.array([0.45, y, 0.25, 0.0, 1.0, 0.0, 0.0], np.float32)).q.numpy()[0]
        for y in (-0.35, 0.35)
    )
    result = planner.solve_offline_numpy([start_q, target_q])
    assert result.motion_time is not None

    # --- viewer: sliders pick the speed and control step; dots mark the hand at every control sample ---
    server = viser.ViserServer()
    box_mesh = trimesh.creation.box(extents=(0.08, 0.5, 0.4)).apply_translation((0.45, 0.0, 0.2))
    server.scene.add_mesh_trimesh("/box", box_mesh)
    visual = ViserUrdf(server, urdf, root_node_name="/robot")
    speed_scale = server.gui.add_slider("Speed scale", min=0.1, max=1.0, step=0.05, initial_value=1.0)
    control_dt = server.gui.add_slider("Control dt (s)", min=0.005, max=0.1, step=0.005, initial_value=0.02)
    info = server.gui.add_markdown("")
    hand = robot.link_names.index("panda_hand")
    while True:
        dt, speed = control_dt.value, speed_scale.value
        q_path = result.interpolate(dt, speed)[0]
        T_world_hand = robot.forward_kinematics_numpy(q_path, device=device)[:, hand]
        server.scene.add_point_cloud(
            "/hand", T_world_hand[:, :3], (1.0, 0.5, 0.0), point_size=0.01, point_shape="circle"
        )
        info.content = f"motion_time={result.motion_time[0] / speed:.3f} s, samples={len(q_path)}"
        for q in q_path:
            if (control_dt.value, speed_scale.value) != (dt, speed):
                break  # a slider moved: restart at once with the new samples
            visual.update_cfg(q)
            time.sleep(dt)
        else:
            time.sleep(0.5)


if __name__ == "__main__":
    main()

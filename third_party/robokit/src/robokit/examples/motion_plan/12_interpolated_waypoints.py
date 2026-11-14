"""Waypoint planning side by side: even frames crowd a short segment, resplit frames move some to the long one."""

import dataclasses
import time

import numpy as np
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import franka_panda
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot


CONTROL_DT = 0.02


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")
    robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH)
    config = presets.offline_joint_lbfgs
    planners = {
        "before": MotionPlanner(
            dataclasses.replace(config, resplit_waypoints=False), robot, "panda_hand", device=device
        ),
        "after": MotionPlanner(config, robot, "panda_hand", device=device),
    }

    # --- waypoints: A-B about 0.2 m, B-C about 0.9 m; IK rests at midrange, whose open fingers keep apart ---
    T_world_waypoints = np.array(
        [[0.5, -0.4, 0.4, 0, 1, 0, 0], [0.5, -0.2, 0.4, 0, 1, 0, 0], [0.3, 0.5, 0.75, 0, 1, 0, 0]], np.float32
    )
    q_rest = robot.spec.midrange_q.astype(np.float32)[None]
    rest = robot.state(q=wp.array(q_rest, dtype=wp.float32, device=device))
    q_waypoints = [
        planners["after"].ik.solve_numpy(T, init_state=rest, rest_state=rest).q.numpy()[0] for T in T_world_waypoints
    ]

    # --- solve both, then interpolate each to the control rate through every waypoint ---
    q_frames, q_paths, info = {}, {}, []
    for name, planner in planners.items():
        result = planner.solve_offline_numpy(q_waypoints)
        assert result.motion_time is not None and result.waypoint_indices is not None
        q_frames[name], q_paths[name] = result.q_traj[0], result.interpolate(CONTROL_DT)[0]
        b, n = result.waypoint_indices[1], result.waypoint_indices[-1]
        motion_time = float(result.motion_time[0])
        info.append(f"{name}: B at frame {b} of {n}, A-B {b * motion_time / n:.2f} s, total {motion_time:.2f} s")
    print("\n".join(info))

    # --- viewer: both robots play at once, side by side; dots mark the hand at each planned frame before interpolation ---
    server = viser.ViserServer()
    server.gui.add_markdown("\n\n".join(info))
    hand = robot.link_names.index("panda_hand")
    visuals = {}
    for (name, q_path), y, color in zip(q_paths.items(), (-0.6, 0.6), ((0.9, 0.2, 0.2), (0.2, 0.6, 0.9))):
        server.scene.add_frame(f"/{name}", position=(0.0, y, 0.0), show_axes=False)
        server.scene.add_label(f"/{name}/label", name, position=(0.0, 0.0, 1.1))
        for i, T_world_waypoint in enumerate(T_world_waypoints):
            server.scene.add_frame(
                f"/{name}/waypoint_{i}",
                position=T_world_waypoint[:3],
                wxyz=T_world_waypoint[3:],
                axes_length=0.08,
                axes_radius=0.004,
            )
        T_world_hand = robot.forward_kinematics_numpy(q_frames[name], device=device)[:, hand]
        server.scene.add_point_cloud(f"/{name}/hand", T_world_hand[:, :3], color, point_size=0.02, point_shape="circle")
        visuals[name] = ViserUrdf(server, urdf, root_node_name=f"/{name}/robot")
    while True:
        for k in range(max(len(q) for q in q_paths.values())):
            for name, q_path in q_paths.items():
                visuals[name].update_cfg(q_path[min(k, len(q_path) - 1)])  # the faster one holds its goal
            time.sleep(CONTROL_DT)
        time.sleep(0.5)


if __name__ == "__main__":
    main()

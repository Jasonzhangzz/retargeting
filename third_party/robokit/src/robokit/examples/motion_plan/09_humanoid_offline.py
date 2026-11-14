"""Offline G1 L-BFGS planning: hand goals while both feet stay planted at their home poses."""

import time
from copy import deepcopy

import numpy as np
import trimesh
import tyro
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.assets.robots.humanoids import unitree_g1
from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot
from robokit.xform.numpy import quaternion_to_matrix


TARGET_LINKS = ("right_rubber_hand_link", "left_rubber_hand_link")
FOOT_LINKS = ("right_ankle_roll_link", "left_ankle_roll_link")
BASE_HEIGHT = 0.7996  # with zero joints the G1 stands on the floor at z=0 with its base this high


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    # --- robot ---
    urdf = yourdfpy.URDF.load(str(unitree_g1.URDF_PATH))
    robot = Robot.load(
        urdf,
        load_collision_spheres=True,
        collision_spheres_path=unitree_g1.COLLISION_SPHERE_PATH,
        self_collision_ignore_path=unitree_g1.SPHERE_SELF_COLLISION_IGNORE_PATH,
    )
    q = np.zeros((1, robot.num_actuated_joints), np.float32)
    base = np.array([[0, 0, BASE_HEIGHT, 1, 0, 0, 0]], np.float32)
    home = robot.forward_kinematics_numpy(q, base, device=device)[0]
    hand_indices = [robot.link_names.index(n) for n in TARGET_LINKS]
    foot_indices = [robot.link_names.index(n) for n in FOOT_LINKS]
    T_world_hands, T_world_feet = home[hand_indices], home[foot_indices]
    com = robot.compute_center_of_mass_numpy(q, base, device=device)[0]
    on_feet = np.isin(robot.spec.collision_spheres_link_indices, foot_indices)

    # --- scene ---
    box_centers = np.array([[0, 0, -0.025], [0.4, 0, 0.3]], np.float32)
    box_half_sizes = np.array([[0.7, 0.7, 0.025], [0.05, 0.05, 0.05]], np.float32)
    T_world_box = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    T_world_box[:, :3, 3] = box_centers
    scene = WarpScene(1, device).add(
        BoxGeom(box_half_sizes, np.array([0, 2], np.int32), poses=wp.array(T_world_box, dtype=wp.mat44, device=device))
    )

    # --- planner: the preset holds every humanoid task and weight; goal IK and every frame hold the feet and CoM ---
    config = deepcopy(presets.humanoid_offline_lbfgs)
    for stage in (config.goal_ik, *config.stages):
        stage.task("feet_position").set_frames(robot, foot_indices, T_world_feet)
        stage.task("feet_rotation").set_frames(robot, foot_indices, T_world_feet)
        stage.task("balance").set_target(com)
    collision = config.stages[0].task("collision")
    collision.margin = np.where(on_feet, 0.0, collision.margin)  # standing on the floor costs nothing
    planner = MotionPlanner(config, robot, TARGET_LINKS, scene=scene, device=device)
    planner.warmup(1)

    # --- viewer ---
    server = viser.ViserServer(port=8899)
    root = server.scene.add_frame("/robot", show_axes=False, position=(0, 0, BASE_HEIGHT))
    visual = ViserUrdf(server, urdf, root_node_name="/robot")
    visual.update_cfg(q[0])
    controls = [
        server.scene.add_transform_controls(f"/target/{name}", scale=0.12, position=t[:3], wxyz=t[3:])
        for name, t in zip(TARGET_LINKS, T_world_hands)
    ]
    obstacles = [
        server.scene.add_transform_controls(f"/scene/{name}", scale=0.12, position=center)
        for name, center in zip(("ground", "cube"), box_centers)
    ]
    for name, half_size in zip(("ground", "cube"), box_half_sizes):
        server.scene.add_mesh_trimesh(f"/scene/{name}/mesh", trimesh.creation.box(half_size * 2))
    status = server.gui.add_markdown("Ready")
    replan = server.gui.add_button("Replan")
    # viser calls back on worker threads: only queue requests, so GPU work stays on this thread
    pending, start_q, start_base = [True], q, base
    replan.on_click(lambda _: pending.append(True))
    while True:
        time.sleep(0.05)
        if not pending:
            continue
        pending.clear()
        before = time.perf_counter()

        # --- update scene and targets ---
        for i, control in enumerate(obstacles):
            T_world_box[i, :3, 3] = control.position
            T_world_box[i, :3, :3] = quaternion_to_matrix(np.asarray(control.wxyz))
        scene.geoms[0].update(poses=wp.array(T_world_box, dtype=wp.mat44, device=device))
        T_world_target = np.array([np.r_[c.position, c.wxyz] for c in controls], np.float32)
        # --- solve: like 02, goal IK inside the planner seeds a start-to-goal path ---
        result = planner.solve_offline_numpy(
            [start_q, dict(zip(TARGET_LINKS, T_world_target))], start_T_world_base=start_base
        )
        assert result.T_world_base_traj is not None and result.motion_time is not None  # offline floating base
        q_path, base_path = result.q_traj[0], result.T_world_base_traj[0]

        # --- viewer: how far the hands end from their goals, and the largest foot displacement from home ---
        path = robot.forward_kinematics_numpy(q_path, base_path, device=device)
        endpoint = np.linalg.norm(path[-1, hand_indices, :3] - T_world_target[:, :3], axis=-1).max()
        drift = np.linalg.norm(path[:, foot_indices, :3] - T_world_feet[:, :3], axis=-1).max()
        status.content = (
            f"{(time.perf_counter() - before) * 1000:.1f} ms | endpoint={endpoint * 1000:.1f} mm | "
            f"foot drift={drift * 1000:.1f} mm"
        )
        # like 02, play the plan once, then start the next one where it ended
        for q_frame, base_frame in zip(q_path, base_path):
            with server.atomic():
                visual.update_cfg(q_frame)
                root.position, root.wxyz = base_frame[:3], base_frame[3:]
            time.sleep(float(result.motion_time[0]) / (len(q_path) - 1))
        start_q, start_base = q_path[-1:], base_path[-1:]


if __name__ == "__main__":
    tyro.cli(main)

"""Track G1 hand goals while keeping feet planted; MPPI samples accelerations around the planted stance."""

import time
from copy import deepcopy
from typing import Sequence

import numpy as np
import trimesh
import viser
import warp as wp
import yourdfpy
from viser.extras import ViserUrdf

from robokit.assets.robots.humanoids import unitree_g1
from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.helpers.motion_plan.motion_planner import MotionPlannerConfig
from robokit.robo import Robot
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.xform.numpy import quaternion_to_matrix


TARGET_LINKS = ("right_rubber_hand_link", "left_rubber_hand_link")
FOOT_LINKS = ("right_ankle_roll_link", "left_ankle_roll_link")
BASE_HEIGHT = 0.7996  # with zero joints the G1 stands on the floor at z=0 with its base this high


def stand(
    config: MotionPlannerConfig, robot: Robot, feet: Sequence[str], q: np.ndarray, T_world_base: np.ndarray
) -> MotionPlannerConfig:
    """Build a humanoid preset for standing q and T_world_base; each example keeps its own copy.

    Standing constraints:
    - Feet: held during goal IK and at every trajectory frame; foot spheres incur no floor-collision cost.
    - CoM/posture: held wherever the preset defines those terms.
    - MPPI sampling: exclude the base and joints supporting the feet, preserving the planted stance.
    """
    config, spec = deepcopy(config), robot.spec
    q, T_world_base = np.asarray(q, np.float32)[None], np.asarray(T_world_base, np.float32)[None]
    foot_indices = [robot.link_names.index(name) for name in feet]
    T_world_feet = robot.forward_kinematics_numpy(q, T_world_base)[0, foot_indices]
    com = robot.compute_center_of_mass_numpy(q, T_world_base)[0]
    on_feet = np.isin(spec.collision_spheres_link_indices, foot_indices)
    for stage in (config.goal_ik, *config.stages):
        for term in stage.terms:
            if term.name in ("feet_position", "feet_rotation"):
                term.set_frames(robot, foot_indices, T_world_feet)
            elif term.name == "balance":
                term.set_target(com)
            elif term.name == "posture":
                term.set_target(rest_q=q[0], T_world_base_rest=T_world_base)
            elif isinstance(term, TrajectoryCollisionTask):
                term.weight = np.where(on_feet, 0.0, term.weight).astype(np.float32)
    if config.stages[0].method == "mppi":
        joints = spec.link_ancestor_joints_mask[foot_indices].any(axis=0) & (spec.actuated_joint_indices >= 0)
        carried = np.isin(np.arange(robot.num_actuated_joints), spec.actuated_joint_indices[joints])
        config.stages[0].std_scale = (~carried).astype(np.float32)
    return config


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
    home_q = robot.spec.zero_q.astype(np.float32)[None]
    home_base = np.array([[0, 0, BASE_HEIGHT, 1, 0, 0, 0]], np.float32)
    home = robot.forward_kinematics_numpy(home_q, home_base, device=device)[0]
    hand_indices = [robot.link_names.index(n) for n in TARGET_LINKS]
    foot_indices = [robot.link_names.index(n) for n in FOOT_LINKS]
    T_world_hands, T_world_feet = home[hand_indices], home[foot_indices]

    # --- scene ---
    box_centers = np.array([[0, 0, -0.025], [0.4, 0, 0.3]], np.float32)
    box_half_sizes = np.array([[0.7, 0.7, 0.025], [0.05, 0.05, 0.05]], np.float32)
    T_world_box = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    T_world_box[:, :3, 3] = box_centers
    scene = WarpScene(1, device).add(
        BoxGeom(box_half_sizes, np.array([0, 2], np.int32), poses=wp.array(T_world_box, dtype=wp.mat44, device=device))
    )

    # --- planner: the preset holds every humanoid task and weight; `stand` plants it at home ---
    config = stand(presets.humanoid_online_mppi, robot, FOOT_LINKS, home_q[0], home_base[0])
    planner = MotionPlanner(config, robot, TARGET_LINKS, scene=scene, device=device)
    planner.warmup(1)

    # --- viewer ---
    server = viser.ViserServer(port=8899)
    root = server.scene.add_frame("/robot", show_axes=False, position=home_base[0, :3])
    visual = ViserUrdf(server, urdf, root_node_name="/robot")
    visual.update_cfg(home_q[0])
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
    status = server.gui.add_markdown("")
    tick, online_state, q, base = 0.0, None, home_q.copy(), home_base.copy()
    while True:
        before = time.perf_counter()
        # --- update scene and targets ---
        for i, control in enumerate(obstacles):
            T_world_box[i, :3, 3] = control.position
            T_world_box[i, :3, :3] = quaternion_to_matrix(np.asarray(control.wxyz))
        scene.geoms[0].update(poses=wp.array(T_world_box, dtype=wp.mat44, device=device))
        T_world_target = np.array([np.r_[c.position, c.wxyz] for c in controls], np.float32)
        # --- solve, then execute the next waypoint ---
        result = planner.solve_online_numpy(
            q, start_T_world_base=base, T_world_target=T_world_target, online_state=online_state
        )
        online_state = result.online_state
        q = result.q_traj[:, 1].copy()
        base = result.T_world_base_traj[:, 1].copy()  # pyright: ignore[reportOptionalSubscript]
        tick = 0.99 * tick + 0.01 * (time.perf_counter() - before) * 1000

        # --- viewer: the executed configuration, its target error, and foot displacement from home ---
        T_world_link = robot.forward_kinematics_numpy(q, base, device=device)[0]
        reached = T_world_link[hand_indices]
        error = np.linalg.norm(reached[:, :3] - T_world_target[:, :3], axis=-1).max()
        angle = 2 * np.arccos(np.clip(np.abs((reached[:, 3:] * T_world_target[:, 3:]).sum(-1)), 0, 1)).max()
        drift = np.linalg.norm(T_world_link[foot_indices, :3] - T_world_feet[:, :3], axis=-1).max()
        with server.atomic():
            status.content = (
                f"{tick:.1f} ms | target={error * 1000:.1f} mm, {angle:.3f} rad | foot drift={drift * 1000:.1f} mm"
            )
            visual.update_cfg(q[0])
            root.position, root.wxyz = base[0, :3], base[0, 3:]
        time.sleep(max(0.0, planner.config.dt - (time.perf_counter() - before)))


if __name__ == "__main__":
    main()

"""Batch motion planning around one shared box."""

import time

import numpy as np
import trimesh
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf


GRID_SIZE = 3
BATCH_SIZE = GRID_SIZE**2


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

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
    robot = Robot.load(
        urdf, load_meshes=True, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH
    )
    planner = MotionPlanner(
        presets.offline_pose_lm,
        robot,
        "panda_hand",
        scene=scene,
        device=device,
    )

    # --- setup problems ---
    xy = np.stack(np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE)), -1).reshape(-1, 2).astype(np.float32)
    goal_wxyz = np.tile(np.array([0.0, 1.0, 0.0, 0.0], np.float32), (BATCH_SIZE, 1))
    start_target = np.concatenate([np.array([0.28, 0.0, 0.2], np.float32), goal_wxyz[0]])
    start_q = planner.ik.solve_numpy(np.tile(start_target, (BATCH_SIZE, 1))).q.numpy()[:: planner.ik.num_solutions]
    goal_position = np.column_stack([0.46 + 0.06 * xy[:, 1], -0.2 + 0.2 * xy[:, 0], 0.5 + 0.05 * xy[:, 1]]).astype(
        np.float32
    )
    T_world_target = np.concatenate([goal_position, goal_wxyz], axis=1)

    # --- solve ---
    result = planner.solve_offline_numpy(
        [start_q, {"panda_hand": T_world_target}], scene_indices=np.zeros(BATCH_SIZE, np.int32)
    )

    metrics = planner.compute_metrics_numpy(result.q_traj, T_world_target=T_world_target)

    # --- visualize ---
    T_world_base = np.zeros((BATCH_SIZE, 7), np.float32)
    T_world_base[:, :2], T_world_base[:, 3] = (xy - (GRID_SIZE - 1) / 2) * 1.6, 1.0
    server = viser.ViserServer()
    server.gui.add_text("Success rate", f"{100.0 * metrics.success.mean():.0f}%", disabled=True)
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=BATCH_SIZE, root_node_name="/robots")
    box_mesh = trimesh.creation.box(extents=(0.08, 0.5, 0.4)).apply_translation((0.45, 0.0, 0.2))
    for i, t in enumerate(T_world_base[:, :3]):
        server.scene.add_mesh_trimesh(f"/boxes/{i}", box_mesh.copy().apply_translation(t))
    server.scene.add_batched_axes(
        "/targets",
        batched_wxyzs=goal_wxyz,
        batched_positions=goal_position + T_world_base[:, :3],
        axes_length=0.1,
        axes_radius=0.005,
    )
    while True:
        for f in range(result.q_traj.shape[1]):
            with server.atomic():
                batch_urdf.update_cfg(result.q_traj[:, f], T_world_base=T_world_base)
            time.sleep(0.05)
        time.sleep(0.3)


if __name__ == "__main__":
    main()

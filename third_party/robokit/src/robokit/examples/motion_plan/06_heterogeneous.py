"""Batch pose-goal planning across heterogeneous collision scenes."""

import time

import numpy as np
import trimesh
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, MeshGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf


SCENE_IDS = np.array([0, 0, 0, 1, 2, 2, 2, 2], np.int32)  # per-query scene assignment (batch 8, unequal 3/1/4)
BATCH_SIZE = len(SCENE_IDS)

# Each scene combines primitive boxes and sphere meshes into one collision set.
SCENE_BOXES = [
    [((0.04, 0.05, 0.3), (0.45, 0.16, 0.3))],
    [((0.06, 0.06, 0.28), (0.45, 0.0, 0.28))],
    [((0.05, 0.05, 0.25), (0.45, 0.0, 0.28))],
]
SCENE_SPHERES = [
    [(0.1, (0.45, -0.16, 0.35))],
    [],
    [(0.09, (0.4, 0.24, 0.42))],
]

# Collision-free config per query in its own scene (sampled offline); its FK pose is the plan's pose goal.
GOAL_Q = np.array(
    [
        [0.4762, -0.4870, -1.5959, -1.8133, 1.0892, 2.0542, 0.3707, 0.0175],
        [0.1517, 0.9203, 1.0981, -1.8381, 1.2426, 0.0655, 0.7985, 0.0042],
        [1.2627, 0.0877, -0.6964, -1.0817, -1.6399, 0.2706, 0.5932, 0.0155],
        [0.4012, -0.2461, 1.7287, -0.0764, 0.6451, 1.4608, 0.6552, 0.0093],
        [-1.2687, 0.4685, 0.0882, -1.2843, -0.0492, 2.0015, 1.5091, 0.0086],
        [0.2487, -0.3768, 0.3279, -1.2344, -0.3768, 2.0033, -0.9486, 0.0150],
        [-1.4463, 0.7037, 0.9982, -1.4119, 1.3089, 0.1220, -0.5698, 0.0036],
        [-0.1727, 0.6268, -0.9365, -1.7494, -0.3319, 0.4385, -1.4229, 0.0139],
    ],
    dtype=np.float32,
)


def build_scene(device: str) -> WarpScene:
    """WarpScene with CSR-partitioned box and sphere-mesh geometry."""
    hes, box_centers, box_first, meshes, mesh_first = [], [], [], [], []
    for boxes, spheres in zip(SCENE_BOXES, SCENE_SPHERES):
        box_first.append(len(hes))
        mesh_first.append(len(meshes))
        for he, c in boxes:
            hes.append(he)
            box_centers.append(c)
        meshes.extend(trimesh.creation.icosphere(radius=r).apply_translation(c) for r, c in spheres)
    T_world_box = np.tile(np.eye(4, dtype=np.float32), (len(hes), 1, 1))
    T_world_box[:, :3, 3] = np.asarray(box_centers, np.float32)
    scene = WarpScene(len(SCENE_BOXES), device).add(
        BoxGeom(
            np.asarray(hes, np.float32),
            np.asarray(box_first + [len(hes)], np.int32),
            poses=wp.from_numpy(T_world_box, dtype=wp.mat44, device=device),
        )
    )
    return scene.add(MeshGeom(meshes, np.asarray(mesh_first + [len(meshes)], np.int32)))


def main():
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")
    robot = Robot.load(
        urdf, load_meshes=True, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH
    )

    # --- build problems: get target poses ---
    ee = robot.link_names.index("panda_hand")
    goal = robot.state(q=wp.from_numpy(GOAL_Q, dtype=wp.float32, device=device))
    robot.forward_kinematics(goal)
    T_world_ee = goal.T_world_link.numpy().reshape(BATCH_SIZE, robot.spec.num_links, 7)[:, ee]
    t_world_goal, q_wxyz_world_goal = T_world_ee[:, :3].copy(), T_world_ee[:, 3:].copy()
    start_q = np.tile(robot.spec.midrange_q.astype(np.float32), (BATCH_SIZE, 1))

    # --- scene ---
    scene = build_scene(device)

    # --- solve ---
    planner = MotionPlanner(
        presets.offline_pose_lm,
        robot,
        "panda_hand",
        scene=scene,
        device=device,
    )
    result = planner.solve_offline_numpy([start_q, {"panda_hand": T_world_ee}], scene_indices=SCENE_IDS)
    q_traj = result.q_traj  # (BATCH_SIZE, num_frames, dof)

    # --- viewer ---
    xy = np.stack(np.meshgrid(np.arange(4), np.arange(2)), -1).reshape(-1, 2).astype(np.float32)
    T_world_base = np.zeros((BATCH_SIZE, 7), np.float32)
    T_world_base[:, :2], T_world_base[:, 3] = (xy - np.array([1.5, 0.5])) * 1.4, 1.0
    server = viser.ViserServer()
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=BATCH_SIZE, root_node_name="/robots")
    server.scene.add_batched_axes(
        "/targets",
        batched_wxyzs=q_wxyz_world_goal,
        batched_positions=t_world_goal + T_world_base[:, :3],
        axes_length=0.1,
        axes_radius=0.005,
    )
    for i, sid in enumerate(SCENE_IDS):
        base = T_world_base[i, :3]
        for j, (he, c) in enumerate(SCENE_BOXES[sid]):
            box = trimesh.creation.box(extents=np.array(he) * 2).apply_translation(np.asarray(c) + base)
            server.scene.add_mesh_trimesh(f"/obstacles/{i}/box{j}", box)
        for j, (r, c) in enumerate(SCENE_SPHERES[sid]):
            ball = trimesh.creation.icosphere(radius=r).apply_translation(np.asarray(c) + base)
            server.scene.add_mesh_trimesh(f"/obstacles/{i}/ball{j}", ball)

    while True:
        for f in range(q_traj.shape[1]):
            with server.atomic():
                batch_urdf.update_cfg(q_traj[:, f], T_world_base=T_world_base)
            time.sleep(0.05)
        time.sleep(0.3)


if __name__ == "__main__":
    main()

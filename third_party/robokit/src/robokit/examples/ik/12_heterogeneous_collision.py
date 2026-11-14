"""Heterogeneous batched collision IK: each query selects one scene with `scene_indices`."""

import time

import numpy as np
import trimesh
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, MeshGeom, WarpScene
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.utils.visualize_utils import ViserBatchUrdf


EE_LINK = "panda_hand"
SCENE_IDS = [0, 0, 0, 1, 2, 2, 2, 2]  # per-query scene assignment (batch 8)

# Put boxes in primitives and spheres in meshes; one collision task avoids their union.
SCENE_BOXES = [
    [((0.05, 0.05, 0.3), (0.45, 0.16, 0.3)), ((0.05, 0.05, 0.3), (0.45, -0.16, 0.3))],
    [((0.08, 0.35, 0.04), (0.45, 0.0, 0.25))],
    [((0.05, 0.05, 0.25), (0.45, 0.0, 0.28))],
]
SCENE_SPHERES = [
    [],
    [(0.11, (0.45, 0.0, 0.5))],
    [(0.1, (0.4, 0.22, 0.42)), (0.1, (0.4, -0.22, 0.42))],
]

# Collision-free config per query in its own scene (sampled offline); its FK pose is the IK target.
GOAL_Q = np.array(
    [
        [0.4762, -0.4870, -1.5959, -1.8133, 1.0892, 2.0542, 0.3707, 0.0175],
        [0.1517, 0.9203, 1.0981, -1.8381, 1.2426, 0.0655, 0.7985, 0.0042],
        [1.2627, 0.0877, -0.6964, -1.0817, -1.6399, 0.2706, 0.5932, 0.0155],
        [0.4012, -0.2461, 1.7287, -0.0764, 0.6451, 1.4608, 0.6552, 0.0093],
        [-1.2687, 0.4685, 0.0882, -1.2843, -0.0492, 2.0015, 1.5091, 0.0086],
        [0.2487, -0.3768, 0.3279, -1.2344, -0.3768, 2.0033, -0.9486, 0.0150],
        [-1.4463, 0.7037, 0.9982, -1.4119, 1.3089, 0.1220, -0.5698, 0.0036],
        [-0.6999, 0.3638, -1.0447, -0.1461, -0.4690, 0.2281, 0.4489, 0.0223],
    ],
    dtype=np.float32,
)


def build_multi_scene(device: str) -> WarpScene:
    """WarpScene(3): boxes as a BoxGeom, spheres as icosphere meshes in a separate geom, both CSR-partitioned."""
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
    scene = WarpScene(3, device).add(
        BoxGeom(
            np.asarray(hes, np.float32),
            np.asarray(box_first + [len(hes)], np.int32),  # CSR offsets: append the total box count
            poses=wp.from_numpy(T_world_box, dtype=wp.mat44, device=device),
        )
    )
    return scene.add(MeshGeom(meshes, np.asarray(mesh_first + [len(meshes)], np.int32)))


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")
    robot = Robot.load(
        urdf, load_meshes=True, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH
    )
    batch_size = len(SCENE_IDS)
    scene = build_multi_scene(device)

    # target pose per query = FK of its baked collision-free goal config
    ee = robot.spec.link_names.index(EE_LINK)
    goal = robot.state(q=wp.from_numpy(GOAL_Q, dtype=wp.float32, device=device))
    robot.forward_kinematics(goal)
    T_world_ee = goal.T_world_link.numpy().reshape(batch_size, robot.spec.num_links, 7)[:, ee]
    target_pos, target_wxyz = T_world_ee[:, :3].copy(), T_world_ee[:, 3:].copy()

    # one batched IK solve, each query against its own scene
    config = (
        IKSolverConfig(solver=presets.smooth.solver, sampler=RobertsSamplerConfig(keep_first=True, scale=1.0))
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
        .add(
            SceneCollisionTask(
                robot,
                scene=scene,
                scene_indices=wp.array(SCENE_IDS, dtype=wp.int32, device=device),
                weight=100.0,
                margin=0.02,
            )
        )
    )
    ik = IKSolver(config, robot=robot, target_links=EE_LINK, device=device)
    q_sol = ik.solve_numpy(np.concatenate([target_pos, target_wxyz], axis=-1)).q.numpy()

    # viewer: grid of robot bases, each showing its own scene's box + sphere obstacles
    xy = np.stack(np.meshgrid(np.arange(4), np.arange(2)), axis=-1).reshape(-1, 2).astype(np.float32)
    T_world_base = np.zeros((batch_size, 7), np.float32)
    T_world_base[:, :2], T_world_base[:, 3] = (xy - np.array([1.5, 0.5])) * 1.4, 1.0
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=8, height=8)
    batch_urdf = ViserBatchUrdf(server, robot, batch_size=batch_size, root_node_name="/robots")
    server.scene.add_batched_axes(
        "/targets",
        batched_wxyzs=target_wxyz,
        batched_positions=target_pos + T_world_base[:, :3],
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
        batch_urdf.update_cfg(q_sol, T_world_base=T_world_base)
        time.sleep(0.05)


if __name__ == "__main__":
    main()

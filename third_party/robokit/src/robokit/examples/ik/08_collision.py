"""Collision-aware IK example: draggable obstacle + self-collision."""

import time

import numpy as np
import trimesh
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import franka_panda
from robokit.geom import SphereGeom, WarpScene
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.robo import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


FRANKA_SPHERES = franka_panda.COLLISION_SPHERE_PATH
OBSTACLE_RADIUS = 0.1


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")

    # --- IK ---
    robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=FRANKA_SPHERES)
    # world obstacle: one sphere primitive, posed in place each frame so the cached IK graph stays valid
    identity = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
    obstacle_geom = SphereGeom(
        np.array([OBSTACLE_RADIUS], dtype=np.float32), np.array([0, 1], dtype=np.int32), poses=identity
    )
    obstacle_scene = WarpScene(1, device).add(obstacle_geom)
    config = (
        IKSolverConfig(solver=presets.smooth.solver, sampler=RobertsSamplerConfig(keep_first=True, scale=1.0))
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
        .add(SceneCollisionTask(robot, scene=obstacle_scene, weight=100.0, margin=0.02))
        .add(SelfCollisionTask(robot, representation="sphere", weight=5.0, margin=0.01))
    )
    ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device)
    ik.warmup(batch_size=1)  # optional
    init_state = robot.forward_kinematics(
        robot.state(q=wp.from_numpy(robot.spec.midrange_q[None], dtype=wp.float32, device=device))
    )

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(robot.spec.midrange_q)
    target = server.scene.add_transform_controls("/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0))
    obstacle = server.scene.add_transform_controls("/obstacle", scale=0.2, position=(0.45, 0.0, 0.45))
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=OBSTACLE_RADIUS)
    server.scene.add_mesh_simple(
        "/obstacle/mesh",
        vertices=np.array(sphere_mesh.vertices, dtype=np.float32),
        faces=np.array(sphere_mesh.faces, dtype=np.uint32),
        color=(220, 80, 80),
        opacity=0.6,
    )
    timing = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    T_world_obstacle = np.eye(4, dtype=np.float32)
    prev_state = init_state
    while True:
        # --- IK: one solve per frame ---
        T_world_obstacle[:3, 3] = obstacle.position
        obstacle_geom.update(poses=wp.from_numpy(T_world_obstacle[None], dtype=wp.mat44, device=device))
        t0 = time.time()
        state = ik.solve_numpy(
            np.concatenate([target.position, target.wxyz]), prev_state=prev_state, init_state=prev_state
        )
        timing.value = 0.99 * timing.value + 0.01 * (time.time() - t0) * 1000

        # --- viewer: visualize the result ---
        urdf_vis.update_cfg(state.q.numpy()[0])
        prev_state = state
        time.sleep(0.001)


if __name__ == "__main__":
    main()

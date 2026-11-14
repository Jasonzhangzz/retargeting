"""Query SDF at two robots' collision spheres and color by signed distance."""

import time

import numpy as np
import trimesh
import viser
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, WarpScene
from robokit.robo import Robot
from robokit.utils.visualize_utils import ViserBatchUrdf
from robokit.utils.warp_utils import wp_vec7


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    urdf = load_robot_description("panda_description")
    robot = Robot.load(
        urdf, load_meshes=True, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH
    )

    # --- two robot instances side by side; collision-sphere centers computed once ---
    q = np.tile(robot.spec.midrange_q.astype(np.float32), (2, 1))
    base = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (2, 1))
    base[:, 0] = [-0.5, 0.5]
    state = robot.state(q=q, T_world_base=wp.from_numpy(base, dtype=wp_vec7, device=device))
    robot.transform_collision_spheres(state)
    centers = state.collision_sphere_centers_world  # (2, num_spheres), queried as-is
    radii = np.tile(robot.spec.collision_sphere_radii.astype(np.float32), 2)

    # --- scene: one draggable box obstacle shared by both robots ---
    obstacle_he = np.array([0.06, 0.06, 0.06], np.float32)
    identity = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
    obstacle_geom = BoxGeom(obstacle_he[None], np.array([0, 1], np.int32), poses=identity)
    scene = WarpScene(1, device).add(obstacle_geom)
    scene_indices = wp.zeros(centers.shape, dtype=wp.int32, device=device)  # one scene -> every point routes to 0

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=2)
    robots_vis = ViserBatchUrdf(server, robot, batch_size=2, root_node_name="/robots")
    robots_vis.update_cfg(q, T_world_base=base)
    obstacle = server.scene.add_transform_controls("/obstacle", scale=0.3, position=(0.0, 0.0, 0.35))
    obs_mesh = trimesh.creation.box(extents=obstacle_he * 2)
    obs_vis = server.scene.add_mesh_simple("/obstacle/box", obs_mesh.vertices, obs_mesh.faces, opacity=0.6)
    min_gui = server.gui.add_number("Min signed distance (m)", 0.0, disabled=True)

    while True:
        # --- query every sphere against the moving obstacle; recolor the cube red once it touches a robot ---
        T_world_obstacle = np.eye(4, dtype=np.float32)
        T_world_obstacle[:3, 3] = obstacle.position
        obstacle_geom.update(poses=wp.from_numpy(T_world_obstacle[None], dtype=wp.mat44, device=device))
        signed_dists, _, _ = scene.query_sdf(centers, scene_indices=scene_indices)
        min_signed_dist = float((signed_dists.numpy() - radii).min())
        with server.atomic():
            min_gui.value = round(min_signed_dist, 4)
            obs_vis.color = (220, 80, 80) if min_signed_dist < 0 else (80, 200, 120)
        time.sleep(0.01)


if __name__ == "__main__":
    main()

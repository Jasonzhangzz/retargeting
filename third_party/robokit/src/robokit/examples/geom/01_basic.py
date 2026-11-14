"""Basic geom example: build a scene (floor + mesh + primitive), query the SDF at a draggable sphere."""

import time

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.geom import BoxGeom, MeshGeom, PlaneGeom, WarpScene


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    # --- scene: floor plane + box + torus mesh ---
    box_he = np.array([0.08, 0.08, 0.15], np.float32)
    box_center = np.array([0.35, 0.25, 0.15], np.float32)
    T_world_box = np.eye(4, dtype=np.float32)
    T_world_box[:3, 3] = box_center
    T_world_floor = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
    T_world_box_wp = wp.from_numpy(T_world_box[None], dtype=wp.mat44, device=device)
    torus = trimesh.creation.torus(major_radius=0.13, minor_radius=0.045).apply_translation((0.1, -0.2, 0.35))
    scene = WarpScene(1, device)
    scene.add(PlaneGeom(np.array([0, 1], np.int32), poses=T_world_floor))
    scene.add(BoxGeom(box_he[None], np.array([0, 1], np.int32), poses=T_world_box_wp))
    scene.add(MeshGeom([torus], np.array([0, 1], np.int32)))
    scene_offsets = wp.from_numpy(np.array([0, 1], np.int32), dtype=wp.int32, device=device)
    probe_radius = 0.05

    # --- viewer ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    server.scene.add_mesh_trimesh("/scene/box", trimesh.creation.box(extents=box_he * 2).apply_translation(box_center))
    server.scene.add_mesh_trimesh("/scene/torus", torus)
    probe = server.scene.add_transform_controls("/probe", scale=0.1, position=(0.2, 0.0, 0.5))
    ball = trimesh.creation.icosphere(radius=probe_radius)
    ball_vis = server.scene.add_mesh_simple(
        "/probe/ball", ball.vertices, ball.faces, color=(120, 160, 230), opacity=0.6
    )
    signed_dist_gui = server.gui.add_number("Signed distance (m)", 0.0, disabled=True)
    marker = server.scene.add_icosphere("/closest", radius=0.012)
    link = server.scene.add_spline_catmull_rom("/link", np.zeros((2, 3), np.float32))

    while True:
        # --- query the probe center against the whole scene; signed distance = distance - probe radius ---
        points = wp.from_numpy(np.asarray(probe.position, np.float32)[None], dtype=wp.vec3, device=device)
        signed_dists, _, closest_points = scene.query_sdf(points, scene_offsets)
        signed_dist = float(signed_dists.numpy()[0]) - probe_radius
        closest = closest_points.numpy()[0]
        color = (220, 80, 80) if signed_dist < 0 else (80, 200, 120)
        with server.atomic():
            signed_dist_gui.value = round(signed_dist, 4)
            ball_vis.color = color
            marker.position, marker.color = tuple(closest), color
            link.positions, link.color = np.stack([probe.position, closest]), color
        time.sleep(0.01)


if __name__ == "__main__":
    main()

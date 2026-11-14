"""Heterogeneous geom example: three scenes with different geometry, all queried in one batched call."""

import time

import numpy as np
import trimesh
import viser
import warp as wp

from robokit.geom import BoxGeom, MeshGeom, WarpScene


# Per scene: boxes go in a "primitives" group, spheres as icosphere "meshes" -- both CSR-partitioned.
SCENE_BOXES = [
    [((0.05, 0.05, 0.2), (0.0, 0.0, 0.2))],  # scene 0: a single pillar
    [],  # scene 1: none
    [((0.18, 0.05, 0.05), (0.0, 0.0, 0.32))],  # scene 2: a bar
]
SCENE_SPHERES = [
    [],  # scene 0: none
    [(0.15, (0.0, 0.0, 0.25))],  # scene 1: one big ball
    [(0.07, (0.0, 0.16, 0.2)), (0.07, (0.0, -0.16, 0.2))],  # scene 2: two small balls
]


def build_multi_scene(device: str) -> WarpScene:
    """WarpScene(3): boxes as a BoxGeom, spheres as icosphere meshes, each CSR-partitioned per scene."""
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
    scene = build_multi_scene(device)

    # --- query grid: the same local slice, tiled once per scene; scene_offsets routes each block ---
    m = 60
    line = np.linspace(-0.3, 0.3, m, dtype=np.float32)
    gx, gy = np.meshgrid(line, line)
    slice_local = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.22, np.float32)]).astype(np.float32)
    per_scene = len(slice_local)
    points = wp.from_numpy(np.tile(slice_local, (3, 1)), dtype=wp.vec3, device=device)
    scene_offsets = wp.from_numpy(
        np.array([0, per_scene, 2 * per_scene, 3 * per_scene], np.int32), dtype=wp.int32, device=device
    )

    # --- viewer: three scenes side by side ---
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=5, height=2)
    bases = [np.array([1.2 * s, 0.0, 0.0], np.float32) for s in range(3)]
    for s, base in enumerate(bases):
        for j, (he, c) in enumerate(SCENE_BOXES[s]):
            box = trimesh.creation.box(extents=np.array(he) * 2).apply_translation(np.asarray(c) + base)
            server.scene.add_mesh_trimesh(f"/scene{s}/box{j}", box)
        for j, (r, c) in enumerate(SCENE_SPHERES[s]):
            ball = trimesh.creation.icosphere(radius=r).apply_translation(np.asarray(c) + base)
            server.scene.add_mesh_trimesh(f"/scene{s}/ball{j}", ball)
    timing = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # --- query: all three scenes' point blocks in one kernel launch ---
        t0 = time.time()
        signed_dists = scene.query_sdf(points, scene_offsets, distance_only=True)
        wp.synchronize()
        timing.value = 0.99 * timing.value + 0.01 * (time.time() - t0) * 1000

        # --- viewer: each scene's block colored by its own SDF (diverging, white at the surface) ---
        d = signed_dists.numpy()
        white, red, teal = np.array([240, 240, 240]), np.array([220, 70, 70]), np.array([70, 150, 210])
        for s, base in enumerate(bases):
            ds = d[s * per_scene : (s + 1) * per_scene]
            t = np.clip(ds / 0.15, -1.0, 1.0)[:, None]
            colors = np.where(t > 0, white * (1 - t) + teal * t, white * (1 + t) - red * t).astype(np.uint8)
            server.scene.add_point_cloud(
                f"/field/scene{s}", slice_local + base, colors=colors, point_size=0.008, point_shape="circle"
            )
        time.sleep(0.03)


if __name__ == "__main__":
    main()

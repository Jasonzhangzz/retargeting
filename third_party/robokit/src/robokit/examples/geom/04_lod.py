"""LOD speedup example: dense mesh with vs without the auto-baked SDF mid-tier."""

import time
from typing import Tuple

import numpy as np
import trimesh
import warp as wp

from robokit.geom import MeshGeom, WarpScene


def main():
    wp.init()
    assert wp.is_cuda_available(), "04_lod requires a CUDA GPU (SDF volumes are GPU-only)."
    device = "cuda"
    voxel_size = 0.005

    # --- scenes: the same dense mesh, with and without the SDF mid-tier ---
    torus = trimesh.creation.torus(major_radius=0.15, minor_radius=0.05, major_sections=512, minor_sections=256)
    mesh_scene = WarpScene(1, device).add(MeshGeom([torus], np.array([0, 1], np.int32)))
    # SDF mid-tier baked (and disk-cached) at add time; the exact mesh only answers near the surface.
    tiered_scene = WarpScene(1, device).add(
        MeshGeom([torus], np.array([0, 1], np.int32), enable_sdf=True, sdf_voxel_size=voxel_size)
    )

    # --- query points: random, inside the baked volume's padded AABB, mostly far from the surface ---
    n = 1_000_000
    lo, hi = torus.bounds
    points_np = np.random.default_rng(0).uniform(lo - 0.08, hi + 0.08, size=(n, 3)).astype(np.float32)
    points = wp.from_numpy(points_np, dtype=wp.vec3, device=device)
    scene_offsets = wp.from_numpy(np.array([0, n], np.int32), dtype=wp.int32, device=device)

    # --- benchmark: identical query on both scenes ---
    def timed_signed_dists(scene: WarpScene, repeats: int = 20) -> Tuple[np.ndarray, float]:
        d = scene.query_sdf(points, scene_offsets, distance_only=True)  # warmup / kernel compile
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            d = scene.query_sdf(points, scene_offsets, distance_only=True)
        wp.synchronize()
        return d.numpy(), (time.perf_counter() - t0) / repeats * 1e3

    ref, ms_mesh = timed_signed_dists(mesh_scene)
    tiered, ms_tiered = timed_signed_dists(tiered_scene)
    err_mm = np.abs(tiered - ref).max() * 1000
    # --- collision accuracy: near the surface the tier refines to the exact mesh ---
    near = np.abs(ref) < 2 * voxel_size
    near_err_mm = np.abs(tiered - ref)[near].max() * 1000
    sign_match = ((tiered < 0) == (ref < 0)).mean() * 100
    print(f"{len(torus.faces)} triangles, {n} query points")
    print(f"exact mesh: {ms_mesh:7.2f} ms")
    print(f"mesh + sdf: {ms_tiered:7.2f} ms  ({ms_mesh / ms_tiered:.1f}x faster, max err {err_mm:.2f} mm)")
    print(f"collision accuracy: {sign_match:.2f}% inside/outside agreement, near-surface max err {near_err_mm:.4f} mm")


if __name__ == "__main__":
    main()

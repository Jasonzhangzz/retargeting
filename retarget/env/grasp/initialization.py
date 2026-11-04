"""Original object mesh and initial-pose sampling utilities for GraspADMM."""

import numpy as np


def matching_mesh(surface):
    """Exact input triangles, with independent normal patches at sharp edges."""
    import trimesh

    vertices, inverse = np.unique(
        surface.triangles.reshape(-1, 3), axis=0, return_inverse=True
    )
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=inverse.reshape(-1, 3), process=False
    )
    return trimesh.graph.smooth_shade(mesh, angle=np.pi / 4, facet_minarea=None)


def sample_original_poses(obj_loader, mesh, points, inplane, seed):
    """Run the original sampler with deterministic Trimesh random streams.

    Recent Trimesh samplers use default_rng, independent of np.random.seed.
    Only supply their RNG seed; upstream sampling/pose math is unchanged.
    """
    import trimesh

    even, ordinary = trimesh.sample.sample_surface_even, trimesh.sample.sample_surface

    def seeded_even(*a, **kw):
        if kw.get("seed") is None:
            kw["seed"] = seed
        return even(*a, **kw)

    def seeded_ordinary(*a, **kw):
        if kw.get("seed") is None:
            kw["seed"] = seed + 1
        return ordinary(*a, **kw)

    try:
        trimesh.sample.sample_surface_even = seeded_even
        trimesh.sample.sample_surface = seeded_ordinary
        return obj_loader.sample_init_pose(mesh, points, inplane)
    finally:
        trimesh.sample.sample_surface_even = even
        trimesh.sample.sample_surface = ordinary

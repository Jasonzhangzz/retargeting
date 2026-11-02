"""Retained Dexonomy input pose sampling (CC BY-NC 4.0).

Only the initialization sampler is retained; datasets and task pipelines are
not imported. See source.json and licenses/Dexonomy-NOTICE.txt.
"""

import numpy as np
import trimesh

from retarget.initialization.dexonomy.util.np_util import (
    np_normal_to_rot,
    np_axis_angle_rotation,
)


def sample_init_pose(
    tm_obj: trimesh.Trimesh, n_init_point: int, n_init_inplane: int
) -> tuple[np.ndarray, np.ndarray]:
    # Sample contact points and corresponding normals
    points, tri_ind = trimesh.sample.sample_surface_even(tm_obj, n_init_point)
    n_remain = n_init_point - points.shape[0]
    if n_remain > 0:
        new_points, new_tri_ind = trimesh.sample.sample_surface(tm_obj, n_remain)
        points = np.concatenate([points, new_points], axis=0)
        tri_ind = np.concatenate([tri_ind, new_tri_ind], axis=0)
    normals = -tm_obj.face_normals[tri_ind]

    # Contact point and normal to relative pose
    rot_base = np_normal_to_rot(normals)[None]
    angles = np.linspace(-np.pi, np.pi, n_init_inplane)
    delta_rot = np_axis_angle_rotation("X", angles).reshape(-1, 1, 3, 3)
    rot_o2c_init = (rot_base @ delta_rot).reshape(-1, 3, 3)
    trans_o2c_init = np.tile(points, (n_init_inplane, 1))

    return rot_o2c_init, trans_o2c_init

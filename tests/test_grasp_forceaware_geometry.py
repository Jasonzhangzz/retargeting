"""A contained physical source cloud fixes tangent-sphere back-side overhang."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from retarget.env.grasp.forceaware_geometry import (
    CONTAINMENT_TOLERANCE_M, MAX_INWARD_INSET_M, RADIUS_FLOOR_M,
    constrain_tangent_spheres,
)


def thin_box(*, repeated=False):
    vertices = np.asarray([[x, y, z] for x in (-.01, .01) for y in (-.01, .01)
                           for z in (-.0003, .0003)])
    text = " ".join(map(str, vertices.reshape(-1)))
    second = '<geom name="physical_second" type="mesh" mesh="thin" pos=".1 0 0"/>' if repeated else ''
    return mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="thin" vertex="{text}"/></asset><worldbody><body name="skin">
      <geom name="physical" type="mesh" mesh="thin"/>{second}</body></worldbody></mujoco>''')


def owner_vertices(model, owner, geom):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mesh = int(model.geom_dataid[geom])
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    world = model.mesh_vert[start:start + count].astype(float) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return (world - data.xpos[owner]) @ data.xmat[owner].reshape(3, 3)


def assert_contained(model, owner, geom_ids, centers, radii):
    for geom in np.unique(geom_ids):
        ids = np.flatnonzero(np.asarray(geom_ids) == geom)
        planes = ConvexHull(owner_vertices(model, owner, int(geom))).equations
        values = centers[ids].astype(float) @ planes[:, :3].T + planes[:, 3] + radii[ids].astype(float)[:, None]
        assert np.max(values) <= CONTAINMENT_TOLERANCE_M


def test_thin_box_old_sphere_protrudes_but_new_serialized_sphere_is_contained():
    model = thin_box(); owner = model.body("skin").id
    points = np.array([[0., 0., .0003], [.004, -.003, .0003]])
    normals = np.tile([0., 0., 1.], (2, 1))
    old_center = points - .001 * normals
    planes = ConvexHull(owner_vertices(model, owner, 0)).equations
    assert np.max(old_center @ planes[:, :3].T + planes[:, 3] + .001) > .001
    before = points.copy()
    centers, radii, report = constrain_tangent_spheres(model, owner, [0], points, normals,
        ["thin"] * 2, ["skin"] * 2, .001)
    assert centers.dtype == radii.dtype == np.float32
    assert np.all(radii >= RADIUS_FLOOR_M) and np.all(radii <= .000300001)
    assert_contained(model, owner, report["selected_physical_geom_ids"], centers, radii)
    np.testing.assert_array_equal(points, before)
    assert report["input_samples"] == report["output_samples"] == 2
    assert report["max_effective_tangent_point_displacement_m"] <= MAX_INWARD_INSET_M


def test_sharp_corner_keeps_positive_radius_with_bounded_inset_and_no_query_mutation():
    model = thin_box(); owner = model.body("skin").id
    points = np.array([[.01, .01, .0003]])
    normals = np.array([[0., 0., 1.]])
    original = points.copy()
    centers, radii, report = constrain_tangent_spheres(model, owner, [0], points, normals,
        ["thin"], ["skin"], .001)
    assert radii[0] >= RADIUS_FLOOR_M
    assert report["inset_samples"] == 1
    assert 0. < report["max_effective_tangent_point_displacement_m"] <= MAX_INWARD_INSET_M
    np.testing.assert_array_equal(points, original)
    assert_contained(model, owner, [0], centers, radii)


def test_unmatched_visual_provenance_and_obviously_outside_points_fail_closed():
    model = thin_box(); owner = model.body("skin").id
    with pytest.raises(ValueError, match="no matching physical geom"):
        constrain_tangent_spheres(model, owner, [0], [[0., 0., .0003]], [[0., 0., 1.]],
                                  ["visual_only"], ["skin"], .001)
    with pytest.raises(ValueError, match="Cannot contain positive"):
        constrain_tangent_spheres(model, owner, [0], [[0., 0., .0013]], [[0., 0., 1.]],
                                  ["thin"], ["skin"], .001)
    model.geom_contype[0] = model.geom_conaffinity[0] = 0
    with pytest.raises(ValueError, match="visual-only"):
        constrain_tangent_spheres(model, owner, [0], [[0., 0., .0003]], [[0., 0., 1.]],
                                  ["thin"], ["skin"], .001)


def test_repeated_physical_mesh_body_key_selects_the_containing_placement():
    model = thin_box(repeated=True); owner = model.body("skin").id
    points = np.array([[0., 0., .0003], [.1, 0., .0003]])
    centers, radii, report = constrain_tangent_spheres(model, owner, [0, 1], points,
        np.tile([0., 0., 1.], (2, 1)), ["thin"] * 2, ["skin"] * 2, .001)
    assert report["selected_physical_geom_ids"] == [0, 1]
    assert_contained(model, owner, [0, 1], centers, radii)


@pytest.mark.parametrize("finger", range(1, 6))
def test_actual_wuji_distal_and_pad_samples_are_all_retained_and_contained(finger):
    root = Path(__file__).resolve().parents[1] / "data/grasp/assets/wuji2"
    if not (root / "right.xml").is_file():
        pytest.skip("Prepared official WUJI2 asset is required")
    model = mujoco.MjModel.from_xml_path(str(root / "right.xml"))
    name = f"right_finger{finger}_link4"; owner = model.body(name).id
    with np.load(root / "qp_right" / (name + ".npz"), allow_pickle=False) as raw:
        points, normals = raw["local_pos"].copy(), raw["local_normal"].copy()
        meshes, bodies = raw["source_mesh"].copy(), raw["source_body"].copy()
    actual_bodies = {model.body(str(body)).id for body in bodies}
    geoms = [int(geom) for geom in np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
             if int(model.geom_bodyid[geom]) in actual_bodies]
    original_points, original_normals = points.copy(), normals.copy()
    centers, radii, report = constrain_tangent_spheres(model, owner, geoms, points, normals, meshes, bodies, .001)
    assert len(centers) == len(radii) == len(points) == 768
    assert np.all(radii >= RADIUS_FLOOR_M) and np.all(radii <= .001)
    assert not report["semantic_filter_applied"] and not report["physical_provenance_filter_applied"]
    assert report["radius_reduced_samples"] > 0
    assert report["max_effective_tangent_point_displacement_m"] <= MAX_INWARD_INSET_M
    assert_contained(model, owner, report["selected_physical_geom_ids"], centers, radii)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(normals, original_normals)

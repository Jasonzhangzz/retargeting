"""WUJI 2 adapter checked against the unmodified, pinned official MJCF."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from retarget.grasp_hand import HAND_BODIES, asset_root, hand_tree, model_hand, query_path


def compiled(hand='wuji2'):
    tree = hand_tree(hand)
    for parent in tree.iter():
        for element in list(parent):
            if element.tag == 'geom' and element.get('type') == 'querypoint':
                parent.remove(element)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))


def test_hand2_physical_frames_limits_and_actuators_match_official():
    root = asset_root('wuji2')
    reference = mujoco.MjModel.from_xml_path(str(root/'official/right.xml'))
    actual = compiled()
    aliases = json.loads((root/'source.json').read_text())['aliases']
    assert model_hand(actual) == 'wuji2'
    assert actual.nq == actual.nu == 26
    assert reference.nq == reference.nu == 20
    np.testing.assert_array_equal(actual.jnt_range[6:], reference.jnt_range)
    np.testing.assert_array_equal(actual.actuator_ctrlrange[6:], reference.actuator_ctrlrange)
    np.testing.assert_array_equal(actual.actuator_forcerange[6:], reference.actuator_forcerange)
    np.testing.assert_array_equal(actual.actuator_gainprm[6:], reference.actuator_gainprm)
    np.testing.assert_array_equal(actual.actuator_biasprm[6:], reference.actuator_biasprm)
    rng = np.random.default_rng(31)
    for _ in range(12):
        expected, state = mujoco.MjData(reference), mujoco.MjData(actual)
        expected.qpos[:] = rng.uniform(*reference.jnt_range.T)
        mujoco.mj_kinematics(reference, expected)
        wrist = reference.body('r_wrist').id
        state.qpos[:3] = expected.xpos[wrist]
        state.qpos[3:6] = Rotation.from_matrix(expected.xmat[wrist].reshape(3, 3)).as_euler('XYZ')
        state.qpos[6:] = expected.qpos
        mujoco.mj_kinematics(actual, state)
        for body in range(1, reference.nbody):
            name = reference.body(body).name
            target = actual.body(aliases.get(name, name)).id
            np.testing.assert_allclose(state.xpos[target], expected.xpos[body], atol=1.e-12)
            np.testing.assert_allclose(state.xmat[target], expected.xmat[body], atol=1.e-12)
            assert actual.body_mass[target] == reference.body_mass[body]
        for site in range(reference.nsite):
            name = reference.site(site).name
            target = actual.site(aliases.get(name, name)).id
            np.testing.assert_allclose(state.site_xpos[target], expected.site_xpos[site], atol=1.e-12)


def test_hand2_queries_lie_on_official_meshes_including_sensor_pads():
    root = asset_root('wuji2')
    reference = mujoco.MjModel.from_xml_path(str(root/'official/right.xml'))
    aliases = json.loads((root/'source.json').read_text())['aliases']
    inverse = {v: k for k, v in aliases.items()}
    data = mujoco.MjData(reference)
    mujoco.mj_kinematics(reference, data)
    actual = compiled()
    for owner in HAND_BODIES:
        body = reference.body(inverse[owner]).id
        attached = {body}
        if owner.endswith('link4'):
            attached.add(reference.body(inverse[owner].replace('_distal', '_tip_sensor_frame')).id)
        if owner == 'right_palm_link':
            attached.add(reference.body('r_mount').id)
        triangles = []
        for geom in range(reference.ngeom):
            if reference.geom_group[geom] != 1 or reference.geom_bodyid[geom] not in attached:
                continue
            mesh = reference.geom_dataid[geom]
            va, vn = reference.mesh_vertadr[mesh], reference.mesh_vertnum[mesh]
            fa, fn = reference.mesh_faceadr[mesh], reference.mesh_facenum[mesh]
            world = reference.mesh_vert[va:va+vn] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            vertices = (world-data.xpos[body]) @ data.xmat[body].reshape(3, 3)
            triangles.extend(vertices[reference.mesh_face[fa:fa+fn]])
        triangles = np.array(triangles)
        # Millimetres keep Trimesh's absolute degeneracy tolerance below the
        # small STL facets. Positions and normal comparisons remain independent.
        mesh = trimesh.Trimesh(triangles.reshape(-1, 3)*1000,
                               np.arange(len(triangles)*3).reshape(-1, 3), process=False)
        with np.load(query_path(actual, owner)) as queries:
            points = queries['local_pos'][::48]
            normal = queries['local_normal'][::48]
            _, distance, face = trimesh.proximity.closest_point_naive(mesh, points*1000)
            assert distance.max() < 2.e-5
            np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1., atol=1.e-6)
            for point, n in zip(points*1000, normal):
                nearest = trimesh.triangles.closest_point(mesh.triangles, np.tile(point, (len(mesh.faces), 1)))
                coincident = np.linalg.norm(nearest-point, axis=1) < 2.e-5
                # Some official STL parts have coincident opposite faces.
                assert (mesh.face_normals[coincident]@n).max() > .99


def test_hand2_palmar_convention_and_fixed_pad_collision_enclosure():
    from retarget.grasp_cem import pad_chart, enclosing_capsule_radii
    from forceaware.mpc_loss import _fit_collision_capsule
    from retarget.grasp_hand import rigid_mesh_arrays
    model = compiled()
    for name in HAND_BODIES:
        body = model.body(name).id
        with np.load(query_path(model, name)) as queries:
            p, n = queries['local_pos'], queries['local_normal']
        mask, belly, endpoint = pad_chart(model, body, p, n)
        assert mask.any()
        if name in ('right_palm_link', 'right_finger2_link2'):
            np.testing.assert_allclose(belly, [0., 1., 0.], atol=1.e-8)
        a, b, radius = _fit_collision_capsule(np.c_[p, np.zeros(len(p))])
        r = enclosing_capsule_radii(model, [body], np.array([a]), np.array([b]), [radius])[0]
        v, _ = rigid_mesh_arrays(model, body)
        t = np.clip((v-a)@(b-a)/max(np.dot(b-a, b-a), 1.e-20), 0., 1.)
        assert np.linalg.norm(v-a-t[:, None]*(b-a), axis=1).max() <= r
        if name.endswith('link4'):
            # Fixed tactile pad is included, not just the distal metal segment.
            own_geoms = np.flatnonzero((model.geom_bodyid == body)&(model.geom_group == 1))
            assert len(v) > sum(model.mesh_vertnum[model.geom_dataid[g]] for g in own_geoms)


def test_generation_marker_survives_xml_snapshot_and_legacy_is_independent(tmp_path):
    from retarget.grasp import build_scene
    from retarget.grasp_primitives import primitive_mesh
    path = primitive_mesh(tmp_path/'object.obj', 'sphere', .025, .05)
    for hand in ('wuji1', 'wuji2'):
        model, _, xml, _ = build_scene(path, hand_model=hand)
        assert model_hand(model) == model_hand(mujoco.MjModel.from_xml_string(xml)) == hand
        assert query_path(model, 'right_palm_link').is_file()
    assert model_hand(compiled('wuji1')) == 'wuji1'


def test_obb_certificate_removes_capsule_false_rejection_and_rejects_overlap():
    import warp as wp
    import retarget.grasp_cem as card
    from tests.test_grasp_cem import cost_data, state, array, launch
    d, s = cost_data(), state(n=1)
    d.box_center = array(np.zeros((2, 3)), wp.vec3)
    d.box_half = array([[.003, .01, .005]]*2, wp.vec3)
    d.box_certificate = 1
    d.rotation.assign(np.tile(np.eye(3), (1, 3, 1, 1)))
    d.position.assign([[[0, 0, 0], [0, 0, 0], [.015, 0, 0]]])
    launch(card.capsule_cost, (1, 1), d)
    launch(card.finish_cost, 1, d, s)
    assert np.isfinite(s.scores.numpy()[0])
    assert d.self_gap.numpy()[0] == pytest.approx(9., abs=1.e-5)
    assert d.parts.numpy()[0, 3] > 0  # capsule soft penalty remains active
    d.self_gap.fill_(np.inf)
    d.position.assign([[[0, 0, 0], [0, 0, 0], [.004, 0, 0]]])
    launch(card.capsule_cost, (1, 1), d)
    launch(card.finish_cost, 1, d, s)
    assert not np.isfinite(s.scores.numpy()[0])
    assert d.self_gap.numpy()[0] < 0


def test_obb_separation_is_a_lower_bound_on_independent_box_distance():
    import warp as wp
    import retarget.grasp_cem as card
    from scipy.optimize import lsq_linear
    from tests.test_grasp_cem import cost_data, array, launch
    d = cost_data()
    d.box_center = array(np.zeros((2, 3)), wp.vec3)
    d.box_half = array(np.zeros((2, 3)), wp.vec3)
    d.box_certificate = 1
    d.capsule_radius.fill_(1.)  # force the hard certificate to use OBBs
    rng = np.random.default_rng(18)
    for _ in range(16):
        rotations = Rotation.random(2, random_state=rng).as_matrix()
        half = rng.uniform(.001, .01, (2, 3))
        delta = rng.uniform(-.025, .025, 3)
        d.box_half.assign(half)
        d.rotation.assign(np.array([[np.eye(3), *rotations]]))
        d.position.assign(np.array([[np.zeros(3), np.zeros(3), delta]]))
        d.self_gap.fill_(np.inf)
        launch(card.capsule_cost, (1, 1), d)
        certificate = d.self_gap.numpy()[0]/1000
        # Box coordinates lie in intervals. Bounded linear least squares
        # gives their actual Euclidean separation independently of SAT.
        matrix = np.concatenate((rotations[0], -rotations[1]), axis=1)
        result = lsq_linear(matrix, delta, bounds=(-half.flatten(), half.flatten()), tol=1.e-12)
        actual = np.linalg.norm(matrix@result.x-delta)
        assert certificate <= actual+1.e-8

import mujoco
import numpy as np
import pytest
from retarget.env.contact.prediction import distance_query, predict_pair


def scene(kind='sphere'):
    shape = {'sphere': 'size=".05"', 'box': 'size=".05 .05 .05"', 'capsule': 'size=".05 .1"'}[kind]
    m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><body pos="-.3 0 0"><freejoint/><geom name="a" type="{kind}" {shape}/></body><body><geom name="b" type="box" size=".05 .2 .2"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.parametrize('kind', ['sphere', 'box', 'capsule'])
def test_linear_toi_and_state_isolation(kind):
    m, d = scene(kind)
    before = d.qpos.copy(), d.geom_xpos.copy()
    pos = np.array([[[-.3, 0, 0], [0, 0, 0]], [[.3, 0, 0], [0, 0, 0]]])
    quat = np.tile([1., 0, 0, 0], (2, 2, 1))
    prediction = predict_pair(m, d, 0, 1, np.array([0., 1.]), pos, quat)
    assert prediction.status == 'hit'
    assert abs(prediction.toi - 1 / 3) < 1e-4
    assert np.allclose(d.qpos, before[0]) and np.allclose(d.geom_xpos, before[1])
    dist, p1, p2, n = distance_query(m, d, 0, 1)
    assert abs(dist - .2) < 1e-6
    assert np.allclose(p2 - p1, dist * n)


def test_separating_and_uncertain():
    m, d = scene()
    q = np.tile([1., 0, 0, 0], (2, 2, 1))
    pos = np.array([[[-.3, 0, 0], [0, 0, 0]], [[-.6, 0, 0], [0, 0, 0]]])
    assert predict_pair(m, d, 0, 1, np.array([0., 1.]), pos, q).status == 'no_collision'
    pos[-1, 0] = [.3, .15, 0]
    out = predict_pair(m, d, 0, 1, np.array([0., 1.]), pos, q, max_iterations=1)
    assert out.status == 'uncertain'


def test_native_speculative_arrests_thin_wall_crossing():
    if not hasattr(mujoco.mjtEnableBit, 'mjENBL_SPECULATIVE'):
        pytest.skip('Run with .venv-contact4 for native speculative verification')
    xml = '<mujoco><option timestep=".02" gravity="0 0 0"/><worldbody><geom type="box" size=".002 .2 .2"/><body pos="-.1 0 0"><freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>'
    positions = []
    for enabled in (False, True):
        m = mujoco.MjModel.from_xml_string(xml)
        if enabled:
            m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SPECULATIVE
        d = mujoco.MjData(m); d.qvel[0] = 10
        mujoco.mj_step(m, d)
        positions.append(d.qpos[0])
        if enabled:
            assert d.ncon and d.nefc
            geometric = distance_query(m, d, 0, 1)[0]
            assert abs(geometric - d.contact[0].dist) > 1e-5
    assert positions[0] > 0 and positions[1] < positions[0]


def test_grazing_parallel_motion_does_not_report_impact():
    m, d = scene()
    pos = np.array([[[-.3, .3, 0], [0, 0, 0]], [[.3, .3, 0], [0, 0, 0]]])
    quat = np.tile([1., 0, 0, 0], (2, 2, 1))
    assert predict_pair(m, d, 0, 1, np.array([0., 1.]), pos, quat).status == 'no_collision'


def test_glancing_sphere_box_corner_has_analytic_toi():
    m, d = scene()
    # Offset .04 above the box edge, leaving .03 of the sphere radius in
    # the travel direction: first centre x = -.05 - sqrt(.05² - .04²).
    pos = np.array([[[-.3, .24, 0], [0, 0, 0]], [[.3, .24, 0], [0, 0, 0]]])
    quat = np.tile([1., 0, 0, 0], (2, 2, 1))
    prediction = predict_pair(m, d, 0, 1, np.array([0., 1.]), pos, quat)
    assert prediction.status == 'hit'
    assert abs(prediction.toi - (.3 - .08) / .6) < 1e-4


def test_rotating_thin_plate_matches_high_resolution_query():
    from scipy.spatial.transform import Rotation
    xml = '<mujoco><worldbody><body><freejoint/><geom type="box" size=".2 .005 .02"/></body><body pos="0 .15 0"><geom type="sphere" size=".02"/></body></worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    pos = np.array([[[0., 0, 0], [0, .15, 0]], [[0., 0, 0], [0, .15, 0]]])
    quat = np.tile([1., 0, 0, 0], (2, 2, 1)); quat[1, 0] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    result = predict_pair(m, d, 0, 1, np.array([0., 1.]), pos, quat, max_iterations=64)
    assert result.status == 'hit'
    first = None
    for t in np.linspace(0, 1, 2001):
        d.geom_xmat[0] = Rotation.from_rotvec([0, 0, t * np.pi / 2]).as_matrix().reshape(9)
        if distance_query(m, d, 0, 1)[0] <= 0:
            first = t; break
    assert first is not None and abs(result.toi - first) < .002


def test_third_geom_occludes_target():
    xml = '<mujoco><worldbody><body pos="-.3 0 0"><freejoint/><geom type="sphere" size=".02"/></body><body><geom type="box" size=".02 .1 .1"/></body><geom type="box" pos="-.15 0 0" size=".01 .1 .1"/></worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    # World geoms are compiled before body geoms; resolve names via body ownership.
    moving = int(np.flatnonzero(m.geom_bodyid == 1)[0])
    target = int(np.flatnonzero(m.geom_bodyid == 2)[0])
    pos = np.array([[[-.3, 0., 0.], [0., 0., 0.]], [[.1, 0., 0.], [0., 0., 0.]]])
    q = np.tile([1., 0, 0, 0], (2, 2, 1))
    out = predict_pair(m, d, moving, target, np.array([0., 1.]), pos, q)
    assert out.status == 'occluded' and out.occluded


def test_native_plane_produces_multiple_contact_constraints():
    if not hasattr(mujoco.mjtEnableBit, 'mjENBL_SPECULATIVE'):
        pytest.skip('Native MuJoCo 4 required')
    m = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".02" gravity="0 0 0"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .4"><freejoint/><geom type="box" size=".1 .1 .005"/></body></worldbody></mujoco>')
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SPECULATIVE
    d = mujoco.MjData(m); d.qvel[2] = -30
    mujoco.mj_step(m, d)
    assert d.ncon >= 3 and d.nefc >= 3


def test_native_pre_step_velocity_does_not_include_new_acceleration():
    if not hasattr(mujoco.mjtEnableBit, 'mjENBL_SPECULATIVE'):
        pytest.skip('Native MuJoCo 4 required')
    m = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".02" gravity="0 0 0"/><worldbody><geom type="box" size=".002 .2 .2"/><body pos="-.1 0 0"><freejoint/><geom type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>')
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SPECULATIVE
    d = mujoco.MjData(m); d.qfrc_applied[0] = 1000
    mujoco.mj_step(m, d)
    assert d.ncon == 0 and d.qpos[0] > 0


def test_exact_plane_contact_keeps_the_oriented_normal_in_both_orders():
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .05"><freejoint/><geom type="sphere" size=".05"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    for first, second, expected in [(0, 1, [0, 0, 1]), (1, 0, [0, 0, -1])]:
        distance, p1, p2, normal = distance_query(m, d, first, second)
        assert abs(distance) < 1e-12
        np.testing.assert_allclose(p1, p2, atol=1e-12)
        np.testing.assert_allclose(normal, expected, atol=1e-12)


def test_coincident_witnesses_recover_a_nonvertical_normal_without_mutation():
    m, d = scene()
    d.qpos[0] = -.1
    mujoco.mj_forward(m, d)
    before = d.geom_xpos.copy()
    distance, _, _, normal = distance_query(m, d, 0, 1)
    assert abs(distance) < 1e-8
    assert normal[0] > .99
    np.testing.assert_array_equal(d.geom_xpos, before)


def test_swept_sphere_filter_rejects_distant_motion_before_gjk():
    m, d = scene()
    pos = np.array([[[-1., 0, 0], [0, 0, 0]], [[-2., 0, 0], [0, 0, 0]]])
    q = np.tile([1., 0, 0, 0], (2, 2, 1))
    result = predict_pair(m, d, 0, 1, np.array([0., 1.]), pos, q)
    assert result.status == 'no_collision' and result.iterations == 0

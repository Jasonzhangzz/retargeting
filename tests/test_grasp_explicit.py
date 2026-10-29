"""Independent reference equations and finite differences for explicit prediction."""
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from retarget import grasp_explicit as explicit


def array(value, dtype=float, grad=False):
    return wp.array(np.asarray(value), dtype=dtype, device='cpu', requires_grad=grad)


def fixture():
    rng = np.random.default_rng(46)
    p = explicit.Parameters()
    p.object_qpos, p.dt, p.sigma, p.weight, p.friction = 26, .1, .5, 1., .6
    p.hand_qpos = array(np.arange(26), int)
    for name in ('joints', 'joint_slide', 'bodies', 'starts', 'ends'):
        setattr(p, name, array([], int))
    p.ancestor, p.samples = array(np.zeros((0, 26)), int), array(np.zeros((0, 3)), wp.vec3)
    p.sample_normals, p.forbidden = array(np.zeros((0, 3)), wp.vec3), array([], int)
    p.body_mass, p.body_com = array([]), array(np.zeros((0, 3)), wp.vec3)
    p.body_ancestor = array(np.zeros((0, 26)), int)
    p.force_low, p.force_high = array(np.full(26, -np.inf)), array(np.full(26, np.inf))
    p.stiffness = array(rng.uniform(2., 20., 26))
    p.command_low, p.command_high = array(np.full(26, -np.inf)), array(np.full(26, np.inf))
    p.inverse = array(np.r_[np.full(3, 1/50), np.full(3, 10.), 1/p.stiffness.numpy()])
    p.gravity_force = wp.vec3(.03, -.02, -.981)
    seed = np.r_[np.zeros(26), [.01, -.02, .03], Rotation.from_euler('xyz', [.3, -.2, .1]).as_quat(scalar_first=True)]
    p.seed = array(seed)
    qpos = seed[None].copy()
    hold = qpos.copy(); hold[0, :26] += rng.uniform(-.002, .002, 26)
    phi = rng.uniform(-.02, .01, (1, 8))
    jac = rng.normal(0, .1, (1, 8, 32))
    v = SimpleNamespace(p=p, qpos=array(qpos, grad=True), hold=array(hold, grad=True),
        phi=array(phi, grad=True), jac=array(jac, grad=True),
        free=wp.zeros((1, 32), dtype=float, device='cpu', requires_grad=True),
        projected=wp.zeros((1, 8), dtype=float, device='cpu', requires_grad=True),
        force=wp.zeros((1, 8), dtype=float, device='cpu', requires_grad=True),
        velocity=wp.zeros((1, 32), dtype=float, device='cpu', requires_grad=True),
        next=wp.zeros((1, 33), dtype=float, device='cpu', requires_grad=True),
        cost=wp.zeros(1, dtype=float, device='cpu', requires_grad=True))
    v.gravity = array(np.zeros((1, 26)), grad=True)
    v.trace, v.rate = array([0.], grad=True), array([0.], grad=True)
    v.forces = [wp.zeros((1, 8), dtype=float, device='cpu', requires_grad=True) for _ in range(65)]
    v.responses = [wp.zeros((1, 32), dtype=float, device='cpu', requires_grad=True) for _ in range(64)]
    v.projections = [wp.zeros((1, 8), dtype=float, device='cpu', requires_grad=True) for _ in range(64)]
    v.force = v.forces[-1]
    return v


def run(v):
    def launch(kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device='cpu')
    launch(explicit.free_displacement, (1, 32), [v.p, v.qpos, v.hold, v.gravity, v.free])
    launch(explicit.contact_trace, 1, [v.p, v.jac, v.trace])
    launch(explicit.contact_step_size, 1, [v.p, v.trace, v.rate])
    for i in range(64):
        launch(explicit.projected_response, (1, 32), [v.p, v.jac, v.forces[i], v.responses[i]])
        launch(explicit.projected_force, (1, 8), [v.jac, v.free, v.phi, v.responses[i], v.projections[i]])
        launch(explicit.projected_update, (1, 2), [v.p, v.rate, v.forces[i], v.projections[i], v.forces[i+1]])
    launch(explicit.contact_velocity, (1, 32), [v.p, v.free, v.jac, v.force, v.velocity])
    launch(explicit.integrate, 1, [v.p, v.qpos, v.velocity, v.next])
    launch(explicit.terminal_pose_cost, 1, [v.p, v.next, v.cost])


def contact_reference(matrix, free, gaps, inverse, mu, sigma, iterations=64):
    """Independent normal complementarity and tangential Coulomb projection."""
    grouped = matrix.reshape(-1, 4, matrix.shape[-1])
    normal_jac = grouped.mean(axis=1)
    t1 = (grouped[:, 0]-grouped[:, 2])/(2*mu)
    t2 = (grouped[:, 1]-grouped[:, 3])/(2*mu)
    trace = np.sum((normal_jac**2+t1**2+t2**2)*inverse)
    rate = sigma/max(trace, 1.e-8)
    forces = np.zeros((len(grouped), 4))
    for _ in range(iterations):
        response = inverse*(matrix.T@forces.reshape(-1))
        grad = (matrix@(free+response)+gaps).reshape(-1, 4)
        normal = np.maximum(forces.sum(axis=1)-rate*grad.mean(axis=1), 0.)
        tangent = mu*np.c_[forces[:, 0]-forces[:, 2], forces[:, 1]-forces[:, 3]]
        tangent -= rate*np.c_[grad[:, 0]-grad[:, 2], grad[:, 1]-grad[:, 3]]/(2*mu)
        for i in range(len(normal)):
            ax, ay = abs(tangent[i]); radius = mu*normal[i]
            if ax+ay > radius:
                if ax-ay >= radius:
                    tangent[i] = [np.sign(tangent[i,0])*radius, 0.]
                elif ay-ax >= radius:
                    tangent[i] = [0., np.sign(tangent[i,1])*radius]
                else:
                    tangent[i] = np.sign(tangent[i])*(np.abs(tangent[i])-(ax+ay-radius)/2)
        scaled = tangent/mu
        shared = np.maximum(normal-np.abs(scaled).sum(axis=1), 0.)/4
        forces = shared[:,None]+np.maximum(np.c_[scaled[:,0], scaled[:,1], -scaled[:,0], -scaled[:,1]], 0.)
    return forces.reshape(-1)


def reference(v, phi=None, jac=None, hold=None):
    p = v.p
    qpos = v.qpos.numpy()[0].astype(float)
    held = v.hold.numpy()[0].astype(float) if hold is None else hold
    matrix = v.jac.numpy()[0].astype(float) if jac is None else jac
    gaps = v.phi.numpy()[0].astype(float) if phi is None else phi
    target = np.clip(held[:26], p.command_low.numpy(), p.command_high.numpy())
    b = np.r_[np.array(list(p.gravity_force)), np.zeros(3), np.clip(p.stiffness.numpy()*(target-qpos[:26]),p.force_low.numpy(),p.force_high.numpy())+v.gravity.numpy()[0]]
    inv = p.inverse.numpy().astype(float)
    free = inv*b
    force = contact_reference(matrix, free, gaps, inv, p.friction, p.sigma)
    velocity = (free+inv*(matrix.T@force))/p.dt
    out = qpos.copy(); out[26:29] += p.dt*velocity[:3]
    q = qpos[29:33]; omega = velocity[3:6]
    dq = np.r_[-np.dot(q[1:], omega), q[0]*omega+np.cross(q[1:], omega)]
    next_q = q+.5*p.dt*dq; next_q /= np.linalg.norm(next_q)
    out[29:33] = next_q
    out[:26] += p.dt*velocity[6:]
    goal = p.seed.numpy()[29:33].astype(float); goal /= np.linalg.norm(goal)
    cost = p.weight*(5000*np.sum((out[26:29]-p.seed.numpy()[26:29])**2)+50*(1-np.dot(next_q, goal)**2))
    return free, force, velocity, out, cost


def test_explicit_force_motion_and_bowl_terminal_cost_match_reference():
    v = fixture(); run(v)
    expected = reference(v)
    for actual, value in zip((v.free, v.force, v.velocity, v.next), expected[:4]):
        np.testing.assert_allclose(actual.numpy()[0], value, atol=2.e-7, rtol=2.e-5)
    assert v.cost.numpy()[0] == pytest.approx(expected[4], abs=2.e-5, rel=2.e-4)
    assert not np.allclose(v.next.numpy()[0, 26:29], v.p.seed.numpy()[26:29])
    assert np.linalg.norm(v.next.numpy()[0, 29:33]) == pytest.approx(1., abs=1.e-6)


def test_contact_force_to_pose_gradient_matches_independent_finite_differences():
    v = fixture()
    with wp.Tape() as tape:
        run(v)
    tape.backward(v.cost)
    rng = np.random.default_rng(2)
    for name in ('phi', 'jac', 'hold'):
        values = getattr(v, name).numpy()[0].astype(float)
        direction = rng.normal(size=values.shape); direction /= np.linalg.norm(direction)
        analytic = np.sum(getattr(v, name).grad.numpy()[0]*direction)
        delta = 1.e-5
        plus = reference(v, **{name: values+delta*direction})[-1]
        minus = reference(v, **{name: values-delta*direction})[-1]
        numerical = (plus-minus)/(2*delta)
        assert abs(numerical) > 1.e-5
        assert analytic == pytest.approx(numerical, rel=3.e-3, abs=2.e-5)


def test_pose_cost_is_nonnegative_and_quaternion_sign_invariant():
    v = fixture(); q = v.p.seed.numpy()[None].copy()
    for scale in (1., -1., 3., -3.):
        q[0, 29:33] = scale*v.p.seed.numpy()[29:33]
        v.next.assign(q)
        wp.launch(explicit.terminal_pose_cost, 1, inputs=[v.p, v.next, v.cost], device='cpu')
        assert 0 <= v.cost.numpy()[0] < 2.e-5
    q[0, 26] += .01
    v.next.assign(q)
    wp.launch(explicit.terminal_pose_cost, 1, inputs=[v.p, v.next, v.cost], device='cpu')
    assert v.cost.numpy()[0] == pytest.approx(.5, abs=2.e-5)


def test_trajectory_cost_penalizes_transient_motion_even_after_terminal_return():
    v = fixture()
    displaced = v.p.seed.numpy()[None].copy(); displaced[0, 26] += .01
    intermediate = array(displaced, grad=True)
    restored = array(v.p.seed.numpy()[None], grad=True)
    costs = array(np.zeros((1, 2)), grad=True)
    with wp.Tape() as tape:
        wp.launch(explicit.step_pose_cost, 1, inputs=[v.p, intermediate, 0, costs], device='cpu')
        wp.launch(explicit.step_pose_cost, 1, inputs=[v.p, restored, 1, costs], device='cpu')
        wp.launch(explicit.average_pose_cost, 1, inputs=[costs, v.cost], device='cpu')
    tape.backward(v.cost)
    assert v.cost.numpy()[0] == pytest.approx(.25, abs=2.e-5)
    assert intermediate.grad.numpy()[0, 26] == pytest.approx(50., rel=1.e-4)


def test_projected_contact_large_values_remain_finite():
    v = fixture(); v.phi.assign([[-1000.]*4+[1000.]*4]); run(v)
    assert np.isfinite(v.force.numpy()).all()
    assert np.isfinite(v.velocity.numpy()).all()
    np.testing.assert_allclose(v.force.numpy()[0, ::2], reference(v)[1][::2], rtol=1.e-5, atol=5.e-4)


def test_coupled_contact_supports_gravity_without_softplus_bias():
    v = fixture()
    matrix = np.zeros((1, 8, 32), np.float32)
    matrix[0, :4, 2] = 1.  # One bottom contact with four friction rays.
    v.jac.assign(matrix); v.phi.assign(np.zeros((1, 8)))
    v.p.gravity_force = wp.vec3(0., 0., -1.962)
    v.hold.assign(v.qpos.numpy())
    run(v)
    assert v.force.numpy()[0, :4].sum() == pytest.approx(1.962, rel=1.e-5)
    assert abs(v.next.numpy()[0, 28]-v.qpos.numpy()[0, 28]) < 1.e-6
    v.p.gravity_force = wp.vec3(0.)
    run(v)
    np.testing.assert_array_equal(v.force.numpy(), 0.)


def test_tangential_slip_cannot_create_normal_pressure():
    v = fixture()
    matrix = np.zeros((1, 8, 32))
    matrix[0, :4, :3] = [[.6, 0., 1.], [0., .6, 1.], [-.6, 0., 1.], [0., -.6, 1.]]
    v.jac.assign(matrix); v.phi.assign(np.zeros((1, 8)))
    v.hold.assign(v.qpos.numpy()); v.p.gravity_force = wp.vec3(1., 0., 0.)
    run(v)
    np.testing.assert_array_equal(v.force.numpy(), 0.)


def test_servo_preload_changes_contact_forces_and_has_motion_gradient():
    v = fixture()
    matrix = np.zeros((1, 8, 32), np.float32)
    for c, sign in enumerate((1., -1.)):
        for edge, tangent in enumerate(((0., 1.), (1., 0.), (0., -1.), (-1., 0.))):
            row = c*4+edge
            matrix[0, row, :3] = [sign, .6*tangent[0], .6*tangent[1]]
            matrix[0, row, c+6] = -.02
    v.jac.assign(matrix); v.phi.assign(np.zeros((1, 8)))
    v.p.gravity_force = wp.vec3(0., 0., -1.962)
    held = v.qpos.numpy().copy(); v.hold.assign(held)
    run(v); baseline = v.force.numpy().sum()
    held[0, :2] += .05; v.hold.assign(held)
    with wp.Tape() as tape:
        run(v)
    tape.backward(v.cost)
    assert v.force.numpy().sum() > baseline
    direction = np.zeros(26); direction[:2] = 1.
    analytic = v.hold.grad.numpy()[0, :26]@direction
    delta = 1.e-5
    plus = reference(v, hold=held[0, :26]+delta*direction)[-1]
    minus = reference(v, hold=held[0, :26]-delta*direction)[-1]
    assert abs(analytic) > 1.e-6
    assert analytic == pytest.approx((plus-minus)/(2*delta), rel=.01, abs=2.e-5)


@pytest.mark.parametrize('gap,forbidden,active', [(.01, False, False), (.0005, False, True), (.0005, True, False)])
def test_real_mesh_contact_rejects_separated_and_forbidden_skin(gap, forbidden, active):
    import trimesh
    mesh = trimesh.creation.box(extents=[.02]*3)
    vertices = array(mesh.vertices, wp.vec3)
    faces = array(mesh.faces.reshape(-1), int)
    bvh = wp.Mesh(points=vertices, indices=faces)
    v = fixture(); p = v.p
    p.mesh, p.object_body, p.cutoff = bvh.id, 0, .001
    p.bodies, p.starts, p.ends = array([1], int), array([0], int), array([1], int)
    p.samples, p.sample_normals = array([[0., 0., 0.]], wp.vec3), array([[0., 0., -1.]], wp.vec3)
    p.forbidden, p.ancestor = array([forbidden], int), array(np.zeros((1, 26)), int)
    position = array([[[0., 0., 0.], [0., 0., .01+gap]]], wp.vec3)
    rotation = array(np.tile(np.eye(3), (1, 2, 1, 1)), wp.mat33)
    selected = array([[0]], int)
    wp.launch(explicit.select_patches, (1, 1), inputs=[p, position, rotation, selected], device='cpu')
    phi, jac, rays = array(np.ones((1, 4))), array(np.zeros((1, 4, 32))), array(np.zeros((1, 4, 3)), wp.vec3)
    anchors = array(np.zeros((1, 1, 3)), wp.vec3)
    wp.launch(explicit.contact_rows, (1, 1, 4), inputs=[p, selected, position, rotation, anchors, anchors, phi, jac, rays], device='cpu')
    assert bool(np.linalg.norm(rays.numpy())) == active
    if active:
        np.testing.assert_allclose(phi.numpy(), gap, atol=1.e-7)


def test_hold_command_uses_actual_actuator_limits_and_clipped_derivative():
    v = fixture()
    v.p.command_low.assign(np.full(26, -.001))
    v.p.command_high.assign(np.full(26, .001))
    held = v.hold.numpy(); held[0, 0] = 1.
    v.hold.assign(held)
    with wp.Tape() as tape:
        run(v)
    tape.backward(v.cost)
    assert v.cost.numpy()[0] == pytest.approx(reference(v)[-1], abs=2.e-5, rel=2.e-4)
    assert v.hold.grad.numpy()[0, 0] == 0.

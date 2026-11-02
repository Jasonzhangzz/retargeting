"""Supported PD integration preserves force clamps and their local derivatives."""

import sys

import mujoco
import numpy as np
import pytest

from retarget.dynamics.dexforge_servo_integration import validate_servo_model


def servo_xml(*, force_limited=True, gear=1.):
    limits = 'forcelimited="true" forcerange="-.5 .5"' if force_limited else 'forcelimited="false"'
    return f'''<mujoco><option timestep=".01" gravity="0 0 0" integrator="Euler"/>
      <worldbody><body name="rotor"><joint name="hinge" type="hinge" axis="0 0 1"
        damping=".4" armature=".02"/><geom type="sphere" size=".1" mass="2"
        contype="0" conaffinity="0"/></body></worldbody>
      <actuator><general name="pd" joint="hinge" gear="{gear}" gainprm="10"
        biasprm="0 -10 -2" biastype="affine" ctrllimited="true" ctrlrange="-1 1"
        {limits}/></actuator></mujoco>'''


def test_supported_pd_contract_keeps_original_parameters_and_limits():
    model = mujoco.MjModel.from_xml_string(servo_xml())
    before = {field: np.asarray(getattr(model, field)).copy()
              for field in ("body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm",
                            "actuator_forcelimited", "actuator_forcerange", "dof_damping")}
    contract = validate_servo_model(model)
    assert contract["actuator_dof_ids"] == [0]
    assert contract["kp"] == [10.] and contract["kv"] == [2.]
    assert contract["actuator_forcelimited"] == [True]
    for field, value in before.items():
        np.testing.assert_array_equal(getattr(model, field), value)


def test_pd_contract_rejects_semantics_without_correct_local_derivative():
    with pytest.raises(ValueError, match="unit scalar"):
        validate_servo_model(mujoco.MjModel.from_xml_string(servo_xml(gear=2.)))
    model = mujoco.MjModel.from_xml_string(servo_xml())
    model.actuator_biasprm[0, 2] = 1.
    with pytest.raises(ValueError, match="nonnegative kv"):
        validate_servo_model(model)
    model = mujoco.MjModel.from_xml_string(servo_xml())
    model.jnt_actfrclimited[0] = True
    model.jnt_actfrcrange[0] = [-.1, .1]
    with pytest.raises(ValueError, match="tighter joint"):
        validate_servo_model(model)


@pytest.fixture
def native_runtime():
    if sys.version_info[:2] != (3, 10):
        pytest.skip("Use .venv-forceaware for native integration tests")
    from retarget.dynamics.dexforge_upstream import require_environment
    require_environment()
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Original tile mass solve requires CUDA")
    from retarget.dynamics.dexforge_servo_integration import activate_servo_integration, deactivate_servo_integration
    activate_servo_integration()
    yield
    deactivate_servo_integration()


def setup_native(tmp_path, *, force_limited=True):
    import warp as wp
    import comfree_warp
    from comfree_warp.native_adjoint.dynamics import DynamicsInput
    from comfree_warp.native_adjoint.fast_step import compile_fast_step
    from comfree_warp.native_adjoint.fast_step_adjoint import allocate_workspace
    from comfree_warp.native_adjoint.runtime import StepInput

    path = tmp_path / "servo.xml"
    # The pinned native loader requires a query-point graph. A zero-mass cloud
    # and a plane 100 m away satisfy that input contract without active contact.
    np.savez(tmp_path / "cloud.npz", local_pos=np.zeros((1, 3), np.float32),
             local_normal=np.array([[0., 0., 1.]], np.float32),
             radius=np.array([.001], np.float32), group_id=np.zeros(1, np.int32))
    xml = servo_xml(force_limited=force_limited)
    xml = xml.replace("<worldbody>", '<worldbody><geom type="plane" size="1 1 .1" '
                      'pos="0 0 -100" contype="4" conaffinity="1"/>')
    xml = xml.replace("</body>", '<geom type="gs" file="cloud.npz" mass="0" '
                      'contype="1" conaffinity="4"/></body>')
    path.write_text(xml)
    device_model, cpu = comfree_warp.load_model(str(path), device="cuda:0")
    compiled = compile_fast_step(cpu, device_model)
    matrix = lambda value: wp.array(np.asarray(value, np.float32), dtype=float,
                                    device="cuda:0", requires_grad=True, retain_grad=True)
    dynamics = DynamicsInput()
    dynamics.qpos = matrix([[.02]])
    dynamics.qvel = matrix([[.01]])
    dynamics.ctrl = matrix([[.03]])
    dynamics.qfrc_applied = matrix([[0.]])
    inputs = StepInput(dynamics, matrix([0.]), matrix([0.]))
    return cpu, compiled, inputs, allocate_workspace(compiled, 1)


def analytic_step_and_gradient(model, q, v, u):
    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0], data.ctrl[0] = q, v, u
    mujoco.mj_forward(model, data)
    mass = np.empty((1, 1))
    mujoco.mj_fullM(model, mass, data.qM)
    control = np.clip(u, -1., 1.)
    raw = 10. * (control - q) - 2. * v
    interior = not model.actuator_forcelimited[0] or -.5 < raw < .5
    force = float(np.clip(raw, -.5, .5)) if model.actuator_forcelimited[0] else raw
    denominator = mass[0, 0] + .01 * (.4 + 2. * interior)
    next_v = v + .01 * (force - .4 * v) / denominator
    next_q = q + .01 * next_v
    # Include the physical acceleration output used by the unchanged FA state
    # regularizer: 1.3*q_next + .7*v_next + .05*qacc_physical.
    seed = 1.3 * .01 + .7
    grad_q = 1.3 + seed * .01 * (-10. * interior) / denominator
    grad_v = seed * (1. + .01 * (-.4 - 2. * interior) / denominator)
    grad_u = seed * .01 * 10. * interior * (-1. < u < 1.) / denominator
    grad_q += .05 * (-10. * interior) / mass[0, 0]
    grad_v += .05 * (-.4 - 2. * interior) / mass[0, 0]
    grad_u += .05 * 10. * interior * (-1. < u < 1.) / mass[0, 0]
    return next_q, next_v, np.array([grad_q, grad_v, grad_u]), force, interior


@pytest.mark.parametrize("u,force_limited", [(.03, True), (.5, True), (2., False)])
def test_real_native_forward_and_vjp_match_clamped_pd_and_mujoco(tmp_path, native_runtime, u, force_limited):
    import warp as wp
    from comfree_warp.native_adjoint.fast_step_adjoint import BackwardCall, StepCotangent, record, backward
    from retarget.dynamics.dexforge_servo_integration import servo_integration_diagnostics

    cpu, compiled, inputs, workspace = setup_native(tmp_path, force_limited=force_limited)
    inputs.dynamics.ctrl.assign(np.array([[u]], np.float32))
    recorded = record(compiled, inputs, workspace)
    expected_q, expected_v, expected_gradient, expected_force, interior = analytic_step_and_gradient(cpu, .02, .01, u)
    np.testing.assert_allclose(recorded.result.qpos.numpy()[0, 0], expected_q, atol=3e-7)
    np.testing.assert_allclose(recorded.result.qvel.numpy()[0, 0], expected_v, atol=3e-6)
    np.testing.assert_allclose(workspace.forward.base.dynamics.actuator_force.numpy()[0, 0], expected_force, atol=3e-7)

    reference = mujoco.MjData(cpu)
    cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    reference.qpos[0], reference.qvel[0], reference.ctrl[0] = .02, .01, u
    mujoco.mj_step(cpu, reference)
    np.testing.assert_allclose(recorded.result.qpos.numpy()[0], reference.qpos, atol=3e-7)
    np.testing.assert_allclose(recorded.result.qvel.numpy()[0], reference.qvel, atol=3e-6)
    np.testing.assert_allclose(recorded.result.qacc.numpy()[0], reference.qacc, atol=3e-5)

    zeros = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device="cuda:0")
    count = compiled.base.collision.contact_count
    cotangent = StepCotangent(zeros((1, 1)), zeros((1, 1)), zeros((1, 1)), zeros(1),
        zeros((1, 1)), zeros((1, cpu.nbody), wp.vec3), zeros((1, cpu.nbody), wp.mat33),
        zeros((1, count)), zeros((1, count), wp.vec3), zeros((1, count), wp.mat33))
    cotangent.qpos.assign(np.array([[1.3]], np.float32))
    cotangent.qvel.assign(np.array([[.7]], np.float32))
    cotangent.qacc.assign(np.array([[.05]], np.float32))
    gradient = backward(BackwardCall(compiled, inputs, workspace, recorded, cotangent))
    actual = np.array([gradient.qpos.numpy()[0, 0], gradient.qvel.numpy()[0, 0], gradient.ctrl.numpy()[0, 0]])
    np.testing.assert_allclose(actual, expected_gradient, rtol=3e-5, atol=3e-6)
    report = servo_integration_diagnostics(compiled, workspace, inputs=inputs)
    assert report["actuator_implicit_damping_active"] == [[bool(interior)]]
    assert not report["local_actuator_nonsmooth"]


def test_diagnostic_detects_force_branch_change_and_clamp_boundary(tmp_path, native_runtime):
    from comfree_warp.native_adjoint.fast_step_adjoint import forward
    from retarget.dynamics.dexforge_servo_integration import servo_integration_diagnostics

    _, compiled, inputs, workspace = setup_native(tmp_path)
    # raw force = 10*(u-.02)-2*.01, so upper force boundary is u=.072.
    reports = []
    for control in (.0719, .072, .0721):
        inputs.dynamics.ctrl.assign(np.array([[control]], np.float32))
        forward(compiled, inputs, workspace)
        reports.append(servo_integration_diagnostics(compiled, workspace, inputs=inputs,
                                                     reference=reports[0] if reports else None))
    assert reports[0]["actuator_implicit_damping_active"] == [[True]]
    assert reports[-1]["actuator_implicit_damping_active"] == [[False]]
    assert reports[-1]["actuator_mask_changes"] == 1
    assert reports[1]["near_force_clamp_boundary"] == [[True]]


def test_activation_covers_imported_original_fa_consumers_and_preserves_source(native_runtime):
    from comfree_warp.native_adjoint import fast_step, fast_step_adjoint
    from forceaware import executor, optimizer
    from retarget.dynamics.dexforge_servo_integration import activate_servo_integration
    from retarget.dynamics.dexforge_upstream import verify_upstream

    metadata = activate_servo_integration()
    assert metadata["active"] and not metadata["upstream_source_modified"]
    assert len(metadata["implementation_sha256"]) == 64
    assert fast_step.integration_acceleration is fast_step_adjoint.integration_acceleration
    assert executor.compile_fast_step is fast_step.compile_fast_step
    assert optimizer.compile_fast_step is fast_step.compile_fast_step
    assert verify_upstream()["verified_files"] > 0

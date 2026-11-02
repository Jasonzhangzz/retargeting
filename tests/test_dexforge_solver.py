"""Official FA contract and humanoid objective/adjoint regression checks."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from retarget.solver.dexforge import (
    DexForgeProblem, DexForgeSettings, make_config, solve,
)


def test_settings_import_does_not_load_numerical_or_upstream_packages():
    code = """import sys
from retarget.solver.dexforge import DexForgeSettings
assert DexForgeSettings().n_iter == 40
assert not any(name in sys.modules for name in ('numpy', 'torch', 'warp', 'forceaware', 'comfree_warp'))
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_official_configuration_defaults_and_joint_units():
    cfg = make_config(DexForgeSettings(), xml="data/humanoid/move_largebox/scene.xml", ref_dt=.0333333)
    assert (cfg.optimizer.horizon, cfg.optimizer.n_iter, cfg.optimizer.multistart_count) == (5, 40, 64)
    assert cfg.simulator.time_grid.knot_substeps == 6
    assert cfg.simulator.time_grid.action_substeps == 3
    assert cfg.optimizer.horizon * cfg.simulator.time_grid.knot_substeps == 30
    assert cfg.loss.action_ref_base == cfg.loss.action_ref_base_rot == cfg.loss.action_ref_finger == .03
    assert cfg.bounds.init_base_pos == cfg.bounds.init_base_rot == cfg.bounds.init_finger == .24
    assert cfg.loss.contact_robustness == cfg.loss.marked_query_pos == 0
    assert cfg.simulator.exec_backend == "comfree"


@pytest.mark.parametrize("values", [dict(n_iter=0), dict(multistart_count=True), dict(mpc_dt=0),
    dict(action_dt=.035), dict(lr_min=.1), dict(root_pos_weight=float("nan")),
    dict(adam_beta1=1), dict(penetration_limit=.001), dict(contact_topk=9)])
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        DexForgeSettings(**values)


def test_shared_dispatch_uses_actual_official_plan_fields():
    pytest.importorskip("torch")
    plan = SimpleNamespace(raw=np.arange(58, dtype=np.float32), loss=2.5, grad_norm=.3,
                           diag={"selected": 0, "scores": np.array([2.5])})
    seen = []

    class Optimizer:
        def solve(self, *args, **kwargs):
            seen.append((args, kwargs))
            return plan

    problem = DexForgeProblem(Optimizer(), np.zeros(43), np.zeros(41), object(),
                             np.zeros(58), np.zeros(29), allow_init=True, start_time=.2)
    settings = DexForgeSettings(horizon=1, multistart_count=1)
    result = solve(problem, settings, seed=42)
    assert result.cost == 2.5 and result.plan is plan
    assert result.history[0]["grad_norm"] == .3
    assert seen[0][1]["seed"] == 42 and seen[0][1]["execution_steps"] == 3
    assert seen[0][1]["allow_init"] and seen[0][1]["start_time"] == .2
    with pytest.raises(TypeError, match="DexForgeProblem"):
        solve(object(), settings)


def _native():
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment
    try:
        require_environment()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    activate_upstream()
    import warp as wp
    wp.init()
    return wp


def _isolated_workspace(device="cpu"):
    wp = _native()
    from retarget.dynamics.dexforge_loss import HumanoidObjective
    settings = DexForgeSettings()
    cfg = make_config(settings, xml="data/humanoid/move_largebox/scene.xml", ref_dt=.0333333)
    q = np.zeros((1, 43), np.float32)
    q[:, 3] = q[:, 39] = 1
    q[0, 0], q[0, 7], q[0, 36] = .1, .5, .2
    q[0, 3], q[0, 6] = np.cos(.1), np.sin(.1)
    result = SimpleNamespace(
        qpos=wp.array(q, dtype=float, device=device, requires_grad=True),
        body_position=wp.zeros((1, 2), dtype=wp.vec3, device=device, requires_grad=True),
        body_matrix=wp.zeros((1, 2), dtype=wp.mat33, device=device, requires_grad=True),
        contact_distance=wp.array(np.array([[-.012, -.02]], np.float32), dtype=float, device=device, requires_grad=True),
        contact_position=wp.zeros((1, 2), dtype=wp.vec3, device=device, requires_grad=True),
        contact_frame=wp.zeros((1, 2), dtype=wp.mat33, device=device, requires_grad=True),
    )
    active = wp.array(np.array([[1, 0]], np.int32), dtype=int, device=device)
    runtime = SimpleNamespace(recorded=SimpleNamespace(result=result), workspace=SimpleNamespace(
        forward=SimpleNamespace(base=SimpleNamespace(collision=SimpleNamespace(contacts=SimpleNamespace(active=active))))))
    objective = HumanoidObjective.__new__(HumanoidObjective)
    objective.settings = settings
    objective.cfg, objective.worlds, objective.device = cfg, 1, device
    objective.index = SimpleNamespace(hand_qpos=np.arange(7,36,dtype=np.int32), object_qpos=36, object_body=1, action_dim=29)
    objective.root_pos_weight, objective.root_rot_weight = 10., 3.
    objective.guidance_count, objective.penetration_output_indices = 4, None
    objective._root_jobs, objective.steps = {}, [runtime]
    workspace = objective._workspace(runtime)
    workspace.target.action.zero_()
    workspace.target.object_qpos.assign(np.array([0,0,0,1,0,0,0], np.float32))
    return objective, workspace, result


def test_humanoid_loss_propagates_root_quaternion_joint_object_and_actual_penetration():
    wp = _native()
    from forceaware import mpc_loss
    from retarget.dynamics.dexforge_loss import _humanoid_loss
    objective, workspace, state = _isolated_workspace()
    root = objective._root_jobs[id(workspace)]
    wp.launch(_humanoid_loss, dim=1, inputs=[workspace.job, root], outputs=[workspace.loss], device="cpu")
    mpc_loss.backward(workspace)
    gradient = state.qpos.grad.numpy()[0]
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[[0,7,36]], [20., .0009, 40.], rtol=1e-5, atol=1e-6)
    assert np.linalg.norm(gradient[3:7]) > .1
    np.testing.assert_allclose(state.contact_distance.grad.numpy()[0], [-.84, 0], rtol=1e-5, atol=1e-6)
    baseline = state.qpos.numpy().copy()
    eps = 1e-4
    costs = []
    for sign in [-1,1]:
        trial = baseline.copy(); trial[0,6] += sign*eps
        state.qpos.assign(trial)
        wp.launch(_humanoid_loss, dim=1, inputs=[workspace.job, root], outputs=[workspace.loss], device="cpu")
        costs.append(float(workspace.loss.numpy()[0]) * 1e-6)
    np.testing.assert_allclose(gradient[6], (costs[1]-costs[0])/(2*eps), rtol=.01, atol=.002)


def test_joint_regularizer_uses_uniform_angular_arrays():
    _native()
    from retarget.dynamics.dexforge_loss import HumanoidObjective
    import mujoco
    xml = "<mujoco><worldbody><body><joint name='j'/><geom type='sphere' size='.1'/></body></worldbody></mujoco>"
    # Regularizer indexes are independent of collision geometry. Repeating a
    # valid scalar coordinate tests the 29-entry unit profile without a scene.
    model = mujoco.MjModel.from_xml_string(xml)
    index = SimpleNamespace(hand_qpos=np.zeros(29,np.int32), action_dim=29)
    settings = DexForgeSettings()
    obj = HumanoidObjective.__new__(HumanoidObjective); obj.settings = settings
    cfg = make_config(settings, xml="data/humanoid/move_largebox/scene.xml", ref_dt=.0333333)
    regularizer = obj._make_regularizer(model, index, cfg, worlds=1, device="cpu")
    expected = {"init_bound":.24,"control_velocity_scale":1.,"control_velocity_weight":.3,
                "control_acceleration_scale":500.,"state_velocity_scale":4.,
                "state_acceleration_scale":100.,"state_component_ratio":.3}
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(regularizer.model,name).numpy(), np.full(29,value), rtol=1e-7)


def test_thin_adapter_inherits_every_production_optimizer_method(monkeypatch):
    _native()
    from forceaware.servo_prior import ServoPriorOptimizer
    from forceaware.optimizer import WindowOptimizer
    from retarget.dynamics.dexforge_loss import create_optimizer
    monkeypatch.setattr(ServoPriorOptimizer, "__init__", lambda self,*args,**kwargs: None)
    cpu = SimpleNamespace(nq=43,nv=41)
    index = SimpleNamespace(action_dim=29, hand_qpos=np.arange(7,36), object_qpos=36)
    instance = create_optimizer(cpu,None,index,cfg=object(),settings=DexForgeSettings())
    subclass = type(instance)
    functions = {name for name,value in subclass.__dict__.items() if callable(value)}
    assert functions == {"_make_objective"}
    for name in ("solve","_forward","_iteration","_backward","_multistart","_prepare_window"):
        assert getattr(subclass,name) is getattr(ServoPriorOptimizer,name)
    assert subclass.solve is WindowOptimizer.solve


def test_initial_correction_records_joint_edits_and_rejects_root_or_object_changes():
    from retarget.dynamics.dexforge_diagnostics import initial_correction
    index = SimpleNamespace(hand_qpos=np.arange(7,36))
    before = np.zeros(43,np.float32); after=before.copy(); after[7]=.1
    result = initial_correction(before,after,index,bound=.24)
    assert result["initial_joint_delta_max_rad"] == pytest.approx(.1)
    assert result["initial_root_unchanged"] and result["initial_object_unchanged"]
    for coordinate in [0,3,36,39]:
        invalid=after.copy();invalid[coordinate]=.01
        with pytest.raises(RuntimeError,match="fixed root or object"):
            initial_correction(before,invalid,index,bound=.24)
    with pytest.raises(RuntimeError,match="allowed bound"):
        initial_correction(before,after,index,bound=.24,allow_init=False)


def test_tracking_diagnostic_uses_dense_poststates_and_terminal_weight():
    from retarget.dynamics.dexforge_diagnostics import tracking_components
    cfg = make_config(DexForgeSettings(),xml="data/humanoid/move_largebox/scene.xml",ref_dt=.0333333)
    index=SimpleNamespace(action_dim=29,hand_qpos=np.arange(7,36),object_qpos=36)
    q=np.zeros((2,43));q[:,3]=q[:,39]=1
    root=q[:,:7].copy();obj=q[:,36:].copy()
    q[0,0]=.1;q[1,36]=.1;q[:,7]=.5
    targets=SimpleNamespace(root_qpos=root,object_qpos=obj,dense_actions=np.zeros((2,29)))
    result=tracking_components(q,targets,index,cfg,root_pos_weight=10.,root_rot_weight=3.)
    assert result["root_tracking_loss"] == pytest.approx(1e6,rel=1e-6)
    assert result["object_tracking_loss"] == pytest.approx(2e6,rel=1e-6)
    assert result["joint_tracking_loss"] == pytest.approx(675.,rel=1e-6)
    assert result["root_position_rms_m"] == pytest.approx(.1/np.sqrt(2))


def test_contact_diagnostic_decodes_native_friction_edges_and_masks_inactive_rows():
    from retarget.dynamics.dexforge_diagnostics import _contact_forces

    class Array:
        def __init__(self,value):self.value=np.asarray(value)
        def numpy(self):return self.value

    params=SimpleNamespace(row_contact=Array([0,0,0,0,1,1,1,1,-1]),
                           row_edge=Array([0,1,2,3,0,1,2,3,-1]),
                           contact_friction=Array([[.5,.5,0,0,0],[.5,.5,0,0,0]]))
    optimizer=SimpleNamespace(compiled=SimpleNamespace(base=SimpleNamespace(contact_rows=SimpleNamespace(parameters=params))))
    frames=np.broadcast_to(np.eye(3),(1,2,3,3)).copy()
    forces=_contact_forces(optimizer,frames,np.array([[2,1,1,1,10,10,10,10,999]]),np.array([[True,False]]))
    np.testing.assert_allclose(forces,[[[5,.5,0],[0,0,0]]])
    both=_contact_forces(optimizer,frames,np.array([[2,1,1,1,10,10,10,10,999]]),np.array([[True,True]]))
    np.testing.assert_allclose(both,[[[5,.5,0],[40,0,0]]])

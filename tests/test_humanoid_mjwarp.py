"""Independent checks for the production MJWarp dynamics migration."""
import ast
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import warp as wp
import mujoco_warp as mjwarp

from retarget.humanoid import HumanoidProblem, ProjectionADMM, Settings
from retarget.humanoid_mjwarp import MJWarpHumanoid
from run_humanoid import task_paths

ROOT = Path(__file__).resolve().parents[1]
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_humanoid_runtime_has_no_native_dynamics_imports():
    assert not (ROOT/'retarget/humanoid_native.py').exists()
    forbidden = ('comfree_warp.native_adjoint', 'forceaware.optimizer', 'retarget.humanoid_native')
    for path in (ROOT/'retarget').glob('humanoid*.py'):
        tree = ast.parse(path.read_text())
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(name.startswith(forbidden) for name in imports), path


def problem(**overrides):
    values = dict(cost_mode='spider', horizon=2, knot_stride=1, multistarts=2,
                  mjwarp_fd_batch=16, contact_samples=3)
    values.update(overrides)
    return HumanoidProblem(*task_paths(ROOT/'data/spider', 'humanoid-object'), settings=Settings(**values))


@cuda
def test_execution_matches_independent_mjwarp_steps_and_keeps_warmstart():
    p = problem(action_iterations=1)
    solver = ProjectionADMM(p)
    initial = p.initial_state(2)
    rollout, history = solver.solve(initial, [3], p.aref[2])
    b = solver.backend
    assert isinstance(b, MJWarpHumanoid)
    assert not hasattr(b, 'compiled') and not hasattr(b, '_backward')
    with wp.ScopedDevice(b.device):
        model = mjwarp.put_model(p.model)
        data = mjwarp.put_data(p.model, initial, nworld=1, nconmax=100, njmax=350)
        for control in rollout['controls_dense']:
            data.ctrl.assign(control[None].astype(np.float32))
            mjwarp.step(model, data)
    np.testing.assert_allclose(data.qpos.numpy()[0], rollout['next_state'].qpos, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(data.qvel.numpy()[0], rollout['next_state'].qvel, atol=2e-4, rtol=2e-4)
    selected = history[-1]['selected']
    np.testing.assert_array_equal(rollout['next_state'].qacc_warmstart,
                                  b.warmstart_history[p.settings.substeps].numpy()[selected])
    b.prepare(rollout['next_state'], 4, solver.actions[0])
    np.testing.assert_array_equal(b.warmstart0.numpy()[0], rollout['next_state'].qacc_warmstart)
    np.testing.assert_array_equal(data.qfrc_applied.numpy(), 0.)


@cuda
def test_central_gradient_matches_independent_coordinate_probes():
    p = problem(horizon=1)
    b = MJWarpHumanoid(p)
    b.prepare(p.initial_state(2), 3, p.aref[2])
    z = np.broadcast_to(b.anchors[:, None], b.targets.z.shape+(3,)).copy()
    z[:, 1, :, 0] += .001
    b.targets.z.assign(z.astype(np.float32))
    x = np.zeros(b.variables.shape); x[1] = .02
    _, gradient = b.evaluate(x, True)
    before = b.stages[1].job.qpos.numpy().copy()
    h = b.settings.mjwarp_fd_step
    for coordinate in (0, p.model.nu-1):
        plus, minus = x.copy(), x.copy()
        plus[:, coordinate] += h; minus[:, coordinate] -= h
        fp = b.evaluate(plus, False)[0]
        fm = b.evaluate(minus, False)[0]
        np.testing.assert_allclose(gradient[:, coordinate], (fp-fm)/(2*h), rtol=.005, atol=2e-4)
    b.evaluate(x, False)
    np.testing.assert_allclose(b.stages[1].job.qpos.numpy(), before, rtol=0, atol=1e-7)


@cuda
def test_observation_does_not_mutate_simulator_state():
    p = problem(horizon=1, multistarts=1)
    b = MJWarpHumanoid(p)
    b.prepare(p.initial_state(2), 3, p.aref[2])
    b.evaluate(np.zeros(b.variables.shape), False)
    state = b.data.qpos.numpy().copy()
    velocity = b.data.qvel.numpy().copy()
    b.diagnose(); b.snapshots()
    np.testing.assert_array_equal(b.data.qpos.numpy(), state)
    np.testing.assert_array_equal(b.data.qvel.numpy(), velocity)

"""Check the strict CLI problem against the pinned upstream implementation."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import warp as wp
import yaml

from retarget.humanoid_spider_exact import SpiderTask, SpiderADMM
from retarget.spider_source import SOURCE, modules

ROOT = Path(__file__).resolve().parents[1]
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_api_adapter_keeps_original_source_functions():
    up = modules()
    from spider.simulators import mjwp
    for name in ('setup_mj_model', 'setup_env', 'step_env', 'get_reward',
                 'get_terminal_reward', 'load_env_params', 'save_state', 'load_state', 'sync_env'):
        assert getattr(up.mjwp, name).__code__ is getattr(mjwp, name).__code__


def test_missing_sprint_contact_labels_are_not_silently_disabled():
    up = modules()
    values = yaml.safe_load((SOURCE/'examples/config/override/humanoid.yaml').read_text())
    values.update(dataset_dir=str(ROOT/'data/spider'), device='cpu')
    with pytest.raises(ValueError, match='contact_site_ids not found|contact data not found'):
        c = up.config.process_config(up.config.Config(**values))
        up.io.load_data(c, c.data_path)


@pytest.fixture(scope='module')
def task():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.set_num_threads(1)
    return SpiderTask('humanoid-object', ROOT/'data/spider', batch=32)


@cuda
def test_reference_initialization_and_physics_parameters_match_source(task):
    c, up = task.config, task.up
    assert (c.horizon_steps, c.ctrl_steps, c.knot_steps, c.ref_steps) == (48, 6, 6, 2)
    assert c.sim_dt == .0166667 and c.ref_dt == .0333333
    assert c.num_dr == c.num_dyn == 1
    assert c.contact_rew_scale == c.vel_rew_scale == 0
    assert c.rot_rew_scale == .3 and c.terminal_rew_scale == 1
    assert task.env_params == [dict(pair_margin=-.005, xy_offset=-.005)]
    assert not c.terminate_resample
    with torch.cuda.stream(task.stream):
        refs = up.io.load_data(c, c.data_path)
        for actual, expected in zip(task.refs, refs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        raw = np.load(c.data_path)
        assert len(task.refs[0]) == len(raw['qpos'])*2+48+6
        expected = up.interp.interp(torch.as_tensor(raw['qpos'], device=c.device,
            dtype=torch.float32)[None], 2)[0]
        torch.testing.assert_close(task.refs[0][:-54], expected, rtol=0, atol=0)
        torch.testing.assert_close(task.nominal, refs[2][:48], rtol=0, atol=0)
    model = up.mjwp.setup_mj_model(c)
    for name in ('pair_margin', 'pair_friction', 'pair_solref', 'pair_solimp',
                 'body_mass', 'body_inertia', 'actuator_gainprm', 'actuator_biasprm'):
        np.testing.assert_array_equal(getattr(task.env.model_cpu, name), getattr(model, name))
    assert task.env.model_cpu.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.iterations == 5 and model.opt.ls_iterations == 10
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = [r[0].cpu().numpy() for r in task.refs[:3]]
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(task.env.data_wp.qpos.numpy()[0], data.qpos, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(task.env.data_wp.qvel.numpy()[0], data.qvel, atol=1e-6, rtol=1e-6)
    assert task.first_step == 0
    assert float(task.env.data_wp.time.numpy()[0]) == pytest.approx(c.sim_dt)


@cuda
def test_dense_control_parameterization_matches_upstream_sampling_without_extra_clamp(task):
    with torch.cuda.stream(task.stream):
        c = task.config
        x = torch.arange(2*task.knots*c.nu, device=task.device, dtype=torch.float64).reshape(2, -1)/10
        actual = task.controls(x)
        expected = task.nominal+task.up.interp.interp(x.float().reshape(2, task.knots, c.nu)*.05, c.knot_steps)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The source applies raw actuator controls; MuJoCo handles saturation.
        assert bool((task.controls(torch.full_like(x, 10000)) >
                     torch.as_tensor(task.env.model_cpu.actuator_ctrlrange[:, 1], device=task.device)).all())
        original = task.nominal.clone()
        plan = actual[0]
        task.shift(plan, c.ctrl_steps)
        torch.testing.assert_close(task.nominal, torch.cat((plan[6:], task.refs[2][48:54])), rtol=0, atol=0)
        task.nominal, task.first_step = original, 0


@cuda
def test_full_horizon_cost_matches_upstream_rewards_on_actual_recorded_states(task):
    c, mj = task.config, task.up.mjwp
    with torch.cuda.stream(task.stream):
        initial_q = wp.to_torch(task.env.data_wp.qpos).clone()
        initial_v = wp.to_torch(task.env.data_wp.qvel).clone()
        x = torch.zeros((2, task.knots*c.nu), device=task.device, dtype=torch.float64)
        x[1] = .1
        actual = task.score(x)
        # Score the recorded states with the source reward independently of
        # make_rollout_fn's reduction and the ADMM callback.
        cumulative = torch.zeros(task.batch, device=task.device)
        for step in range(1, c.horizon_steps+1):
            fake = SimpleNamespace(data_wp=SimpleNamespace(qpos=wp.from_torch(task.q[step]),
                                                         qvel=wp.from_torch(task.v[step])))
            refs = tuple(r[step] for r in task.refs)
            reward = mj.get_terminal_reward if step == c.horizon_steps else mj.get_reward
            cumulative += reward(c, fake, refs)[0]
        torch.testing.assert_close(actual, (-cumulative/c.horizon_steps)[:2].double(), rtol=0, atol=0)
        torch.testing.assert_close(wp.to_torch(task.env.data_wp.qpos), initial_q, rtol=0, atol=0)
        torch.testing.assert_close(wp.to_torch(task.env.data_wp.qvel), initial_v, rtol=0, atol=0)
        assert np.all(task.env.model_wp.pair_margin.numpy() == 0)


@cuda
def test_execution_and_warmstart_match_original_step_and_sync(task):
    mj, c = task.up.mjwp, task.config
    with torch.cuda.stream(task.stream):
        saved = mj.save_state(task.env)
        plan = task.nominal[:2].clone()
        q, v, warm = task.execute(plan)
        mj.load_state(task.env, saved)
        for i, control in enumerate(plan):
            mj.step_env(c, task.env, control)
            np.testing.assert_allclose(task.env.data_wp.qpos.numpy()[0], q[i], atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(task.env.data_wp.qvel.numpy()[0], v[i], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(task.env.data_wp.qacc_warmstart.numpy()[0], warm[i], atol=2e-3, rtol=2e-4)
        mj.load_state(task.env, saved)


@cuda
def test_admm_reports_pure_upstream_cost_and_preserves_feasible_control_space(task):
    args = SimpleNamespace(device=task.device, adamw_lr=.3, adamw_weight_decay=.01,
        adamw_persist=True, multistarts=2, multistart_noise_std=None, seed=1,
        admm_iterations=2, action_iterations=2, mjwarp_fd_step=.01, rho=1.)
    solver = SpiderADMM(task, args)
    plan, record = solver.solve()
    assert plan.shape == (48, task.config.nu)
    assert np.isfinite(record['task_cost_by_candidate']).all()
    assert record['task_cost'] == min(record['task_cost_by_candidate'])
    assert len(record['iterations']) == 2
    assert all(o['continued_within_window'] for o in record['iterations'][1]['block_optimizers'])
    assert all(h['split'] == 'symmetric_upstream_objective_control_consensus' for h in record['iterations'])
    assert all(sum(o['iterations'] for o in h['block_optimizers']) == args.action_iterations for h in record['iterations'])
    # Final ranking uses F rather than augmented values, with the retained
    # optimizer state belonging to that selected physical candidate.
    index = record['selected_start']
    for optimizer in solver.optimizers:
        torch.testing.assert_close(optimizer.memory[0], optimizer._cuda_window_memory[0][index])

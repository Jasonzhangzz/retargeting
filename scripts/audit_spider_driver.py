#!/usr/bin/env python3
"""Audit the local adapter against the actual pinned upstream MPC main body.

Only optimizer/viewer/output callbacks are injected into the upstream function.
The main AST body, loader, simulator and reward code are left unchanged.
This is a same-control audit, not a test of solver performance.
"""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch
import warp as wp
import yaml

from retarget.humanoid_spider_exact import SpiderTask
from retarget.spider_source import SOURCE, COMMIT, modules


def audit(output, windows=3, batch=64):
    torch.set_num_threads(1)
    up = modules()
    values = yaml.safe_load((SOURCE/'examples/config/override/humanoid_object.yaml').read_text())
    values.update(dataset_dir=str(ROOT/'data/spider'), num_samples=batch,
                  show_viewer=False, save_video=False, save_config=False,
                  save_info=False, sanity_check_seconds=0., use_torch_compile=False,
                  num_trace_uniform_samples=0, num_trace_topk_samples=0)
    c = up.config.Config(**values)
    tree = ast.parse((SOURCE/'examples/run_mjwp.py').read_text())
    names = {'main', '_assert_object_actuator_gains_zero', '_get_bimanual_hand_indices',
             '_apply_noise_mask', '_save_config_yaml'}
    functions = ast.Module(body=[deepcopy(n) for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    for n in functions.body:
        n.decorator_list = []
    scope = dict(vars(up.mjwp))
    scope.update(Config=up.config.Config, process_config=up.config.process_config,
                 load_data=up.io.load_data, get_slice=up.interp.get_slice,
                 make_rollout_fn=up.sampling.make_rollout_fn, time=time)
    baseline, environment = [], []
    rng = torch.Generator(device=c.device).manual_seed(731)
    # Solver inputs are generated once, then passed identically to both drivers.
    probes = torch.randn((windows, batch, 8*29), generator=rng,
                         device=c.device, dtype=torch.float64)*.03

    def setup_env(config, refs):
        env = up.mjwp.setup_env(config, refs)
        environment.append(env)
        return env

    def optimize_factory(rollout):
        def optimize(config, env, nominal, refs):
            index = len(baseline)
            x = probes[index]
            plan = nominal+up.interp.interp(x.float().reshape(batch, 8, config.nu)*.05, config.knot_steps)
            _, reward, _, _ = rollout(config, env, plan.clone(), refs, config.env_params_list[0][0])
            first_score = (-reward).double().cpu().numpy().copy()
            _, repeated, _, _ = rollout(config, env, plan.clone(), refs, config.env_params_list[0][0])
            baseline.append(dict(nominal=nominal.cpu().numpy().copy(),
                refs=[r.cpu().numpy().copy() for r in refs],
                score=first_score,
                self_repeat_max_abs=float(np.max(np.abs(first_score-(-repeated).double().cpu().numpy()))),
                plan=plan[0].cpu().numpy().copy()))
            return plan[0], dict(opt_steps=np.array([0]))
        return optimize

    class Viewer:
        def __enter__(self):
            self.calls = 0
            return self

        def __exit__(self, *args):
            pass

        def is_running(self):
            self.calls += 1
            return self.calls <= windows

    def update_viewer(*args):
        env = environment[0]
        baseline[-1].update(qpos=env.data_wp.qpos.numpy().copy(),
            qvel=env.data_wp.qvel.numpy().copy(),
            warmstart=env.data_wp.qacc_warmstart.numpy().copy(),
            physical_time=env.data_wp.time.numpy().copy(),
            logical_time=float(args[3].time))

    scope.update(setup_env=setup_env, setup_viewer=lambda *args: Viewer,
                 setup_renderer=lambda *args: None, update_viewer=update_viewer,
                 log_frame=lambda *args, **kwargs: None,
                 make_optimize_once_fn=lambda rollout: rollout,
                 make_optimize_fn=optimize_factory,
                 compute_object_tracking_error=lambda *args: dict(obj_pos_err=0., obj_quat_err=0.))
    exec(compile(functions, str(SOURCE/'examples/run_mjwp.py'), 'exec'), scope)
    # Use the same WP/Torch stream for both sides; only the display and reporting
    # functions above are stubs. The upstream main's entire control loop runs.
    stream = torch.cuda.ExternalStream(wp.get_stream(c.device).cuda_stream, device=c.device)
    stream.wait_stream(torch.cuda.current_stream(c.device))
    with torch.no_grad(), torch.cuda.stream(stream):
        scope['main'](c)

    task = SpiderTask('humanoid-object', ROOT/'data/spider', batch=batch)
    checks = []
    with torch.no_grad(), torch.cuda.stream(task.stream):
        for index, expected in enumerate(baseline):
            nominal_delta = float(np.max(np.abs(task.nominal.cpu().numpy()-expected['nominal'])))
            refs_equal = all(np.array_equal(r[task.first_step+1:task.first_step+49].cpu().numpy(), ref)
                             for r, ref in zip(task.refs, expected['refs']))
            state_names = ('qpos', 'qvel', 'ctrl', 'qacc', 'qacc_warmstart', 'time',
                           'qfrc_applied', 'xfrc_applied')
            initial = {name:getattr(task.env.data_wp, name).numpy().copy() for name in state_names}
            actual = task.score(probes[index]).cpu().numpy()
            repeated = task.score(probes[index]).cpu().numpy()
            restored = {name:bool(np.array_equal(getattr(task.env.data_wp, name).numpy(), value,
                                                equal_nan=True)) for name, value in initial.items()}
            controls = task.controls(probes[index, :1])[0]
            control_delta = float(np.max(np.abs(controls.cpu().numpy()-expected['plan'])))
            task.execute(controls[:6]); task.shift(controls, 6)
            difference = lambda name, key: float(np.max(np.abs(
                getattr(task.env.data_wp, name).numpy()-expected[key])))
            checks.append(dict(window=index, nominal_max_abs=nominal_delta,
                references_exact=refs_equal, controls_max_abs=control_delta,
                objective_max_abs=float(np.max(np.abs(actual-expected['score']))),
                upstream_self_repeat_cost_max_abs=expected['self_repeat_max_abs'],
                adapter_self_repeat_cost_max_abs=float(np.max(np.abs(actual-repeated))),
                executed_qpos_max_abs=difference('qpos', 'qpos'),
                executed_qvel_max_abs=difference('qvel', 'qvel'),
                executed_warmstart_max_abs=difference('qacc_warmstart', 'warmstart'),
                physical_time_max_abs=difference('time', 'physical_time'),
                source_logical_time=expected['logical_time'], local_logical_step=task.first_step,
                state_restored_exact=restored))
    source = asdict(c); local = asdict(task.config)
    differences = {key:dict(upstream=source[key], adapter=local[key]) for key in source
        if key not in ('noise_scale', 'env_params_list') and source[key] != local[key]}
    # The upstream sanity check is deliberately tested separately at its default
    # 3 s setting. It never modifies the GPU environment.
    sanity_config = up.config.Config(**{**values, 'sanity_check_seconds':3.})
    sanity_config = up.config.process_config(sanity_config)
    model = up.mjwp.setup_mj_model(sanity_config)
    sanity = dict(seconds=3., passed=False, error=None)
    try:
        up.mjwp._initial_state_sanity_check(model, mujoco.MjData(model),
            *task.refs[:3], sanity_config)
        sanity['passed'] = True
    except Exception as error:
        sanity['error'] = f'{type(error).__name__}: {error}'
    result = dict(source_commit=COMMIT, windows=windows, worlds=batch,
        upstream_main_body_unmodified=True,
        injected_callbacks=['fixed-control optimizer', 'viewer', 'output-only tracking metric'],
        shared_overrides=['headless/output flags', 'sanity_check_seconds=0',
                          'use_torch_compile=False', 'num_samples', 'trace sample counts'],
        configuration_differences=differences, checks=checks, upstream_default_sanity_check=sanity)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    audit(args.output, args.windows, args.batch)

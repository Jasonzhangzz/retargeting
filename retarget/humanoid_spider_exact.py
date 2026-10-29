"""The upstream SPIDER problem with an equivalent ADMM control-variable split.

min (F(x)+F(z))/2 subject to x=z has precisely the same feasible controls
and objective as min F(x). Both ADMM blocks evaluate actual upstream physics;
contact anchors/patches are not constraints in this mode.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import warp as wp
import yaml

from retarget.fa.adamw import AdamW
from retarget.spider_source import SOURCE, COMMIT, modules, engine_version_audit


def write_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


class SpiderTask:
    """One shared upstream loader, simulator, rollout, reward and interpolation."""
    def __init__(self, task, data_root, device='cuda:0', batch=64, scene=None, reference=None):
        self.up = modules()
        name = 'humanoid_object' if task == 'humanoid-object' else 'humanoid'
        values = yaml.safe_load((SOURCE/f'examples/config/override/{name}.yaml').read_text())
        values.update(dataset_dir=str(Path(data_root).resolve()), device=device,
                      num_samples=batch, use_torch_compile=False, show_viewer=False,
                      save_video=False, save_config=False, sanity_check_seconds=0.,
                      num_trace_uniform_samples=0, num_trace_topk_samples=0)
        self.config = c = self.up.config.process_config(self.up.config.Config(**values))
        if scene is not None:
            c.model_path = str(Path(scene).resolve())
        if reference is not None:
            c.data_path = str(Path(reference).resolve())
        model = self.up.mjwp.setup_mj_model(c)
        c.nq, c.nv, c.nu, c.npair = model.nq, model.nv, model.nu, model.npair
        # No automatic disabling of a configured reward when labels are absent.
        self.refs = self.up.io.load_data(c, c.data_path)
        self.device, self.batch = device, batch
        wp.init()
        self.stream = torch.cuda.ExternalStream(wp.get_stream(device).cuda_stream, device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            self.env = self.up.mjwp.setup_env(c, self.refs)
            self.q = torch.empty((c.horizon_steps+1, batch, c.nq), device=device)
            self.v = torch.empty((c.horizon_steps+1, batch, c.nv), device=device)
        torch.cuda.current_stream(device).wait_stream(self.stream)
        self.index = 0
        mj = self.up.mjwp
        self.rollout = self.up.sampling.make_rollout_fn(
            self._step, mj.save_state, mj.load_state, mj.get_reward,
            mj.get_terminal_reward, mj.get_terminate, mj.get_trace,
            mj.save_env_params, mj.load_env_params, mj.copy_sample_state)
        if c.num_dr == 0:
            self.env_params = [dict(pair_margin=0., xy_offset=0.)]
        else:
            self.env_params = [dict(pair_margin=float(m), xy_offset=float(x)) for m, x in zip(
                np.linspace(*c.pair_margin_range, c.num_dr), np.linspace(*c.xy_offset_range, c.num_dr))]
        self.first_step = 0
        self.nominal = self.refs[2][:c.horizon_steps].clone()
        self.knots = int(round(c.horizon/c.knot_dt))
        self.control_scale = .05
        self.calls = self.trajectories = 0

    def _step(self, config, env, control):
        self.up.mjwp.step_env(config, env, control)
        self.index += 1
        self.q[self.index].copy_(wp.to_torch(env.data_wp.qpos))
        self.v[self.index].copy_(wp.to_torch(env.data_wp.qvel))

    def controls(self, x):
        c = self.config
        knots = x.float().reshape(-1, self.knots, c.nu)*self.control_scale
        # Same dense-reference plus align_corners=True knot interpolation as
        # upstream sample_ctrls. No extra actuator clamp is added here.
        return self.nominal+self.up.interp.interp(knots, c.knot_steps)

    def score(self, x):
        c = self.config
        count = len(x)
        if count > self.batch:
            return torch.cat([self.score(chunk) for chunk in x.split(self.batch)])
        padded = x[:1].expand(self.batch, -1).clone(); padded[:count] = x
        controls = self.controls(padded)
        refs = tuple(r[self.first_step+1:self.first_step+c.horizon_steps+1] for r in self.refs)
        worst_cost = torch.full((self.batch,), -torch.inf, device=self.device)
        for params in self.env_params:
            self.index = 0
            self.q[0].copy_(wp.to_torch(self.env.data_wp.qpos))
            self.v[0].copy_(wp.to_torch(self.env.data_wp.qvel))
            _, reward, _, _ = self.rollout(c, self.env, controls.clone(), refs, params)
            worst_cost = torch.maximum(worst_cost, -reward)
        self.calls += 1; self.trajectories += count
        return worst_cost[:count].double().clone()

    def execute(self, controls):
        states, velocities, warmstarts = [], [], []
        for control in controls:
            self.up.mjwp.step_env(self.config, self.env, control)
            d = self.env.data_wp
            states.append(d.qpos.numpy()[0].copy())
            velocities.append(d.qvel.numpy()[0].copy())
            warmstarts.append(d.qacc_warmstart.numpy()[0].copy())
        self.up.mjwp.sync_env(self.config, self.env, None)
        return states, velocities, warmstarts

    def shift(self, controls, executed):
        self.first_step += executed
        retained = controls[executed:]
        self.nominal = torch.cat((retained, self.refs[2][
            self.first_step+len(retained):self.first_step+self.config.horizon_steps]))


class SpiderADMM:
    """Two nontrivial action blocks, sharing the original per-round update budget."""
    def __init__(self, task, args):
        if args.action_iterations < 2:
            raise ValueError('Strict consensus ADMM needs at least 2 action updates total (one per block)')
        self.task, self.args = task, args
        self.optimizers = [AdamW(learning_rate=args.adamw_lr, weight_decay=args.adamw_weight_decay,
                                persist=args.adamw_persist, device=args.device) for _ in range(2)]
        self.window = 0

    def _update(self, variables, target, optimizer, budget, iteration):
        task, a = self.task, self.args
        w, d = variables.shape
        def evaluate(values, gradient=False):
            values = torch.as_tensor(values, dtype=torch.float64, device=a.device)
            def objective(probes, source):
                return .5*task.score(probes)+a.rho*((probes-target[source])**2).mean(1)
            value = objective(values, torch.arange(w, device=a.device))
            g = None
            if gradient:
                scores = torch.empty(2*w*d, dtype=torch.float64, device=a.device)
                for start in range(0, len(scores), task.batch):
                    ii = torch.arange(start, min(start+task.batch, len(scores)), device=a.device)
                    worlds, dimension = ii % w, ii//(2*w)
                    sign = torch.where((ii//w) % 2 == 0, 1., -1.)
                    probe = values[worlds].clone()
                    probe[torch.arange(len(ii), device=a.device), dimension] += a.mjwarp_fd_step*sign
                    scores[start:start+len(ii)] = objective(probe, worlds)
                pair = scores.reshape(d, 2, w)
                g = ((pair[:, 0]-pair[:, 1])/(2*a.mjwarp_fd_step)).T.contiguous()
            return value, g
        result, _, diag = optimizer.update(variables, evaluate, iterations=budget,
            horizon=task.knots, advance=task.config.ctrl_steps/task.config.knot_steps,
            continue_window=iteration>0, evaluate_on_device=True)
        diag.update(finite_difference_gradients=True, gradient_source='upstream_MJWarp_central_differences')
        return torch.as_tensor(result, device=a.device), diag

    def solve(self):
        task, a = self.task, self.args
        w, d = a.multistarts, task.knots*task.config.nu
        caller = torch.cuda.current_stream(a.device)
        task.stream.wait_stream(caller)
        with torch.no_grad(), torch.cuda.stream(task.stream):
            x = torch.zeros((w, d), dtype=torch.float64, device=a.device)
            sigma = a.multistart_noise_std if a.multistart_noise_std is not None else (.05 if task.config.embodiment_type == 'humanoid_object' else .003)
            rng = torch.Generator(device=a.device).manual_seed(a.seed+self.window)
            x[1:] = torch.randn((w-1, d), generator=rng, device=a.device, dtype=x.dtype)*(sigma/task.control_scale)
            z, dual = x.clone(), torch.zeros_like(x)
            history = []
            for iteration in range(a.admm_iterations):
                # .5 F(x) + .5 F(z), x=z. No new restriction on the original
                # controls; both blocks use the same source reward and physics.
                previous_z = z.clone()
                x, dx = self._update(x, z-dual, self.optimizers[0], (a.action_iterations+1)//2, iteration)
                z, dz = self._update(z, x+dual, self.optimizers[1], a.action_iterations//2, iteration)
                primal = x-z
                dual += primal
                history.append(dict(iteration=iteration, split='symmetric_upstream_objective_control_consensus',
                    primal_max_rad=float(primal.abs().max()*task.control_scale),
                    dual_change_max_rad=float((z-previous_z).abs().max()*task.control_scale),
                    total_action_updates=a.action_iterations, block_optimizers=[dx, dz]))
            # Each block is itself a valid physical control candidate for the
            # original unrestricted problem. Rank both solely by the original F.
            candidates = torch.cat((x, z))
            physical = task.score(candidates)
            valid = torch.isfinite(physical)
            if not bool(valid.any()):
                raise FloatingPointError('All upstream SPIDER task candidates are invalid')
            selected = int(torch.where(valid, physical, torch.inf).argmin())
            if a.adamw_persist:
                for optimizer in self.optimizers:
                    optimizer.retain_candidate_memory(selected % w)
            controls = task.controls(candidates[selected:selected+1])[0].clone()
            self.window += 1
        caller.wait_stream(task.stream)
        return controls, dict(iterations=history, selected=selected, selected_block=selected//w,
                              selected_start=selected % w, task_cost=float(physical[selected]),
                              task_cost_by_candidate=physical.cpu().tolist())


def run(args, scene, reference, output):
    if args.start_frame:
        raise ValueError('Strict SPIDER comparison starts at frame 0; use --cost spider-contact for custom local timing')
    if args.anchor_weight != 0:
        raise ValueError('Strict SPIDER cost has no anchor regularizer; use --anchor-weight 0')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    task = SpiderTask(args.task, args.data_root, args.device,
                      max(args.multistarts, args.mjwarp_fd_batch), scene, reference)
    c = task.config
    if (args.horizon, args.knot_stride, args.substeps) != (task.knots, c.knot_steps//c.ref_steps, c.ref_steps):
        raise ValueError('Strict SPIDER mode uses the upstream horizon/knot/reference grid; use --cost spider-contact for custom grids')
    if args.ref_dt is not None and not np.isclose(args.ref_dt, c.ref_dt, rtol=0, atol=1e-10):
        raise ValueError('Strict SPIDER mode uses the upstream ref_dt')
    if args.viewer:
        raise ValueError('Strict SPIDER comparison is headless; replay the saved trajectory to view it')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    solver = SpiderADMM(task, args)
    max_steps = c.max_sim_steps if c.max_sim_steps>0 else len(task.refs[0])-c.horizon_steps-c.ctrl_steps
    c.max_sim_steps=max_steps
    if args.frames is not None:
        max_steps = min(max_steps, args.frames*c.ref_steps)
    requested = math.ceil(max_steps/c.ctrl_steps)
    files = ['spider/simulators/mjwp.py', 'spider/optimizers/sampling.py', 'spider/io.py',
             'spider/interp.py', 'spider/config.py']
    config = {k:v for k,v in asdict(c).items() if k not in ('noise_scale', 'env_params_list')}
    write_json(output/'config.json', dict(profile='spider_source_problem', source_commit=COMMIT,
        comparison_scope='pinned SPIDER source problem on installed engine; not a full upstream environment reproduction',
        engine_version_audit=engine_version_audit(),
        upstream_environment_fully_reproduced=False,
        non_solver_differences=['initial CPU sanity precheck disabled', 'headless trace/output callbacks',
                               'saved time uses physical GPU clock rather than upstream logical display clock',
                               'dependency versions: see engine_version_audit'],
        source_sha256={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in files},
        versions=dict(mujoco=mujoco.__version__, warp=wp.__version__, torch=torch.__version__),
        upstream_config=config, solver='ADMM_symmetric_objective_control_consensus_and_shared_AdamW',
        equivalent_problem='min (F(x)+F(z))/2 subject to x=z, unrestricted controls',
        action_budget='action_iterations is total shared by both blocks per ADMM round',
        admm_iterations=args.admm_iterations, action_iterations=args.action_iterations,
        multistarts=args.multistarts, rho=args.rho, anchor_weight=0.,
        extra_task_costs=False, extra_contact_constraints=False,
        objective='exact upstream rollout reward negated; worst DR member; post-step mean',
        initial_state='upstream setup_env CPU pre-step', initialization_clock='upstream logical MPC clock starts at zero',
        control_parameterization='dense nominal plus exact upstream interp(knots); no extra clamp',
        scene_sha256=hashlib.sha256(Path(scene).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        finite_difference_gradients=True, fd_step=args.mjwarp_fd_step,
        solver_differences=['ADMM versus sampling, optimizer budget, random starts, gradient oracle, parallel batch size'],
        cost_code='spider.simulators.mjwp.get_reward/get_terminal_reward',
        physics_code='spider.simulators.mjwp.setup_env/step_env/load_env_params/sync_env'))
    d = task.env.data_wp
    qs, vs, times = [d.qpos.numpy()[0].copy()], [d.qvel.numpy()[0].copy()], [float(d.time.numpy()[0])]
    controls, warmstarts, records = [], [], []
    started = time.perf_counter()
    state, error = 'running', None
    write_json(output/'status.json', dict(state=state, completed_windows=0,
        requested_windows=requested, executed_physics_steps=0, seconds=0.))
    try:
        with (output/'iterations.jsonl').open('w') as log:
            for window in range(requested):
                tick = time.perf_counter()
                plan, record = solver.solve()
                count = min(c.ctrl_steps, max_steps-task.first_step) if args.frames is not None else c.ctrl_steps
                with torch.cuda.stream(task.stream):
                    q, v, warm = task.execute(plan[:count])
                    controls.extend(plan[:count].cpu().numpy())
                    task.shift(plan, count)
                qs.extend(q); vs.extend(v); warmstarts.extend(warm)
                last_time=times[-1]
                times.extend([last_time+(i+1)*c.sim_dt for i in range(count)])
                record.update(window=window, sim_step=task.first_step-count, seconds=time.perf_counter()-tick)
                records.append(record); log.write(json.dumps(record)+'\n'); log.flush()
                np.savez_compressed(output/'trajectory.tmp.npz', qpos=qs, qvel=vs, ctrl=controls,
                    qacc_warmstart=warmstarts, time=times, qpos_ref=task.refs[0][:len(qs)].cpu().numpy(),
                    qvel_ref=task.refs[1][:len(qs)].cpu().numpy())
                (output/'trajectory.tmp.npz').replace(output/'trajectory.npz')
                write_json(output/'status.json', dict(state='running', completed_windows=window+1,
                    requested_windows=requested, executed_physics_steps=len(controls), seconds=time.perf_counter()-started))
                print(f'[SPIDER source {window+1}/{requested}] task_cost={record["task_cost"]:.6g} seconds={record["seconds"]:.2f}', flush=True)
        actual = np.asarray(qs); target = task.refs[0][:len(qs)].cpu().numpy()
        model = task.env.model_cpu
        actual_data, reference_data = mujoco.MjData(model), mujoco.MjData(model)
        torso = model.body('torso_link').id
        tilt = []
        for q, r in zip(actual, target):
            actual_data.qpos[:], reference_data.qpos[:] = q, r
            mujoco.mj_forward(model, actual_data); mujoco.mj_forward(model, reference_data)
            tilt.append(float(np.degrees(np.arccos(np.clip(actual_data.xmat[torso].reshape(3,3)[:,2] @ reference_data.xmat[torso].reshape(3,3)[:,2], -1, 1)))))
        failure = (target[:,2]-actual[:,2]>.18) | (np.asarray(tilt)>60)
        summary = dict(completed_windows=len(records), executed_physics_steps=len(controls),
            root_position_rms_m=float(np.sqrt(np.mean(np.sum((actual[:,:3]-target[:,:3])**2, axis=1)))),
            balance_failure_frames=int(failure.sum()), first_balance_failure_frame=int(np.flatnonzero(failure)[0]) if failure.any() else None,
            seconds=time.perf_counter()-started, profile='spider_source_problem', task_success_claimed=False)
        if c.embodiment_type == 'humanoid_object':
            summary.update(box_position_rms_m=float(np.sqrt(np.mean(np.sum((actual[:,-7:-4]-target[:,-7:-4])**2, axis=1)))),
                           box_final_error_m=float(np.linalg.norm(actual[-1,-7:-4]-target[-1,-7:-4])))
        write_json(output/'summary.json', summary)
        state = 'completed'
    except BaseException as exc:
        state, error = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(output/'status.json', dict(state=state, error=error, completed_windows=len(records),
            requested_windows=requested, executed_physics_steps=len(controls), seconds=time.perf_counter()-started))
    return output

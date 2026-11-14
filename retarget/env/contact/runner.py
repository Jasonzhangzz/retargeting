"""Closed-loop GPU proposal search with native MuJoCo 4 selection/execution."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import traceback
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import torch
import warp as wp
from retarget.common.results import clean_json, write_json
from retarget.solver.base import solve
from retarget.solver.contact_cem import ContactProblem, DEFAULTS
from retarget.task.contact.tasks import PROFILES
from retarget.task.contact.robokit_tasks import Task
from retarget.task.contact.metrics import assess
from .native import snapshot
from .client import NativeClient
from .robokit_rollout import GPURollout
from .robokit_planner import ControlPlanner
from .contact_terms import ContactDynamicsTask
from .strategy import ContactStrategy
from .robokit_nominal import SourceNominalPlanner
from .objective import force_penalty_parameters


def validate(config) -> Path:
    if config.task.get('name') not in PROFILES:
        raise ValueError(f'contact task.name must be one of {list(PROFILES)}')
    if config.solver.type not in ('cem', 'contact_cem'):
        raise ValueError('Contact tasks require cem or contact_cem')
    if config.task.get('options', {}).get('baseline') not in (None, 'source_sampling'):
        raise ValueError('Only the source_sampling baseline is supported')
    if config.task.get('options', {}).get('baseline') and config.solver.type != 'cem':
        raise ValueError('source_sampling requires solver.type: cem')
    frames = config.environment.get('frames')
    if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 1):
        raise ValueError('environment.frames must be a positive number of physics steps')
    if not str(config.device).startswith('cuda'):
        raise ValueError('Contact benchmark planning requires a CUDA GPU')
    return Path(config.task['data'])


def run(config, output: Path) -> list:
    validate(config)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[3]
    implementation_files = [*root.joinpath('retarget/env/contact').rglob('*.py'),
        *root.joinpath('retarget/task/contact').rglob('*.py'), root / 'retarget/solver/contact_cem.py',
        root / 'retarget/solver/contact_defaults.py',
        root / 'retarget/solver/robokit_cem.py', root / 'third_party/robokit/SOURCE.json', root / 'scripts/run_contact_benchmark.py']
    implementation_hashes = {}
    for path in implementation_files:
        relative = path.relative_to(root)
        destination = output / 'implementation' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        implementation_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / 'implementation_sha256.json', implementation_hashes)
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    task = Task(config.task['name'], config.seed, str(config.device))
    p, model = task.profile, task.model
    options = config.task.get('options', {})
    method = options.get('baseline', config.solver.type)
    extra = method != 'source_sampling'
    initial = snapshot(task.state)
    reference = options.get('physics_reference')
    if reference:
        if task.name != 'h1_jog':
            raise ValueError('Reference tracking is supported for H1 only')
        with np.load(reference, allow_pickle=False) as raw:
            if 'timestep' not in raw or not np.isclose(float(raw['timestep']), model.opt.timestep, rtol=0., atol=1e-12):
                raise ValueError('Frozen reference timestep must match the H1 simulation timestep')
            task.reference = {k: raw[k].copy() for k in ('qpos', 'qvel', 'ctrl', 'joint_names')}
            if 'sensordata' in raw:
                task.reference['sensordata'] = raw['sensordata'].copy()
        expected = [model.joint(int(j)).name for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)]
        shapes = {'qpos': (p.steps + 1, model.nq), 'qvel': (p.steps + 1, model.nv), 'ctrl': (p.steps, model.nu)}
        if task.reference['joint_names'].tolist() != expected or any(task.reference[k].shape != shape for k, shape in shapes.items()):
            raise ValueError('Reference/model joint names or dimensions do not match')
        if not all(np.isfinite(task.reference[k]).all() for k in ('qpos', 'qvel', 'ctrl')):
            raise ValueError('Nonfinite reference')
        if not np.allclose(np.linalg.norm(task.reference['qpos'][:, 3:7], axis=1), 1., rtol=0., atol=1e-5):
            raise ValueError('Reference root quaternions must be normalized')
        if 'sensordata' not in task.reference:
            probe = mujoco.MjData(model)
            sensors = []
            for i, (q, v) in enumerate(zip(task.reference['qpos'], task.reference['qvel'])):
                probe.qpos[:], probe.qvel[:] = q, v
                probe.ctrl[:] = task.reference['ctrl'][min(i, p.steps - 1)]
                mujoco.mj_forward(model, probe)
                sensors.append(probe.sensordata.copy())
            task.reference['sensordata'] = np.asarray(sensors)
        if task.reference['sensordata'].shape != (p.steps + 1, model.nsensordata) or not np.isfinite(task.reference['sensordata']).all():
            raise ValueError('Reference sensor trajectory does not match the H1 model')
        initial['qpos'] = task.reference['qpos'][0].copy()
        initial['qvel'] = task.reference['qvel'][0].copy()
        initial['ctrl'] = task.reference['ctrl'][0].copy()
        initial['qpos'][0] += .02
        tilt = np.array([np.cos(np.pi / 180), 0., np.sin(np.pi / 180), 0.])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, initial['qpos'][3:7], tilt)
        initial['qpos'][3:7] = quat
        rng = np.random.default_rng(config.seed)
        initial['qpos'][7:] += rng.uniform(-.02, .02, model.nq - 7)
        task.initial['qpos'] = initial['qpos'].copy()
        task.bind_reference()
    # Export source-relative meshes; neither XML depends on an external Robokit tree.
    for scene, name in ((task.scene.copy(), 'scene.xml'), (task.plan_scene.copy(), 'plan_scene.xml')):
        xml = ET.fromstring(scene.to_xml())
        compiler = xml.find('compiler')
        meshdir = Path(compiler.get('meshdir', '') if compiler is not None else '')
        for mesh in xml.findall('.//mesh'):
            if mesh.get('file'):
                path = Path(mesh.get('file'))
                if not path.is_absolute():
                    path = meshdir / path
                mesh.set('file', os.path.relpath(path.resolve(), output.resolve()))
        if compiler is not None:
            compiler.set('meshdir', '')
        (output / name).write_text(ET.tostring(xml, encoding='unicode'))
    steps = min(p.steps, config.environment.get('frames', p.steps))
    state = initial
    rows, records, contact_log, event_log = [], [], [], []
    nominal = previous = previous_ref = previous_events = None
    error = None
    cpu_evaluated = 0
    with NativeClient(output / 'scene.xml', state, output, workers=DEFAULTS.cpu_workers) as native:
        state = native.info['state']
        contacts = native.info['contacts']
        initial = {k: np.array(v, copy=True) if k != 'time' else v for k, v in state.items()}
        packet = task.packet(state, 0, contacts)
        contact_term = ContactDynamicsTask(model, p.horizon, tracking=bool(reference)) if extra else None
        rollout = (GPURollout(task, p.samples, str(config.device), contact_term) if method == 'contact_cem'
                   else ControlPlanner(task, method, str(config.device), contact_term))
        strategy = ContactStrategy(task, rollout) if method == 'contact_cem' else None
        # Re-optimize a whole-body nominal with the source MPC. In tracking,
        # this optimizer reads only frozen targets, never generator internals.
        nominal_planner = (SourceNominalPlanner(task, str(config.device))
            if method == 'contact_cem' and task.name == 'h1_jog' else None)
        write_json(output / 'runtime.json', dict(task=task.metadata(), method=method, defaults=asdict(DEFAULTS),
            objective=dict(force_scale_n=force_penalty_parameters(model)[0],
                force_peak_weight=force_penalty_parameters(model)[1],
                state_stage='integration_input', gpu_aggregation='horizon_mean', cpu4_aggregation='horizon_sum'),
            nominal_solver=(dict(backend='robokit.SimMPC', samples=nominal_planner.num_samples,
                stages=[asdict(stage) for stage in nominal_planner.config.stages],
                initial_iterations=nominal_planner.config.init_iterations,
                selection='best_scored_incumbent' if reference else 'source_mean_or_best_draw',
                target='frozen_artifact' if reference else 'source_task_commands') if nominal_planner else None),
            execution_backend=dict(mujoco=native.info['version'], speculative=True, cpu_workers=native.info['workers']),
            proposal_backend=dict(mujoco=mujoco.__version__, speculative=False, device=config.device),
            source=json.loads(Path(__file__).parents[2].joinpath('task/contact/SOURCE.json').read_text()),
            versions={k: importlib.metadata.version(k) for k in ('torch', 'mujoco', 'mujoco-warp', 'warp-lang', 'numpy', 'scipy')},
            robokit=json.loads((root / 'third_party/robokit/SOURCE.json').read_text()),
            gpu=torch.cuda.get_device_name(config.device), reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest() if reference else None))
        startup_seconds = time.perf_counter() - started
        try:
            for clock in range(0, steps, p.ctrl_steps):
                preparation_started = time.perf_counter()
                packet = task.packet(state, clock, contacts)
                nominal_started = time.perf_counter()
                nominal_before = nominal_planner.trajectories if nominal_planner else 0
                if nominal_planner is not None:
                    nominal = nominal_planner.propose(state, clock, packet)
                elif nominal is None:
                    nominal = packet['ctrl_ref'].copy()
                else:
                    n = p.ctrl_steps
                    shifted = np.concatenate([previous[n:] - previous_ref[n:], np.zeros((n, model.nu))], axis=0)
                    nominal = np.clip(packet['ctrl_ref'] + shifted, *task.ctrl_range.T)
                if strategy is not None:
                    strategy.nominal = nominal
                nominal_seconds = time.perf_counter() - nominal_started
                branches = strategy.branches(state, packet) if strategy is not None else []
                iterations = p.initial_iterations if clock == 0 and p.initial_iterations else p.iterations
                phases = dict(reference_and_candidates=time.perf_counter() - preparation_started,
                              contact_lm=0., gpu_rollout=0., cpu4_validation=0.)
                tick = time.perf_counter()
                gpu_before = rollout.trajectories
                def decode(parameters: np.ndarray, branch: int) -> tuple:
                    before = time.perf_counter()
                    try:
                        return strategy.decode(parameters, branch)
                    finally:
                        phases['contact_lm'] += time.perf_counter() - before
                def evaluate(controls: np.ndarray, events: list, coarse: bool = False) -> np.ndarray:
                    before = time.perf_counter()
                    try:
                        return rollout.evaluate(controls, state, packet, events, extra=extra,
                            coarse=(p.coarse_fraction, p.coarse_stride) if coarse else (1., 1))
                    finally:
                        phases['gpu_rollout'] += time.perf_counter() - before
                def verify(controls: np.ndarray, events: list) -> list:
                    nonlocal cpu_evaluated
                    before = time.perf_counter()
                    checked = native.call('evaluate', controls=controls[:, :p.horizon], state=state,
                                          packet=packet, events=events, dynamics_cost=extra)
                    phases['cpu4_validation'] += time.perf_counter() - before
                    cpu_evaluated += len(checked)
                    with (output / 'native_validations.jsonl').open('a') as log:
                        for i, item in enumerate(checked):
                            log.write(json.dumps(clean_json(dict(step=clock, candidate=i,
                                **{k: v for k, v in item.items() if k not in ('qpos', 'qvel')})), allow_nan=False) + '\n')
                    return checked
                if method == 'contact_cem':
                    elapsed = p.ctrl_steps * packet['dt']
                    retained_events = [dict(e, start_time=max(0., e['start_time'] - elapsed),
                        duration=e['duration'] + min(0., e['start_time'] - elapsed))
                        for e in previous_events or [] if e['start_time'] + e['duration'] > elapsed]
                    retained_controls = None
                    if previous is not None:
                        retained_controls = (np.concatenate([previous[p.ctrl_steps:], packet['ctrl_ref'][-p.ctrl_steps:]])
                            if task.name == 'h1_jog' else nominal)
                    problem = ContactProblem(branches, decode, evaluate, verify, p.samples, iterations,
                        retained_controls, retained_events, str(config.device), retain_nominal=task.name == 'h1_jog')
                    result = solve(problem, config.solver, config.seed + clock)
                else:
                    before_gpu = rollout.gpu_seconds
                    result = rollout.solve(state, clock, packet, [], verify)
                    phases['gpu_rollout'] += rollout.gpu_seconds - before_gpu
                chosen = result.metadata['controls']
                solve_seconds = time.perf_counter() - tick
                phases['search_other'] = max(0., solve_seconds - sum(phases[k] for k in
                    ('contact_lm', 'gpu_rollout', 'cpu4_validation')))
                n = min(p.ctrl_steps, steps - clock)
                tick = time.perf_counter()
                executed = native.call('execute', controls=chosen[:n])
                rows.extend(executed)
                state, contacts = executed[-1]['state'], executed[-1]['contacts']
                previous = chosen.copy()
                previous_events = result.metadata['events']
                previous_ref = packet['ctrl_ref'].copy()
                cpu = result.metadata['cpu4']
                event_log.append(dict(step=clock, branch=result.metadata.get('branch'), events=result.metadata['events'],
                    predictions=cpu.get('predictions', []), realized_predictions=cpu.get('realized_predictions', []),
                    first_hits=cpu.get('first_hits', {})))
                records.append(dict(step=clock, cost=result.cost, solve_seconds=solve_seconds,
                    planning_seconds=solve_seconds + phases['reference_and_candidates'],
                    nominal_seconds=nominal_seconds, nominal_gpu_rollouts=(nominal_planner.trajectories - nominal_before if nominal_planner else 0),
                    nominal_selected=result.metadata.get('nominal_selected'),
                    nominal_control_delta_rms=float(np.sqrt(np.mean((chosen - nominal)**2))) if strategy else None,
                    phase_seconds=phases,
                    execute_seconds=time.perf_counter() - tick, history=result.history, parts=cpu['parts'],
                    gpu_rollouts=rollout.trajectories - gpu_before, cpu4_rollouts=result.metadata['cpu4_evaluated']))
                with (output / 'iterations.jsonl').open('a') as f:
                    f.write(json.dumps(clean_json(records[-1]), allow_nan=False) + '\n')
                write_json(output / 'status.json', dict(state='running', completed_steps=len(rows), requested_steps=steps))
                if clock == 0 or (clock + n) // max(1, round(1 / model.opt.timestep)) != clock // max(1, round(1 / model.opt.timestep)):
                    print(f'[{task.name}/{method}/seed={config.seed}] {clock + n}/{steps} cost={result.cost:.5g} solve={solve_seconds:.3f}s', flush=True)
        except Exception as exc:
            rows.extend(getattr(exc, 'partial_rows', []))
            error = traceback.format_exc()
            (output / 'failure.txt').write_text(error)
    qs = np.asarray([r['state']['qpos'] for r in rows]).reshape(-1, model.nq)
    vs = np.asarray([r['state']['qvel'] for r in rows]).reshape(-1, model.nv)
    controls = np.asarray([r['state']['ctrl'] for r in rows]).reshape(-1, model.nu)
    sensors = np.asarray([r['sensordata'] for r in rows]).reshape(-1, model.nsensordata)
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    trajectory = dict(qpos=qs, qvel=vs, ctrl=controls, sensordata=sensors)
    np.savez_compressed(output / 'trajectory.npz', **trajectory,
        time=np.asarray([r['state']['time'] for r in rows]), initial_qpos=initial['qpos'], initial_qvel=initial['qvel'],
        timestep=model.opt.timestep, joint_names=[model.joint(int(j)).name for j in joint_ids],
        actuator_names=[model.actuator(j).name for j in range(model.nu)],
        qacc_warmstart=np.asarray([r['state']['qacc_warmstart'] for r in rows]),
        actuator_force=np.asarray([r['actuator_force'] for r in rows]))
    for name, data in (('events.jsonl', event_log), ('contacts.jsonl', [dict(step=i, contacts=r['contacts']) for i, r in enumerate(rows)])):
        with (output / name).open('w') as f:
            for row in data:
                f.write(json.dumps(clean_json(row), allow_nan=False) + '\n')
    complete = len(rows) == p.steps and error is None
    summary = assess(task, trajectory, rows, complete)
    summary.update(method=method, seed=config.seed, requested_steps=steps, completed_steps=len(rows),
        total_seconds=time.perf_counter() - started, startup_seconds=startup_seconds,
        first_solve_seconds=records[0]['solve_seconds'] if records else None,
        cold_start_to_first_control_seconds=(startup_seconds + records[0]['phase_seconds']['reference_and_candidates']
            + records[0]['solve_seconds']) if records else None,
        steady_solve_mean_seconds=float(np.mean([r['solve_seconds'] for r in records[1:]])) if len(records) > 1 else None,
        steady_planning_mean_seconds=float(np.mean([r['planning_seconds'] for r in records[1:]])) if len(records) > 1 else None,
        gpu_rollouts=rollout.trajectories, cold_gpu_rollouts=rollout.cold_rollouts, cpu4_rollouts=cpu_evaluated,
        nominal_backend=('original_robokit_SimMPC' if nominal_planner else 'frozen_reference' if reference else None),
        nominal_gpu_rollouts=nominal_planner.trajectories if nominal_planner else 0,
        nominal_cold_gpu_rollouts=nominal_planner.cold_rollouts if nominal_planner else 0,
        nominal_seconds=sum(r['nominal_seconds'] for r in records),
        nominal_selected_windows=sum(bool(r.get('nominal_selected')) for r in records),
        runtime_error=error, parameters=task.metadata(), speculative_execution=True, cpu_workers=native.info['workers'],
        rollout_budget_scope='GPU proposals and CPU4 validations counted separately; LM and prediction time included in solve_seconds')
    write_json(output / 'summary.json', summary)
    if task.name == 'h1_jog' and method == 'source_sampling' and complete:
        # Independent frozen reference includes the initial state; playback/tracking
        # may load only this artifact, never the source planner's internal plan.
        probe = mujoco.MjData(model)
        probe.qpos[:], probe.qvel[:] = initial['qpos'], initial['qvel']
        probe.ctrl[:] = initial['ctrl']
        mujoco.mj_forward(model, probe)
        np.savez_compressed(output / 'reference.npz', qpos=np.vstack([initial['qpos'], qs]),
            qvel=np.vstack([initial['qvel'], vs]), ctrl=controls, timestep=model.opt.timestep,
            sensordata=np.vstack([probe.sensordata, sensors]), source_task_success=summary['task_success'],
            joint_names=[model.joint(int(j)).name for j in joint_ids],
            contact=np.asarray([[any(model.geom('left_foot').id in (c['geom1'], c['geom2']) and c['force'][0] > .002 for c in r['contacts']),
                                 any(model.geom('right_foot').id in (c['geom1'], c['geom2']) and c['force'][0] > .002 for c in r['contacts'])] for r in rows]))
    write_json(output / 'status.json', dict(state='failed' if error else 'completed', completed_steps=len(rows)))
    case = dict(task='contact', name=task.name, method=method, model='scene.xml', trajectory='trajectory.npz',
                summary='summary.json', dt=model.opt.timestep)
    write_json(output / 'manifest.json', dict(version=1, task='contact', method=method, cases=[case]))
    if error:
        raise RuntimeError(f'Contact run failed; partial results saved in {output}: {error.splitlines()[-1]}')
    return [case]

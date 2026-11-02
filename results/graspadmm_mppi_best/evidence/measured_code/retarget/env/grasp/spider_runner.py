"""CA proposals, fixed-object MPPI acquisition and independently audited hold."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from retarget.common.results import write_json
from retarget.solver.spider_mppi import optimize, shift_controls
from retarget.task.grasp.spider_cost import SpiderCostSettings
from .spider_reference import read_input, prepare_model, preload, sha256
from .spider_physics import BatchedPhysics
from .spider_rollout import SamplingRollout


def options(config):
    values = config.environment.get("spider", {})
    if not isinstance(values, dict):
        raise ValueError("environment.spider must be a mapping")
    values = dict(values)
    defaults = dict(duration_seconds=.4, sim_dt=.01, exec_dt=.002, horizon=.3,
        knot_dt=.06, ctrl_dt=.03, pos_noise=.003, rot_noise=.03, joint_noise=.08,
        first_noise=.5, last_noise=1., nconmax=256, njmax=2048,
        solver_iterations=20, ls_iterations=50, preload=True,
        cost={"contact_rew_scale": 1.})
    defaults['acquisition']={}
    if set(values)-set(defaults)-{"case_indices"}:
        raise ValueError(f"Unknown SPIDER settings: {set(values)-set(defaults)-{'case_indices'}}")
    values = {**defaults, **values}
    for key in set(defaults)-{"preload", "cost", "acquisition"}:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"spider.{key} must be finite and positive")
    for key in ("nconmax", "njmax", "solver_iterations", "ls_iterations"):
        if not isinstance(values[key], int):
            raise ValueError(f"spider.{key} must be an integer")
    if not isinstance(values["preload"], bool):
        raise ValueError("spider.preload must be boolean")
    if not isinstance(values["cost"], dict):
        raise ValueError("spider.cost must be a mapping")
    from .spider_acquisition import AcquisitionSettings
    AcquisitionSettings(**values['acquisition'])
    try:
        SpiderCostSettings(**values["cost"])
    except TypeError as error:
        raise ValueError(f"Invalid SPIDER cost settings: {error}") from error
    for outer, inner in (("horizon", "knot_dt"), ("knot_dt", "sim_dt"),
                         ("ctrl_dt", "sim_dt"), ("sim_dt", "exec_dt"),
                         ("duration_seconds", "exec_dt")):
        ratio = values[outer]/values[inner]
        if round(ratio) < 1 or not np.isclose(ratio, round(ratio), atol=1e-8, rtol=0):
            raise ValueError(f"SPIDER {outer}/{inner} must be a positive integer")
    if values["ctrl_dt"] > values["horizon"]:
        raise ValueError("SPIDER execution interval cannot exceed the horizon")
    return values


def validate(config):
    from .spider_source import verify_sources
    verify_sources()
    if (config.task.get("type") != "grasp" or config.task.get("name") != "primitive"
            or config.solver.type != "mppi" or config.environment.get("dynamics") != "mjwarp"):
        raise ValueError("SPIDER grasp requires grasp/primitive, solver mppi and native mjwarp dynamics")
    if config.task.get("options", {}) != {"hand_model": "wuji2"}:
        raise ValueError("SPIDER grasp requires options {hand_model: wuji2}")
    if config.solver.gradient_steps or not str(config.device).startswith("cuda"):
        raise ValueError("SPIDER MPPI requires CUDA and no gradient refinement")
    opts = options(config)
    data = Path(config.task["data"])
    sources = [data] if (data/"card_input.npz").is_file() else sorted(
        p.parent for p in data.glob("**/card_input.npz")
        if (p.parent/"contactaware_reference.json").is_file())
    if (data/'spider_seed.npz').is_file() or (data/'spider_reference.npz').is_file():sources=[data]
    if not sources:
        raise ValueError("input_invalid: no SPIDER seed, acquired reference or accepted legacy CA input found")
    ids = opts.get("case_indices", list(range(len(sources))))
    if (not isinstance(ids, list) or not ids
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(sources) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("SPIDER case_indices must select distinct existing inputs")
    selected = [sources[i] for i in ids]
    for source in selected:
        read_input(source)
    return selected


class FingerForces:
    """Read actual native contact rows from the one-world execution engine."""
    def __init__(self, physics):
        from .forceaware_validation import finger_body_labels
        self.physics = physics
        self.ids = wp.array(np.arange(physics.data.naconmax), dtype=int, device=physics.device)
        self.force = wp.zeros(physics.data.naconmax, dtype=wp.spatial_vector, device=physics.device)
        self.fingers = finger_body_labels(physics.cpu)
        self.object_body = physics.cpu.body("grasp_object").id

    def read(self):
        env = self.physics
        mjw.contact_force(env.model, env.data, self.ids, False, self.force)
        count = int(env.data.nacon.numpy()[0])
        if count > env.data.naconmax:
            raise RuntimeError("Native execution contact buffer overflow")
        geoms = env.data.contact.geom.numpy()[:count]
        bodies = env.cpu.geom_bodyid[geoms]
        pair = (bodies[:, 0] == self.object_body) ^ (bodies[:, 1] == self.object_body)
        hand = np.where(bodies[:, 0] == self.object_body, bodies[:, 1], bodies[:, 0])
        finger = self.fingers[hand]
        pressure = self.force.numpy()[:count, 0]
        result = np.zeros(5)
        mask = pair & (finger >= 0)
        np.add.at(result, finger[mask], pressure[mask])
        return result


def contact_retention_metrics(forces, required, settings):
    """Audit the same local objective on recorded forces, never inferred FK."""
    from .forceaware_validation import FINGER_NAMES
    forces, required = np.asarray(forces, float), np.asarray(required, bool)
    if forces.ndim != 2 or forces.shape[1] != 5 or required.shape != (5,):
        raise ValueError("Contact retention needs [steps,5] loads and a [5] CA mask")
    result = dict(finger_names=FINGER_NAMES, required_finger_mask=required,
        minimum_target_force_n=settings.contact_min_force_n,
        cost_weight=settings.contact_keep_rew_scale,
        physics_steps=len(forces), finite=bool(np.isfinite(forces).all()))
    if not len(forces) or not required.any() or not result["finite"]:
        return dict(result, available=False)
    deficit = np.maximum(1.-np.maximum(forces, 0.)/settings.contact_min_force_n, 0.)
    return dict(result, available=True,
        finger_below_target_steps=(forces < settings.contact_min_force_n).sum(0),
        all_required_positive_fraction=(forces[:, required] > 1e-6).all(1).mean(),
        all_required_at_target_fraction=(forces[:, required] >= settings.contact_min_force_n).all(1).mean(),
        mean_contact_keep_cost=settings.contact_keep_rew_scale*(deficit[:, required]**2).sum(1).mean(),
        normalized_deficit_squared_mean=(deficit[:, required]**2).sum(1).mean())


def _run_case_impl(config, source, case):
    started = time.perf_counter()
    opts = options(config)
    reference = read_input(source)
    original_hashes=reference.source_hashes.copy()
    acquisition=None
    if reference.reference_kind=='ca_proposal':
        from .spider_acquisition import acquire
        acquisition=acquire(source,case/'acquisition',config.solver,opts['acquisition'],config.device)
        if not acquisition['success']:
            # A completed MPPI attempt, not a rejected CA input. Preserve the
            # actual optimized proposal as a clearly failed acquisition trace.
            import shutil
            shutil.copy2(case/'acquisition/acquisition_trajectory.npz',case/'trajectory.npz')
            _, candidate_model = prepare_model(reference,case,opts['exec_dt'])
            summary=dict(task_success=False,native_hold_success=False,independent_mujoco_hold_success=False,
                acquisition=acquisition,hold_stage='not_run',failure_stage='mppi_acquisition',
                ca_is_gate=False,ca_success=acquisition['ca_success'],timing=dict(optimization_seconds=acquisition['seconds'],case_wall_seconds=time.perf_counter()-started))
            selected=acquisition['selected_diagnostics']
            summary['stage_results']=dict(contact_establishment=selected['contact_established'],
                geometric_force_closure=selected['force_closure']['success'],dynamic_hold=None)
            from .grasp_evaluation import apply_evaluation
            with np.load(case/'trajectory.npz', allow_pickle=False) as candidate_trace:
                candidate_qpos = candidate_trace['qpos'][-1].copy()
            apply_evaluation(summary, candidate_model, candidate_qpos, case, method='spider_mppi',
                             selection=dict(kind='optimized_mppi_acquisition', path=str(case/'acquisition/acquisition.json')))
            write_json(case/'summary.json',summary)
            return summary
        reference=read_input(acquisition['reference'])
    cpu, model_path = prepare_model(reference, case, opts["exec_dt"],
        solver_iterations=opts["solver_iterations"], ls_iterations=opts["ls_iterations"])
    knots = round(opts["horizon"]/opts["knot_dt"])
    knot_steps = round(opts["knot_dt"]/opts["sim_dt"])
    horizon_steps = knots*knot_steps
    try:
        warm, load_report = (preload(reference, model_path, knots) if opts["preload"] else
                             (reference.qpos[:26].astype(np.float32), {"enabled": False}))
    except ValueError as error:
        if reference.reference_kind!='mppi_acquired':raise
        data=mujoco.MjData(cpu);data.qpos[:]=reference.qpos;mujoco.mj_forward(cpu,data)
        warm=(reference.qpos[:26]+data.qfrc_bias[:26]/cpu.actuator_gainprm[:,0]).astype(np.float32)
        warm=np.clip(warm,cpu.actuator_ctrlrange[:,0],cpu.actuator_ctrlrange[:,1])
        load_report=dict(enabled=False,fallback='gravity-compensated acquired pose; preload never gates MPPI',reason=str(error))
    settling=None
    if reference.reference_kind=='mppi_acquired':
        from .spider_reference import settle_acquired_grip
        reference,warm,settling=settle_acquired_grip(reference,model_path,warm,case/'grip_establishment')
        from .spider_grip import trim_grip
        reference,warm,trim=trim_grip(reference,model_path,warm,case/'grip_establishment/trim',config.solver,config.device)
        settling['mppi_load_trim']=trim
    write_json(case/"control_initialization.json", load_report)
    write_json(case/"ca_input_hashes.json", reference.source_hashes)
    write_json(case/"reference_manifest.json", dict(source_experiment=str(source),
        reference_source=reference.reference_kind, contactaware_changed=False,
        hand_reference_qpos=reference.qpos[:26], object_reference_qpos=reference.qpos[26:],
        active_slot_indices=reference.active_slots, active_contact_count=len(reference.active_slots),
        qpos_dimension=26, nq=33, nv=32, nu=26,
        virtual_object_forces=False, hand_contact_local=reference.contact_local,
        object_contact_local=reference.object_local,
        observation_stage="post-step qpos/qvel; native pre-integration contact-point FK, matching SPIDER"))
    plan_cpu = mujoco.MjModel.from_xml_path(str(model_path))
    plan_cpu.opt.timestep = opts["sim_dt"]
    cost_values=dict(opts['cost'])
    if reference.required_groups is not None:
        cost_values={'palm_keep_rew_scale':.1,'wrench_rew_scale':.05,
            'contact_min_force_n':.25,'contact_worst_rew_scale':2.,**cost_values}
        if not reference.required_groups[5]:cost_values['palm_keep_rew_scale']=0.
    cost_settings = SpiderCostSettings(**cost_values)
    from .spider_contact import required_finger_mask
    required_fingers = required_finger_mask(cpu, reference.contact_bodies)
    if reference.required_groups is not None:required_fingers=reference.required_groups[:5]
    write_json(case/"contact_retention.json", dict(enabled=cost_settings.contact_keep_rew_scale > 0,
        origin="local MPPI cost extension; no DexForge FA optimization",
        required_finger_mask=required_fingers, active_ca_slots=reference.active_slots,
        contact_keep_rew_scale=cost_settings.contact_keep_rew_scale,
        contact_min_force_n=cost_settings.contact_min_force_n,
        formula="weight * sum_CA_fingers relu(1 - max(actual_normal_force_N, 0) / min_force_N)^2",
        force_source="native solved object-finger contact rows, including fixed pad descendants",
        observation_stage="solver contacts before integration; sampled after each native step",
        requirement_policy="template fingers plus broad palm" if reference.required_groups is not None else "CA active contact body ancestry, deduplicated by physical finger"))
    rollout = SamplingRollout(plan_cpu, reference, warm, worlds=config.solver.worlds,
        horizon_steps=horizon_steps, cost_settings=cost_settings, device=config.device,
        nconmax=opts["nconmax"], njmax=opts["njmax"])
    executor = BatchedPhysics(cpu, 1, config.device, nconmax=opts["nconmax"],
                              njmax=opts["njmax"])
    executor.reset(reference.qpos, reference.qvel, control=warm)
    executor.warmup()
    force_reader = FingerForces(executor) if reference.required_groups is None else AcquiredForces(executor,reference.seed_source)
    mean = torch.as_tensor(warm, device=config.device).expand(horizon_steps, -1).clone()
    noise = torch.tensor([opts["pos_noise"]]*3+[opts["rot_noise"]]*3+[opts["joint_noise"]]*20,
                         device=config.device)
    substeps = round(opts["sim_dt"]/opts["exec_dt"])
    execute_steps = round(opts["ctrl_dt"]/opts["sim_dt"])
    total = round(opts["duration_seconds"]/opts["exec_dt"])
    qtrace, vtrace, controls, forces = [reference.qpos.copy()], [reference.qvel.copy()], [], []
    windows = [];group_forces=[];observations=[]
    executed = 0
    setup_seconds = time.perf_counter()-started
    while executed < total:
        rollout.prepare(executor.snapshot())
        torch.cuda.synchronize(config.device)
        tick = time.perf_counter()
        command, result = optimize(mean, rollout.evaluate, config.solver, knots=knots,
            knot_steps=knot_steps, channel_noise=noise, seed=config.seed+len(windows),
            first_noise=opts["first_noise"], last_noise=opts["last_noise"],
            return_best=reference.reference_kind=='mppi_acquired')
        torch.cuda.synchronize(config.device)
        result["seconds"] = time.perf_counter()-tick
        count = min(execute_steps, int(np.ceil((total-executed)/substeps)))
        for action in command[:count]:
            for _ in range(min(substeps, total-executed)):
                executor.step(action)
                status = executor.check()
                if not bool(status["valid"].all()):
                    raise RuntimeError("Native SPIDER execution is nonfinite or overflowed")
                qtrace.append(executor.qpos[0].cpu().numpy().copy())
                vtrace.append(executor.qvel[0].cpu().numpy().copy())
                controls.append(action.cpu().numpy().copy())
                forces.append(force_reader.read())
                if isinstance(force_reader,AcquiredForces):
                    group_forces.append(force_reader.last_loads.copy());observations.append(force_reader.observation())
                executed += 1
        tail = torch.as_tensor(warm, device=config.device).expand(count, -1)
        mean = shift_controls(command, count, tail)
        windows.append(dict(window=len(windows), executed_steps=executed, **result))
        write_json(case/"iterations.json", windows)
        print(f"[SPIDER MPPI {case.name}] {executed}/{total} cost={result['cost']:.6g} seconds={result['seconds']:.3f}", flush=True)
    trajectory = case/"trajectory.npz"
    np.savez_compressed(trajectory, qpos=qtrace, qvel=vtrace, ctrl=controls,
        time=np.arange(total+1)*opts["exec_dt"], finger_normal_force_n=forces,
        object_reference_qpos=np.repeat(reference.qpos[None, 26:], total+1, 0))
    if observations:
        capacity=force_reader.geometry.capacity
        points=np.zeros((total+1,capacity,3));pressure=np.zeros((total+1,capacity));counts=np.zeros(total+1,int)
        for i,obs in enumerate(observations,1):
            count=len(obs['points']);points[i,:count]=obs['points'];pressure[i,:count]=obs['forces'];counts[i]=count
        with np.load(trajectory) as raw:arrays={k:raw[k].copy() for k in raw.files}
        arrays.update(contact_points=points,contact_normal_force=pressure,contact_counts=counts,group_normal_force_n=np.asarray(group_forces))
        np.savez_compressed(trajectory,**arrays)
    hold_audit=None
    if reference.required_groups is not None:
        from .spider_hold import audit_hold
        hold_audit=audit_hold(model_path,trajectory,reference,observations,case/'group_hold')
    from .spider_validation import audit_replay
    report = audit_replay(model_path, trajectory, case/"replay", reference_qpos=reference.qpos,
        contact_body_ids=reference.contact_bodies, contact_local=reference.contact_local,
        contact_object_local=reference.object_local, required_contact_finger_mask=required_fingers)
    with np.load(case/"replay/replay.npz", allow_pickle=False) as replay:
        cpu_forces = replay["cpu_finger_normal_force_n"].copy()
    retention = dict(enabled=cost_settings.contact_keep_rew_scale > 0,
        gpu=contact_retention_metrics(forces, required_fingers, cost_settings),
        cpu=contact_retention_metrics(cpu_forces, required_fingers, cost_settings))
    if any(sha256(path) != digest for path, digest in reference.source_hashes.items()):
        raise RuntimeError("ContactAware source input was modified during SPIDER optimization")
    if any(sha256(path)!=digest for path,digest in original_hashes.items()):
        raise RuntimeError('Original SPIDER initial proposal was modified during optimization')
    summary = dict(report, method="spider_mppi", contactaware_changed=False,
        forceaware_optimizer_used=False, forceaware_cost_used=False,
        dynamics="native MuJoCo Warp implicitfast; full original ordinary mesh model",
        comparison_scope="end-to-end MPPI/MJWarp versus FA/native-ComFree; not an isolated optimizer ablation",
        cost=asdict(cost_settings), source_experiment=str(source),
        contact_retention=retention,
        cost_origin="SPIDER tracking plus optional local CA-finger contact-load deficit penalty",
        timing=dict(setup_seconds=setup_seconds, optimization_seconds=sum(w["seconds"] for w in windows),
            windows=len(windows), case_wall_seconds=time.perf_counter()-started),
        sampler=dict(population=config.solver.population, worlds=config.solver.worlds,
            iteration_limit=config.solver.iterations, actual_iterations=sum(w["iterations"] for w in windows)),
        profile=opts)
    if hold_audit is not None:
        summary['legacy_five_finger_acceptance']=summary.get('acceptance')
        summary['acceptance']=dict(protocol='template fingers, broad palm, actual loaded force closure and independent free hold',
            required_contact_finger_mask=reference.required_groups[:5],required_groups=reference.required_groups,
            minimum_force_n=.05,position_limit_mm=2.,rotation_limit_deg=5.,hold_seconds=.4)
        summary.update(acquisition=acquisition,grip_establishment=settling,group_hold=hold_audit,ca_is_gate=False,
            task_success=hold_audit['success'],native_hold_success=hold_audit['gpu']['success'],
            independent_mujoco_hold_success=hold_audit['cpu']['success'],
            failure_stage=None if hold_audit['success'] else 'dynamic_hold',
            stage_results=dict(contact_establishment=True,geometric_force_closure=True,dynamic_hold=hold_audit['success']))
        summary['timing']['acquisition_seconds']=acquisition['seconds'] if acquisition else 0.
        summary['timing']['optimization_seconds']+=summary['timing']['acquisition_seconds']
    from .grasp_evaluation import apply_evaluation
    apply_evaluation(summary, model_path, qtrace[-1], case, method="spider_mppi",
                     selection=dict(kind="final_executed_state", path=str(trajectory)))
    write_json(case/"summary.json", summary)
    return summary


def run_case(config,source,case):
    with torch.no_grad(),wp.ScopedDevice(config.device),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(config.device))):
        return _run_case_impl(config,source,case)


def run(config, output):
    wp.init(); wp.set_device(config.device)
    cases, summaries = [], []
    for number, source in enumerate(validate(config)):
        case = Path(output)/f"{number:03d}_{source.parents[1].name}"
        case.mkdir(parents=True, exist_ok=True)
        write_json(Path(output)/"status.json", dict(state="running", case=case.name))
        with torch.no_grad(), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(config.device))):
            summary = run_case(config, source, case)
        summaries.append(dict(name=case.name, **summary))
        cases.append(dict(task="grasp", method="mppi", hand="wuji2", name=case.name,
            model=str((case/"model.xml").relative_to(output)),
            trajectory=str((case/"trajectory.npz").relative_to(output)),
            summary=str((case/"summary.json").relative_to(output)), dt=options(config)["exec_dt"]))
    write_json(Path(output)/"summary.json", dict(cases=summaries, evaluation_protocol="graspadmm_native_v1",
        all_task_success=all(s["task_success"] for s in summaries), method="spider_mppi"))
    return cases


class AcquiredForces:
    def __init__(self,physics,seed_source):
        from .spider_geometry import NativeContactGeometry
        from .spider_seed import load_seed
        self.geometry=NativeContactGeometry(physics,load_seed(seed_source)[2])
    def read(self):
        self.last_loads=self.geometry.sample()[0].cpu().numpy().copy()
        return self.last_loads[:5].copy()
    def observation(self):
        g=self.geometry;count=int(g.count.numpy()[0])
        if count>g.capacity:raise RuntimeError('Execution contact geometry overflow')
        return dict(points=g.points.numpy()[0,:count].copy(),normals=g.normals.numpy()[0,:count].copy(),
            groups=g.groups.numpy()[0,:count].copy(),forces=g.pressures.numpy()[0,:count].copy(),
            bodies=g.hand_bodies.numpy()[0,:count].copy(),friction=float(g.mu.numpy()[0,:count].min()) if count else 0.)

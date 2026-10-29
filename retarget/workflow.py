"""Fixed-template worker with contact-surface AdamW and fast SQP-RTI."""
from dataclasses import replace
import json
import time
import mujoco
import numpy as np

def run(args):
    p = args.experiment.resolve()
    from forceaware.config import load_retarget_config, config_dict, object_xml_stem
    from forceaware.runtime import setup_runtime
    from forceaware.time_grid import SolveRequest
    cfg = load_retarget_config(p / 'sequence', p / 'forceaware.yaml', hand='wuji_right', contactaware_dir=p / 'contactaware')
    if args.iterations is not None:
        cfg = replace(cfg, optimizer=replace(cfg.optimizer, n_iter=args.iterations))
    elif args.hand_solver in ('adamw','acados-rti'):
        cfg = replace(cfg, optimizer=replace(cfg.optimizer, n_iter=1))
    if args.multistarts is not None:
        cfg = replace(cfg, optimizer=replace(cfg.optimizer, multistart_count=args.multistarts))
    cfg = replace(cfg, optimizer=replace(cfg.optimizer, multistart_seed=args.seed))
    case = (args.output_base.resolve() if args.output_base is not None else p) / args.executor / args.case_name
    case.mkdir(parents=True, exist_ok=True)
    print(f'[case] {case}', flush=True)
    if any(((case / n).exists() for n in ('rollout.npz', 'partial.npz', 'windows.jsonl', 'live_state.npz'))):
        raise FileExistsError(f'{case} contains a previous run; use a new --case-name')
    grid = replace(cfg.simulator.time_grid, exec_dt=0.002 if args.executor == 'mujoco' else 0.01)
    cfg = replace(cfg, sequence=replace(cfg.sequence, output_dir_name=str(case)), simulator=replace(cfg.simulator, time_grid=grid, exec_backend=args.executor))
    setup_runtime(cfg)
    from forceaware import workflow as workflow
    from forceaware.executor import ActionExecution, ExecutorRuntime
    from retarget.fa.solver import settings_from_args, optimizer_class
    settings = settings_from_args(args)
    optimizer_base = optimizer_class(settings)
    snapshots = []
    snapshot_indices = []
    cursor = {'window': 0}
    class FixedInitialOptimizer(optimizer_base):

        def solve(self, *a, **kw):
            solve_started = time.perf_counter()
            kw['allow_init'] = False
            plan = super().solve(*a, **kw)
            if args.hand_solver == 'acados-rti':
                snapshots.append(self.contact_snapshot)
                snapshot_indices.append(cursor['window'])
                plan.diag['solver']['reference_gate'] = self.reference_gate_diagnostic if hasattr(self, 'reference_gate_diagnostic') else None
                cursor['window'] += 1
                with (case / 'windows.jsonl').open('a') as f:
                    f.write(json.dumps(dict(
                        window=cursor['window'] - 1,
                        seconds=time.perf_counter() - solve_started,
                        solver_wall_seconds=plan.seconds,
                        solver=plan.diag.get('solver', {}),
                    )) + '\n')
                return plan
            from forceaware.mpc_loss import MpcObjective
            selected = plan.diag['selected']
            snap = self.objective.snapshot(selected)
            snapshots.append(snap)
            snapshot_indices.append(cursor['window'])
            MpcObjective.evaluate(self.objective, self.control, allow_init=False)
            fa = float(self.loss.total.numpy()[selected])
            mask = snap['mask'] > 0
            primal = np.linalg.norm(snap['x'] - snap['z'], axis=-1)[mask] * 1000
            shift = np.linalg.norm(snap['z'] - snap['template'], axis=-1)[mask] * 1000
            record = dict(window=cursor['window'], FA_loss=fa, selection_merit=plan.loss, primal_mean_mm=float(primal.mean()) if len(primal) else 0.0, primal_max_mm=float(primal.max()) if len(primal) else 0.0, object_contact_shift_mean_mm=float(shift.mean()) if len(shift) else 0.0, object_contact_shift_max_mm=float(shift.max()) if len(shift) else 0.0, seconds=time.perf_counter() - solve_started, solver_wall_seconds=plan.seconds, solver=self.last_solver_diag)
            with (case / 'windows.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            cursor['window'] += 1
            return replace(plan, seconds=time.perf_counter() - solve_started)
    workflow.WindowOptimizer = FixedInitialOptimizer
    initial_run = workflow._initial_run

    def initial(*a, **kw):
        result = initial_run(*a, **kw)
        result = replace(result, qvel=np.zeros_like(result.qvel))
        if visualizer is not None:
            visualizer.update(result.qpos, result.qvel, 0.0)
        return result
    workflow._initial_run = initial
    visualizer = None
    if args.viewer:
        from retarget.common.viewer_bridge import ViewerProcess
        visualizer = ViewerProcess(p, case)
    workflow.make_solve_request = lambda config: SolveRequest.create(config.simulator.time_grid, duration_seconds=args.duration)

    class MujocoExecutor:

        def __init__(self, runtime, index, cfg):
            self.model = runtime.cpu_model
            self.data = mujoco.MjData(self.model)
            self.index = index
            self.elapsed = 0.0
            self.substeps = round(cfg.simulator.time_grid.mpc_dt / self.model.opt.timestep)
            self.contact_observer = None

        def enable_contact_reporting(self):
            from forceaware.contact_observation import MujocoContactObserver
            self.contact_observer = MujocoContactObserver(self.model, self.index)

        def execute_action_sequence(self, qpos, qvel, *, actions):
            d = self.data
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            qs = []
            vs = []
            contacts = []
            for action in actions:
                d.ctrl[self.index.hand_ctrl] = action
                d.qfrc_applied[:] = 0.0
                for _ in range(self.substeps):
                    mujoco.mj_step(self.model, d)
                mujoco.mj_forward(self.model, d)
                qs.append(d.qpos.copy())
                vs.append(d.qvel.copy())
                if self.contact_observer is not None:
                    contacts.append(self.contact_observer.read(d))
            self.elapsed += len(actions) * self.substeps * self.model.opt.timestep
            return ActionExecution(qs[-1], vs[-1], np.array(qs), np.array(vs),
                                   tuple(contacts) if self.contact_observer is not None else None)
    if args.executor == 'mujoco':

        def load_executor(config, comfree_warp, **kw):
            scene=p/'sequence/scene/wuji_right'/f'{object_xml_stem(p/"sequence")}_mesh.xml'
            cpu = mujoco.MjModel.from_xml_path(str(scene.resolve()))
            cpu.opt.timestep = 0.002
            return ExecutorRuntime(cpu, None)
        workflow._load_executor_runtime = load_executor
        workflow.build_action_executor = lambda runtime, index, cfg: MujocoExecutor(runtime, index, cfg)

    def save(config, recorder, *, compiled, cpu_model, index, sequence, **kw):
        from forceaware.targets import _object_qpos
        arrays = recorder.arrays()
        arrays['object_reference_qpos'] = _object_qpos(sequence.object_pose_xyzw, np.r_[arrays['qpos_frames'][0], arrays['target_frames']])
        np.savez_compressed(case / 'rollout.npz', **arrays)
        if snapshots:
            np.savez_compressed(case / 'contacts.npz', window_indices=np.array(snapshot_indices), **{k: np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
        (case / 'config.json').write_text(json.dumps(config_dict(config), default=str, indent=2))
        (case / 'audit.json').write_text(json.dumps(dict(executor=args.executor, closed_loop=True, initial_pose_refinement=False, initial_velocity='zero', duration=args.duration, original_FA_optimizer=True, rendering=False), indent=2))
        audit = json.loads((case / 'audit.json').read_text())
        audit.update(original_FA_optimizer=False, base_optimizer=args.hand_solver, variant=args.hand_solver,
            gradient_source=self_gradient_source(args.hand_solver), finite_difference_gradients=False, seed=args.seed,
            solver_settings=vars(settings), force_QP=False, template_query_targets_optimized=False, contact_mask_unchanged=True, force_gates=False, FA_cost_formulas_and_weights_unchanged=not bool(config.loss.obj_vel or config.loss.obj_ang_vel), MANO_role='fixed_template', final_ranking='pure_FA')
        audit.update(auxiliary_snapshot_windows=len(snapshots),
                     object_velocity_tracking=bool(config.loss.obj_vel or config.loss.obj_ang_vel),
                     object_velocity_weights=dict(linear=config.loss.obj_vel,angular=config.loss.obj_ang_vel),
                     object_velocity_frame='world')
        audit.update(contact_solver=settings.contact_solver,
                     force_QP=settings.contact_radius > 0 and settings.contact_solver == 'qp')
        if args.hand_solver == 'adamw':
            explores=settings.contact_radius > 0 and settings.contact_admm and settings.contact_exploration
            admm=settings.contact_radius > 0 and settings.contact_admm
            audit.update(template_query_targets_optimized=explores,
                         MANO_role='local_contact_anchor' if explores else 'fixed_template',
                         contact_admm_enabled=admm, contact_exploration_enabled=explores,
                         FA_cost_formulas_and_weights_unchanged=False,
                         physical_wrench_loss_enabled=True,
                         final_ranking='physical_FA_with_updated_contacts',
                         contact_wrench_metric=('comfree_pose_prediction' if settings.contact_solver == 'sampled-lm' else 'convex_wrench_qp_residual'),
                         contact_consensus='scaled_ADMM_object_frame' if admm else 'disabled',
                         final_ranking_includes_ADMM_penalty=False)
        if args.hand_solver == 'acados-rti':
            audit.update(FA_cost_formulas_and_weights_unchanged=False,
                         template_query_targets_optimized=settings.contact_radius > 0,
                         contact_surface_optimizer='bounded_mesh_wrench_search',
                         contact_surface_input='measured_kinematics_zero_dual',
                         contact_model='reduced_friction_pyramid_not_native_warp',
                         full_adamw_warp_objective=False,
                         final_ranking='single_RTI_trajectory')
        audit['contact_gated_reference'] = args.hand_solver == 'acados-rti' and cfg.sequence.hold_frame < 0
        if getattr(args,'noise_metadata',None):audit['noise']=args.noise_metadata
        audit['prepared_experiment']=str(p)
        (case / 'audit.json').write_text(json.dumps(audit, indent=2))
        return case
    print_step = workflow._print_step

    def progress(window, total, plan, *, recorder):
        print_step(window, total, plan, recorder=recorder)
        if visualizer is not None:
            reference_seconds = recorder.qpos_frames[-1] * cfg.simulator.time_grid.ref_dt
            visualizer.update(recorder.qpos[-1], recorder.qvel[-1], reference_seconds, snapshots[-1] if snapshots else None)
        if window % 10 == 0:
            temporary = case / 'partial_tmp.npz'
            np.savez_compressed(temporary, **recorder.arrays())
            temporary.replace(case / 'partial.npz')
            if snapshots:
                temporary = case / 'partial_contact_tmp.npz'
                np.savez_compressed(temporary, window_indices=np.array(snapshot_indices), **{k: np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
                temporary.replace(case / 'partial_contacts.npz')
    workflow._print_step = progress
    started = time.time()
    def reference_gate_factory(sequence, start_frame, config):
        from retarget.contact_gate import ContactReferenceGate, prelift_frame
        hold = prelift_frame(sequence, start_frame, config.sequence.end)
        print(f'[contact_gate] hold MANO at frame {hold:g} until two fingers carry contact for 0.09 s', flush=True)
        grid = config.simulator.time_grid
        return ContactReferenceGate(start_frame, hold, config.sequence.end,
                                    grid.mpc_dt)
    try:
        workflow.rolling_mpc_retarget(
            cfg, config_path=p / 'forceaware.yaml', save_outputs_fn=save,
            reference_gate_factory=reference_gate_factory if args.hand_solver == 'acados-rti' else None)
    finally:
        if visualizer is not None:
            visualizer.close()
    (case / 'timing.json').write_text(json.dumps(dict(wall_seconds=time.time() - started)))


def self_gradient_source(solver):
    return "casadi_reduced_model_analytic" if solver=="acados-rti" else "analytic_adjoint"

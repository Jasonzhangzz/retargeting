"""Stationary-reference MPC and open-loop retention on the shared CUDA engine."""
from dataclasses import replace
from collections import deque
from retarget.grasp_capture import capture_guard
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp


@wp.kernel(enable_backward=False)
def hold_object(qpos:wp.array2d(dtype=float),qvel:wp.array2d(dtype=float),
                target:wp.array(dtype=float),position:int,velocity:int):
    for i in range(7):
        qpos[0,position+i] = target[i]
    for i in range(6):
        qvel[0,velocity+i] = 0.


def stable_velocity(qvel, hand_dofs, object_dof, *, linear=.005, angular=.05):
    """Check hand base, all finger joints, and the free object's velocities."""
    v = np.asarray(qvel)
    return bool(np.isfinite(v).all()
        and np.linalg.norm(v[hand_dofs[:3]]) <= linear
        and np.max(np.abs(v[hand_dofs[3:]]),initial=0.) <= angular
        and np.linalg.norm(v[object_dof:object_dof+3]) <= linear
        and np.linalg.norm(v[object_dof+3:object_dof+6]) <= angular)


def retention_metrics(qpos, reference, object_qpos, *, distance_mm=30.):
    """Measure object motion relative to the hand, in the settled hand frame."""
    q, baseline = np.asarray(qpos), np.asarray(reference)
    r = Rotation.from_euler('XYZ',q[3:6])
    r0 = Rotation.from_euler('XYZ',baseline[3:6])
    relative = r.inv().apply(q[object_qpos:object_qpos+3]-q[:3])
    relative0 = r0.inv().apply(baseline[object_qpos:object_qpos+3]-baseline[:3])
    delta = relative-relative0
    down = r0.inv().apply([0.,0.,-1.])
    fall = float(delta@down)*1000
    displacement = float(np.linalg.norm(delta))*1000
    return dict(relative_fall_mm=fall,relative_displacement_mm=displacement,
                dropped=fall>=distance_mm,escaped=displacement>=distance_mm)


def stable_pose_window(states, hand_qpos, object_qpos, *, hand_geometry_stable=False):
    """Accept a position plateau despite hard-contact velocity chatter.

    Require the entire 0.2 s window to stay within 0.2 mm / 1 mrad.
    A drifting object cannot pass this alternative stability gate.
    """
    q = np.asarray(states)
    hand = q[:,hand_qpos]
    obj = q[:,object_qpos:object_qpos+7]
    rotations = Rotation.from_quat(obj[:,3:],scalar_first=True)
    return bool(np.isfinite(q).all()
        and np.linalg.norm(np.ptp(hand[:,:3],axis=0))<=.0002
        and (hand_geometry_stable or np.max(np.ptp(hand[:,3:],axis=0))<=.001)
        and np.linalg.norm(np.ptp(obj[:,:3],axis=0))<=.0002
        and np.max((rotations[0].inv()*rotations).magnitude())<=.001)


def hand_surface_motion_bound(positions, matrices, radii):
    """Bound every hand surface point's motion, allowing tiny joint chatter.

    Body translation diameter plus its bounding-sphere rotation diameter is
    conservative for all query points, not just the five selected contacts.
    """
    p,r = np.asarray(positions),np.asarray(matrices)
    relative = np.swapaxes(r[0],-1,-2)[None]@r
    angle = np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1.,1.))
    bound = np.linalg.norm(np.ptp(p,axis=0),axis=-1)+2*np.asarray(radii)*np.sin(
        np.minimum(angle.max(axis=0),np.pi/2))
    return float(np.max(bound))


def run_drop_test(case,args,*,on_frame=None):
    """Settle, then enable gravity without resetting pose or velocity."""
    setup_started = time.perf_counter()
    import comfree_warp
    from forceaware.config import load_retarget_config
    from forceaware.runtime import setup_runtime
    from forceaware.targets import build_index,hand_qvel_indices
    from forceaware.time_grid import TimeGrid
    from forceaware.workflow import _configure_comfree_physics
    from forceaware.executor import WarpActionExecutor
    case = Path(case)
    source = json.loads((case/'result.json').read_text())
    experiment = case/f'input_{source["selected"]["start"]:02d}'
    cfg = load_retarget_config(experiment/'sequence',experiment/'forceaware.yaml',
                              contactaware_dir=experiment/'contactaware')
    grid = cfg.simulator.time_grid
    use_mpc = getattr(args,'hold_controller','fixed') == 'mpc'
    supplied_load = None
    if getattr(args,'load_conditioning',True) and (experiment/'load_input.npz').exists():
        with np.load(experiment/'load_input.npz') as saved:
            supplied_load = {key:saved[key].copy() for key in saved.files}
    conditioned = supplied_load is not None
    grid = TimeGrid.create(mpc_dt=grid.mpc_dt,action_dt=grid.action_dt,knot_dt=grid.knot_dt,
                           exec_dt=getattr(args,'physics_dt',.01 if use_mpc else .002),ref_dt=grid.ref_dt)
    cfg = replace(cfg,simulator=replace(cfg.simulator,time_grid=grid))
    setup_runtime(cfg)
    from retarget.grasp_physics import apply_object_physics,physics_snapshot
    scene = cfg.sequence.xml
    overrides = getattr(args,'physics_overrides',{})
    if overrides:
        import xml.etree.ElementTree as ET
        tree = ET.parse(scene)
        apply_object_physics(tree,overrides)
        scene = case/'drop_test_scene.xml'
        tree.write(scene)
    device,cpu = comfree_warp.load_model(str(scene),device=args.init_device)
    _configure_comfree_physics(cfg,cpu,device,timestep=grid.exec_dt)
    index = build_index(cpu)
    hand_dofs = hand_qvel_indices(cpu,index)
    object_dof = int(cpu.jnt_dofadr[cpu.joint('obj_joint').id])
    class FixtureExecutor(WarpActionExecutor):
        fixture_active = False
        def _advance(self,state,workspace_index):
            if not getattr(args,'physics_graph',True):
                return super()._advance(state,workspace_index)
            cache=getattr(self,'step_graphs',None)
            if cache is None:
                cache=self.step_graphs={}
            # A fixture kernel is part of the captured sequence: release must
            # select a separate graph, with no stale object-holding operation.
            key=(self.fixture_active,self.fixture_pose.ptr,workspace_index,
                 state.qpos.ptr,state.qvel.ptr,state.time.ptr)
            if key not in cache:
                with capture_guard(), wp.ScopedCapture(device=self.device) as capture:
                    result=super()._advance(state,workspace_index)
                cache[key]=(capture.graph,result,self.last_workspace)
            graph,result,last=cache[key]
            wp.capture_launch(graph)
            self.last_workspace=last
            return result
        def _step(self,state,workspace):
            result = super()._step(state,workspace)
            self.last_workspace = workspace
            if self.fixture_active:
                wp.launch(hold_object,dim=1,inputs=[result.qpos,result.qvel,self.fixture_pose,
                    index.object_qpos,object_dof],device=self.device)
            return result
    executor = FixtureExecutor(cpu,device,index=index,cfg=cfg)
    executor.enable_contact_reporting()
    with np.load(case/'grasp.npz') as saved:
        qpos = saved['qpos'].copy()
        qvel = saved['qvel'].copy() if 'qvel' in saved else np.zeros(cpu.nv)
        action = saved['hold_action'].copy() if 'hold_action' in saved else qpos[index.hand_qpos].copy()
        command_source = 'selected planner command' if 'hold_action' in saved else 'saved hand pose (legacy result has no command)'
        if (use_mpc or conditioned) and 'initial_qpos' in saved:
            # The one-window soft prediction is a geometry diagnostic, not
            # an executed state. Roll out the physical hand from Dexonomy.
            qpos = saved['initial_qpos'].copy()
            qvel = np.zeros(cpu.nv)
            action = qpos[index.hand_qpos].copy()
    if conditioned:
        action = supplied_load['gravity_command'].copy() if getattr(args,'preload_ramp_seconds',.3)>0 else supplied_load['command'].copy()
        command_source = 'actuator-feasible CUDA static-load input'
    fixture = getattr(args,'settle_mode','fixed-object')=='fixed-object'
    executor.fixture_active = fixture
    executor.fixture_pose = wp.array(qpos[index.object_qpos:index.object_qpos+7],dtype=float,device=executor.device)
    if fixture:
        qvel[object_dof:object_dof+6] = 0.
    controller = None
    windows = []
    pending = []
    plan = None
    if use_mpc:
        from retarget.grasp_runtime import GraspRuntime
        from forceaware.workflow import RollingController,InitialRun
        from forceaware.time_grid import SolveRequest
        runtime = GraspRuntime(experiment,args,scene=scene,gravity=9.81)
        initial = InitialRun(qpos,qvel,runtime.initial_warm,
            action,0.,SolveRequest.create(grid,duration_seconds=args.settle_timeout+args.drop_seconds))
        controller = RollingController(runtime.cfg,index,runtime.sequence,runtime.optimizer,initial,allow_init=False)
        command_source = ('run.py shared rolling controller; actuator-feasible load warm start'
                          if conditioned else 'run.py shared rolling controller; stationary Dexonomy references')
    log = dict(qpos=[qpos.copy()],qvel=[qvel.copy()],time=[0.],phase=[0],contact_mask=[np.zeros(5,bool)],
               object_contact_force_world=[np.zeros(3)],normal_force_by_finger=[np.zeros(5)],
               hand_actuator_force=[np.zeros(len(hand_dofs))],action=[action.copy()])
    observer = executor.contact_observer
    row_contact = observer.row_contacts
    contact_rows = np.flatnonzero(row_contact>=0)
    row_finger = observer.labels[row_contact[contact_rows]]
    elapsed,quiet = 0.,0.
    plateau_count = int(np.ceil(.2/grid.mpc_dt))+1
    pose_history,rotation_history = deque(maxlen=plateau_count),deque(maxlen=plateau_count)
    surface_bound = None
    plateau_geometry = False
    if conditioned:
        names = ['right_palm_link']+[f'right_finger{f}_link{k}' for f in range(1,6) for k in range(1,5)]
        hand_bodies = np.array([cpu.body(name).id for name in names])
        radii = []
        for name in names:
            from retarget.grasp_hand import query_path
            with np.load(query_path(cpu, name)) as query:
                radii.append(np.linalg.norm(query['local_pos'],axis=-1).max())
    stable = False
    status = 'settling'
    phase_cost={phase:dict(solver_seconds=0.,executor_seconds=0.,diagnostic_seconds=0.,viewer_seconds=0.)
                for phase in (-1,0,1)}
    begun = time.perf_counter()
    setup_seconds = begun-setup_started
    if on_frame:
        on_frame(qpos,qvel,elapsed,status)

    def advance(phase,label,remaining_steps=None):
        nonlocal qpos,qvel,elapsed,action,plan
        active_control = controller is not None and (phase==1 or (phase==-1 and not conditioned))
        if active_control:
            if not pending:
                count = min(grid.action_substeps,remaining_steps) if remaining_steps is not None else grid.action_substeps
                plan,_ = controller.solve(qpos,qvel,execution_steps=count,start_time=executor.elapsed)
                phase_cost[phase]['solver_seconds']+=plan.seconds
                pending.extend(plan.control_sequence)
                diagnostics = runtime.optimizer.last_solver_diag
                windows.append(dict(window=controller.window,time=elapsed,phase=phase,loss=plan.loss,execution_steps=count,
                    solver_seconds=plan.seconds,prepare_seconds=diagnostics['prepare_seconds'],
                    contact_seconds=diagnostics['contact_seconds'],hand_seconds=diagnostics['hand_seconds']))
            action = pending.pop(0)
        if conditioned and phase==-1:
            duration = getattr(args,'preload_ramp_seconds',.3)
            fraction = min(1.,(elapsed+grid.mpc_dt)/duration) if duration>0 else 1.
            action = ((1-fraction)*supplied_load['gravity_command']
                      +fraction*supplied_load['command'])
        execute_begun=time.perf_counter()
        result = executor.execute_action_sequence(qpos,qvel,actions=action[None])
        phase_cost[phase]['executor_seconds']+=time.perf_counter()-execute_begun
        diagnostic_begun=time.perf_counter()
        if active_control and not pending:
            controller.commit(plan,execution_steps=len(plan.control_sequence))
        qpos,qvel = result.next_qpos,result.next_qvel
        elapsed += grid.mpc_dt
        contacts = np.zeros(5,bool)
        for finger in result.contact_fingers[-1]:
            if 1<=finger<=5:
                contacts[finger-1] = True
        workspace = executor.last_workspace
        if conditioned and phase==0:
            pose_history.append(workspace.base.pose.body_position.numpy()[0,hand_bodies])
            rotation_history.append(workspace.base.pose.body_matrix.numpy()[0,hand_bodies])
        row_force = workspace.contact.output.force.numpy()[0]
        normal_force = np.array([np.maximum(row_force[contact_rows[row_finger==i]],0.).sum()
                                 for i in range(1,6)])
        object_force = workspace.contact.output.constraint_force.numpy()[0,object_dof:object_dof+3]
        actuator_force = workspace.base.dynamics.actuator_force.numpy()[0,hand_dofs]
        for name,value in (('qpos',qpos.copy()),('qvel',qvel.copy()),('time',elapsed),
                           ('phase',phase),('contact_mask',contacts),
                           ('object_contact_force_world',object_force),
                           ('normal_force_by_finger',normal_force),('hand_actuator_force',actuator_force),
                           ('action',action.copy())):
            log[name].append(value)
        phase_cost[phase]['diagnostic_seconds']+=time.perf_counter()-diagnostic_begun
        if on_frame:
            viewer_begun=time.perf_counter()
            on_frame(qpos,qvel,elapsed,label)
            phase_cost[phase]['viewer_seconds']+=time.perf_counter()-viewer_begun

    # The temporary fixture is preparation only. The planner already predicts
    # gravity so its stationary object target has a load to resist.
    gravity_compensated = bool(source.get('selected', {}).get('solver_diagnostics', {})
        .get('explicit_prediction', {}).get('hand_gravity_compensated_at_candidate', False))
    preparation_gravity = 9.81 if conditioned or gravity_compensated else getattr(args,'preparation_gravity',0.)
    device.opt.gravity.fill_((0.,0.,-preparation_gravity))
    phase_begun=time.perf_counter()
    if use_mpc or conditioned or gravity_compensated:
        for _ in range(int(np.ceil(getattr(args,'formation_seconds',1.)/grid.action_dt))):
            for _ in range(grid.action_substeps):
                advance(-1,'building physical preload' if conditioned else
                    'forming stationary grasp (MPC)' if use_mpc else 'forming grasp with saved command')
    formation_time = elapsed
    formation_wall=time.perf_counter()-phase_begun
    phase_begun=time.perf_counter()
    for _ in range(int(np.ceil(args.settle_timeout/grid.mpc_dt))):
        advance(0,'settling')
        quiet = quiet+grid.mpc_dt if stable_velocity(qvel,hand_dofs,object_dof) else 0.
        if conditioned and len(pose_history)==plateau_count:
            surface_bound = hand_surface_motion_bound(pose_history,rotation_history,radii)
            plateau_geometry = surface_bound<=.0002
        plateau = (elapsed-formation_time+1e-9>=.2 and
            stable_pose_window(log['qpos'][-plateau_count:],index.hand_qpos,index.object_qpos,
                               hand_geometry_stable=plateau_geometry))
        if elapsed-formation_time+1.e-9>=args.settle_seconds and (quiet+1.e-9>=.2 or plateau):
            stable = True
            break
    settling_wall=time.perf_counter()-phase_begun
    audit_begun=time.perf_counter()
    settled = qpos.copy()
    release_action = action.copy()
    formed_geometry = None
    formed_arrays = {}
    if stable and (use_mpc or gravity_compensated):
        with np.load(case/'grasp.npz') as saved:
            if all(key in saved for key in ('hand_body_ids','hand_local_points','hand_local_normals')):
                if not use_mpc:
                    from retarget.grasp_runtime import GraspRuntime
                    runtime = GraspRuntime(experiment,args,scene=scene,gravity=9.81)
                from retarget.grasp import StaticHand
                from retarget.grasp_audit import GraspAudit
                from retarget.contact_exploration import ObjectMeshSurface
                surface = ObjectMeshSurface.from_model(cpu,index.object_body)
                hand = StaticHand(cpu,surface,saved['hand_body_ids'],saved['hand_local_points'],saved['hand_local_normals'])
                auditor = GraspAudit(surface,args.init_device)
                formed_geometry,formed_arrays = auditor.evaluate_state(runtime,hand,args,qpos,qvel)
    if stable:
        np.savez_compressed(case/'formed_grasp.npz',qpos=qpos,qvel=qvel,
                            hold_action=release_action,**formed_arrays)
    formed_audit_wall=time.perf_counter()-audit_begun
    settle_time = elapsed
    gravity_start = len(log['time'])-1
    max_fall,max_motion = 0.,0.
    dropped,escaped = False,False
    first_drop = None
    phase_begun=time.perf_counter()
    if stable:
        # Release the settling fixture and change gravity. qpos/qvel and
        # command are continuous; no applied force supports the free object.
        executor.fixture_active = False
        device.opt.gravity.fill_((0.,0.,-9.81))
        if use_mpc:
            # The fixture/zero-gravity settling interval has no planner
            # windows. Start feedback from its measured state and command.
            initial = replace(initial,qpos=qpos,qvel=qvel,previous_action=action)
            if conditioned:
                initial = replace(initial,warm=np.r_[np.zeros(index.action_dim),np.tile(
                    action-runtime.sequence.hand[-1],cfg.optimizer.horizon)].astype(np.float32))
            controller = RollingController(runtime.cfg,index,runtime.sequence,runtime.optimizer,initial,allow_init=False)
            controller.executed_steps = int(round(elapsed/grid.mpc_dt))
            controller.window = len(windows)
            if hasattr(runtime.optimizer,'_contact_memory'):
                del runtime.optimizer._contact_memory
            runtime.optimizer.objective.commit_window(action)
            if runtime.optimizer.adamw is not None:
                runtime.optimizer.adamw.memory = None
        gravity_steps = int(np.ceil(args.drop_seconds/grid.mpc_dt))
        for step in range(gravity_steps):
            advance(1,'gravity test',remaining_steps=gravity_steps-step)
            measured = retention_metrics(qpos,settled,index.object_qpos,distance_mm=args.drop_distance_mm)
            max_fall = max(max_fall,measured['relative_fall_mm'])
            max_motion = max(max_motion,measured['relative_displacement_mm'])
            if measured['dropped'] and first_drop is None:
                first_drop = elapsed-settle_time
            dropped |= measured['dropped']
            escaped |= measured['escaped']
        status = 'dropped' if dropped else ('escaped' if escaped else 'retained')
    else:
        status = 'not_settled'
    gravity_wall=time.perf_counter()-phase_begun if stable else 0.
    report = dict(status=status,settled=stable,gravity_test_ran=stable,
        retained=bool(not dropped and not escaped) if stable else None,dropped=bool(dropped) if stable else None,
        escaped=bool(escaped) if stable else None,settle_seconds=settle_time-formation_time,
        formation_seconds=formation_time,
        gravity_seconds=elapsed-settle_time,first_drop_seconds=first_drop,
        maximum_relative_fall_mm=max_fall if stable else None,
        maximum_relative_displacement_mm=max_motion if stable else None,
        drop_distance_mm=args.drop_distance_mm,gravity_world=[0.,0.,-9.81],
        stability_linear_speed_m_s=.005,stability_angular_speed_rad_s=.05,
        stability_continuous_seconds=.2,settle_timeout_seconds=args.settle_timeout,
        stability_pose_window_translation_m=.0002,stability_pose_window_angle_rad=.001,
        stability_criterion='velocity' if quiet+1e-9>=.2 else (
            'surface_position_plateau' if stable and plateau_geometry else ('position_plateau' if stable else None)),
        maximum_hand_surface_motion_window_mm=None if surface_bound is None else surface_bound*1000,
        load_conditioned=conditioned,preparation_gravity_world=[0.,0.,-preparation_gravity],
        preload_ramp_seconds=getattr(args,'preload_ramp_seconds',.3) if conditioned else None,
        formed_geometry=formed_geometry,
        settling_mode='fixed-object' if fixture else 'free',settling_object_fixture=fixture,
        gravity_test_object_fixture=False,
        command_source=command_source,control_policy='shared run.py rolling MPC' if use_mpc else 'fixed selected position command; no MPC or extra squeeze',
        object_support_force=False,physics_backend='shared native ComFree CUDA hard-contact executor',
        physics_dt=grid.exec_dt,sample_dt=grid.mpc_dt,setup_seconds=setup_seconds,
        physics_graph=getattr(args,'physics_graph',True),
        physics_graph_count=len(getattr(executor,'step_graphs',{})),
        formed_audit_seconds=formed_audit_wall,
        phase_timings={
            'preload':dict(simulated_seconds=formation_time,wall_seconds=formation_wall,**phase_cost[-1]),
            'settling':dict(simulated_seconds=settle_time-formation_time,wall_seconds=settling_wall,**phase_cost[0]),
            'gravity':dict(simulated_seconds=elapsed-settle_time,wall_seconds=gravity_wall,**phase_cost[1])},
        simulation_seconds=time.perf_counter()-begun,wall_seconds=time.perf_counter()-setup_started,
        object_weight_N=float(cpu.body_mass[index.object_body]*9.81),
        physics=physics_snapshot(cpu,device,index),physics_overrides=overrides,
        settled_normal_force_by_finger_N=log['normal_force_by_finger'][gravity_start].tolist(),
        settled_normal_force_total_N=float(np.sum(log['normal_force_by_finger'][gravity_start])),
        loaded_fingers_at_release=(np.flatnonzero(log['contact_mask'][gravity_start])+1).tolist(),
        settled_object_contact_force_world_N=log['object_contact_force_world'][gravity_start].tolist(),
        mean_upward_contact_force_first_100ms_N=float(np.maximum(
            np.asarray(log['object_contact_force_world'])[gravity_start+1:gravity_start+11,2],0.).mean()) if stable else None,
        settled_finger_actuator_torque_Nm=log['hand_actuator_force'][gravity_start][6:].tolist(),
        settled_finger_command_error_rad=(release_action[6:]-settled[index.hand_qpos][6:]).tolist(),
        loaded_fingers_final=(np.flatnonzero(log['contact_mask'][-1])+1).tolist())
    report['controller_windows'] = len(windows)
    report['verified_grasp'] = bool(report['retained'] and formed_geometry and formed_geometry['feasible'])
    report['controller_solver_seconds'] = float(sum(w['solver_seconds'] for w in windows))
    report['controller_warm_window_mean_seconds'] = float(np.mean([w['solver_seconds'] for w in windows[1:]])) if len(windows)>1 else None
    report['controller_planning_gravity_world'] = [0.,0.,-9.81] if use_mpc else None
    report['controller_action_dt'] = grid.action_dt if use_mpc else None
    report['controller_warm_window_mean_breakdown_seconds'] = {
        name:float(np.mean([w[name] for w in windows[1:]]))
        for name in ('prepare_seconds','contact_seconds','hand_seconds')} if len(windows)>1 else None
    if use_mpc:
        (case/'hold_windows.json').write_text(json.dumps(windows,indent=2))
    arrays = {key:np.asarray(value) for key,value in log.items()}
    arrays.update(hold_action=release_action,settled_qpos=settled,gravity_start_index=np.asarray(gravity_start))
    np.savez_compressed(case/'drop_test.npz',**arrays)
    (case/'drop_test.json').write_text(json.dumps(report,indent=2))
    source['drop_test'] = report
    (case/'result.json').write_text(json.dumps(source,indent=2))
    print(f'  [physics] {status}: settle={settle_time:.2f}s gravity={elapsed-settle_time:.2f}s '
          f'max fall={report["maximum_relative_fall_mm"]} mm',flush=True)
    print('  [physics timing] '+', '.join(
        f'{name}: sim {values["simulated_seconds"]:.2f}s / wall {values["wall_seconds"]:.2f}s'
        for name,values in report['phase_timings'].items()),flush=True)
    if on_frame:
        on_frame(qpos,qvel,elapsed,status)
    return report,arrays

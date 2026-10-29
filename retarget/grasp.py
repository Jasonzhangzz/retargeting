"""Static grasp inputs, CARD/AdamW dispatch, audits and result export."""
import csv
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from retarget.contact_exploration import ObjectMeshSurface
from retarget.contact_solvers import skew
from retarget.dexonomy import ROOT, load_template, map_contacts, anatomical_joints, retarget_from_mano

def build_scene(mesh, scale=1., hand_model='wuji1'):
    """Independent fixed object and 26-DOF Wuji, without custom query geoms."""
    from retarget.grasp_hand import hand_tree
    tree = hand_tree(hand_model)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == 'geom' and child.get('type') == 'querypoint':
                parent.remove(child)
            elif child.tag == 'geom':
                visual = child.get('group') == '1'
                child.set('contype', '0' if visual else '1')
                child.set('conaffinity', '0' if visual else '1')
    ET.SubElement(tree.find('asset'), 'mesh', name='grasp_object_mesh', file=str(Path(mesh).resolve()),
                  scale=' '.join([str(scale)]*3))
    body = ET.SubElement(tree.find('worldbody'), 'body', name='grasp_object')
    geom = ET.SubElement(body, 'geom', name='grasp_object_geom', mesh='grasp_object_mesh',
                         type='mesh', contype='2', conaffinity='1', rgba='.25 .65 .85 1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    surface = ObjectMeshSurface.from_model(model, model.body('grasp_object').id)
    vertices = surface.triangles.reshape(-1, 3)
    center = (vertices.min(0)+vertices.max(0))/2
    geom.set('pos', ' '.join(map(str, -center)))
    xml = ET.tostring(tree, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    surface = ObjectMeshSurface.from_model(model, model.body('grasp_object').id)
    return model, surface, xml, center


class StaticHand:
    """Input-only Wuji kinematics and query assets; contains no grasp solver."""
    def __init__(self, model, surface, bodies, local, normals):
        from retarget.grasp_hand import HAND_BODIES, query_path, model_hand
        self.model, self.surface = model, surface
        self.hand_model = model_hand(model)
        self.data = mujoco.MjData(model)
        self.bodies, self.local, self.normals = bodies, local, normals
        self.all_bodies, self.all_local = [], []
        for name in HAND_BODIES:
            with np.load(query_path(model, name)) as asset:
                queries = asset['local_pos']
            self.all_bodies.extend([model.body(name).id]*len(queries))
            self.all_local.extend(queries)
        self.all_bodies, self.all_local = np.asarray(self.all_bodies), np.asarray(self.all_local)

    def set(self, q):
        self.data.qpos[:26] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.jacobians = {}

    def points(self, bodies, local, jacobian=False):
        rotation = self.data.xmat[bodies].reshape(-1, 3, 3)
        offsets = np.einsum('nij,nj->ni', rotation, local)
        points = self.data.xpos[bodies]+offsets
        if not jacobian:
            return points
        for body in np.unique(bodies):
            if body not in self.jacobians:
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jp, jr, int(body))
                self.jacobians[body] = jp[:, :26], jr[:, :26]
        jp = np.array([self.jacobians[b][0] for b in bodies])
        jr = np.array([self.jacobians[b][1] for b in bodies])
        return points, jp-skew(offsets) @ jr, jr



def display(model, q, case, *, viewer=False, render=False, image_name='grasp.png'):
    data = mujoco.MjData(model)
    data.qpos[:len(q)] = q
    mujoco.mj_forward(model, data)
    obj = model.body('grasp_object').id
    camera = mujoco.MjvCamera()
    palm = data.xpos[model.body('right_palm_link').id]
    camera.lookat[:] = .85*data.xpos[obj]+.15*palm
    direction = data.xpos[obj]-palm
    camera.distance = .32
    camera.azimuth = np.degrees(np.arctan2(direction[1], direction[0]))+200
    camera.elevation = -np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2])))
    model.vis.headlight.ambient[:] = .4
    if render:
        import imageio.v2 as imageio
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera=camera)
            imageio.imwrite(case / image_name, renderer.render())
    if viewer:
        from mujoco import viewer as viewer_module
        with viewer_module.launch_passive(model, data) as window:
            window.cam.lookat[:] = camera.lookat
            window.cam.distance = camera.distance
            window.cam.azimuth = camera.azimuth; window.cam.elevation = camera.elevation
            while window.is_running():
                window.sync(); time.sleep(.02)


def replay(case, *, viewer=True, render=False, stage='grasp'):
    model = mujoco.MjModel.from_xml_path(str(case / 'scene.xml'))
    report_path=case/'result.json'
    report=json.loads(report_path.read_text()) if report_path.is_file() else {}
    legacy_prediction=report.get('output_pose')=='terminal predicted state of selected common-solver rollout'
    if report.get('selected', {}).get('termination') == 'initialization_failed' and stage != 'initial':
        raise ValueError('Initialization failed; only --stage initial is available')
    if stage=='grasp' and legacy_prediction:
        raise ValueError('This older result exported the full prediction as a grasp. Regenerate it with '
                         'run_grasp.py, or inspect it explicitly with --stage prediction.')
    with np.load(case / ('formed_grasp.npz' if stage=='formed' else 'grasp.npz')) as saved:
        field = {'grasp':'qpos','initial':'initial_qpos','prediction':'predicted_terminal_qpos','formed':'qpos'}[stage]
        if stage=='prediction' and legacy_prediction:
            q=saved['qpos']
        elif stage=='initial' and 'initial_qpos' not in saved and 'seed_qpos' in saved:
            q=model.qpos0.copy();q[:len(saved['seed_qpos'])]=saved['seed_qpos']
        elif field not in saved:
            raise ValueError(f'{field} is missing in this older result; rerun grasp synthesis to export all stages')
        else:
            q = saved[field]
    display(model, q, case, viewer=False, render=render,
            image_name={'grasp':'grasp.png','initial':'initial.png','prediction':'prediction.png','formed':'formed.png'}[stage])
    if viewer:
        from retarget.grasp_preview import GraspPreview
        preview=GraspPreview(case,True)
        preview.update(q,np.zeros(model.nv),0.,stage)
        preview.close()


def _run_cases(args):
    from retarget.grasp_primitives import jobs, catalog
    from retarget.grasp_initialization import (global_alignment, pose_in_hand,
        validate_initialization, failed_initialization_attempt)
    from retarget.grasp_runtime import prepare_experiment, GraspRuntime
    from retarget.grasp_audit import GraspAudit
    from retarget.grasp_template import fit_contact_template
    from retarget.grasp_input_geometry import dense_power_template, fit_power_cylinder_input
    from scipy.spatial.transform import Rotation
    import warp as wp
    output = args.output.expanduser().resolve()/args.name
    output.mkdir(parents=True, exist_ok=False)
    if hasattr(args,'_tasks'):
        tasks = args._tasks
    elif args.primitives:
        tasks, skipped = jobs(output,args.templates,args)
        (output/'primitive_catalog.json').write_text(json.dumps(catalog(args.dexonomy_root),indent=2))
        (output/'skipped.json').write_text(json.dumps(skipped,indent=2))
    else:
        tasks = [(p.expanduser().resolve(),name,None) for p in args.object for name in args.templates]
    rng = np.random.default_rng(args.seed)
    if hasattr(args,'_rng_state'):
        rng.bit_generator.state=args._rng_state
    summary = []
    for object_index,(mesh,name,primitive) in enumerate(tasks):
        import copy
        object_rng_state=copy.deepcopy(rng.bit_generator.state)
        begun = time.perf_counter()
        model,surface,input_xml,offset = build_scene(mesh,args.scale,getattr(args,'hand_model','wuji1'))
        shadow,posed,contacts,source = load_template(args.dexonomy_root,name)
        bodies,local,normals,mapping = map_contacts(shadow,contacts,model)
        hand = StaticHand(model,surface,bodies,local,normals)
        if hasattr(args, 'gpu_nodes') and not getattr(args, 'legacy_serial', False):
            from retarget.grasp_seed import bone_seed_cuda
            fitted, bone_seed = bone_seed_cuda(model, anatomical_joints(shadow,posed)[None], device=args.init_device)
        else:
            fitted, bone_seed = retarget_from_mano(model,anatomical_joints(shadow,posed)[None])
        reference_qpos, reference_normals = fitted[0].copy(), normals.copy()
        hand.ik_reference_qpos = reference_qpos
        hand.contact_reference_normals = reference_normals
        print(f'[grasp] {mesh.name} / {name}',flush=True)
        template_fit = dict(method='bone_direction_IK_only', enabled=False)
        initialization_seconds = 0.
        if not getattr(args,'no_template_fit',False):
            opening = (2*primitive['radius_m'] if primitive and primitive['kind'] in ('cylinder','sphere') else None)
            fitted_q,template_fit = fit_contact_template(hand,shadow,posed,contacts,fitted[0],
                device=args.init_device,opening=opening,
                joint_prior=1. if getattr(args,'load_conditioning',False) else None,
                reference_qpos=reference_qpos,reference_normals=reference_normals)
            fitted[0] = fitted_q
            final_fit = template_fit
        guesses = global_alignment(hand,fitted[0],args,rng)
        for guess in guesses:
            validate_initialization(hand,guess,args,reference_normals)
        initialization_seconds += guesses[0].metadata.get('seconds',0.)
        if not getattr(args,'no_template_fit',False):
            # The first original initialization supplies attainable object
            # surface targets. Fit the Wuji hand to those contacts, then run
            # the original initializer again so all final poses/contacts remain
            # its unmodified outputs. No extra ADMM or object-pose optimizer.
            def quality(guess):
                info = guess.metadata
                return (not info.get('initialization_valid',False),
                        max(1.-info.get('forbidden_region_min_gap_mm',1.),0.),
                        max(max(info.get('initial_penetration_mm',float('inf')),
                                info.get('initial_self_penetration_mm',0.))-args.penetration_tolerance_mm,0.),
                        info.get('bone_direction_max_deviation_deg',0.), info.get('objective',float('inf')))
            template_fit['object_contact_refinements'] = []
            for refinement_index in range(getattr(args,'template_refinements',2)):
                guide = min(guesses,key=quality)
                if guide.metadata.get('initialization_valid', False):
                    break
                saved_local,saved_normals = hand.local.copy(),hand.normals.copy()
                reserve_weight = None
                if getattr(args,'load_conditioning',False):
                    reserve_weight = (1. if (guide.metadata.get('load_conditioning') or {}).get('feasible')
                                      else 2.**(refinement_index+1))
                power_cylinder = dense_power_template(hand,primitive)
                if power_cylinder:
                    refined_q,refinement = fit_power_cylinder_input(hand,guide.hand_qpos,
                        device=args.init_device,joint_prior=reserve_weight,
                        reference_qpos=reference_qpos,reference_normals=reference_normals)
                else:
                    refined_q,refinement = fit_contact_template(hand,shadow,posed,contacts,guide.hand_qpos,
                        device=args.init_device,object_contacts=(guide.object_points,-guide.object_normals),
                        joint_prior=reserve_weight,
                        reference_qpos=reference_qpos,reference_normals=reference_normals)
                refined_guesses = global_alignment(hand,refined_q,args,rng,pose_seed=refined_q)
                for guess in refined_guesses:
                    validate_initialization(hand,guess,args,reference_normals)
                initialization_seconds += refined_guesses[0].metadata.get('seconds',0.)
                accepted = min(map(quality,refined_guesses)) <= min(map(quality,guesses))
                template_fit['seconds'] += refinement['seconds']
                template_fit['object_contact_refinements'].append(dict(refinement,accepted=accepted,
                    source_initialization=guide.metadata.copy(),
                    refined_initialization=min(refined_guesses,key=quality).metadata.copy(),
                    source_object_points=guide.object_points.tolist(),
                    source_object_inward_normals=(-guide.object_normals).tolist()))
                if accepted:
                    guesses = refined_guesses
                    fitted[0],final_fit = refined_q,refinement
                    template_fit['after'] = refinement['after']
                else:
                    hand.local,hand.normals = saved_local,saved_normals
                    if (getattr(args,'load_conditioning',False) and
                            not (guide.metadata.get('load_conditioning') or {}).get('feasible')):
                        continue
                    break
                best_input = min(guesses,key=quality).metadata
                if (best_input.get('initialization_valid',False) and
                        max(best_input.get('matching',{}).get('gap_mm',[float('inf')]))<=args.contact_tolerance_mm):
                    break
            local,normals = hand.local,hand.normals
            for i,record in enumerate(mapping):
                record['initial_query_index'] = record['query_index']
                record['initial_mapping_distance_mm'] = record.pop('mapping_distance_mm')
                record.update(query_index=final_fit['surface_query_indices'][i],
                              wuji_local=local[i].tolist(),wuji_local_normal=normals[i].tolist(),
                              template_position_error_mm=final_fit['contact_position_error_mm'][i],
                              template_normal_error_deg=final_fit['contact_normal_error_deg'][i])
        for guess in guesses:
            guess.metadata['template_fit'] = template_fit
            guess.metadata['bone_seed'] = bone_seed
        case = output/f'{object_index:03d}_{mesh.stem}'/name
        case.mkdir(parents=True)
        auditor = GraspAudit(surface,args.init_device)
        attempts = []
        prepared = []
        for start,guess in enumerate(guesses):
            if guess.metadata.get('initialization_valid') is False:
                print(f'  initialization INVALID: reasons={",".join(guess.metadata["failure_reasons"])}; '
                      'saving IK diagnostics and skipping optimization',flush=True)
                attempts.append(failed_initialization_attempt(hand,guess,args,auditor,input_xml,start))
                continue
            input_started = time.perf_counter()
            experiment = prepare_experiment(case/f'input_{start:02d}',hand,guess,args)
            input_seconds = time.perf_counter()-input_started
            runtime = GraspRuntime(experiment,args)
            runtime.audit_context = (auditor, hand, args)
            prepared.append((start, guess, runtime, input_seconds))
        solved = yield [runtime for _, _, runtime, _ in prepared]
        for (start, guess, runtime, input_seconds), (plan, snapshot, qpos) in zip(prepared, solved):
            audit_started = time.perf_counter()
            if hasattr(runtime, 'device_audit_result'):
                audit, arrays = runtime.device_audit_result
            else:
                audit,arrays = auditor.evaluate(runtime,hand,args,int(plan.diag['selected']))
            audit_seconds = time.perf_counter()-audit_started
            # Express the selected next-step hand in the same-step object frame.
            obj = qpos[runtime.index.object_qpos:runtime.index.object_qpos+7]
            root = qpos[runtime.index.hand_qpos]
            rotation = Rotation.from_quat(obj[3:],scalar_first=True).inv()
            canonical = np.r_[rotation.apply(root[:3]-obj[:3]),
                (rotation*Rotation.from_euler('XYZ',root[3:6])).as_euler('XYZ'),root[6:]]
            audit.update(start=start,rounds=0 if runtime.grasp_optimizer == 'cem' else args.contact_rounds,
                         initialization=guess.metadata,seconds=plan.seconds,
                         setup_seconds=runtime.setup_seconds,audit_seconds=audit_seconds,
                         input_seconds=input_seconds,
                         output_time_seconds=runtime.output_step*runtime.cfg.simulator.time_grid.mpc_dt,
                         prediction_horizon_seconds=getattr(runtime,'prediction_horizon_seconds',
                             len(runtime.optimizer.backend.steps)*runtime.cfg.simulator.time_grid.mpc_dt),
                         object_shift_mm=float(np.linalg.norm(obj[:3]-runtime.qpos[runtime.index.object_qpos:runtime.index.object_qpos+3])*1000),
                         predicted_terminal_object_shift_mm=float(np.linalg.norm(
                             runtime.predicted_terminal_qpos[runtime.index.object_qpos:runtime.index.object_qpos+3]
                             -runtime.qpos[runtime.index.object_qpos:runtime.index.object_qpos+3])*1000),
                         termination=runtime.optimizer.last_solver_diag.get('termination','single_shared_solver_window'),
                         solver_diagnostics=runtime.optimizer.last_solver_diag)
            attempts.append(dict(audit=audit,arrays=arrays,qpos=qpos,canonical=canonical,
                                 initial_qpos=runtime.qpos.copy(),prediction=runtime.predicted_terminal_qpos,
                                 snapshot=snapshot,guess=guess,model=runtime.model,plan=plan,
                                 scene_xml=runtime.scene_xml))
            print(f'  start {start+1}/{len(guesses)}: solve={plan.seconds*1000:.1f} ms '
                  f'FC={audit["force_closure"]["force_closure"]} feasible={audit["feasible"]}',flush=True)
            # All starts are independent nodes; discard local rollout handles.
            del runtime
            import gc
            gc.collect()
        prepared.clear()
        best = min(attempts,key=lambda item:(not item['audit']['feasible'],
                   not item['audit'].get('optimizer_success',True),
                   not item['audit']['force_closure']['force_closure'],item['plan'].loss))
        guess,snap = best['guess'],best['snapshot']
        initialization_failed = best['audit']['termination'] == 'initialization_failed'
        hand.set(guess.hand_qpos)
        initial_points = hand.points(bodies,local)
        np.savez_compressed(case/'grasp.npz',qpos=best['qpos'],hand_qpos=best['canonical'],
            stage_valid=np.array([1,not initialization_failed],dtype=np.uint8),
            qvel=best['plan'].next_qvel,hold_action=best['plan'].action,
            initial_qpos=best['initial_qpos'],predicted_terminal_qpos=best['prediction'],
            optimized_initial_qpos=best['plan'].initial_qpos,
            seed_qpos=guess.hand_qpos,initial_object_pose_in_hand=pose_in_hand(guess.hand_qpos),
            final_object_pose_in_hand=pose_in_hand(best['canonical']),
            initial_hand_points=initial_points,initial_hand_local_points=local,
            initial_hand_local_normals=normals,initial_object_points=guess.object_points,
            initial_object_normals=guess.object_normals,template_object_points=guess.object_points,
            object_points=snap['z'][0],dual=snap['dual'][0],
            hand_body_ids=bodies,hand_local_points=snap.get('hand_local_points',local),
            hand_local_normals=snap.get('hand_local_normals',normals),
            mesh_center_before_recentering=offset,**best['arrays'],
            **{'dexonomy_'+key:value for key,value in guess.raw.items()})
        np.savez_compressed(case/'contacts.npz',**snap)
        (case/'scene.xml').write_text(best['scene_xml'])
        timings = dict(total_seconds=time.perf_counter()-begun,
            template_fit_seconds=template_fit.get('seconds',0.),
            initialization_seconds=initialization_seconds,
            input_seconds=sum(a['audit']['input_seconds'] for a in attempts),
            setup_seconds=sum(a['audit']['setup_seconds'] for a in attempts),
            solve_seconds=sum(a['audit']['seconds'] for a in attempts),
            audit_seconds=sum(a['audit']['audit_seconds'] for a in attempts))
        if initialization_failed:
            solver_description = 'initialization_failed: optimizer not run'
            output_description = 'failed initialization diagnostic; no optimized grasp or prediction'
        elif getattr(args,'grasp_optimizer','adamw') == 'cem':
            solver_description = ('retarget.grasp_cem: CARD CEM + compliant contact prediction and posture/preload refinement'
                if getattr(args,'explicit_pose',False) else 'retarget.grasp_cem: static CARD annealed CEM')
            output_description = ('static grasp configuration; explicit model prediction saved separately'
                if getattr(args,'explicit_pose',False) else 'static CARD configuration; no execution or prediction')
        else:
            solver_description = ('retarget.grasp_device: native CUDA graph AdamW/contact ADMM'
                if best['audit']['solver_diagnostics'].get('device_resident_iterations') else
                'retarget.fa.solver.optimizer_class: same as run.py --adamw')
            output_description = 'plan.next_qpos: next executed interval endpoint; full prediction saved separately'
        report = dict(object=str(mesh),mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
            scale=args.scale,hand_model=hand.hand_model,template=name,primitive=primitive,source_annotation=str(source),
            annotation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            upstream_source_sha256={relative:hashlib.sha256((args.dexonomy_root/relative).read_bytes()).hexdigest()
                for relative in ('dexonomy/data/obj_loader.py','dexonomy/data/hand_loader.py',
                                 'dexonomy/op/gen_init.py','dexonomy/util/warp_util.py')
                if (args.dexonomy_root/relative).is_file()},
            solver=solver_description,
            solver_device=args.init_device,task=f'stationary force closure; planning gravity={getattr(args,"planning_gravity",0.):g}; no floor or lift',
            output_pose=output_description,
            contact_frame='hand_points/object_points in output object frame; qpos in world frame',
            mapping=mapping,selected=best['audit'],starts=[a['audit'] for a in attempts],timings=timings,
            initialization_rng_state=object_rng_state,
            settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                      if k!='object' and not k.startswith('_')})
        (case/'result.json').write_text(json.dumps(report,indent=2))
        drop_report = None
        if getattr(args,'drop_test',False) and not initialization_failed:
            from retarget.grasp_preview import simulate_case
            drop_report = simulate_case(case,args)
        row = dict(object=str(mesh),hand_model=hand.hand_model,template=name,primitive=primitive,case=str(case),
                   **best['audit'],timings=timings,drop_test=drop_report)
        attempts_physical = [dict(case=str(case),seed=args.seed,
            candidate_policy=getattr(args,'init_selection','matching'),
            status=(drop_report or {}).get('status'),verified_grasp=(drop_report or {}).get('verified_grasp'))]
        if (getattr(args,'load_conditioning',False) and drop_report is not None
                and not drop_report.get('verified_grasp') and getattr(args,'initialization_retries',0)>0):
            retry = copy.copy(args)
            retry.init_selection='matching' if getattr(args,'init_selection','matching')=='interior' else 'interior'
            if not getattr(args,'_alternate_done',False):
                retry._rng_state=object_rng_state
                retry._alternate_done=True
            else:
                retry.seed=args.seed+1
                retry._alternate_done=False
                if hasattr(retry,'_rng_state'):
                    del retry._rng_state
            retry.output,retry.name = output,f'retry_{object_index:03d}_{retry.init_selection}_seed{retry.seed}'
            retry.initialization_retries -= 1
            retry._tasks = [(mesh,name,primitive)]
            print('  [input retry] input/physical validation failed; preserving this attempt and rebuilding the original initialization',flush=True)
            retried = json.loads((run(retry)/'summary.json').read_text())[0]
            attempts_physical.extend(retried.get('physical_initialization_attempts',[]))
            if (retried.get('drop_test') or {}).get('verified_grasp'):
                row = retried
        row['physical_initialization_attempts'] = attempts_physical
        summary.append(row)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        columns=['template','primitive','radius_m','height_m','force_closure','feasible','rank',
                 'solve_ms','setup_seconds','initialization_seconds','audit_seconds',
                 'drop_status','retained','dropped','formed_feasible','verified_grasp','input_attempts','case']
        with (output/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for row in summary:
                shape=row['primitive'] or {}
                writer.writerow(dict(template=row['template'],primitive=shape.get('kind','mesh'),
                    radius_m=shape.get('radius_m'),height_m=shape.get('height_m'),
                    force_closure=row['force_closure']['force_closure'],feasible=row['feasible'],
                    rank=row['force_closure']['rank'],solve_ms=row['timings']['solve_seconds']*1000,
                    drop_status=(row['drop_test'] or {}).get('status'),
                    retained=(row['drop_test'] or {}).get('retained'),dropped=(row['drop_test'] or {}).get('dropped'),
                    formed_feasible=((row['drop_test'] or {}).get('formed_geometry') or {}).get('feasible'),
                    verified_grasp=(row['drop_test'] or {}).get('verified_grasp'),
                    input_attempts=len(row.get('physical_initialization_attempts',[])),
                    **{key:row['timings'][key] for key in ('setup_seconds','initialization_seconds','audit_seconds')},
                    case=row['case']))
        if args.viewer or args.render:
            if args.render:
                display(best['model'],best['initial_qpos'],case,render=True,image_name='initial.png')
            if initialization_failed:
                if args.viewer:
                    display(best['model'],best['initial_qpos'],case,viewer=True)
                continue
            if args.render:
                display(best['model'],best['prediction'],case,render=True,image_name='prediction.png')
            display(best['model'],best['qpos'],case,render=args.render)
            if args.viewer and not getattr(args,'drop_test',False):
                from retarget.grasp_preview import GraspPreview
                preview=GraspPreview(case,True)
                preview.update(best['qpos'],best['plan'].next_qvel,0.,'static grasp')
                preview.close()
    print(f'[saved] {output}',flush=True)
    return output


def run(args):
    """Prepare scene nodes, execute their graphs together, then export results."""
    if getattr(args, 'legacy_serial', False) or not hasattr(args, 'gpu_nodes'):
        pending = _run_cases(args)
        try:
            runtimes = next(pending)
            while True:
                runtimes = pending.send([runtime.solve() for runtime in runtimes])
        except StopIteration as complete:
            return complete.value
    import copy
    from retarget.grasp_device import solve_nodes
    from retarget.grasp_primitives import jobs, catalog
    output = args.output.expanduser().resolve()/args.name
    output.mkdir(parents=True, exist_ok=False)
    if hasattr(args, '_tasks'):
        tasks = args._tasks
    elif args.primitives:
        tasks, skipped = jobs(output, args.templates, args)
        (output/'primitive_catalog.json').write_text(json.dumps(catalog(args.dexonomy_root), indent=2))
        (output/'skipped.json').write_text(json.dumps(skipped, indent=2))
    else:
        tasks = [(p.expanduser().resolve(), name, None) for p in args.object for name in args.templates]
    width = args.gpu_nodes or max(len(tasks)*args.starts, 1)
    scenes_per_group = max(1, width//args.starts)
    summary, dispatches = [], []
    # Copy each task's RNG state from one deterministic stream. Node preparation
    # remains a boundary operation; optimization graphs never share buffers.
    rng = np.random.default_rng(args.seed)
    for offset in range(0, len(tasks), scenes_per_group):
        active = []
        node_outputs = []
        for index, task in enumerate(tasks[offset:offset+scenes_per_group], start=offset):
            child = copy.copy(args)
            child.output, child.name = output, f'node_{index:04d}'
            child._tasks = [task]
            seed = int(rng.integers(0, 2**31-1)) if index else args.seed
            child.seed = args._task_seeds[index] if hasattr(args, '_task_seeds') else seed
            pending = _run_cases(child)
            active.append((pending, next(pending)))
        while active:
            runtimes = [runtime for _, group in active for runtime in group]
            results = []
            for begin in range(0, len(runtimes), width):
                group = runtimes[begin:begin+width]
                results.extend(solve_nodes(group))
                dispatches.append(dict(nodes=len(group), worlds=sum(runtime.optimizer.last_solver_diag['evaluation_worlds']
                    for runtime in group), scheduling='independent_cuda_streams',
                    completion_barriers=1, host_iteration_syncs=0, host_graph_launches=1,
                    observed_overlapping_compute_graphs=max(runtime.optimizer.last_solver_diag['node_dispatch']
                        ['observed_overlapping_compute_graphs'] for runtime in group)))
            waiting = []
            cursor = 0
            for pending, group in active:
                result = results[cursor:cursor+len(group)]
                cursor += len(group)
                try:
                    waiting.append((pending, pending.send(result)))
                except StopIteration as complete:
                    node_outputs.append(complete.value)
            active = waiting
        for node_output in node_outputs:
            summary.extend(json.loads((node_output/'summary.json').read_text()))
        (output/'summary.json').write_text(json.dumps(summary, indent=2))
        if hasattr(args, '_compact_output'):
            from retarget.grasp_compact import write_batch
            from retarget.grasp_results import method_name
            write_batch(summary, args._compact_output,
                method=method_name(args.grasp_optimizer, args.explicit_pose))
        (output/'gpu_dispatch.json').write_text(json.dumps(dispatches, indent=2))
        # Preserve the established CSV contract, including failed grasps.
        with (output/'summary.csv').open('w', newline='') as stream:
            writer = None
            for node_output in sorted(output.glob('node_*')):
                csv_path = node_output/'summary.csv'
                if not csv_path.exists():
                    continue
                with csv_path.open(newline='') as source:
                    reader = csv.DictReader(source)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    writer.writerows(reader)
        import gc
        gc.collect()
    print(f'[saved GPU nodes] {output}', flush=True)
    return output

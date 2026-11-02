"""Memory-budgeted CUDA MPPI along adjacent size grids, anchored at solved 1x."""
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from copy import deepcopy
from functools import partial
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from retarget.common.results import write_json
from .template_mppi_continuation import POLICY, MotionLimits, densify, outward_chains, pose_change
from .template_mppi_reference import pose_matrix


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(config):
    from .template_mppi_runner import validate as base_validate
    from .template_mppi_parallel import options
    child=deepcopy(config)
    values=child.environment['template_mppi'].pop('continuation')
    if not isinstance(values,dict) or set(values)-{'source_batches','subdivisions','types'}:
        raise ValueError('continuation requires source_batches and optional subdivisions/types')
    subdivisions=values.get('subdivisions',4)
    densify([],subdivisions)
    sources=values.get('source_batches')
    if not isinstance(sources,list) or not sources:
        raise ValueError('continuation.source_batches must contain audited batches')
    for root in sources:
        audit=read(Path(root)/'independent_result_audit.json')
        if not audit['passed'] or not audit['complete']:
            raise ValueError(f'Continuation requires a complete audited source: {root}')
    args=base_validate(child)
    options(child.environment['template_mppi'].get('parallel',{}))
    if args.settings.objective.pose_weight or not args.settings.size_ik.enabled:
        raise ValueError('Continuation requires GPU size IK and no fixed 1x pose prior')
    return args,values,child


def absolute_entry(root,entry):
    e=dict(entry)
    for key in ('model','trajectory','summary','patch_coverage'):
        if e.get(key):e[key]=str((root/e[key]).resolve())
    return e


def reference_from_parent(directory,parent):
    """Freeze parent contacts/identity, then let the existing GPU IK scale them."""
    case=Path(parent['trajectory']).parent
    initial=read(case/'initialization.json')
    record=read(initial['reference'])
    # Older exact-1x seeds predate canonical-grasp bootstrap. Preserve their
    # original bone pose before replacing the initial q with the solved parent.
    record.setdefault('root_match_initial_qpos',record['initial_qpos'])
    d=read(case/'projection_force_closure.json')
    q=np.load(case/'grasp.npz')['qpos']
    directory.mkdir(parents=True,exist_ok=True)
    record.update(policy=POLICY, canonical_primitive=parent['primitive'],
        canonical_size_scale=parent['primitive']['size_scale'], initial_qpos=q.tolist(),
        canonical_grasp_qpos=q.tolist(),object_points=d['projected_points'],
        object_outward_normals=d['object_normals'],wrist_in_object=pose_matrix(q).tolist(),
        object_in_wrist=np.linalg.inv(pose_matrix(q)).tolist(),
        object_contact_reference='solved_adjacent_grasp',
        canonical_grasp_reference=dict(source=str(case/'projection_force_closure.json'),
            sha256=sha(case/'projection_force_closure.json'),qpos_source=str(case/'grasp.npz'),
            qpos_sha256=sha(case/'grasp.npz'),policy=POLICY,from_cache=True),
        pose_reuse='adjacent solved object-in-wrist transform; no translation scaling',
        parent_case=parent['name'],parent_scale=parent['primitive']['size_scale'])
    write_json(directory/'reference.json',record)
    write_json(directory/'template_contacts.json',read(case/'template_contacts.json'))
    np.savez_compressed(directory/'candidates.npz',qpos=q[None])
    return directory/'reference.json',q,d.get('side_contact_slots')


def solve_node(config,root,node,parent):
    """One CUDA child; retries remain reviewable and never change a fixed anchor."""
    import torch
    import warp as wp
    from .template_mppi_runner import run
    from .template_mppi_continuation import ContinuationObjective
    root=Path(root);base=root/'cases'/node['name'];base.mkdir(parents=True,exist_ok=True)
    reference,parent_q,slots=reference_from_parent(base/'adjacent_reference',parent)
    reports=[];started=time.perf_counter()
    with (base/'worker.log').open('a',buffering=1) as log:
        saved=os.dup(1),os.dup(2)
        try:
            os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
            wp.config.kernel_cache_dir='/tmp/retarget-template-mppi-warp'
            # Prefer exactly the same object pose. Relax only when the first
            # result lacks projected closure or anatomical patch coverage.
            limits=(MotionLimits(),MotionLimits(.002,2.,12.),MotionLimits(.003,3.,16.))
            for attempt,limit in enumerate(limits):
                child=deepcopy(config)
                child.environment['template_mppi'].pop('parallel',None)
                child.task['object']=node['mesh'];child.task['templates']=[node['template']]
                child.seed=config.seed+node['index']*1009
                target=base/f'attempt_{attempt}'
                entries=run(child,target,initialization_reference=str(reference),primitive_override=node['primitive'],
                    objective_factory=partial(ContinuationObjective,parent_q=parent_q,limits=limit,side_slots=slots))
                summary=read(target/'summary.json')['cases'][0]
                entry=absolute_entry(target,entries[0]) if entries else None
                d=summary.get('geometry_diagnostics',{})
                report=dict(attempt=attempt,limits=vars(limit),entry=entry,summary=summary)
                reports.append(report)
                if summary['solver_completed'] and d['soft_contact_success'] and d['coverage']>=.8-1e-7:
                    break
                gc.collect();torch.cuda.empty_cache()
            def rank(r):
                d=r['summary'].get('geometry_diagnostics',{})
                return (not r['summary']['solver_completed'],not d.get('valid',False),
                    not d.get('hand_hand_penetration_accepted',False),not d.get('projected_closure',False),
                    -d.get('coverage',0),d.get('cost',float('inf')),r['attempt'])
            chosen=min(reports,key=rank)
            if not chosen['summary']['solver_completed']:
                raise RuntimeError(f"All continuation attempts failed: {chosen['summary'].get('error')}")
            entry=dict(chosen['entry'],name=node['name'],continuation_parent=parent['name'])
            summary=dict(chosen['summary'],index=node['index'],case_id=node['name'])
            q=np.load(Path(entry['trajectory']).with_name('grasp.npz'))['qpos']
            motion=pose_change(parent_q,q)
            continuity=dict(policy=POLICY,parent_case=parent['name'],parent_entry=parent,
                parent_primitive=parent['primitive'],primitive=node['primitive'],
                parent_reference=str(reference),reference_sha256=sha(reference),
                motion=motion,selected_attempt=chosen['attempt'],limits=chosen['limits'],
                attempts=[dict(attempt=r['attempt'],limits=r['limits'],entry=r['entry'],
                    solver_completed=r['summary']['solver_completed'],
                    projected_closure=r['summary'].get('geometry_diagnostics',{}).get('projected_closure'),
                    coverage=r['summary'].get('geometry_diagnostics',{}).get('coverage')) for r in reports],
                seconds=time.perf_counter()-started)
            write_json(base/'continuation.json',continuity)
            entry['continuation']=str(base/'continuation.json')
            return dict(index=node['index'],entries=[entry],summary=summary,continuity=continuity)
        finally:
            gc.collect();torch.cuda.empty_cache()
            for target,source in zip((1,2),saved):os.dup2(source,target);os.close(source)


def make_plan(config,output,values):
    from .primitives import primitive_mesh
    sources=[Path(p).resolve() for p in values['source_batches']]
    source_entries=[]
    for source in sources:
        source_entries.extend(absolute_entry(source,e) for e in read(source/'manifest.json')['cases'])
    # Only the first batch supplies the requested ranges; extra batches supply
    # exact 1x grasps when a historical coarse sweep omitted that sample.
    first=[absolute_entry(sources[0],e) for e in read(sources[0]/'manifest.json')['cases']]
    names=list(dict.fromkeys(e['template'] for e in first))
    if values.get('types'):
        names=[n for n in names if int(n.partition('_')[0]) in values['types']]
    nodes=[];anchors={};groups=[]
    for name in names:
        family=[e for e in first if e['template']==name]
        kinds=list(dict.fromkeys(e['primitive']['kind'] for e in family))
        for kind in kinds:
            coarse=[e for e in family if e['primitive']['kind']==kind]
            anchor=next((e for e in source_entries if e['template']==name and e['primitive']['kind']==kind
                and abs(e['primitive']['size_scale']-1)<1e-10),None)
            if anchor is None:raise ValueError(f'No audited exact 1x grasp: {name}/{kind}')
            primitive_grid=densify([e['primitive'] for e in coarse],values.get('subdivisions',4))
            grid,chains,pivot=outward_chains(primitive_grid,anchor['primitive'])
            indices=[]
            for p in grid:
                index=len(nodes);case_id=f'{index:04d}_{name}'
                mesh=primitive_mesh(output/'meshes'/f'{case_id}.obj',kind,p['radius_m'],p['height_m'])
                nodes.append(dict(index=index,name=case_id,template=name,primitive=p,mesh=str(mesh.resolve())))
                indices.append(index)
            i=indices[pivot]
            anchors[i]=dict(index=i,entries=[dict(anchor,name=nodes[i]['name'],source_case_name=anchor['name'],continuation_anchor=True)],
                summary=dict(read(anchor['summary']),index=i,case_id=nodes[i]['name']))
            for chain in chains:
                parent=i
                for position in chain:
                    j=indices[position];nodes[j]['parent_index']=parent;parent=j
            groups.append(dict(template=name,kind=kind,anchor_index=i,indices=indices,
                original_scales=[e['primitive']['size_scale'] for e in coarse],scales=[p['size_scale'] for p in grid]))
    write_json(output/'continuation_plan.json',dict(policy=POLICY,subdivisions=values.get('subdivisions',4),
        planned=len(nodes),anchors=len(anchors),recomputed=len(nodes)-len(anchors),groups=groups,nodes=nodes,
        source_audit_sha256={str(p/'independent_result_audit.json'):sha(p/'independent_result_audit.json') for p in sources}))
    return nodes,anchors


def run(config,output):
    import torch
    from .template_mppi_parallel import options,worker_count
    from .template_mppi_runner import OBJECTIVE
    from retarget.common.provenance import ROOT
    args,values,child=validate(config);output=Path(output);started=time.perf_counter()
    settings=options(child.environment['template_mppi'].get('parallel',{}))
    nodes,completed=make_plan(config,output,values)
    write_json(output/'skipped.json',read(Path(values['source_batches'][0])/'skipped.json'))
    free,total=torch.cuda.mem_get_info(config.device);budget=int(free*settings['memory_fraction'])
    resource=dict(device=torch.cuda.get_device_name(config.device),total_bytes=total,
        baseline_free_bytes=free,budget_bytes=budget,**settings,minimum_free_bytes=free,
        peak_incremental_bytes=0,peak_fraction_of_initial_free=0.,physical_evaluation=False,
        response_backend=args.settings.objective.closure.equilibrium_backend,
        scheduling='adjacent-size dependency DAG; sum of measured CUDA worker peaks within budget',
        worker_lifetime='one adjacent grid including bounded retries',peak_active_workers=0)
    sources=list((ROOT/'retarget').rglob('*.py'))+[ROOT/'run.py',config.source]
    write_json(output/'source_sha256_at_start.json',{str(p):sha(p) for p in sources})
    def memory():
        available=torch.cuda.mem_get_info(config.device)[0]
        resource['minimum_free_bytes']=min(resource['minimum_free_bytes'],available)
        resource['peak_incremental_bytes']=max(resource['peak_incremental_bytes'],free-available)
        resource['peak_fraction_of_initial_free']=resource['peak_incremental_bytes']/free
    def save(state='running'):
        ordered=[completed[i] for i in sorted(completed)]
        entries=[e for r in ordered for e in r['entries']];rows=[r['summary'] for r in ordered]
        write_json(output/'manifest.json',dict(version=1,task='grasp',method='mppi',cases=entries))
        write_json(output/'summary.json',dict(method='mppi',objective=OBJECTIVE,planned=len(nodes),finished=len(rows),
            solver_completed=sum(r['solver_completed'] for r in rows),native_evaluated=0,
            successes=sum(r['task_success'] for r in rows),cases=rows))
        write_json(output/'resources.json',resource)
        write_json(output/'status.json',dict(state=state,planned=len(nodes),finished=len(rows),wall_seconds=time.perf_counter()-started))
        return entries
    # Contact-count profiles are measured with this exact solver and retries.
    counts={}
    for r in completed.values():
        e=r['entries'][0]
        counts[e['template']]=len(read(Path(e['trajectory']).with_name('template_contacts.json'))['body_names'])
    remaining={n['index'] for n in nodes}-set(completed)
    context=mp.get_context('spawn');profiles={}
    representatives={}
    for i in sorted(remaining):
        if nodes[i]['parent_index'] in completed:
            representatives.setdefault(counts[nodes[i]['template']],i)
    save('calibrating')
    for n,i in sorted(representatives.items(),reverse=True):
        before=torch.cuda.mem_get_info(config.device)[0];minimum=before
        with ProcessPoolExecutor(max_workers=1,mp_context=context) as pool:
            f=pool.submit(solve_node,child,str(output),nodes[i],completed[nodes[i]['parent_index']]['entries'][0])
            while not f.done():
                wait([f],timeout=.1);memory();minimum=min(minimum,torch.cuda.mem_get_info(config.device)[0])
            completed[i]=f.result();remaining.remove(i)
        profiles[n]=before-minimum
        worker_count(budget,profiles[n],settings['max_workers'],settings['safety_factor'])
        resource['calibration_profiles']={str(k):dict(peak_bytes=v,case=representatives[k]) for k,v in profiles.items()}
        print(f'[dense_mppi] calibrated n={n}, peak={profiles[n]/2**30:.2f} GiB; {len(completed)}/{len(nodes)}',flush=True)
        save()
    workers=min(settings['max_workers'],len(remaining) or 1)
    with ProcessPoolExecutor(max_workers=workers,mp_context=context,max_tasks_per_child=1) as pool:
        pending={}
        while remaining or pending:
            committed=sum(profiles[counts[nodes[i]['template']]]*settings['safety_factor'] for i in pending.values())
            # Start long remaining branches first; every dependency must have a
            # completed final grasp, never an unevaluated or interpolated pose.
            ready=[i for i in sorted(remaining) if nodes[i]['parent_index'] in completed]
            for i in ready:
                predicted=profiles[counts[nodes[i]['template']]]*settings['safety_factor']
                if len(pending)>=workers:break
                if committed+predicted>budget or torch.cuda.mem_get_info(config.device)[0]<free-budget:continue
                parent=completed[nodes[i]['parent_index']]['entries'][0]
                pending[pool.submit(solve_node,child,str(output),nodes[i],parent)]=i
                remaining.remove(i);committed+=predicted
            if not pending:raise RuntimeError('No dependency-ready solve fits the current CUDA memory budget')
            resource['peak_active_workers']=max(resource['peak_active_workers'],len(pending))
            resource['active_workers']=len(pending)
            done,_=wait(pending,timeout=.2,return_when=FIRST_COMPLETED);memory()
            for f in done:
                i=pending.pop(f);completed[i]=f.result();d=completed[i]['summary']['geometry_diagnostics']
                print(f"[dense_mppi] {len(completed)}/{len(nodes)} {nodes[i]['template']} "
                    f"scale={nodes[i]['primitive']['size_scale']:.6g} closure={d['projected_closure']} "
                    f"patch={d['coverage']:.0%} attempt={completed[i]['continuity']['selected_attempt']}",flush=True)
                save()
    memory();resource.update(wall_seconds=time.perf_counter()-started,active_workers=0)
    entries=save('completed')
    write_json(output/'template_initialization/manifest.json',dict(policy=POLICY,fits=len(counts),
        cases=len(nodes),anchors=len(counts),reference='per-node continuation parent; fixed canonical anchor per family',
        fk_backend='torch_cuda_batched_WujiContactKinematics'))
    return entries

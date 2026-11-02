#!/usr/bin/env python3
"""Paired, resumable MPPI local refinement of saved final GraspADMM poses."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from retarget.common.results import write_json
from retarget.env.grasp.grasp_evaluation import PROTOCOL, NativeEvaluationWorker, evaluate_native_grasp
from retarget.env.grasp.graspadmm_mppi import LocalPoseObjective, RefinementSettings, local_to_native


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_ready(directory):
    return all((Path(directory)/name).is_file() for name in (
        'native_candidate.json','native_candidate.npz','graspadmm_evaluation/evaluation.json',
        'graspadmm_evaluation/candidate.npz'))


def transition(before,after):
    return ('retained' if after else 'lost') if before else ('recovered' if after else 'neither')


def size_index(case):
    return int(re.search(r'_s(\d+)_seed',case['case_id'])[1])


def resolve_case_paths(cases, source):
    """Batch manifests use source-relative paths; merged archives may be absolute."""
    result=[]
    for case in cases:
        row=dict(case)
        for key in ('summary','model','trajectory'):
            if row.get(key):
                path=Path(row[key])
                row[key]=str(path if path.is_absolute() else (Path(source)/path).resolve())
        result.append(row)
    return result


def aggregate(rows):
    n=len(rows);before=sum(bool(x['baseline_success']) for x in rows)
    after=sum(bool(x.get('task_success')) for x in rows)
    counts=Counter(transition(x['baseline_success'],x.get('task_success',False)) for x in rows)
    times=[x['optimization_seconds'] for x in rows if x.get('optimization_seconds') is not None]
    return dict(trials=n,baseline_successes=before,refined_successes=after,
        baseline_success_rate=before/n if n else None,refined_success_rate=after/n if n else None,
        validated_fallback_successes=before+counts['recovered'],
        validated_fallback_success_rate=(before+counts['recovered'])/n if n else None,
        percentage_point_change=100*(after-before)/n if n else None,
        transitions={k:counts[k] for k in ('retained','lost','recovered','neither')},
        failure_stages=dict(Counter(x.get('failure_stage') or 'passed' for x in rows)),
        semantic_evaluated=sum('semantic_success' in x for x in rows),
        semantic_successes=sum(bool(x.get('semantic_success')) for x in rows),
        optimization_seconds_median=float(np.median(times)) if times else None,
        optimization_seconds_p95=float(np.percentile(times,95)) if times else None,
        optimization_seconds_sum=sum(times))


def report(output,cases,rows,wall_seconds,settings):
    groups={};bytype={}
    for row in rows:
        key=(row['type_id'],row['size_index'])
        groups.setdefault(key,[]).append(row);bytype.setdefault(row['type_id'],[]).append(row)
    grouped=[dict(type_id=k[0],size_index=k[1],primitive=v[0]['primitive'],**aggregate(v)) for k,v in sorted(groups.items())]
    typed=[dict(type_id=k,**aggregate(v)) for k,v in sorted(bytype.items())]
    document=dict(protocol=PROTOCOL,method='graspadmm_mppi',planned_trials=len(cases),finished=len(rows),
        state='completed' if len(rows)==len(cases) else 'running',wall_seconds=wall_seconds,
        settings=asdict(settings),**aggregate(rows),by_type=typed,by_size=grouped,
        denominator='All planned ADMM cases, with runtime/missing-output failures included at completion',
        raw_policy='Always use the MPPI best geometry-cost candidate; never use native success in optimization',
        fallback_policy='Explicit separate policy: retain original ADMM only if refined candidate fails native evaluation',
        comparison_scope='Paired identical final ADMM inputs and actual meshes; historical MPPI inventory is a different cohort')
    write_json(output/'report.json',document)
    write_json(output/'summary.json',dict(method='graspadmm_mppi',protocol=PROTOCOL,jobs=rows))
    fields=['id','type_id','size_index','seed','baseline_success','task_success','transition','validated_fallback_success',
        'failure_stage','semantic_success','reference_orientation_error_deg','initial_cost','final_cost',
        'optimization_seconds','evaluation_seconds','wall_seconds']
    with (output/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    for name,data in [('by_type',typed),('by_size',grouped)]:
        values=[]
        for row in data:
            value={k:v for k,v in row.items() if not isinstance(v,dict)}
            value.update(row['transitions'])
            if 'primitive' in row:
                value.update(primitive=row['primitive']['kind'],size_axis=row['primitive']['size_axis'],
                    size_scale=row['primitive']['size_scale'],dimensions_m=json.dumps(row['primitive']['dimensions_m']))
            values.append(value)
        if values:
            with (output/f'{name}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    manifest=[]
    source_by_id={x['case_id']:x for x in cases}
    for row in rows:
        if not row.get('model'):continue
        source=source_by_id[row['id']]
        manifest.append(dict(source,method='graspadmm_mppi',name=row['id'],model=row['model'],
            trajectory=row['trajectory'],summary=str(output/'cases'/row['id']/'summary.json'),
            dt=1.,phase='refinement',task_success=bool(row['task_success']),
            baseline_success=row['baseline_success'],source_admm=source['summary']))
        evaldir=output/'cases'/row['id']/'evaluation'
        if (evaldir/'evaluation_trajectory.npz').is_file():
            manifest.append(dict(manifest[-1],name=row['id']+'_native_test',phase='native_test',
                model=str(evaldir/'evaluation_scene.xml'),trajectory=str(evaldir/'evaluation_trajectory.npz'),dt=.004))
    write_json(output/'manifest.json',dict(version=1,task='grasp',method='graspadmm_mppi',cases=manifest))
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/graspadmm_mppi_refine.yaml')
    parser.add_argument('--source',type=Path,default=ROOT/'output/graspadmm_wuji2_results')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--types',type=int,nargs='+')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--size-indices',type=int,nargs='+')
    parser.add_argument('--seed-offset',type=int,default=200000)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve()
    if output.parent!=ROOT/'output' or output.is_symlink() or output==source:
        raise ValueError('Use a separate real direct child of output/')
    config=yaml.safe_load(args.config.read_text());settings=RefinementSettings(**config['refinement'])
    torch.set_num_threads(1)
    cases=resolve_case_paths(json.loads((source/'manifest.json').read_text())['cases'],source)
    cases=[c for c in cases if (not args.types or int(c['template'].split('_')[0]) in args.types)
        and (not args.seeds or c['seed'] in args.seeds)
        and (not args.size_indices or size_index(c) in args.size_indices)]
    cases=sorted(cases,key=lambda c:c['case_id'])
    if args.limit is not None:cases=cases[:args.limit]
    if not cases:raise ValueError('No source cases')
    codefiles=[Path(__file__),*sorted((ROOT/'retarget').rglob('*.py'))]
    inputs={}
    for c in cases:
        p=Path(c['summary']).parent
        if not source_ready(p):
            inputs[c['case_id']]={str(p/'summary.json'):sha(p/'summary.json')}
            continue
        meta=json.loads((p/'native_candidate.json').read_text())
        files=[p/'native_candidate.npz',p/'native_candidate.json',p/'summary.json',
            p/'graspadmm_evaluation/evaluation.json',Path(meta['mesh_path'])]
        inputs[c['case_id']]={str(f):sha(f) for f in files}
    plan=dict(source=str(source),source_manifest_sha256=sha(source/'manifest.json'),settings=asdict(settings),
        seed_offset=args.seed_offset,case_ids=[c['case_id'] for c in cases],source_hashes=inputs,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in codefiles},protocol=PROTOCOL)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text())!=plan:
            raise ValueError('Plan/source/code changed; use a new output directory')
    else:
        write_json(output/'plan.json',plan)
        for p in codefiles:
            target=output/'code'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        shutil.copy2(args.config,output/'config.yaml')
        versions={name:importlib.metadata.version(name) for name in ('torch','numpy','mujoco','mujoco-warp','warp-lang','trimesh')}
        write_json(output/'environment.json',dict(versions=versions,python=sys.executable,gpu=torch.cuda.get_device_name(),
            native_evaluator=str(ROOT/'.venv-graspadmm/bin/python'),threads=1))
    rows=[];objective=None;object_key=None;started=time.perf_counter()
    prior_wall=json.loads((output/'report.json').read_text()).get('wall_seconds',0) if (output/'report.json').exists() else 0
    with NativeEvaluationWorker(output/'native_worker.log') as worker:
        for c in cases:
            dest=output/'cases'/c['case_id'];summary_path=dest/'summary.json'
            if summary_path.is_file():
                rows.append(json.loads(summary_path.read_text()));continue
            t=time.perf_counter();dest.mkdir(parents=True,exist_ok=True)
            p=Path(c['summary']).parent
            if not source_ready(p):
                failed=json.loads((p/'summary.json').read_text())
                row=dict(id=c['case_id'],type_id=int(c['template'].split('_')[0]),size_index=size_index(c),
                    seed=c['seed'],primitive=c['primitive'],baseline_success=False,task_success=False,
                    state='failed',failure_stage=failed.get('failure_stage','source_missing_candidate'),
                    error=failed.get('error'),source_summary=str(p/'summary.json'),source_hashes=inputs[c['case_id']],
                    protocol=PROTOCOL,transition='neither',validated_fallback_success=False,wall_seconds=time.perf_counter()-t)
                write_json(summary_path,row);rows.append(row)
                report(output,cases,rows,prior_wall+time.perf_counter()-started,settings)
                print(f"[{len(rows)}/{len(cases)}] {row['id']} source initialization failed",flush=True)
                continue
            meta=json.loads((p/'native_candidate.json').read_text())
            baseline=json.loads((p/'graspadmm_evaluation/evaluation.json').read_text())
            if baseline['protocol']!=PROTOCOL:raise ValueError('Baseline uses a different acceptance protocol')
            # Check that the evaluated original really is the saved final pose.
            with np.load(p/'native_candidate.npz') as raw:grasp=raw['grasp_qpos'].copy()
            with np.load(p/'graspadmm_evaluation/candidate.npz') as raw:
                if not np.array_equal(grasp,raw['grasp_qpos']):raise ValueError('Baseline evaluated a different grasp')
            if sha(meta['mesh_path'])!=sha(p/'graspadmm_evaluation/object.obj'):
                raise ValueError('Baseline evaluated a different object mesh')
            row=dict(id=c['case_id'],type_id=int(c['template'].split('_')[0]),
                size_index=size_index(c),seed=c['seed'],primitive=c['primitive'],
                baseline_success=bool(baseline['success']),source_summary=str(p/'summary.json'),
                source_hashes=inputs[c['case_id']],protocol=PROTOCOL,task_success=False,state='failed',failure_stage='refinement_runtime')
            try:
                key=sha(meta['mesh_path'])
                if object_key!=key:
                    objective=None;gc.collect();torch.cuda.empty_cache()
                    setup=time.perf_counter();objective=LocalPoseObjective(meta['mesh_path'],settings)
                    row['geometry_setup_seconds']=time.perf_counter()-setup;object_key=key
                with torch.no_grad():
                    q,detail=objective.refine(grasp,meta['required_cbody'],args.seed_offset+c['seed'],
                        contact_regions=meta.get('contact_regions'),
                        reference_orientation_wxyz=meta.get('reference_orientation_wxyz'),
                        gesture_reference=meta.get('gesture_reference'))
                trace=detail.pop('trajectory');native=local_to_native(q)
                if meta.get('reference_orientation_wxyz') is not None:
                    from retarget.env.grasp.contact_regions import orientation_error_deg
                    detail['reference_orientation_error_deg']=orientation_error_deg(native,meta['reference_orientation_wxyz'])
                    row['reference_orientation_error_deg']=detail['reference_orientation_error_deg']
                    detail['max_trace_orientation_error_deg']=max(
                        orientation_error_deg(local_to_native(pose),meta['reference_orientation_wxyz']) for pose in trace)
                detail['contact_regions']=meta.get('contact_regions')
                detail['reference_orientation_wxyz']=meta.get('reference_orientation_wxyz')
                if meta.get('gesture_reference') is not None:
                    from retarget.env.grasp.graspadmm_gesture import reference_diagnostics
                    detail['gesture_diagnostics']=reference_diagnostics(native,meta['gesture_reference'])
                    detail['gesture_reference']=meta['gesture_reference']
                write_json(dest/'refinement.json',detail)
                np.savez_compressed(dest/'grasp.npz',qpos=q,grasp_qpos=native)
                np.savez_compressed(dest/'native_candidate.npz',grasp_qpos=native)
                write_json(dest/'native_candidate.json',dict(mesh_path=meta['mesh_path'],required_cbody=meta['required_cbody'],
                    template=c['template'],primitive=c['primitive'],method='graspadmm_mppi',hand='wuji2',seed=c['seed'],
                    **{key:meta[key] for key in ('contact_regions','reference_orientation_wxyz','gesture_reference') if key in meta}))
                np.savez_compressed(dest/'trajectory.npz',qpos=trace,time=np.arange(len(trace),dtype=float))
                (dest/'scene.xml').write_text(objective.xml)
                row.update(model=str(dest/'scene.xml'),trajectory=str(dest/'trajectory.npz'),
                    optimization_seconds=detail['optimization_seconds'],initial_cost=detail['initial_cost'],final_cost=detail['final_cost'])
                evaluation=evaluate_native_grasp(native,meta['mesh_path'],dest/'evaluation',method='graspadmm_mppi',
                    contact_regions=meta.get('contact_regions'),
                    required_cbody=meta['required_cbody'],selection=dict(source_admm=str(p/'native_candidate.npz'),
                        policy='MPPI geometry-cost optimum; no native test within optimization'),worker=worker)
                row.update(state=evaluation['state'],task_success=bool(evaluation['success']),
                    failure_stage=evaluation.get('failure_stage'),evaluation_seconds=evaluation['evaluation_wall_seconds'])
                if evaluation.get('contact_region_diagnostics') is not None:
                    row['contact_region_diagnostics']=evaluation['contact_region_diagnostics']
                    row['semantic_success']=evaluation['contact_region_diagnostics']['semantic_success']
            except Exception as error:
                (dest/'error.log').write_text(traceback.format_exc());row['error']=f'{type(error).__name__}: {error}'
                print(row['error'],flush=True)
            row.update(transition=transition(row['baseline_success'],row['task_success']),
                validated_fallback_success=row['baseline_success'] or row['task_success'],wall_seconds=time.perf_counter()-t)
            chosen=dest/'grasp.npz' if row['task_success'] else p/'native_candidate.npz'
            write_json(dest/'validated_fallback.json',dict(success=row['validated_fallback_success'],
                source=str(chosen),uses_refined_pose=row['task_success'],baseline_success=row['baseline_success'],
                policy='separately reported native-validation fallback; never counted as raw refinement'))
            write_json(summary_path,row);rows.append(row)
            report(output,cases,rows,prior_wall+time.perf_counter()-started,settings)
            print(f"[{len(rows)}/{len(cases)}] {row['id']} {row['transition']} stage={row['failure_stage']} seconds={row['wall_seconds']:.3f}",flush=True)
    # An explicit end-to-end read-only-source audit, including baseline test and geometry.
    changed=[str(f) for hashes in inputs.values() for f,digest in hashes.items() if sha(f)!=digest]
    write_json(output/'source_audit.json',dict(passed=not changed,changed=changed,source_cases=len(cases)))
    if changed:raise RuntimeError('Source files changed during experiment')
    document=report(output,cases,rows,prior_wall+time.perf_counter()-started,settings)
    print(json.dumps({k:v for k,v in document.items() if k not in ('by_type','by_size','settings')}),flush=True)


if __name__=='__main__':main()

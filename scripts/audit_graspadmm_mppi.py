#!/usr/bin/env python3
"""Verify every paired trial's provenance, budget, acceptance and denominator."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from retarget.common.results import write_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(batch):
    batch=Path(batch).resolve();plan=json.loads((batch/'plan.json').read_text())
    report=json.loads((batch/'report.json').read_text());rows=json.loads((batch/'summary.json').read_text())['jobs']
    errors=[];checks=Counter();grids=defaultdict(list);counts=Counter();samples=0
    def check(value,label,case='batch'):
        checks[label]+=1
        if not value:errors.append(dict(case=case,check=label))
    check(report['state']=='completed','complete')
    check(len(rows)==len(plan['case_ids'])==len(set(plan['case_ids'])),'unique_denominator')
    check([r['id'] for r in rows]==plan['case_ids'],'exact_plan_coverage')
    by_id={row['id']:row for row in rows}
    manifest=json.loads((batch/'manifest.json').read_text())
    for case in manifest['cases']:
        check(case['task_success']==by_id[case['case_id']]['task_success'],'replay_acceptance_matches_native',case['case_id'])
    for relative,digest in plan['code_hashes'].items():
        check(sha(ROOT/relative)==digest,'runtime_code_unchanged',relative)
        check(sha(batch/'code'/relative)==digest,'snapshot_matches',relative)
    for row in rows:
        cid=row['id'];p=batch/'cases'/cid;grids[(row['type_id'],row['size_index'])].append(row['seed'])
        for filename,digest in plan['source_hashes'][cid].items():
            check(sha(filename)==digest,'source_unchanged',cid)
        baseline_dir=Path(row['source_summary']).parent
        source=json.loads((baseline_dir/'native_candidate.json').read_text())
        baseline=json.loads((baseline_dir/'graspadmm_evaluation/evaluation.json').read_text())
        check(row['baseline_success']==baseline['success'],'baseline_output_matches',cid)
        before=bool(row['baseline_success']);after=bool(row['task_success'])
        transition=('retained' if after else 'lost') if before else ('recovered' if after else 'neither')
        counts[transition]+=1
        check(row['transition']==transition,'paired_transition',cid)
        check(row['validated_fallback_success']==(before or after),'fallback_is_separate',cid)
        if not (p/'refinement.json').exists():
            check(not after and bool(row.get('error')),'runtime_failure_in_denominator',cid)
            continue
        detail=json.loads((p/'refinement.json').read_text());s=plan['settings']
        check(detail['settings']==s,'fixed_settings',cid)
        check(detail['seed']==plan['seed_offset']+row['seed'],'fixed_seed',cid)
        check(detail['sampled']==s['population']*s['iterations']*s['stages'],'fixed_sampling_budget',cid)
        check(all(h['iterations']==s['iterations'] for h in detail['history']),'fixed_iterations',cid)
        check(detail['final_cost']<=detail['initial_cost']+1e-6,'cost_monotonic',cid)
        check(detail['acceptance_during_optimization'] is False,'no_validation_selection_in_optimizer',cid)
        samples+=detail['sampled']
        with np.load(p/'grasp.npz') as raw:
            q=raw['qpos'];native=raw['grasp_qpos']
        expected=np.r_[q[:3],Rotation.from_euler('XYZ',q[3:6]).as_quat(scalar_first=True),q[6:]]
        check(np.array_equal(expected,native),'root_and_finger_export',cid)
        trace=np.load(p/'trajectory.npz')['qpos']
        check(np.array_equal(trace[-1],q),'final_trace_is_output',cid)
        with np.load(p/'evaluation/candidate.npz') as raw:
            check(np.array_equal(raw['grasp_qpos'],native),'evaluated_raw_mppi_output',cid)
        candidate=json.loads((p/'evaluation/candidate.json').read_text())
        check(candidate['required_cbody']==source['required_cbody'],'original_required_groups',cid)
        check(sha(candidate['mesh'])==sha(source['mesh_path']),'actual_original_geometry',cid)
        evaluation=json.loads((p/'evaluation/evaluation.json').read_text())
        check(evaluation['protocol']=='graspadmm_native_v1','common_protocol',cid)
        check(evaluation['success']==after,'only_native_success',cid)
        if evaluation['state']=='completed':
            check(evaluation['upstream_commit']=='d561ff9df8545deef7a3f9e7cfb4045ef66f6913','pinned_upstream',cid)
        else:
            check(not after and row['failure_stage']=='evaluation_runtime','native_runtime_failure_in_denominator',cid)
        if after:
            check(evaluation.get('six_direction_tested') is True,'real_six_direction_test',cid)
            check(len(evaluation['directions'])==6,'six_directions_all_present',cid)
            check({d['direction'] for d in evaluation['directions']}=={'+x','-x','+y','-y','+z','-z'},'all_force_axes',cid)
            for direction in evaluation['directions']:
                check(direction['tested'] and direction['success'],'every_direction_passed',cid)
                check(len(direction['checks'])==10,'ten_periodic_checks_per_direction',cid)
                check(all(c['success'] and c['translation_m']<.05 and c['rotation_deg']<15.
                    for c in direction['checks']),'strict_native_thresholds',cid)
        fallback=json.loads((p/'validated_fallback.json').read_text())
        check(fallback['uses_refined_pose']==after,'fallback_pose_selection',cid)
    check(dict(counts)=={k:v for k,v in report['transitions'].items() if v},'reported_transitions')
    check(sum(r['task_success'] for r in rows)==report['refined_successes'],'reported_raw_successes')
    check(sum(r['baseline_success'] for r in rows)==report['baseline_successes'],'reported_baseline_successes')
    full=len(rows)==1610
    if full:
        check(len({key[0] for key in grids})==31,'all_supported_types')
        check(len(grids)==161,'all_size_combinations')
        check(all(sorted(seeds)==list(range(10)) for seeds in grids.values()),'ten_seeds_every_size')
        check(report['baseline_successes']==524,'full_paired_baseline_524')
    result=dict(passed=not errors,trials=len(rows),full_sweep=full,checks=dict(checks),errors=errors,
        transitions=dict(counts),sampled_candidates=samples)
    write_json(batch/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,required=True)
    result=audit(parser.parse_args().batch);print(json.dumps(result));sys.exit(0 if result['passed'] else 1)

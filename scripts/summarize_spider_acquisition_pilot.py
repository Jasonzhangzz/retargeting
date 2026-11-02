"""Summarize actual acquisition certificates and independent physical replay."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json
from retarget.env.grasp.spider_seed import GROUPS
from scripts.run_spider_grasp_batch import promotion_gate


def summarize(batch):
    batch=Path(batch).resolve();source=json.loads((batch/'summary.json').read_text())
    rows=[];acquisition_reasons=Counter();hold_reasons=Counter();timings=[]
    for status in source['jobs']:
        row={k:status.get(k) for k in ('id','type_id','state','primitive','elapsed_seconds')}
        row.update(task_success=False,acquisition_success=False)
        if status.get('work'):
            output=Path(status['work'])/'solve';manifest=output/'manifest.json'
            if manifest.is_file():
                case=(output/json.loads(manifest.read_text())['cases'][0]['summary']).parent
                result=json.loads((case/'summary.json').read_text());acq=result.get('acquisition',{})
                row.update(task_success=result['task_success'],acquisition_success=acq.get('success',False),
                    ca_success=acq.get('ca_success'),source=str(case),timing=result.get('timing'))
                diagnostics=acq.get('selected_diagnostics',{})
                row['contact_established']=diagnostics.get('contact_established',False)
                row['geometric_force_closure']=diagnostics.get('force_closure',{}).get('success',False)
                row['acquisition_diagnostics']=diagnostics
                if acq and not acq.get('success'):
                    for name,required,present in zip(GROUPS,diagnostics['required_groups'],diagnostics['present_groups']):
                        if required and not present:acquisition_reasons['missing_'+name]+=1
                    if diagnostics['penetration_mm']>.5:acquisition_reasons['penetration']+=1
                    if diagnostics['self_penetration_mm']>.2:acquisition_reasons['self_collision']+=1
                    if not row['geometric_force_closure']:acquisition_reasons['force_closure']+=1
                    if not diagnostics['middle_in_center_band']:acquisition_reasons['middle_center_band']+=1
                hold=case/'group_hold/summary.json'
                if hold.is_file():
                    data=json.loads(hold.read_text());row['dynamic_hold_success']=data['success']
                    row['hold']={k:{a:v for a,v in data[k].items() if a!='force_closure_checks'} for k in ('gpu','cpu')}
                    for backend in ('gpu','cpu'):
                        report=data[backend]
                        if not report['object_hold']:hold_reasons[backend+'_object_motion']+=1
                        if not report['force_closure']:hold_reasons[backend+'_force_closure']+=1
                        for name,required,n in zip(GROUPS,data['required_groups'],report['below_target_steps']):
                            if required and n:hold_reasons[backend+'_unloaded_'+name]+=1
                if status.get('elapsed_seconds') is not None:timings.append(status['elapsed_seconds'])
        rows.append(row)
    passed=[r['id'] for r in rows if r['task_success']]
    result=dict(batch=str(batch),state=source['state'],total=len(rows),
        solver_completed=sum(r['state']=='completed' for r in rows),
        ca_failed_but_mppi_attempted=sum(r.get('ca_success') is False for r in rows),
        contact_established=sum(r.get('contact_established',False) for r in rows),
        geometric_force_closure=sum(r.get('geometric_force_closure',False) for r in rows),
        acquisition_successes=sum(r['acquisition_success'] for r in rows),
        complete_successes=len(passed),successful_cases=passed,
        acquisition_failure_counts=dict(acquisition_reasons),hold_failure_counts=dict(hold_reasons),
        completed_case_wall_seconds=dict(mean=float(np.mean(timings)),median=float(np.median(timings)),
            min=float(min(timings)),max=float(max(timings))) if timings else {},
        promotion=promotion_gate(batch),cases=rows)
    write_json(batch/'validation_summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    args=parser.parse_args();result=summarize(args.batch)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2))

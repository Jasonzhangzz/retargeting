#!/usr/bin/env python3
"""Finite 490-input GPU restart cohort, separate from the production dataset."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import argparse,collections,fcntl,hashlib,json,shutil,sys,time,traceback
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import restart_native_mano_gpu as task


def prepare(a):
    import numpy as np
    if (a.root/'MANIFEST.json').exists():raise ValueError('Cohort is already prepared')
    source=json.loads((a.source/'initialization.json').read_text())['inputs']
    reused=json.loads((a.first/'inputs.json').read_text())
    first=json.loads((a.first/'solve_receipt.json').read_text())
    if first['completed']!=len(reused) or first['finite']!=len(reused):raise ValueError('First batch has not produced finite committed states')
    used={r['input']:r['sha256'] for r in reused}
    if len(used)!=len(reused):raise ValueError('Duplicate reused input')
    all_ids={r['path']:r['sha256'] for r in source}
    if any(all_ids.get(p)!=h for p,h in used.items()):raise ValueError('First batch does not belong to this cohort')
    groups=collections.defaultdict(list)
    for row in source:
        if task.sha(row['path'])!=row['sha256']:raise ValueError('Frozen source changed')
        if row['path'] in used:continue
        data=np.load(row['path'],allow_pickle=True).item()
        key=(data['scene_path'],tuple(data['hand_cbody']),json.dumps(data['required_cbody']))
        groups[key].append(row)
    jobs=[]
    for key,rows in sorted(groups.items()):
        for start in range(0,len(rows),a.batch_size):
            chunk=rows[start:start+a.batch_size];name=f'{len(jobs):04d}'
            view=a.root/'views'/name;view.mkdir(parents=True)
            for asset in ('hand','upstream','mapping.json'):
                (view/asset).symlink_to((a.source/asset).resolve(),target_is_directory=(a.source/asset).is_dir())
            task.write(view/'initialization.json',dict(inputs=chunk))
            jobs.append(dict(id=name,source_view=str(view),root=str(a.root/'jobs'/name),samples=len(chunk),inputs=chunk))
    if len(used)+sum(j['samples'] for j in jobs)!=len(source):raise ValueError('Cohort denominator mismatch')
    task.write(a.root/'jobs.json',jobs)
    files=['scripts/run_native_mano_gpu_cohort.py','scripts/validate_native_mano_gpu.py',*json.loads((a.first/'MANIFEST.json').read_text())['source_sha256']]
    task.write(a.root/'MANIFEST.json',dict(authorization='User: 修改后重新启动求解',source=str(a.source),
        samples=len(source),first_batch=str(a.first),reused_samples=len(used),jobs=len(jobs),batch_size=a.batch_size,
        scope='All fixed Shadow-GSR-source MANO inputs from the existing 490-case diagnostic experiment; not the million-pose production dataset',
        sources_sha256={f:task.sha(ROOT/f) for f in files},jobs_sha256=task.sha(a.root/'jobs.json'),
        first_poses_sha256=task.sha(a.first/'poses.npz'),first_receipt_sha256=task.sha(a.first/'solve_receipt.json'),
        gpu_physics='Eager CUDA native mjwarp',qp='Original six-ray objective/constraints, GPU batched backend',
        objectives_changed=False,production_launch_allowed=False,initialization='Existing mapped MANO poses from verified Shadow sources; no new CPU pose fitting',
        failure_policy='No failed-row resampling; native rejections retained; unexpected technical exceptions stop dispatch',
        created_unix=time.time()))
    task.write(a.root/'prepare_status.json',dict(state='prepared',jobs=len(jobs),samples=len(source)))
    task.write(a.root/'status.json',dict(state='prepared',completed=len(used),total=len(source),native_output=first['native_output']))


def verify(a):
    m=json.loads((a.root/'MANIFEST.json').read_text())
    for p,h in m['sources_sha256'].items():
        if task.sha(ROOT/p)!=h:raise ValueError('Frozen cohort source changed: '+p)
    if task.sha(a.root/'jobs.json')!=m['jobs_sha256']:raise ValueError('Frozen job list changed')
    first=Path(m['first_batch'])
    if task.sha(first/'poses.npz')!=m['first_poses_sha256'] or task.sha(first/'solve_receipt.json')!=m['first_receipt_sha256']:
        raise ValueError('First committed batch changed')
    return m,json.loads((a.root/'jobs.json').read_text())


def build_assets(a):
    _,jobs=verify(a)
    for i,j in enumerate(jobs):
        if (a.root/'STOP').exists():raise InterruptedError('Cohort STOP')
        root=Path(j['root'])
        if not (root/'MANIFEST.json').exists():
            root.mkdir(parents=True,exist_ok=True)
            if not (root/'STOP').is_symlink():(root/'STOP').symlink_to(a.root/'STOP')
            task.prepare(SimpleNamespace(root=root,source=Path(j['source_view']),samples=j['samples'],device=a.device))
        task.write(a.root/'prepare_status.json',dict(state='preparing',completed=i+1,total=len(jobs)))
    task.write(a.root/'prepare_status.json',dict(state='complete',completed=len(jobs),total=len(jobs)))


def solve(a):
    import validate_native_mano_gpu as validator
    m,jobs=verify(a);first=json.loads((Path(m['first_batch'])/'solve_receipt.json').read_text())
    completed=m['reused_samples'];native=first['native_output'];closure=first['projected_closure'];receipts=[]
    first_root=Path(m['first_batch'])
    while not (first_root/'dynamic_receipt.json').exists():
        if (a.root/'STOP').exists():raise InterruptedError('Cohort STOP')
        if (first_root/'dynamic_failure.json').exists():raise RuntimeError('First batch dynamic validation failed technically')
        task.write(a.root/'status.json',dict(state='waiting_for_first_dynamic_validation',completed=completed,total=m['samples']))
        time.sleep(5)
    for j in jobs:
        root=Path(j['root'])
        while not (root/'status.json').exists():
            if (a.root/'STOP').exists():raise InterruptedError('Cohort STOP')
            s=json.loads((a.root/'prepare_status.json').read_text())
            if s['state']=='failed':raise RuntimeError('Asset preparation failed')
            time.sleep(5)
        if (a.root/'STOP').exists():raise InterruptedError('Cohort STOP')
        task.write(a.root/'status.json',dict(state='solving',job=j['id'],completed=completed,total=m['samples'],native_output=native,projected_closure=closure))
        if (root/'solve_receipt.json').exists():raise ValueError('Unexpected existing result without a cohort commit; inspect before replaying')
        with (root/'run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            task.write(root/'process.json',dict(pid=os.getpid(),mode='cohort_gpu_solve',started_unix=time.time()))
            try:
                task.run(SimpleNamespace(root=root,device=a.device,capture_physics=False))
                validator.run(SimpleNamespace(root=root,device=a.device,wait_timeout=0.))
            except BaseException:
                task.write(root/'failure.json',dict(exception=traceback.format_exc(),state_may_be_inflight=True))
                task.write(root/'status.json',dict(state='failed'));raise
        r=json.loads((root/'solve_receipt.json').read_text());completed+=r['completed'];native+=r['native_output'];closure+=r['projected_closure']
        receipts.append(dict(job=j['id'],samples=j['samples'],poses_sha256=task.sha(root/'poses.npz'),receipt_sha256=task.sha(root/'solve_receipt.json'),
            dynamic_receipt_sha256=task.sha(root/'dynamic_receipt.json')))
        task.write(a.root/'commits.json',receipts)
        task.write(a.root/'status.json',dict(state='solving',completed=completed,total=m['samples'],native_output=native,projected_closure=closure))
    task.write(a.root/'status.json',dict(state='solve_and_dynamic_validation_complete_pending_quality_review',completed=completed,total=m['samples'],
        native_output=native,projected_closure=closure,production_launch_allowed=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','assets','solve'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--first',type=Path)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    a.root=a.root.resolve();a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/(a.mode+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        task.write(a.root/(a.mode+'_process.json'),dict(pid=os.getpid(),argv=sys.argv,started_unix=time.time()))
        try:{'prepare':prepare,'assets':build_assets,'solve':solve}[a.mode](a)
        except BaseException as error:
            task.write(a.root/(a.mode+'_failure.json'),dict(exception=traceback.format_exc()))
            task.write(a.root/('prepare_status.json' if a.mode=='assets' else 'status.json'),dict(state='stopped_by_user' if isinstance(error,InterruptedError) else 'failed',phase=a.mode));raise


if __name__=='__main__':main()

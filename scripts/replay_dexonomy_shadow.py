#!/usr/bin/env python3
"""Replay original Shadow grasps, cache contacts by grasp type, audit closure."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.env.grasp.dexonomy_replay import run_pilot,run_full,verify_cache,write_json,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/dexonomy');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['pilot','full','verify'],default='pilot');p.add_argument('--count',type=int,default=100)
    p.add_argument('--full-dataset',type=Path,default=ROOT/'data/dexonomy')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--workers',type=int,default=8);p.add_argument('--shard-size',type=int,default=256)
    p.add_argument('--auto-full',action='store_true');p.add_argument('--full-output',type=Path);p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--skip-dynamic',action='store_true');p.add_argument('--wait-assets',action='store_true')
    a=p.parse_args()
    if a.count<1 or a.workers<1 or a.shard_size<1 or not 0<a.threshold<=1:p.error('positive counts and threshold in (0,1] required')
    if a.wait_assets:
        if a.mode=='pilot':
            while not (a.dataset/'ready.json').exists():
                print('waiting for stratified pilot assets',flush=True);time.sleep(30)
        else:
            a.output.mkdir(parents=True,exist_ok=True)
            write_json(a.output/'status.json',{'state':'preparing_assets','pid':os.getpid(),'workers':a.workers,'shard_size':a.shard_size,'all_closure_labels_retained':True})
            subprocess.run([sys.executable,'-I',str(ROOT/'scripts/extract_dexonomy_archives.py')],check=True)
    if a.mode=='verify':
        reports=[verify_cache(f.parent) for f in a.output.glob('*/manifest.json')];print(json.dumps(reports,indent=2));return
    if a.mode=='full':run_full(a.dataset,a.output,a.workers,a.shard_size,not a.skip_dynamic);return
    report=run_pilot(a.dataset,a.output,a.count,a.seed,not a.skip_dynamic,a.workers)
    verified=[verify_cache(f.parent) for f in a.output.glob('*/manifest.json')]
    verification={'success':len(verified)==report['grasp_types'] and sum(v['cases'] for v in verified)==report['replayed'],'groups':verified}
    write_json(a.output/'verification.json',verification)
    report['full_batch_threshold']=a.threshold
    report['eligible_for_full_batch']=report['closure_pass_rate']>=a.threshold and not report['errors'] and verification['success']
    report['code_sha256']={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'retarget/env/grasp/dexonomy_replay.py',ROOT/'retarget/env/grasp/closure_types.py']}
    report['python']=sys.executable;report['full_batch_started']=False
    if a.auto_full and report['eligible_for_full_batch']:
        output=a.full_output or a.output.with_name(a.output.name+'_full');output.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--mode','full','--dataset',str(a.full_dataset.resolve()),'--wait-assets','--output',str(output.resolve()),'--workers',str(a.workers),'--shard-size',str(a.shard_size)]
        if a.skip_dynamic:cmd.append('--skip-dynamic')
        with (output/'run.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=ROOT)
        report.update(full_batch_started=True,full_batch_pid=proc.pid,full_batch_output=str(output))
    write_json(a.output/'summary.json',report);write_json(a.output/'status.json',{'state':'completed',**report});print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

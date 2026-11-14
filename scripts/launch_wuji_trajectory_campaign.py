#!/usr/bin/env python3
"""Single-owner finite queue: no automatic retry, no neural optimization."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path


def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();r=a.root;lock=(r/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name in ['algebra_preflight.json','causal_human_preflight.json','crocoddyl_preflight.json','pipeline_preflight.json','structural_validation.json','SDK_COMPLETE.json']:
  if not json.loads((r/name).read_text()).get('passed',json.loads((r/name).read_text()).get('complete',False)):raise ValueError('Unpassed prerequisite '+name)
 manifest=json.loads((r/'MANIFEST.json').read_text());control=json.loads((r/'CONTROL.json').read_text());assert control['authorized'] and not control['stop_requested'];revision=control['revision']
 source={str(p.relative_to(r/'runtime')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'runtime').rglob('*') if p.suffix in ['.py','.cpp']}
 signature=hashlib.sha256(json.dumps(dict(source=source,selection=json.loads((r/'selection.json').read_text())),sort_keys=True).encode()).hexdigest()
 if manifest.get('candidate_signature') and manifest['candidate_signature']!=signature:raise ValueError('Runtime changed; explicitly record a new candidate revision before resuming')
 manifest['candidate_signature']=signature;manifest['runtime_sha256']=source
 jobs=[]
 for split in ['development','heldout']:
  for backend in ['banded_lm','trajectory_lbfgs','gpu_ilqr','crocoddyl_boxfddp','causal_lm','causal_lbfgs']:
   for config in ['builtin','hl_urdf']:
    for side in ['left','right']:
     jobs.append(dict(id=f'{split}_{backend}_{config}_{side}',split=split,backend=backend,configuration=config,side=side,budgets=[16] if backend.startswith('causal_') else [1,2,4,8,16,32],reference=False))
 for backend in ['banded_lm','crocoddyl_boxfddp']:
  for config in ['builtin','hl_urdf']:
   for side in ['left','right']:jobs.append(dict(id=f'reference_{backend}_{config}_{side}',split='development',backend=backend,configuration=config,side=side,budgets=[200],reference=True))
 manifest['jobs']=jobs;manifest['expected_parameter_cases']=sum(len(j['budgets']) for j in jobs);manifest['neural_optimizer_updates']=0;atomic(r/'MANIFEST.json',manifest)
 stop=False;child=None
 def requested(*_):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
 def state(status,**kw):atomic(r/'STATUS.json',dict(status=status,pid=os.getpid(),control_revision=revision,candidate_signature=signature,neural_optimizer_updates=0,**kw))
 completed=[]
 try:
  for job in jobs:
   current=json.loads((r/'CONTROL.json').read_text())
   if stop or current['stop_requested'] or current['revision']!=revision:raise InterruptedError('Stop/control revision changed')
   unit=r/'jobs'/job['id']/'UNIT.json'
   if unit.exists() and json.loads(unit.read_text()).get('complete') and json.loads(unit.read_text()).get('candidate_signature')==signature:completed.append(job['id']);continue
   logs=r/'logs';logs.mkdir(exist_ok=True);log=logs/(job['id']+'.log')
   if log.exists() and not a.resume:raise ValueError('Existing execution log; use explicit --resume after inspecting failure')
   argv=[sys.executable,'-u',str(r/'runtime/scripts/benchmark_wuji_trajectory.py'),'run','--root',str(r),'--backend',job['backend'],'--configuration',job['configuration'],'--side',job['side'],'--split',job['split'],'--job-id',job['id'],'--repeats','10','--budgets',*map(str,job['budgets'])]+(['--reference'] if job['reference'] else [])
   with log.open('a') as f:child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   receipt=dict(job=job,pid=child.pid,argv=argv,started=time.time(),candidate_signature=signature,owner_uid=os.getuid());atomic(log.with_suffix('.receipt.json'),receipt)
   while child.poll() is None:
    current=json.loads((r/'CONTROL.json').read_text())
    if stop or current['stop_requested'] or current['revision']!=revision:raise InterruptedError('Stop requested')
    state('running',completed=completed,active_job=job,child_pid=child.pid,heartbeat=time.time());time.sleep(5)
   if child.returncode:raise RuntimeError(job['id']+' failed with exit '+str(child.returncode))
   if not unit.exists() or not json.loads(unit.read_text()).get('complete'):raise RuntimeError('Missing completion receipt '+job['id'])
   completed.append(job['id']);child=None
  state('complete',complete=True,completed=completed,expected_jobs=len(jobs),expected_parameter_cases=manifest['expected_parameter_cases'])
 except BaseException as e:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  state('stopped' if isinstance(e,InterruptedError) else 'awaiting_analysis',complete=False,completed=completed,reason=str(e));raise


if __name__=='__main__':main()

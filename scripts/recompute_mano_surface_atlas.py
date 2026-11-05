#!/usr/bin/env python3
"""Finite atlas rebuild for a corrected frozen surface, using only shared CEM.

Each independent shard has private receipts, checkpoints and outputs. This
controller contains no grasp objective, sampler or solver implementation.
"""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.mano_patch_tools import write

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'RUN.json').exists():raise RuntimeError('Existing run: inspect its receipts before resuming')
    candidate=json.loads(args.candidate.read_text());report=candidate['proximal_width_report'];validation=json.loads(args.candidate.with_name('held_out_geometry.json').read_text())
    if not (report['accepted'] and validation['passed']):raise ValueError('Surface candidate has not passed independent checks')
    source=args.parent/'saved_left';saved=out/'saved_left';shutil.copytree(source,saved)
    parent_manifest=json.loads((saved/'manifest.json').read_text());(saved/'manifest.json').unlink()
    # Stale posed snapshots/topologies belong to the original surface. The
    # complete original immutable archive remains at the parent path.
    for name in ('live-state.json','topology.json'):
        (saved/name).unlink(missing_ok=True)
    write(saved/'current.json',candidate);shutil.copy2(args.candidate.with_name('proximal_geometry.npz'),saved/'geometry.npz')
    for name in ('mano-view.js','app.js','style.css','taxel-view.js'):shutil.copy2(ROOT/'moxian_web/static'/name,saved/name)
    write(saved/'manifest.json',{'saved_at_ms':time.time()*1000,'hand_side':'left','geometry_version':report['geometry_version'],'model_revision':candidate.get('revision'),'parent_snapshot':str(source.resolve()),'parent_manifest_sha256':sha(source/'manifest.json'),'purpose':'static proximal breadth correction; original snapshot retained','sha256':{p.name:sha(p) for p in saved.iterdir() if p.is_file()}})
    shutil.copytree(args.parent/'web',out/'web',ignore=shutil.ignore_patterns('data','saved-left.zip'));model=json.loads((out/'web/model.json').read_text());geo=np.load(saved/'geometry.npz');model['mano']['rest']=(geo['rest_vertices']*[-1,1,1]).tolist();model['mano']['joints']=(geo['rest_joints']*[-1,1,1]).tolist();write(out/'web/model.json',model)
    frozen=out/'frozen';old_manifest=json.loads((args.parent/'MANIFEST.json').read_text());files=list(old_manifest['source_code_sha256'])
    for f in files:
        dest=frozen/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    launcher=out/'frozen_launcher.py';shutil.copy2(args.parent/'frozen_launcher.py',launcher)
    names=[e['name'] for e in json.loads((args.previous/'web/index.json').read_text())['entries']]
    receipt={'state':'running','pid':os.getpid(),'started_at_ms':time.time()*1000,'types':len(names),'knots':1247,'workers':3,'solver':'retarget.solver.cem.solve_patch_cem','solver_sha256':sha(frozen/'retarget/solver/cem.py'),'source_code_sha256':{f:sha(frozen/f) for f in files},'scope':'31 types, one finite rebuild; existing population/iterations/early stop','parent':str(args.parent.resolve()),'candidate_sha256':sha(args.candidate),'publication':'after validation only'};write(out/'RUN.json',receipt)
    jobs=[];stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        for job,_,_ in jobs:
            if job.poll() is None:job.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for i in range(3):
        shard=out/f'shard-{i}';shard.mkdir();shutil.copytree(saved,shard/'saved_left');log=(shard/'worker.log').open('w')
        command=[sys.executable,str(launcher),str(frozen),str(frozen/'scripts/fix_personalized_grasp_grid.py'),'--previous',str(args.previous.resolve()),'--output',str(shard),'--only',','.join(names[i::3])]
        job=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((job,shard,log));write(shard/'job.json',{'pid':job.pid,'command':command,'types':names[i::3]})
    while any(j.poll() is None for j,_,_ in jobs):
        if any(j.poll() not in (None,0) for j,_,_ in jobs):stop()
        completed=sum(len(list(s.glob('grids_fixed/*/complete.json'))) for _,s,_ in jobs);write(out/'status.json',{'state':'stopping' if stopping else 'running','types_complete':completed,'types_total':len(names)});time.sleep(2)
    for _,_,log in jobs:log.close()
    if stopping or any(j.returncode for j,_,_ in jobs):
        receipt.update(state='stopped' if stopping else 'failed',exit_codes=[j.returncode for j,_,_ in jobs]);write(out/'RUN.json',receipt);raise RuntimeError('Atlas incomplete; old published dataset retained')
    grids=out/'grids_fixed';grids.mkdir()
    for _,shard,_ in jobs:
        for folder in (shard/'grids_fixed').iterdir():shutil.copytree(folder,grids/folder.name)
    shutil.copy2(jobs[0][1]/'MANIFEST.json',out/'MANIFEST.json')
    for script,extra in [('export_fixed_primitive_grid.py',['--strip-legacy-fields']),('export_personalized_hand_assets.py',[]),('publish_personalized_grasp_viewer.py',[]),('export_mano_taxels.py',[])]:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--output',str(out),*extra],cwd=ROOT,env=env,check=True)
    receipt.update(state='computed',finished_at_ms=time.time()*1000,exit_codes=[j.returncode for j,_,_ in jobs]);write(out/'RUN.json',receipt);write(out/'status.json',{'state':'computed_awaiting_browser_and_data_validation','types':31,'knots':1247});print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

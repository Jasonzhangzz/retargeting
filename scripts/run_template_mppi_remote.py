#!/usr/bin/env python3
"""Launch independent type shards on large-memory GPUs, then audit and merge."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json
from retarget.env.grasp.template_mppi_continuation import densify


def read(p):return json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/grasp_template_mppi_dense.yaml')
    parser.add_argument('--devices',type=int,nargs='+',default=[0,1,2,3])
    parser.add_argument('--population',type=int,default=1024)
    parser.add_argument('--batch',default='grasp_template_mppi_a800_20261102')
    args=parser.parse_args()
    if args.population<256 or len(set(args.devices))!=len(args.devices):raise ValueError('Invalid population/devices')
    os.chdir(ROOT);root=ROOT/'output'/args.batch
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    base=yaml.safe_load(args.config.read_text())
    source=Path(base['environment']['template_mppi']['continuation']['source_batches'][0])
    entries=read(source/'manifest.json')['cases'];families={}
    for e in entries:families.setdefault(e['template'],[]).append(e)
    jobs=[]
    for name,rows in families.items():
        count=len(read((source/rows[0]['trajectory']).with_name('template_contacts.json'))['body_names'])
        weight=len(densify([r['primitive'] for r in rows]))*count
        jobs.append((weight,int(name.partition('_')[0]),name))
    assignments={d:[] for d in args.devices};weights={d:0 for d in args.devices}
    for weight,number,name in sorted(jobs,reverse=True):
        device=min(args.devices,key=lambda d:weights[d]);assignments[device].append(number);weights[device]+=weight
    generated=ROOT/'configs/generated';generated.mkdir(exist_ok=True)
    children=[];started=time.perf_counter()
    for device in args.devices:
        config=copy.deepcopy(base)
        config['task']['data']=str(ROOT/'data/grasp')
        config['solver'].update(population=args.population,worlds=args.population)
        config['environment']['device']='cuda:0'
        config['environment']['template_mppi']['continuation']['types']=sorted(assignments[device])
        config['environment']['template_mppi']['parallel'].update(memory_fraction=.8,max_workers=24,safety_factor=1.1)
        config['output']['name']=f'{args.batch}_gpu{device}'
        path=generated/f'{args.batch}_gpu{device}.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(device),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
             'PYTHONUNBUFFERED':'1','MUJOCO_GL':'egl'}
        log=root/f'gpu{device}.log'
        with log.open('wb') as f:
            process=subprocess.Popen([sys.executable,'run.py','--config',str(path)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(device=device,pid=process.pid,types=sorted(assignments[device]),weight=weights[device],
            config=str(path),output=str(ROOT/'output'/config['output']['name']),log=str(log),process=process))
    def save(state):
        records=[{k:v for k,v in c.items() if k!='process'} for c in children]
        for c in records:
            status=Path(c['output'])/'status.json'
            if status.exists():
                try:c['status']=read(status)
                except ValueError:pass
        write_json(root/'remote_job.json',dict(state=state,population=args.population,worlds=args.population,
            memory_fraction_per_device=.8,children=records,wall_seconds=time.perf_counter()-started))
    save('running');print('REMOTE_JOB',root,flush=True)
    for c in children:print('GPU',c['device'],'pid',c['pid'],'types',c['types'],flush=True)
    while any(c['process'].poll() is None for c in children):
        save('running');time.sleep(10)
    save('auditing')
    for child in children:
        if child['process'].returncode:
            save('failed');raise RuntimeError(f"GPU {child['device']} shard failed; inspect {child['log']}")
        with (root/f"audit_gpu{child['device']}.log").open('w') as log:
            subprocess.run([sys.executable,'scripts/audit_template_mppi_results.py',child['output']],stdout=log,stderr=subprocess.STDOUT,check=True)
    from merge_template_mppi_shards import merge
    merge([Path(c['output']) for c in children],root)
    with (root/'audit.log').open('w') as log:
        subprocess.run([sys.executable,'scripts/audit_template_mppi_results.py',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'scripts/report_template_mppi_dense.py',str(root)],check=True)
    save('completed')
    print('COMPLETED',root,flush=True)


if __name__=='__main__':main()

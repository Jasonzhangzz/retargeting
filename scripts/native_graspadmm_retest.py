#!/usr/bin/env python3
"""Unmodified upstream Shadow GraspADMM: bounded initialization, quality and timing."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,copy,datetime,fcntl,hashlib,importlib,importlib.metadata,inspect,json,logging,multiprocessing as mp
import random,signal,statistics,subprocess,sys,time,traceback
from types import SimpleNamespace

REPOSITORY_ROOT=Path(__file__).resolve().parents[1]
if (REPOSITORY_ROOT/'retarget/solver/graspadmm.py').is_file():sys.path.insert(0,str(REPOSITORY_ROOT))

COMMIT='d561ff9df8545deef7a3f9e7cfb4045ef66f6913'
MENAGERIE='bf04290ac7911fa7a39339c7e507792fd464c438'
TYPE='7_Prismatic_3_Finger'
VERSIONS={'torch':'2.5.1','mujoco':'3.3.3','numpy':'2.2.6','scipy':'1.15.3',
    'clarabel':'0.11.1','qpsolvers':'4.8.0','warp-lang':'1.8.0','trimesh':'4.5.3',
    'matplotlib':'3.10.3','usd-core':'25.2'}


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()
def write(path,value):
    def clean(x):
        if hasattr(x,'tolist'):return clean(x.tolist())
        if hasattr(x,'item'):return clean(x.item())
        if isinstance(x,float) and not __import__('math').isfinite(x):return None
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,Path):return str(x)
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load_upstream(root):
    root=Path(root).resolve();manifest=json.loads((root/'GRASPADMM_UPSTREAM.json').read_text())
    if manifest['commit']!=COMMIT:raise RuntimeError('Unexpected upstream revision')
    for name,expected in manifest['sha256'].items():
        if sha(root/name)!=expected:raise RuntimeError('Upstream source changed: '+name)
    if 'dexonomy' in sys.modules and str(root/'dexonomy') not in sys.modules['dexonomy'].__path__:
        raise RuntimeError('Wrong Dexonomy package already loaded')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from omegaconf import OmegaConf
    return SimpleNamespace(core=importlib.import_module('dexonomy.op.gen_grasp_admm'),
        init=importlib.import_module('dexonomy.op.gen_init'),convert=importlib.import_module('dexonomy.op.convert_template'),
        evaluation=importlib.import_module('dexonomy.op.eval_data'),sim=importlib.import_module('dexonomy.sim'),
        files=importlib.import_module('dexonomy.util.file_util'),OmegaConf=OmegaConf,manifest=manifest,root=root)


def configuration(source,root,stage,output=None,workers=32):
    O=source.OmegaConf;root=Path(root).resolve();output=Path(output or root/'quality').resolve()
    base=O.load(source.root/'dexonomy/config/base.yaml')
    for key in ('defaults','hydra'):base.pop(key,None)
    base.asset_root=str(source.root/'assets');base.hand_name='shadow';base.hand=O.load(source.root/'dexonomy/config/hand/shadow.yaml')
    base.op_name=stage;base.op=O.load(source.root/f'dexonomy/config/op/{stage}.yaml') if stage!='tmpl' else O.create({})
    base.tmpl_name=TYPE;base.init_tmpl_dir=str(root/'initial_templates');base.new_tmpl_dir=str(root/'new_templates')
    base.init_dir=str(root/'init_data');base.grasp_dir=str(output/'grasp');base.succ_grasp_dir=str(output/'succ_grasp')
    base.traj_dir=str(output/'traj');base.succ_traj_dir=str(output/'succ_traj')
    base.debug_dir=str(output/'debug');base.log_path=str(output/'path_log.txt')
    base.n_worker=workers;base.debug_view=False;base.debug_render=False;base.debug_name=None
    if stage=='init':
        base.op.object.cfg_path=str(root/'scenes/*.npy');base.op.object.n_cfg=16;base.init_gpu=[1]
    from retarget.solver.graspadmm import DEFAULTS,settings
    if stage=='grasp_admm':
        for key,value in settings().items():
            if not key.startswith('init_'):base.op.grasp[key]=value
    if stage=='init':
        base.op.epoch=DEFAULTS.init_epoch
        base.op.object.n_init_point=DEFAULTS.init_points
        base.op.object.n_init_inplane=DEFAULTS.init_inplane
        base.op.matcher.opt_step=DEFAULTS.init_iterations
        base.op.filter.general.n_final=DEFAULTS.init_n_final
    return O.create(O.to_container(base,resolve=True))


def prepare(root,production,source_manifest,upstream):
    import numpy as np
    root=Path(root);production=Path(production);root.mkdir(parents=True,exist_ok=True)
    original=json.loads(Path(source_manifest).read_text());cache={};scenes=[];input_assets={}
    official=json.loads((root/'OFFICIAL_FILES.json').read_text())
    for name,expected in official.items():
        if sha(Path(upstream)/name)!=expected:raise RuntimeError('Transfer differs from fixed Git checkout: '+name)
    for item in original['cases']:
        path=item['source_path']
        if path not in cache:cache[path]=json.loads((Path(path)/'manifest.json').read_text())['cases']
        entry=cache[path][item['row']]
        relative=str(entry['scene_path']).split('assets/object/',1)[-1]
        scene_path=(production/'data/dexonomy/extracted'/relative).resolve()
        scene=np.load(scene_path,allow_pickle=True).item()
        def absolutize(value):
            for key,v in value.items():
                if isinstance(v,dict):absolutize(v)
                elif key.endswith('_path') and isinstance(v,str):value[key]=str((scene_path.parent/v).resolve())
        absolutize(scene['scene'])
        for obj in scene['scene'].values():
            for key,value in obj.items():
                if key.endswith('_path') and isinstance(value,str):
                    if not Path(value).is_file():raise RuntimeError('Missing original scene asset: '+value)
                    input_assets[value]=sha(value)
                    if key in ('xml_path','urdf_path'):
                        for asset in Path(value).parent.rglob('*'):
                            if asset.is_file():input_assets[str(asset)]=sha(asset)
        factor=float(entry['scene_scale'])
        for cfg in scene['scene'].values():
            if cfg['type']=='rigid_object':
                cfg['scale']=np.asarray(cfg['scale'])*factor
                cfg['pose']=np.asarray(cfg['pose']).copy();cfg['pose'][:3]*=factor
        original_id=scene['scene_id'];scene['scene_id']=f'case_{item["id"]:02d}'
        path=root/'scenes'/f'case_{item["id"]:02d}.npy';path.parent.mkdir(exist_ok=True)
        np.save(path,scene)
        scenes.append(dict(id=item['id'],source_scene=str(scene_path),source_scene_sha256=sha(scene_path),
            source_scene_id=original_id,scene_scale=factor,scene_path=str(path),sha256=sha(path)))
    assets={str(p.relative_to(upstream)):sha(p) for directory in ('assets/hand/shadow','third_party/mujoco_menagerie/shadow_hand') for p in (Path(upstream)/directory).rglob('*') if p.is_file()}
    if not assets:raise RuntimeError('Official hand assets missing')
    write(root/'scenes.json',scenes)
    sources={str(p.relative_to(upstream)):sha(p) for p in (Path(upstream)/'dexonomy').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')}
    script_names=['native_graspadmm_retest.py','launch_native_graspadmm_retest.py','diagnose_native_graspadmm.py','profile_native_graspadmm.py','report_native_graspadmm.py',
        'setup_native_graspadmm.sh','fetch_graspadmm_torch.py','fetch_graspadmm_dependencies.py','requirements-graspadmm.txt']
    scripts={name:sha(root/name) for name in script_names}
    from retarget.solver.graspadmm import DEFAULTS as parameters
    write(root/'MANIFEST.json',dict(commit=COMMIT,menagerie_commit=MENAGERIE,
        authorization='User approved full original Shadow retest',source_manifest_sha256=sha(source_manifest),
        assets_sha256=assets,sources_sha256=sources,input_assets_sha256=input_assets,official_files_sha256=official,
        source_manifest=json.loads((Path(upstream)/'GRASPADMM_UPSTREAM.json').read_text()),
        seed=20261104,type=TYPE,scene_count=16,max_initialization_outputs=len(scenes)*parameters.init_epoch*parameters.init_n_final,
        init_budget=dict(epoch=parameters.init_epoch,n_init_point=parameters.init_points,n_init_inplane=parameters.init_inplane,matcher_steps=parameters.init_iterations,n_final=parameters.init_n_final),
        grasp_budget=dict(step=parameters.step,substep=parameters.substep),timing=dict(latency_warmup=1,latency_repeats=5,throughput_max_inputs=128,workers=[1,32],repeats=3),
        callback_required=False,created_utc=stamp(),script_sha256=sha(__file__),runner_scripts_sha256=scripts,production_solver='cem'))


def seed():
    import numpy as np,torch
    random.seed(20261104);np.random.seed(20261104);torch.manual_seed(20261104);torch.set_num_threads(1)


def verify_assets(root,source):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,expected in manifest['assets_sha256'].items():
        if sha(source.root/name)!=expected:raise RuntimeError('Official asset changed: '+name)
    for name,expected in source.manifest['sha256'].items():
        if sha(source.root/name)!=expected:raise RuntimeError('Upstream source changed: '+name)
    for name,expected in manifest.get('sources_sha256',{}).items():
        if sha(source.root/name)!=expected:raise RuntimeError('Upstream source changed: '+name)
    for name,expected in manifest.get('official_files_sha256',{}).items():
        if sha(source.root/name)!=expected:raise RuntimeError('Upstream source/asset differs from official Git checkout: '+name)
    for name,expected in manifest.get('input_assets_sha256',{}).items():
        if sha(name)!=expected:raise RuntimeError('Original scene asset changed: '+name)
    for name,expected in manifest.get('runner_scripts_sha256',{}).items():
        if sha(root/name)!=expected:raise RuntimeError('Frozen experiment script changed: '+name)
    receipt=root/'template_receipt.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if sha(value['path'])!=value['sha256']:raise RuntimeError('Frozen initial template changed')
    receipt=root/'initialization.json'
    if receipt.exists():
        inputs=json.loads(receipt.read_text())['inputs']
        if list_inputs(root)!=sorted(r['path'] for r in inputs):raise RuntimeError('Frozen initialization list changed')
        for item in inputs:
            if sha(item['path'])!=item['sha256']:raise RuntimeError('Frozen initialization pose changed')


def observe(function,call,filter_code=None):
    import numpy as np
    captured={};filters=[];code=getattr(function,'__wrapped__',function).__code__
    previous=sys.getprofile()
    lines,start_line=inspect.getsourcelines(getattr(function,'__wrapped__',function))
    export_line=next((start_line+i for i,line in enumerate(lines) if 'grasp_data["grasp_qpos"] = np_array32' in line),None)
    def profile(frame,event,arg):
        if event!='return':return
        if frame.f_code is code:
            local=frame.f_locals
            for key in ('cp_o','cp_h','lam','hand_cpn_b','hand_cbody','grasp_data','terminated','succ_flag'):
                if key in local:
                    value=local[key]
                    if hasattr(value,'detach'):value=value.detach().cpu().numpy().copy()
                    captured[key]=copy.deepcopy(value)
            if export_line is not None:
                captured['pose_export_completed']=frame.f_lineno>export_line
                if 'sim_env' in local:
                    try:captured['last_available_qpos']=local['sim_env'].get_hand_qpos().copy()
                    except Exception as error:captured['state_read_error']=str(error)
                captured['admm_loop_started']='iter' in local
        elif frame.f_code.co_name=='_safe_wrapper_func' and 'error_traceback' in frame.f_locals:
            captured['exception']=frame.f_locals['error_traceback']
        elif filter_code is not None and frame.f_code is filter_code:
            filt=frame.f_locals['self'];filters.append(dict(name=filt.log_name,passed=bool(arg[0]) if arg is not None else False))
        elif frame.f_code.co_name in ('_limit_filter','_collision_filter','_contact_filter','_qp_filter'):
            filters.append(dict(check=frame.f_code.co_name,passed=bool(arg),name=frame.f_locals['self'].log_name))
    try:sys.setprofile(profile);call()
    finally:sys.setprofile(previous)
    return captured,filters


SOURCE=None
BASE=None
def worker_init(upstream,cfg,warm_queue=None,ready=None):
    global SOURCE,BASE
    SOURCE=load_upstream(upstream);BASE=SOURCE.OmegaConf.create(cfg);seed()
    import torch
    torch.set_default_device('cpu')  # Same CPU branch setup as operate_grasp_admm.
    if warm_queue is not None:
        path=warm_queue.get();warm=copy.deepcopy(BASE);warm.grasp_dir+=f'/warmup_{os.getpid()}';warm.debug_dir+=f'/warmup_{os.getpid()}'
        SOURCE.core._single_grasp((path,warm));ready.put(os.getpid())


def execute_job(job):
    import numpy as np
    path,cfg_dict,mode,folder=job;cfg=SOURCE.OmegaConf.create(cfg_dict);folder=Path(folder)
    output=Path(path.replace(cfg.init_dir,cfg.grasp_dir));start=time.perf_counter();observed=None;filters=[]
    if mode=='quality':
        observed,filters=observe(SOURCE.core._single_grasp,lambda:SOURCE.core._single_grasp((path,cfg)),SOURCE.core.GraspFilter.forward.__code__)
    else:
        if sys.gettrace() is not None or sys.getprofile() is not None:raise RuntimeError('Timing cannot use an observer')
        SOURCE.core._single_grasp((path,cfg))
    seconds=time.perf_counter()-start
    result=dict(input=path,native_operation_output=output.is_file(),operation_seconds=seconds,filters=filters,
                mode=mode,pid=os.getpid())
    if output.is_file():
        saved=np.load(output,allow_pickle=True).item()
        result['pose_sha256']=hashlib.sha256(b''.join(np.ascontiguousarray(saved[k]).tobytes() for k in ('grasp_qpos','squeeze_qpos','pregrasp_qpos'))).hexdigest()
        result['output']=str(output)
    if mode=='quality':
        folder.mkdir(parents=True,exist_ok=True)
        observed=observed or {}
        arrays={key:value for key,value in observed.items() if isinstance(value,np.ndarray)}
        data=observed.get('grasp_data',{})
        if observed.get('pose_export_completed') and 'grasp_qpos' in data:
            arrays['final_grasp_qpos']=np.asarray(data['grasp_qpos'])
        result.update(fallback_used=bool(observed.get('terminated',False)),captured_final_state='final_grasp_qpos' in arrays,
                      hand_cbody=observed.get('hand_cbody',[]),required_cbody=data.get('required_cbody',[]),
                      exception=observed.get('exception'),admm_loop_started=observed.get('admm_loop_started',False))
        result['input_sha256']=sha(path)
        np.savez_compressed(folder/'final_state.npz',**arrays)
        if 'final_grasp_qpos' in arrays:
            result['captured_final_sha256']=hashlib.sha256(np.ascontiguousarray(arrays['final_grasp_qpos']).tobytes()).hexdigest()
        write(folder/'result.json',result)
    return result


def evaluate_job(job):
    path,cfg_dict=job;cfg=SOURCE.OmegaConf.create(cfg_dict)
    observed,_=observe(SOURCE.evaluation._single_eval,lambda:SOURCE.evaluation._single_eval((path,cfg)))
    return dict(output=path,success=bool(observed.get('succ_flag',False)),observed='succ_flag' in observed,exception=observed.get('exception'))


def list_inputs(root):return sorted(str(p) for p in (root/'init_data').rglob('*.npy'))


def output_receipt(path,cfg):
    """Called only after the timing interval has ended."""
    import numpy as np
    output=Path(path.replace(cfg.init_dir,cfg.grasp_dir))
    row=dict(input=path,native_operation_output=output.is_file())
    if output.is_file():
        data=np.load(output,allow_pickle=True).item()
        row.update(output=str(output),pose_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(data[k]).tobytes()
            for k in ('grasp_qpos','squeeze_qpos','pregrasp_qpos'))).hexdigest())
    return row


def round_robin(paths,limit=128):
    groups={}
    for path in paths:
        scene=Path(path).parent.name;groups.setdefault(scene,[]).append(path)
    selected=[]
    for i in range(max(map(len,groups.values()),default=0)):
        for key in sorted(groups):
            if i<len(groups[key]):selected.append(groups[key][i])
            if len(selected)>=limit:return selected
    return selected


def status(root,stage,**kw):write(root/'status.json',dict(stage=stage,updated_utc=stamp(),pid=os.getpid(),**kw))
def stop_check(root):
    if (root/'STOP').exists():raise RuntimeError('STOP requested')


def stage_template(root,source):
    cfg=configuration(source,root,'tmpl',workers=1);cfg.debug_name=TYPE
    source.convert.operate_tmpl(cfg)
    path=root/'initial_templates'/f'{TYPE}.npy'
    if not path.is_file():raise RuntimeError('Original template conversion produced no template')
    write(root/'template_receipt.json',dict(path=str(path),sha256=sha(path)))
    (root/'new_templates').mkdir(exist_ok=True)


def resume_worker(value):
    path=Path('/proc')/str(value['pid'])
    if path.exists() and (path/'stat').read_text().split()[21]==value['identity']:os.kill(value['pid'],signal.SIGCONT)


def guardian(path,owner):
    while Path('/proc',str(owner)).exists():
        if path.exists():
            value=json.loads(path.read_text())
            if value.get('paused') and time.monotonic()>value['deadline']:
                resume_worker(value);write(path,dict(value,paused=False,expired=True))
        time.sleep(1)
    if path.exists():
        value=json.loads(path.read_text())
        if value.get('paused'):resume_worker(value);write(path,dict(value,paused=False,owner_exited=True))


def stage_init(root,source,production):
    import torch
    cfg=configuration(source,root,'init');write(root/'init_configuration.json',source.OmegaConf.to_container(cfg))
    probes=[]
    def probe(label):
        result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
        probes.append(dict(label=label,utc=stamp(),gpu=result.stdout))
    probe('before_initialize');start=time.perf_counter()
    try:source.init.operate_init(cfg);torch.cuda.synchronize(1)
    finally:
        probe('after_initialize')
        write(root/'initialization_resources.json',dict(probes=probes,production_workers_paused=False))
    paths=list_inputs(root);counts={p.stem:0 for p in (root/'scenes').glob('*.npy')}
    for path in paths:counts[Path(path).parent.name]+=1
    write(root/'initialization.json',dict(seconds=time.perf_counter()-start,samples=len(paths),scene_counts=counts,
        inputs=[dict(path=p,sha256=sha(p)) for p in paths],production_workers_paused=False))
    if not paths:raise RuntimeError('Original initialization emitted zero candidates; no fallback inputs permitted')


def stage_quality(root,source):
    cfg=configuration(source,root,'grasp_admm');cfg_dict=source.OmegaConf.to_container(cfg)
    inputs=list_inputs(root);jobs=[];cached={}
    for i,path in enumerate(inputs):
        folder=root/'quality/cases'/f'{i:05d}';receipt=folder/'result.json'
        if receipt.is_file():
            row=json.loads(receipt.read_text())
            if row['input']!=path or row['input_sha256']!=sha(path):raise RuntimeError('Cannot reuse mismatching quality receipt')
            cached[path]=row
        else:jobs.append((path,cfg_dict,'quality',str(folder)))
    rows=list(cached.values());start=time.perf_counter()
    with mp.Pool(32,initializer=worker_init,initargs=(str(source.root),cfg_dict)) as pool:
        for result in pool.imap(execute_job,jobs,chunksize=1):
            rows.append(result);stop_check(root)
            if len(rows)%32==0:status(root,'quality',completed=len(rows),total=len(jobs));print('QUALITY',len(rows),len(jobs),flush=True)
    by_input={r['input']:r for r in rows};rows=[by_input[path] for path in inputs]
    write(root/'quality_operation.json',dict(samples=len(inputs),rows=rows,wall_seconds=time.perf_counter()-start,reused_cases=len(cached)))
    cfg_eval=configuration(source,root,'eval');eval_dict=source.OmegaConf.to_container(cfg_eval)
    successful=[r['output'] for r in rows if r['native_operation_output']];start=time.perf_counter()
    with mp.Pool(32,initializer=worker_init,initargs=(str(source.root),eval_dict)) as pool:
        evaluation=list(pool.imap(evaluate_job,[(p,eval_dict) for p in successful],chunksize=1))
    write(root/'native_evaluation.json',dict(input_attempts=len(inputs),native_operation_outputs=len(successful),
        native_gsr_passed=sum(r['success'] for r in evaluation),rows=evaluation,wall_seconds=time.perf_counter()-start))


def stage_timing(root,source):
    inputs=list_inputs(root);subset=round_robin(inputs);write(root/'timing_inputs.json',[dict(path=p,sha256=sha(p)) for p in subset])
    if not subset:raise RuntimeError('No timing inputs')
    rows=[]
    worker_init(str(source.root),source.OmegaConf.to_container(configuration(source,root,'grasp_admm')))
    for repetition in range(6):
        cfg=configuration(source,root,'grasp_admm',root/'timing/latency'/f'{repetition:02d}',1)
        if sys.gettrace() is not None or sys.getprofile() is not None:raise RuntimeError('Timing observer detected')
        start=time.perf_counter();source.core._single_grasp((subset[0],cfg));seconds=time.perf_counter()-start
        row=dict(output_receipt(subset[0],cfg),operation_seconds=seconds)
        rows.append(dict(row,repetition=repetition,warmup=repetition==0));stop_check(root)
    write(root/'latency.json',dict(rows=rows,median_seconds=statistics.median(r['operation_seconds'] for r in rows if not r['warmup'])))
    throughput=[]
    for workers in (1,32):
        base=configuration(source,root,'grasp_admm',root/'timing'/f'workers_{workers}',workers)
        warm=mp.Queue();ready=mp.Queue()
        for i in range(workers):warm.put(subset[i%len(subset)])
        start=time.perf_counter()
        with mp.Pool(workers,initializer=worker_init,initargs=(str(source.root),source.OmegaConf.to_container(base),warm,ready)) as pool:
            warm_pids=[ready.get(timeout=900) for _ in range(workers)];startup=time.perf_counter()-start
            for repetition in range(3):
                stop_check(root);cfg=configuration(source,root,'grasp_admm',root/'timing'/f'workers_{workers}'/f'repeat_{repetition}',workers)
                jobs=[(path,source.OmegaConf.to_container(cfg)) for path in subset]
                start=time.perf_counter();pool.map(source.core._single_grasp,jobs,chunksize=1);seconds=time.perf_counter()-start
                results=[output_receipt(path,cfg) for path in subset]
                throughput.append(dict(workers=workers,repetition=repetition,samples=len(subset),seconds=seconds,
                    samples_per_second=len(subset)/seconds,startup_and_warmup_seconds=startup,warm_pids=warm_pids,rows=results))
                write(root/'throughput_progress.json',throughput);print('THROUGHPUT',workers,repetition,seconds,flush=True)
    signatures={}
    for record in throughput:
        for result in record['rows']:
            key=result['input'];sig=(result['native_operation_output'],result.get('pose_sha256'))
            signatures.setdefault(key,set()).add(sig)
    differences=[p for p,s in signatures.items() if len(s)!=1]
    write(root/'throughput.json',dict(records=throughput,different_final_outputs=differences))


def stage_consistency(root,source):
    """Observe failures in an independent, untimed one-worker run as well."""
    inputs=[r['path'] for r in json.loads((root/'timing_inputs.json').read_text())]
    reference={r['input']:r for r in json.loads((root/'quality_operation.json').read_text())['rows']}
    cfg=configuration(source,root,'grasp_admm',root/'consistency',1);cfg_dict=source.OmegaConf.to_container(cfg)
    jobs=[(p,cfg_dict,'quality',str(root/'consistency/cases'/f'{i:05d}')) for i,p in enumerate(inputs)]
    with mp.Pool(1,initializer=worker_init,initargs=(str(source.root),cfg_dict)) as pool:rows=list(pool.imap(execute_job,jobs))
    def signature(row):
        forwards=[(r['name'].split()[-1],r['passed']) for r in row['filters'] if 'check' not in r]
        return row['native_operation_output'],row.get('captured_final_sha256'),row.get('pose_sha256'),row['fallback_used'],forwards
    different=[row['input'] for row in rows if signature(row)!=signature(reference[row['input']])]
    write(root/'consistency.json',dict(samples=len(inputs),workers=[1,32],untimed=True,different_final_states_or_native_judgments=different,rows=rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--production',type=Path);p.add_argument('--source-manifest',type=Path)
    p.add_argument('--stage',choices=['prepare','template','initialize','quality','timing','consistency','diagnostics','profile','report','all'],default='all')
    p.add_argument('--guardian',type=Path);p.add_argument('--owner',type=int)
    a=p.parse_args()
    if getattr(a,'production',None):sys.path.insert(0,str(a.production));a.root=a.root.resolve();a.upstream=a.upstream.resolve()
    if a.guardian:guardian(a.guardian,a.owner);return
    if a.stage in ('quality','timing','consistency','diagnostics'):
        mp.set_start_method('spawn',force=True)
    a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.root/f'.{a.stage}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(process)d %(levelname)s %(message)s')
    if a.stage=='prepare':prepare(a.root,a.production,a.source_manifest,a.upstream);return
    if a.stage=='all':
        receipt_path=a.root/'stage_receipts.json'
        receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        fingerprint=sha(a.root/'MANIFEST.json')
        artifacts={'template':'template_receipt.json','initialize':'initialization.json','quality':'native_evaluation.json',
            'timing':'throughput.json','consistency':'consistency.json','diagnostics':'diagnostics.json',
            'profile':'phase_diagnostic.json','report':'summary.json'}
        for stage in ('template','initialize','quality','timing','consistency','diagnostics','profile','report'):
            stop_check(a.root);status(a.root,stage)
            if stage in receipts:
                receipt=receipts[stage]
                if receipt['manifest_sha256']!=fingerprint or sha(a.root/artifacts[stage])!=receipt['artifact_sha256']:
                    raise RuntimeError('Cannot reuse mismatching stage receipt: '+stage)
                continue
            env=os.environ.copy()
            if stage!='initialize':env['CUDA_VISIBLE_DEVICES']=''
            cmd=[sys.executable,__file__,'--root',str(a.root),'--upstream',str(a.upstream),'--stage',stage]
            if a.production:cmd+=['--production',str(a.production)]
            try:
                with (a.root/f'{stage}.log').open('ab') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            except BaseException as e:
                status(a.root,'failed',failed_stage=stage,error=str(e));raise
            receipts[stage]=dict(manifest_sha256=fingerprint,artifact_sha256=sha(a.root/artifacts[stage]),completed_utc=stamp())
            write(receipt_path,receipts)
        status(a.root,'complete');return
    versions={k:importlib.metadata.version(k) for k in VERSIONS}
    for name,expected in VERSIONS.items():
        if versions[name].split('+')[0]!=expected:raise RuntimeError(f'Pinned environment mismatch: {name} {versions[name]} != {expected}')
    if sys.prefix==sys.base_prefix or 'include-system-site-packages = false' not in (Path(sys.prefix)/'pyvenv.cfg').read_text():
        raise RuntimeError('An isolated environment without inherited packages is required')
    start=time.perf_counter();source=load_upstream(a.upstream);module_load_seconds=time.perf_counter()-start
    verify_assets(a.root,source);os.chdir(a.upstream);seed()
    origins={}
    for name in ('torch','mujoco','numpy','scipy','clarabel','qpsolvers','warp','trimesh'):
        module=importlib.import_module(name);origins[name]=module.__file__
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError('Package escaped isolated environment: '+name)
    runtime=dict(versions=versions,python=sys.executable,cpu_affinity=sorted(os.sched_getaffinity(0)),
        multiprocessing_start_method=mp.get_start_method(),
        module_origins=origins,torch_cuda=importlib.import_module('torch').version.cuda,
        module_load_seconds=module_load_seconds,
        cpu_loadavg=os.getloadavg(),cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None),
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
    write(a.root/'runtime.json',runtime);write(a.root/f'runtime_{a.stage}.json',runtime)
    try:
        if a.stage=='template':stage_template(a.root,source)
        elif a.stage=='initialize':stage_init(a.root,source,a.production)
        elif a.stage=='quality':stage_quality(a.root,source)
        elif a.stage=='timing':stage_timing(a.root,source)
        elif a.stage=='consistency':stage_consistency(a.root,source)
        elif a.stage=='diagnostics':
            from diagnose_native_graspadmm import run
            run(a.root,a.upstream,a.production)
        elif a.stage=='profile':
            from profile_native_graspadmm import run
            run(a.root,source)
        elif a.stage=='report':
            from report_native_graspadmm import run
            run(a.root)
        verify_assets(a.root,source)
    except BaseException as e:status(a.root,'failed',failed_stage=a.stage,error=str(e));raise


if __name__=='__main__':main()

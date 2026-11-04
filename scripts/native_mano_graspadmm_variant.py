#!/usr/bin/env python3
"""MANO rigid-body GraspADMM variant: explicit 2 mm filters and torque rollouts."""
from pathlib import Path
import argparse,copy,fcntl,json,multiprocessing as mp,os,shutil,statistics,subprocess,sys,time,traceback
import numpy as np
import native_graspadmm_retest as runner
try:
    from native_grasp_extensions import Settings,configure_penetration,evaluate_wrenches
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from retarget.env.grasp.native_grasp_extensions import Settings,configure_penetration,evaluate_wrenches

BASE_CONFIGURATION=runner.configuration
SETTINGS=Settings(hand_object_penetration_m=.002,self_penetration_m=.002)


def configuration(source,root,stage,output=None,workers=32):
    cfg=BASE_CONFIGURATION(source,root,stage,output,workers)
    cfg.hand_name='mano';cfg.hand=source.OmegaConf.load(source.root/'dexonomy/config/hand/mano.yaml')
    cfg.hand.xml_path=str(root/'hand/mano/right.xml')
    cfg.raw_anno_dir=str(root/'hand/mano/type7_annotation')
    cfg.hand_skt_path=str(root/'hand/mano/skeleton.yaml');cfg.hand_kp_path=str(root/'hand/mano/keypoint.yaml')
    cfg.hand_bgroup_path=str(root/'hand/mano/body_group.yaml')
    if stage=='grasp_admm':configure_penetration(cfg,SETTINGS)
    return source.OmegaConf.create(source.OmegaConf.to_container(cfg,resolve=True))


def prepare(root,baseline):
    if (root/'MANIFEST.json').exists():return
    upstream=(baseline/'upstream').resolve()
    if not (root/'upstream').exists():(root/'upstream').symlink_to(upstream,target_is_directory=True)
    if not (root/'scenes').exists():(root/'scenes').symlink_to(baseline/'scenes',target_is_directory=True)
    official=root/'mano_original';hand=root/'hand/mano'
    official_receipt=json.loads((root/'MANO_OFFICIAL_FILES.json').read_text())
    if official_receipt['commit']!=runner.COMMIT:raise RuntimeError('MANO asset commit mismatch')
    for name,expected in official_receipt['sha256'].items():
        if runner.sha(official/name.split('assets/hand/mano/',1)[1])!=expected:raise RuntimeError('MANO asset transfer mismatch: '+name)
    shutil.copytree(official,hand,dirs_exist_ok=True)
    xml=hand/'right.xml';original=xml.read_text()
    if original.count('noslip_stepations')!=1:raise RuntimeError('Unexpected upstream MANO XML; refusing guessed repairs')
    xml.write_text(original.replace('noslip_stepations','noslip_iterations'))
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(xml))
    if (model.nq,model.nu,model.ngeom)!=(20,20,16):raise RuntimeError('Unexpected MANO dimensions')
    source=runner.load_upstream(upstream);O=source.OmegaConf
    annotation=O.load(official/'raw_anno/4_Adducted_Thumb.yaml')
    keep=['index3','middle3','ring3','thumb3']
    annotation.contact=O.create({name:O.to_container(annotation.contact[name]) for name in keep})
    annotation.required_cbody=[[name] for name in keep]
    target=hand/'type7_annotation'/f'{runner.TYPE}.yaml';target.parent.mkdir(exist_ok=True)
    O.save(annotation,target)
    manifest=json.loads((baseline/'MANIFEST.json').read_text())
    manifest.update(authorization='User selected upstream rigid MANO; repair assets and adapt initialization; allow hand-object and self penetration 2 mm; add six pure torque tests',
        variant='rigid_mano_penetration_2mm_torque_v1',parent_manifest_sha256=runner.sha(baseline/'MANIFEST.json'),
        parent_directory=str(baseline),created_utc=runner.stamp(),settings=__import__('dataclasses').asdict(SETTINGS),
        template_adapter=dict(original='MANO 4_Adducted_Thumb.yaml',target_type=runner.TYPE,
            unchanged_joint_seed=True,retained_original_contact_annotations=keep,
            note='Adapted four-fingertip type7 initialization; not an official MANO type7 template; no Shadow qpos retargeting'),
        xml_repairs=[dict(old='noslip_stepations',new='noslip_iterations',reason='Invalid MuJoCo option attribute spelling')],
        hand_model='Upstream 20-joint segmented rigid MANO; not the 778-vertex PCA skin model',mano_official_source=official_receipt,
        multiprocessing_start_method='spawn',callback_required=False)
    frozen=['native_mano_graspadmm_variant.py','native_graspadmm_retest.py','native_grasp_extensions.py',
            'diagnose_native_graspadmm.py','export_native_graspadmm_viewer.py','MANO_OFFICIAL_FILES.json']
    manifest['runner_scripts_sha256']={name:runner.sha(root/name) for name in frozen}
    manifest['input_assets_sha256'].update({str(p.resolve()):runner.sha(p) for folder in (official,hand) for p in folder.rglob('*') if p.is_file()})
    runner.write(root/'MANIFEST.json',manifest)
    runner.write(root/'asset_adapter.json',dict(original_sha256={str(p.relative_to(official)):runner.sha(p) for p in official.rglob('*') if p.is_file()},
        adapted_sha256={str(p.relative_to(hand)):runner.sha(p) for p in hand.rglob('*') if p.is_file()},
        dimensions=dict(qpos=model.nq,actuators=model.nu,geoms=model.ngeom),template=manifest['template_adapter'],xml_repairs=manifest['xml_repairs']))


def extended_eval_job(job):
    index,path,cfg_dict=job;source=runner.SOURCE;cfg=source.OmegaConf.create(cfg_dict)
    import importlib
    helpers=importlib.import_module('dexonomy.util.traj_util')
    row=dict(index=index,input=path,complete=False,force_passed=False,torque_passed=False,combined_passed=False)
    try:
        data=np.load(path,allow_pickle=True).item();scene=source.files.load_scene_cfg(data['scene_path'])
        for o in scene['scene'].values():
            if o['type']=='rigid_object':
                info=source.files.load_json(o['info_path']);coef=info['mass']/(info['density']*info['scale']**3)
                o['density']=cfg.obj_mass/(coef*np.prod(o['scale']))
        env=source.sim.MuJoCo_EvalEnv(source.sim.HandCfg(cfg.hand.xml_path,freejoint=True),scene,source.sim.MuJoCo_EvalCfg(miu_coef=list(cfg.op.miu_coef)),debug_view=False,hard=cfg.op.hard)
        reference=env.get_active_obj_pose().copy();planner=helpers.get_planner(scene['task'])
        trajectory=planner.plan_trajectory(reference,data['pregrasp_qpos'],data['grasp_qpos'],data['squeeze_qpos'],None)
        mesh,*_=source.core.load_obj_trimesh(data);radius=float(np.linalg.norm(mesh.vertices-mesh.center_mass,axis=-1).max())
        result=evaluate_wrenches(env,*trajectory,radius,SETTINGS)
        row.update(result,complete=True)
    except Exception:row['exception']=traceback.format_exc()
    return row


def extended_evaluation(root,source):
    quality=json.loads((root/'quality_operation.json').read_text());cfg=configuration(source,root,'eval')
    jobs=[(i,r['output'],source.OmegaConf.to_container(cfg)) for i,r in enumerate(quality['rows']) if r['native_operation_output']]
    start=time.perf_counter();rows=[]
    with mp.Pool(32,initializer=runner.worker_init,initargs=(str(source.root),source.OmegaConf.to_container(cfg))) as pool:
        for row in pool.imap(extended_eval_job,jobs,chunksize=1):
            rows.append(row);runner.stop_check(root)
            if len(rows)%32==0:runner.status(root,'wrench_evaluation',completed=len(rows),total=len(jobs))
    runner.write(root/'wrench_evaluation.json',dict(input_attempts=quality['samples'],evaluated_outputs=len(rows),
        missing_or_rejected_inputs=quality['samples']-len(rows),force_passed=sum(r['force_passed'] for r in rows),
        torque_passed=sum(r['torque_passed'] for r in rows),combined_passed=sum(r['combined_passed'] for r in rows),
        exceptions=sum(not r['complete'] for r in rows),wall_seconds=time.perf_counter()-start,rows=rows,
        torque_scale='1 * mass * 9.8 * max vertex radius about object COM; magnitude also saved in N m per input',
        note='Extended tests; no added grasp objective. Original six-force GSR retained separately. Rejected outputs count as failures.'))


def timing(root,source):
    inputs=runner.list_inputs(root);selected=runner.round_robin(inputs,128);rows=[]
    runner.worker_init(str(source.root),source.OmegaConf.to_container(configuration(source,root,'grasp_admm')))
    for rep in range(6):
        cfg=configuration(source,root,'grasp_admm',root/'timing/latency'/str(rep),1)
        start=time.perf_counter();source.core._single_grasp((selected[0],cfg));seconds=time.perf_counter()-start
        rows.append(dict(runner.output_receipt(selected[0],cfg),operation_seconds=seconds,warmup=rep==0));runner.stop_check(root)
    runner.write(root/'latency.json',dict(rows=rows,median_seconds=statistics.median(r['operation_seconds'] for r in rows[1:])))
    cfg=configuration(source,root,'grasp_admm',root/'timing/throughput',32);warm=mp.Queue();ready=mp.Queue()
    for i in range(32):warm.put(selected[i%len(selected)])
    records=[];start=time.perf_counter()
    with mp.Pool(32,initializer=runner.worker_init,initargs=(str(source.root),source.OmegaConf.to_container(cfg),warm,ready)) as pool:
        pids=[ready.get(timeout=900) for _ in range(32)];startup=time.perf_counter()-start
        for rep in range(3):
            cfg=configuration(source,root,'grasp_admm',root/'timing/throughput'/str(rep),32)
            jobs=[(p,source.OmegaConf.to_container(cfg)) for p in selected]
            start=time.perf_counter();pool.map(source.core._single_grasp,jobs,chunksize=1);seconds=time.perf_counter()-start
            records.append(dict(workers=32,repetition=rep,samples=len(selected),seconds=seconds,samples_per_second=len(selected)/seconds,
                startup_and_warmup_seconds=startup,warm_pids=pids));runner.stop_check(root)
    runner.write(root/'throughput.json',dict(records=records))


def diagnostics(root,source,production):
    import diagnose_native_graspadmm
    diagnose_native_graspadmm.run(root,source.root,production)
    d=json.loads((root/'diagnostics.json').read_text())
    for r in d['rows']:
        r['zero_penetration_pass']=r.get('native_collision_pass',False)
        r['tolerated_collision_pass']=(r.get('valid_final_state',False) and r.get('hand_object_depth_m',np.inf)<=SETTINGS.hand_object_penetration_m and r.get('self_depth_m',np.inf)<=SETTINGS.self_penetration_m)
        r['diagnostic_joint_pass']=r['projected_hand'] and r['slot_joint_pass'] and r['tolerated_collision_pass'] and r.get('native_contact_count',0)>0
    d.update(tolerated_collision_passed=sum(r['tolerated_collision_pass'] for r in d['rows']),
        diagnostic_joint_passed=sum(r['diagnostic_joint_pass'] for r in d['rows']),penetration_tolerances_m=dict(hand_object=.002,self=.002))
    runner.write(root/'diagnostics.json',d)


def report(root):
    read=lambda n:json.loads((root/n).read_text())
    q=read('quality_operation.json');d=read('diagnostics.json');e=read('native_evaluation.json');w=read('wrench_evaluation.json');lat=read('latency.json');tp=read('throughput.json')
    n=q['samples'];wrows={r['index']:r for r in w['rows']};entries=[]
    for i,(qr,dr) in enumerate(zip(q['rows'],d['rows'])):
        entries.append(dict(index=i,**dr,operation=qr,wrench=wrows.get(i),extended_wrench_pass=wrows.get(i,{}).get('combined_passed',False)))
    runner.write(root/'per_input_results.json',entries)
    measured=[r['operation_seconds'] for r in lat['rows'] if not r['warmup']]
    summary=dict(complete=True,model='Upstream rigid MANO (20 joints)',inputs=n,native_outputs=e['native_operation_outputs'],gsr_passed=e['native_gsr_passed'],
        projected_hand_passed=d['projected_hand_passed'],projected_targets_passed=d['projected_targets_passed'],slot_joint_passed=d['slot_joint_passed'],
        tolerated_collision_passed=d['tolerated_collision_passed'],diagnostic_joint_passed=d['diagnostic_joint_passed'],
        extended_force_passed=w['force_passed'],torque_passed=w['torque_passed'],combined_wrench_passed=w['combined_passed'],
        latency_median_seconds=statistics.median(measured),throughput_median=statistics.median(r['samples_per_second'] for r in tp['records']))
    runner.write(root/'summary.json',summary)
    lines=['# MANO 刚体 GraspADMM：2 mm 穿透与扭矩扩展','',
        '使用原固定提交的 MANO 20 关节刚体网格，非 778 顶点 PCA 蒙皮 MANO。修复 XML 的 noslip_stepations 拼写；原文件保留于 mano_original。',
        'type7 初始化适配：从原 MANO type4 关节 seed 与原始标注保留 index3/middle3/ring3/thumb3 四槽。不是上游官方 MANO type7 模板，也不是 Shadow 最终姿态转换。',
        'ADMM 接触搜索、Clarabel QP、MuJoCo 虚拟力推进、对偶更新、fallback、squeeze/pregrasp 均调用原函数；grasp filter 的手物/自碰撞阈值设为 -2 mm，pregrasp 自碰撞也允许2 mm；pregrasp 手物4 cm净空保留。',
        '接触物理响应未通过 margin 或 solref 调软；2 mm 是允许的验收深度，完整深度仍逐条记录。未新增目标函数。','',
        f'16 个固定物体尺度组合，初始化与实际送入样本共 {n} 条；以下通过率分母为全部输入。','',
        '| 指标 | 通过数 |','|---|---:|']
    labels={'native_outputs':'原 operation 输出','gsr_passed':'原生六方向外力 GSR','projected_hand_passed':'最终手点投影闭合','projected_targets_passed':'ADMM 目标点投影闭合',
        'slot_joint_passed':'距离/法向联合覆盖≥80%','tolerated_collision_passed':'手物/自穿透均≤2 mm','diagnostic_joint_passed':'闭合+覆盖+2 mm碰撞验收',
        'extended_force_passed':'新增框架六方向纯外力','torque_passed':'新增六方向纯扭矩','combined_wrench_passed':'12 工况全部通过'}
    lines += [f'| {label} | {summary[key]}/{n} ({summary[key]/n:.2%}) |' for key,label in labels.items()]
    lines += ['',f'独立单条完整 operation，预热一次、测五次，中位数 {summary["latency_median_seconds"]:.6f} s；范围 {min(measured):.6f}–{max(measured):.6f} s。',
        f'32 worker 固定128条三轮吞吐中位数 {summary["throughput_median"]:.4f} 条/s。主计时无 profiler；加载与原 operation 文件保存包含其中；扭矩评测单独计时。',
        f'新增12工况评测 wall time {w["wall_seconds"]:.3f} s，异常 {w["exceptions"]} 条。每方向独立复位并执行原接近/抓握/挤压轨迹，然后施加500步扰动，每50步检查，仍用5 cm/15°阈值。',
        '纯扭矩为世界坐标 ±XYZ；大小采用 m·9.8·r（r为物体网格相对质心的最大半径），每条实际 N·m 保存在 wrench_evaluation.json；这是本实验尺度约定，不宣称来自论文标准。纯扭矩不附加净外力，也不加入 pose 求解目标。',
        '全部方向保留最终 qpos 供回放，不保存完整仿真历史。目标点闭合不表示手已贴合；投影闭合不替代动力学稳定性。新旧手模型与初始化不同，不能据此宣称同初值质量提升。','',
        '四卡生产 CEM 及原生 Shadow 基线不变。来源、依赖、适配资产与脚本指纹在 MANIFEST.json / asset_adapter.json；逐条诊断与失败见 per_input_results.json。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--production',type=Path,required=True)
    p.add_argument('--stage',default='all',choices=['all','prepare','template','initialize','quality','diagnostics','wrench','timing','report','export'])
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);baseline=a.baseline.resolve()
    lock=(root/('.'+a.stage+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=baseline/'runtime.json'
    if runtime.exists():
        allowed=set(os.sched_getaffinity(0)) & set(json.loads(runtime.read_text())['cpu_affinity'])
        if not allowed:raise RuntimeError('Baseline CPU affinity unavailable')
        os.sched_setaffinity(0,allowed)
    runner.configuration=configuration;mp.set_start_method('spawn',force=True)
    if a.stage=='prepare':prepare(root,baseline);return
    if a.stage=='all':
        receipts=json.loads((root/'stage_receipts.json').read_text()) if (root/'stage_receipts.json').exists() else {}
        fingerprint=runner.sha(root/'MANIFEST.json')
        artifacts=dict(template='template_receipt.json',initialize='initialization.json',quality='quality_operation.json',diagnostics='diagnostics.json',wrench='wrench_evaluation.json',timing='throughput.json',report='summary.json',export='viewer/manifest.json')
        for stage,artifact in artifacts.items():
            runner.stop_check(root)
            if stage in receipts:
                if receipts[stage]['manifest_sha256']!=fingerprint or runner.sha(root/artifact)!=receipts[stage]['artifact_sha256']:raise RuntimeError('Stage identity changed')
                continue
            runner.status(root,stage)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
            if stage!='initialize':env['CUDA_VISIBLE_DEVICES']=''
            cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--baseline',str(baseline),'--production',str(a.production),'--stage',stage]
            with (root/f'{stage}.log').open('ab') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipts[stage]=dict(manifest_sha256=fingerprint,artifact_sha256=runner.sha(root/artifact),completed_utc=runner.stamp());runner.write(root/'stage_receipts.json',receipts)
        runner.status(root,'complete');return
    source=runner.load_upstream(root/'upstream');runner.verify_assets(root,source);runner.seed()
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in runner.VERSIONS}
    if any(versions[k].split('+')[0]!=v for k,v in runner.VERSIONS.items()):raise RuntimeError('Pinned dependency versions differ')
    runner.write(root/f'runtime_{a.stage}.json',dict(python=sys.executable,versions=versions,affinity=sorted(os.sched_getaffinity(0)),loadavg=os.getloadavg()))
    os.chdir(source.root)
    if a.stage=='template':runner.stage_template(root,source)
    elif a.stage=='initialize':runner.stage_init(root,source,a.production)
    elif a.stage=='quality':runner.stage_quality(root,source)
    elif a.stage=='diagnostics':diagnostics(root,source,a.production)
    elif a.stage=='wrench':extended_evaluation(root,source)
    elif a.stage=='timing':timing(root,source)
    elif a.stage=='report':report(root)
    elif a.stage=='export':
        import export_native_graspadmm_viewer
        export_native_graspadmm_viewer.run(root,root/'viewer')
    runner.verify_assets(root,source)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        detail=traceback.format_exc()
        if '--root' in sys.argv:
            failure_root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
            runner.status(failure_root,'failed',error=str(error),traceback=detail)
        print(detail,flush=True);raise

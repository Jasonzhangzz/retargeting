#!/usr/bin/env python3
"""Transfer only GSR-passing Shadow grasps, then run unchanged native MANO ADMM."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib.metadata,json,multiprocessing as mp,shutil,statistics,subprocess,sys,time,traceback
import numpy as np
import native_graspadmm_retest as runner
import native_mano_graspadmm_variant as variant
try:
    from shadow_mano_seed import ShadowManoTransfer,TransferSettings
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from retarget.env.grasp.shadow_mano_seed import ShadowManoTransfer,TransferSettings

TRANSFER=None
OBJECTS={}


def prepare(root,mano,shadow):
    if (root/'MANIFEST.json').exists():return
    root.mkdir(parents=True,exist_ok=True)
    shutil.copytree(mano/'hand',root/'hand',dirs_exist_ok=True)
    if not (root/'upstream').exists():(root/'upstream').symlink_to((shadow/'upstream').resolve(),target_is_directory=True)
    if not (root/'scenes').exists():(root/'scenes').symlink_to((shadow/'scenes').resolve(),target_is_directory=True)
    quality=json.loads((shadow/'quality_operation.json').read_text());evaluation=json.loads((shadow/'native_evaluation.json').read_text())
    passed={r['output'] for r in evaluation['rows'] if r['success'] and r['observed']};selected=[]
    for i,row in enumerate(quality['rows']):
        if row.get('output') not in passed:continue
        state=shadow/'quality/cases'/f'{i:05d}'/'final_state.npz'
        selected.append(dict(shadow_index=i,source=row['output'],source_sha256=runner.sha(row['output']),source_input=row['input'],
            source_state=str(state),source_state_sha256=runner.sha(state),hand_cbody=row['hand_cbody'],source_native_gsr_pass=True,
            scene=Path(row['input']).parent.name))
    if len(selected)!=len(passed):raise RuntimeError('Some passing source grasps have no quality state receipt')
    runner.write(root/'source_selection.json',dict(count=len(selected),policy='All and only observed native six-direction GSR successes; no resampling or padding',rows=selected))
    manifest=json.loads((mano/'MANIFEST.json').read_text())
    manifest.update(authorization='User: use feasible Shadow grasp solutions as MANO initial guesses',variant='shadow_gsr_to_rigid_mano_v1',
        created_utc=runner.stamp(),parent_mano_directory=str(mano),shadow_directory=str(shadow),
        source_selection_sha256=runner.sha(root/'source_selection.json'),source_grasps=len(selected),
        source_gsr_passed=len(selected),max_initialization_outputs=len(selected),
        template_adapter=dict(mode='Per-pose anatomical transfer from verified Shadow GSR success',no_template_sampling=True),
        init_budget=dict(source_poses=len(selected),bone_ik_max_evaluations=80,contact_ik_max_evaluations=80),
        script_sha256=runner.sha(__file__),initialization='Anatomical rigid-hand point IK from actual passing Shadow grasp qpos and moved contact slots; no fresh sampler',
        initialization_ik=__import__('dataclasses').asdict(TransferSettings()),
        contact_ik_terms='Only corresponding Cartesian point positions; no grasp cost, collision penalty, normal cost or force closure term',
        normal_diagnostic_version='actual_hand_surface_v2; do not compare directly to legacy transported template-normal joint counts',
        timing=dict(latency_warmup=1,latency_repeats=5,throughput_max_inputs=128,workers=[32],repeats=3),
        production_gpu_pause='none; all transfer/refinement/evaluation work is CPU',callback_required=False)
    files=['run_shadow_seeded_mano.py','shadow_mano_seed.py','shadow_seed_diagnostics.py','native_mano_graspadmm_variant.py',
        'native_graspadmm_retest.py','native_grasp_extensions.py','diagnose_native_graspadmm.py','export_native_graspadmm_viewer.py']
    manifest['runner_scripts_sha256']={n:runner.sha(root/n) for n in files}
    manifest['input_assets_sha256'].update({str(p):runner.sha(p) for p in (root/'hand').rglob('*') if p.is_file()})
    for row in selected:
        manifest['input_assets_sha256'][row['source']]=row['source_sha256'];manifest['input_assets_sha256'][row['source_state']]=row['source_state_sha256']
    for p in (shadow/'native_evaluation.json',shadow/'quality_operation.json',root/'source_selection.json'):
        manifest['input_assets_sha256'][str(p)]=runner.sha(p)
    runner.write(root/'MANIFEST.json',manifest)


def transfer_job(job):
    global TRANSFER,OBJECTS
    entry,root_text=job;root=Path(root_text);source=runner.SOURCE;record=dict(entry,completed=False)
    folder=root/'mapping/cases'/f'{entry["shadow_index"]:05d}';folder.mkdir(parents=True,exist_ok=True)
    try:
        if runner.sha(entry['source'])!=entry['source_sha256'] or runner.sha(entry['source_state'])!=entry['source_state_sha256']:raise RuntimeError('Source changed')
        if TRANSFER is None:TRANSFER=ShadowManoTransfer(source.root/'assets/hand/shadow/right.xml',runner.BASE.hand.xml_path)
        data=np.load(entry['source'],allow_pickle=True).item();state=np.load(entry['source_state']);q=data['grasp_qpos'].reshape(-1)
        np.testing.assert_array_equal(q,state['final_grasp_qpos'].reshape(-1))
        bodies=entry['hand_cbody'];local=state['hand_cpn_b'][:,:3]
        start=time.perf_counter();mapped,names,contacts,info=TRANSFER.fit(q,bodies,local);seconds=time.perf_counter()-start
        key=data['scene_path']
        if key not in OBJECTS:OBJECTS[key]=source.core.load_obj_trimesh(data)[0]
        mesh=OBJECTS[key];import trimesh
        source_points=np.array(info['source_hand_cpn_w'])[:,:3]
        projected,distance,faces=trimesh.proximity.closest_point(mesh,source_points)
        bary=trimesh.triangles.points_to_barycentric(mesh.triangles[faces],projected)
        normal=(mesh.vertex_normals[mesh.faces[faces]]*bary[...,None]).sum(1);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        initial=copy.deepcopy(data)
        initial.update(hand_name='mano',grasp_qpos=mapped[None].astype(np.float32),hand_cbody=names,hand_cpn_w=contacts.astype(np.float32),
            required_cbody=[[b] for b in names],obj_cpn_w=np.c_[projected,normal].astype(np.float32),n_evo=np.array([0],np.float32),
            shadow_source=entry['source'],shadow_source_sha256=entry['source_sha256'],shadow_native_gsr_pass=True)
        for name in ('squeeze_qpos','pregrasp_qpos','ho_c'):initial.pop(name,None)
        path=root/'init_data'/runner.TYPE/entry['scene']/f'shadow_{entry["shadow_index"]:05d}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,initial)
        record.update(completed=True,input=str(path),input_sha256=runner.sha(path),fit_seconds=seconds,**info)
        runner.write(folder/'mapping.json',record)
    except Exception:record['exception']=traceback.format_exc();runner.write(folder/'mapping.json',record)
    return record


def transfer(root,source):
    entries=json.loads((root/'source_selection.json').read_text())['rows'];cfg=variant.configuration(source,root,'grasp_admm');rows=[];start=time.perf_counter()
    with mp.Pool(32,initializer=runner.worker_init,initargs=(str(source.root),source.OmegaConf.to_container(cfg))) as pool:
        for row in pool.imap(transfer_job,[(e,str(root)) for e in entries],chunksize=1):
            rows.append(row);runner.stop_check(root)
            if len(rows)%32==0:runner.status(root,'shadow_to_mano_ik',completed=len(rows),total=len(entries))
    paths=runner.list_inputs(root);counts={p.stem:0 for p in (root/'scenes').glob('*.npy')}
    for path in paths:counts[Path(path).parent.name]+=1
    runner.write(root/'mapping.json',dict(selected=len(entries),completed=sum(r['completed'] for r in rows),failures=sum(not r['completed'] for r in rows),
        within_2mm=sum(r.get('contact_fit_within_2mm',False) for r in rows),wall_seconds=time.perf_counter()-start,rows=rows))
    runner.write(root/'initialization.json',dict(seconds=time.perf_counter()-start,samples=len(paths),source_grasps=len(entries),scene_counts=counts,
        inputs=[dict(path=p,sha256=runner.sha(p)) for p in paths],worker_pause='none'))
    if not paths:raise RuntimeError('No finite mapped initial poses; no replacement initialization allowed')


def export(root,source,shadow):
    import export_native_graspadmm_viewer
    output=root/'viewer';export_native_graspadmm_viewer.run(root,output)
    manifest=json.loads((output/'manifest.json').read_text());mapping={r['input']:r for r in json.loads((root/'mapping.json').read_text())['rows'] if r['completed']}
    source_geometry=np.load(shadow/'viewer/geometry.npz');source_poses=np.load(shadow/'viewer/poses.npz');source_manifest=json.loads((shadow/'viewer/manifest.json').read_text())
    geometry=dict(np.load(output/'geometry.npz'));poses=dict(np.load(output/'poses.npz'))
    for k in range(source_manifest['hand_geoms']):
        for kind in ('vertices','faces'):geometry[f'shadow_hand_{k}_{kind}']=source_geometry[f'hand_{k}_{kind}']
    for row in manifest['cases']:
        m=mapping[row['input']];i=row['index'];j=m['shadow_index']
        row['source_shadow']=dict(index=j,path=m['source'],native_gsr_pass=True,max_mapping_error_m=m['max_contact_error_m'],mapping_fit_within_2mm=m['contact_fit_within_2mm'])
        for field in ('positions','wxyz'):poses[f'{i}_shadow_{field}']=source_poses[f'{j}_final_{field}']
        poses[f'{i}_shadow_slots']=np.asarray(m['source_hand_cpn_w'])
    manifest.update(seed_kind='shadow_gsr_transfer',shadow_hand_geoms=source_manifest['hand_geoms'],source_selected=json.loads((root/'source_selection.json').read_text())['count'])
    np.savez_compressed(output/'geometry.npz',**geometry);np.savez_compressed(output/'poses.npz',**poses);runner.write(output/'manifest.json',manifest)


def report(root):
    read=lambda n:json.loads((root/n).read_text())
    selection=read('source_selection.json');mapping=read('mapping.json');q=read('quality_operation.json');e=read('native_evaluation.json');before=read('initial_diagnostics.json');after=read('diagnostics.json');w=read('wrench_evaluation.json');lat=read('latency.json');tp=read('throughput.json')
    n=selection['count'];physical={r['output']:r['success'] for r in e['rows']};wr={r['index']:r for r in w['rows']}
    maps={r.get('input'):r for r in mapping['rows'] if r['completed']};joined=[]
    for i,(r,b,a) in enumerate(zip(q['rows'],before['rows'],after['rows'])):
        joined.append(dict(index=i,source=maps[r['input']],operation=r,before=b,after=a,wrench=wr.get(i),native_gsr_pass=physical.get(r.get('output'),False)))
    runner.write(root/'per_input_results.json',joined)
    summary=dict(complete=True,source_shadow_gsr_passed=n,mapped_inputs=mapping['completed'],mapping_failures=mapping['failures'],mapping_contacts_within_2mm=mapping['within_2mm'],
        native_outputs=e['native_operation_outputs'],gsr_passed=e['native_gsr_passed'],torque_passed=w['torque_passed'],combined_wrench_passed=w['combined_passed'],
        before={k:v for k,v in before.items() if k!='rows'},after={k:v for k,v in after.items() if k!='rows'},
        latency_median_seconds=lat['median_seconds'],throughput_median=statistics.median(r['samples_per_second'] for r in tp['records']),normal_version='actual_hand_surface_v2')
    runner.write(root/'summary.json',summary)
    lines=['# Shadow 可行抓姿 → MANO 初值 → 原版 ADMM','',
        f'来源为固定 Shadow 原生实验中已观察到六方向 GSR 成功的全部 {n} 条。没有重新随机采样、候选补齐或从失败来源替换。物体网格、尺度和世界坐标不变。',
        f'MANO 为同一20关节刚体模型。先用等骨长对应点 IK 传递五指形状与腕部，再用对应四个指端点位置 IK 贴合 Shadow 实际接触。两步各最多80次函数评估；不加入法向、碰撞、闭合、力或姿态罚项。ADMM仍为原50×10预算、Clarabel、虚拟力、对偶和fallback。手物/自穿透均允许2 mm。',
        f'映射完成 {mapping["completed"]}/{n}，异常 {mapping["failures"]}；四个源接触点均在2 mm映射误差内 {mapping["within_2mm"]}/{n}。映射误差不等于物理可行，初值实际几何另行测量。','',
        '| 判据 | 映射后初值 | ADMM 后 |','|---|---:|---:|']
    for key,label in [('distance_passed','距离覆盖'),('normal_passed','法向覆盖'),('slot_joint_passed','距离+法向覆盖'),('tolerated_collision_passed','手物/自穿透≤2mm'),('geometry_joint_passed','闭合+覆盖+穿透联合')]:
        lines.append(f'| {label} | {before[key]}/{n} | {after[key]}/{n} |')
    lines+=['',f'原生生成输出 {e["native_operation_outputs"]}/{n}；原六外力 GSR {e["native_gsr_passed"]}/{n}；六纯扭矩稳定 {w["torque_passed"]}/{n}；12工况联合 {w["combined_passed"]}/{n}。映射失败和无输出保留在来源分母中。',
        '法向诊断v2：在实际更新的手面接触点重新查询碰撞三角形并插值法向，不沿用移动前模板法向。旧模板法向与旧角度同时保存；新旧法向联合通过数不可直接混比。覆盖阈值±2mm/30°/80%。',
        f'单条完整ADMM operation预热一次测五次，中位数 {summary["latency_median_seconds"]:.6f} s；32 workers吞吐中位数 {summary["throughput_median"]:.4f} 条/s。初始化IK wall time {mapping["wall_seconds"]:.3f} s另计，未混入ADMM计时。',
        '扭矩测试保留上一版的 m·9.8·r 幅度、500步与5cm/15°阈值，不加入求解目标。来源经过Shadow成功筛选，不能将490条的比例与旧800随机候选直接作为无偏因果提升比较。',
        '全部阶段在CPU；没有暂停任何生产GPU worker。Shadow基线、旧MANO实验和生产CEM结果均保留。全部指纹、逐条映射与源抓姿索引可复核。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mano',type=Path,required=True);p.add_argument('--shadow',type=Path,required=True);p.add_argument('--production',type=Path,required=True)
    p.add_argument('--stage',default='all',choices=['prepare','all','transfer','initial_metrics','quality','final_metrics','wrench','timing','report','export']);a=p.parse_args()
    root=a.root.resolve();mano=a.mano.resolve();shadow=a.shadow.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/('.'+a.stage+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=json.loads((shadow/'runtime.json').read_text());os.sched_setaffinity(0,set(runtime['cpu_affinity'])&set(os.sched_getaffinity(0)))
    runner.configuration=variant.configuration;mp.set_start_method('spawn',force=True)
    if a.stage=='prepare':prepare(root,mano,shadow);return
    if a.stage=='all':
        artifacts=dict(transfer='mapping.json',initial_metrics='initial_diagnostics.json',quality='native_evaluation.json',final_metrics='diagnostics.json',wrench='wrench_evaluation.json',timing='throughput.json',report='summary.json',export='viewer/manifest.json')
        receipts=json.loads((root/'stage_receipts.json').read_text()) if (root/'stage_receipts.json').exists() else {};fingerprint=runner.sha(root/'MANIFEST.json')
        for stage,artifact in artifacts.items():
            runner.stop_check(root)
            if stage in receipts:
                if receipts[stage]['manifest_sha256']!=fingerprint or receipts[stage]['artifact_sha256']!=runner.sha(root/artifact):raise RuntimeError('Stage receipt mismatch')
                continue
            runner.status(root,stage);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--mano',str(mano),'--shadow',str(shadow),'--production',str(a.production),'--stage',stage]
            with (root/f'{stage}.log').open('ab') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipts[stage]=dict(manifest_sha256=fingerprint,artifact_sha256=runner.sha(root/artifact),completed_utc=runner.stamp());runner.write(root/'stage_receipts.json',receipts)
        runner.status(root,'complete');return
    versions={k:importlib.metadata.version(k) for k in runner.VERSIONS}
    if any(versions[k].split('+')[0]!=v for k,v in runner.VERSIONS.items()):raise RuntimeError('Pinned versions differ')
    source=runner.load_upstream(root/'upstream');runner.verify_assets(root,source);runner.seed();os.chdir(source.root)
    cfg=variant.configuration(source,root,'grasp_admm');runner.write(root/f'runtime_{a.stage}.json',dict(python=sys.executable,versions=versions,affinity=sorted(os.sched_getaffinity(0))))
    if a.stage=='transfer':transfer(root,source)
    elif a.stage in ('initial_metrics','final_metrics'):
        import shadow_seed_diagnostics
        shadow_seed_diagnostics.run(root,source,cfg,a.production.resolve(),initial=a.stage=='initial_metrics')
    elif a.stage=='quality':runner.stage_quality(root,source)
    elif a.stage=='wrench':variant.extended_evaluation(root,source)
    elif a.stage=='timing':variant.timing(root,source)
    elif a.stage=='report':report(root)
    elif a.stage=='export':export(root,source,shadow)
    runner.verify_assets(root,source)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--root' in sys.argv:runner.status(Path(sys.argv[sys.argv.index('--root')+1]).resolve(),'failed',error=str(error),traceback=traceback.format_exc())
        raise

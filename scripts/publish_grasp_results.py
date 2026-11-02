#!/usr/bin/env python3
"""Export the entire best verified policy as a portable, auditable Git bundle.

Includes failures, both underlying candidates, exact geometry and recorded
native trajectories. No optimization or success-based frame selection occurs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from retarget.common.results import write_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_pose(q):
    return np.r_[q[:3],Rotation.from_quat(q[3:7],scalar_first=True).as_euler('XYZ'),q[7:]]


def export(source,destination):
    source=source.resolve();destination=destination.resolve()
    if destination.exists():raise ValueError('Export requires a new destination; never overwrite measured results')
    report=json.loads((source/'report.json').read_text())
    if report['state']!='completed':raise ValueError('Only publish completed experiments')
    rows=json.loads((source/'summary.json').read_text())['jobs']
    if not json.loads((source/'audit.json').read_text())['passed']:raise ValueError('Source audit failed')
    destination.mkdir(parents=True)
    for name in ('assets','models','poses','trajectories','cases','evidence'):(destination/name).mkdir()
    model_cache={};manifest=[];index=[];admm=[];mppi=[];selected=[];asset_origins={}

    def model(path):
        path=Path(path)
        if str(path) in model_cache:return model_cache[str(path)]
        tree=ET.parse(path);root=tree.getroot();compiler=root.find('compiler')
        meshdir=Path(compiler.get('meshdir','.')) if compiler is not None else Path('.')
        if not meshdir.is_absolute():meshdir=path.parent/meshdir
        for asset in root.findall('./asset/mesh'):
            old=Path(asset.get('file'));old=old if old.is_absolute() else meshdir/old
            name=sha(old)+old.suffix.lower();target=destination/'assets'/name
            if not target.exists():shutil.copy2(old,target)
            asset_origins.setdefault(name,dict(sha256=sha(old),original_name=old.name))
            asset.set('file','../assets/'+name)
        if compiler is not None:compiler.attrib.pop('meshdir',None)
        data=ET.tostring(root,encoding='utf-8');name=hashlib.sha256(data).hexdigest()+'.xml'
        target=destination/'models'/name
        if not target.exists():target.write_bytes(data)
        result='models/'+name;model_cache[str(path)]=result;return result

    for row in rows:
        cid=row['id'];raw=source/'cases'/cid;baseline=Path(row['source_summary']).parent
        meta=json.loads((baseline/'native_candidate.json').read_text())
        a=np.load(baseline/'native_candidate.npz')['grasp_qpos']
        b=np.load(raw/'native_candidate.npz')['grasp_qpos']
        use_mppi=bool(row['task_success']);q=b if use_mppi else a
        evaluation_dir=raw/'evaluation' if use_mppi else baseline/'graspadmm_evaluation'
        evaluation=json.loads((evaluation_dir/'evaluation.json').read_text())
        expected=bool(row['validated_fallback_success'])
        if bool(evaluation['success'])!=expected:raise ValueError('Selected policy disagrees with recorded native evaluation')
        tested=np.load(evaluation_dir/'candidate.npz')['grasp_qpos']
        if not np.array_equal(q,tested):raise ValueError('Selected pose was not actually evaluated')
        mesh_source=Path(meta['mesh_path']);meshname=sha(mesh_source)+'.obj'
        if not (destination/'assets'/meshname).exists():shutil.copy2(mesh_source,destination/'assets'/meshname)
        posefile='poses/'+cid+'.npz';np.savez_compressed(destination/posefile,qpos=local_pose(q)[None],time=[0.],grasp_qpos=q)
        static_model=model(raw/'scene.xml')
        native_model=native_trajectory=None
        if (evaluation_dir/'evaluation_trajectory.npz').is_file():
            native_model=model(evaluation_dir/'evaluation_scene.xml')
            native_trajectory='trajectories/'+cid+'.npz'
            shutil.copy2(evaluation_dir/'evaluation_trajectory.npz',destination/native_trajectory)
            evaluation.update(scene=native_model,trajectory=native_trajectory)
        summary=dict(task='grasp',method='graspadmm_mppi',policy='native_validation_fallback',
            case_id=cid,template=meta['template'],seed=row['seed'],primitive=row['primitive'],
            task_success=expected,baseline_success=bool(row['baseline_success']),
            raw_mppi_success=bool(row['task_success']),uses_refined_pose=use_mppi,transition=row['transition'],
            failure_stage=evaluation.get('failure_stage'),evaluation_protocol=evaluation['protocol'],
            required_cbody=meta['required_cbody'],mesh='assets/'+meshname,
            model=static_model,trajectory=posefile,evaluation_model=native_model,evaluation_trajectory=native_trajectory,
            source_evaluation_sha256=sha(evaluation_dir/'evaluation.json'),graspadmm_evaluation=evaluation)
        write_json(destination/'cases'/f'{cid}.json',summary)
        entry=dict(task='grasp',method='graspadmm_mppi',hand='wuji2',name=cid+'__validated_fallback',
            case_id=cid,template=meta['template'],primitive=row['primitive'],seed=row['seed'],
            size_scale=row['primitive']['size_scale'],size_axis=row['primitive']['size_axis'],
            model=static_model,trajectory=posefile,summary='cases/'+cid+'.json',dt=1.,phase='selected_pose',
            stage='结果',policy='native_validation_fallback',task_success=expected)
        manifest.append(entry)
        if native_trajectory:
            manifest.append(dict(entry,name=cid+'__native_test',model=native_model,trajectory=native_trajectory,
                phase='native_test',stage='GraspADMM评测',dt=.004))
        index.append({k:summary[k] for k in ('case_id','template','seed','primitive','task_success','baseline_success',
            'raw_mppi_success','uses_refined_pose','required_cbody','mesh','source_evaluation_sha256')})
        admm.append(a);mppi.append(b);selected.append(q)
    np.savez_compressed(destination/'candidates.npz',case_ids=np.array([r['id'] for r in rows]),
        admm_qpos=np.array(admm),mppi_qpos=np.array(mppi),selected_qpos=np.array(selected),
        admm_success=np.array([r['baseline_success'] for r in rows]),
        mppi_success=np.array([r['task_success'] for r in rows]),success=np.array([r['validated_fallback_success'] for r in rows]))
    write_json(destination/'manifest.json',dict(version=1,task='grasp',method='graspadmm_mppi',
        policy='native_validation_fallback',cases=manifest))
    write_json(destination/'index.json',dict(protocol=report['protocol'],trials=len(rows),
        successes=sum(r['task_success'] for r in index),raw_mppi_successes=report['refined_successes'],
        admm_successes=report['baseline_successes'],cases=index))
    write_json(destination/'assets/provenance.json',asset_origins)
    shutil.copy2(ROOT/'data/grasp/assets/wuji2/LICENSE',destination/'assets/WUJI-LICENSE')
    shutil.copy2(ROOT/'licenses/Dexonomy-NOTICE.txt',destination/'assets/Dexonomy-NOTICE.txt')
    for name in ('report.json','plan.json','summary.json','audit.json','source_audit.json','timings.json',
        'deterministic_replay_audit.json','validation.json','config.yaml','environment.json','by_type.csv',
        'by_size.csv','trials.csv','comparison.md','type_comparison.png','size_comparison.png'):
        shutil.copy2(source/name,destination/'evidence'/name)
    shutil.copytree(source/'code',destination/'evidence/measured_code',ignore=shutil.ignore_patterns('__pycache__'))
    (destination/'README.md').write_text('''# Highest verified grasp success policy\n\n647/1610 (40.19%): use the raw local MPPI output when it passes the original native\nGraspADMM protocol; otherwise retain the original ADMM output. This is a separate\nvalidation fallback policy. Raw MPPI is 573/1610 (35.59%); ADMM is 524/1610 (32.55%).\nAll 1610 cases, including 963 failures, are retained. No best trajectory frame is selected.\n\n`candidates.npz` preserves both source candidates and the chosen 27-coordinate pose.\n`index.json` contains the exact required-body groups and actual object mesh for each trial.\n`manifest.json`, `models/`, `poses/`, `trajectories/` and `assets/` are self-contained replay\ninputs with relative paths. Physical traces are copied byte for byte. XML changes only\nmake mesh paths relative; all physical parameters are retained. Static previews use\nthe official WUJI2 geometry and show the chosen pose, without implying a physical test.\n\nOriginal measured reports, configuration, hashes and code are retained under `evidence/`.\nAbsolute paths there are historical provenance, not replay dependencies. Success rates\nfor the original raw MPPI and the validated fallback policy remain separate.\n\n```bash\n.venv-contact/bin/python scripts/publish_grasp_results.py --verify --activate\n.venv-contact/bin/python run.py --config configs/web_grasp_best.yaml\n# Reevaluate selected candidates in the original pinned native environment:\n.venv-contact/bin/python scripts/publish_grasp_results.py --reevaluate --ids t04_cylinder_s01_seed00 t28_sphere_s00_seed00\n```\n\nFull usage and measured timings: `docs/graspadmm_mppi.md`.\n''')
    write_json(destination/'files_sha256.json',{str(p.relative_to(destination)):sha(p)
        for p in sorted(destination.rglob('*')) if p.is_file()})
    return dict(trials=len(rows),successes=sum(x['task_success'] for x in index),path=str(destination))


def verify(bundle):
    import mujoco
    bundle=bundle.resolve();errors=[]
    hashes=json.loads((bundle/'files_sha256.json').read_text())
    for name,digest in hashes.items():
        if sha(bundle/name)!=digest:errors.append(name)
    manifest=json.loads((bundle/'manifest.json').read_text());models={}
    for case in manifest['cases']:
        if case['model'] not in models:
            models[case['model']]=mujoco.MjModel.from_xml_path(str(bundle/case['model']))
        q=np.load(bundle/case['trajectory'])['qpos'];model=models[case['model']]
        if q.shape[1]!=model.nq or not np.isfinite(q).all():errors.append(case['name'])
        for mesh in ET.parse(bundle/case['model']).getroot().findall('./asset/mesh'):
            if Path(mesh.get('file')).is_absolute():errors.append(case['model'])
    data=np.load(bundle/'candidates.npz');index=json.loads((bundle/'index.json').read_text())
    if not np.array_equal(data['success'],data['admm_success']|data['mppi_success']):errors.append('fallback rule')
    if sum(data['success'])!=647 or len(data['success'])!=1610:errors.append('complete denominator')
    result=dict(passed=not errors,trials=len(index['cases']),successes=int(sum(data['success'])),
        verified_files=len(hashes),replay_entries=len(manifest['cases']),models=len(models),errors=errors)
    write_json(bundle/'portable_validation.json',result)
    if errors:raise RuntimeError(str(result))
    return result


def reevaluate(bundle,ids):
    from retarget.env.grasp.grasp_evaluation import NativeEvaluationWorker,evaluate_native_grasp
    bundle=bundle.resolve();index=json.loads((bundle/'index.json').read_text());data=np.load(bundle/'candidates.npz')
    output=ROOT/'output/published_grasp_recheck'
    records=[]
    with NativeEvaluationWorker(output/'worker.log') as worker:
        for i,c in enumerate(index['cases']):
            if ids and c['case_id'] not in ids:continue
            report=evaluate_native_grasp(data['selected_qpos'][i],bundle/c['mesh'],output/c['case_id'],
                required_cbody=c['required_cbody'],method='graspadmm_mppi',worker=worker)
            records.append(dict(case_id=c['case_id'],expected=c['task_success'],actual=bool(report['success']),
                match=c['task_success']==bool(report['success'])))
    result=dict(passed=all(c['match'] for c in records),cases=records)
    write_json(bundle/'native_recheck.json',result)
    if not result['passed']:raise RuntimeError(str(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'output/graspadmm_mppi_wuji2_results')
    parser.add_argument('--bundle',type=Path,default=ROOT/'results/graspadmm_mppi_best')
    parser.add_argument('--export',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--activate',action='store_true');parser.add_argument('--reevaluate',action='store_true')
    parser.add_argument('--ids',nargs='+');args=parser.parse_args()
    if args.export:print(json.dumps(export(args.source,args.bundle)),flush=True)
    if args.verify:print(json.dumps(verify(args.bundle)),flush=True)
    if args.activate:
        target=ROOT/'output/graspadmm_mppi_best';target.parent.mkdir(exist_ok=True)
        if target.is_symlink() and target.resolve()==args.bundle.resolve():pass
        elif target.exists() or target.is_symlink():raise ValueError('Output alias already exists and points elsewhere')
        else:target.symlink_to(args.bundle.resolve(),target_is_directory=True)
        print('Replay alias: '+str(target),flush=True)
    if args.reevaluate:print(json.dumps(reevaluate(args.bundle,args.ids)),flush=True)
    if not any((args.export,args.verify,args.activate,args.reevaluate)):parser.error('Use --export, --verify, --activate or --reevaluate')


if __name__=='__main__':main()

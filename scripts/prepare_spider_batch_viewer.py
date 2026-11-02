"""Expose real MPPI trajectories and clearly labeled failed CA poses to Viser."""

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json
from retarget.env.grasp.inputs import build_scene
from retarget.env.grasp.primitives import CATALOG


def object_only_case(row,work,assets,reason):
    """Keep invalid inputs visible without inventing a hand pose."""
    p=row['primitive'];tree=ET.Element('mujoco');world=ET.SubElement(tree,'worldbody')
    body=ET.SubElement(world,'body',name='grasp_object');mesh=work/(row['id']+'.obj')
    if mesh.is_file():
        asset=ET.SubElement(tree,'asset');ET.SubElement(asset,'mesh',name='object_mesh',file=str(mesh))
        ET.SubElement(body,'geom',type='mesh',mesh='object_mesh',rgba='.9 .55 .2 1')
    else:
        kind=p['kind'];r=p['radius_m'];h=p['height_m']
        typ='sphere' if kind=='sphere' else 'cylinder' if kind in ('disk','cylinder') else 'box'
        size=str(r) if typ=='sphere' else f'{r} {h/2}' if typ=='cylinder' else f'{r} {0.7*r} {h/2}'
        ET.SubElement(body,'geom',type=typ,size=size,rgba='.9 .55 .2 1')
    model=assets/'object_only.xml';ET.ElementTree(tree).write(model,encoding='utf-8')
    trajectory=assets/'unavailable_pose.npz';np.savez_compressed(trajectory,qpos=np.empty((1,0)),time=np.zeros(1))
    summary=assets/'summary.json';write_json(summary,dict(state=row['state'],reason=reason,task_success=False))
    return dict(task='grasp',method='mppi',hand='wuji2',template=row['template'],case_id=row['id'],
        name=f"{row['id']} | 无有限手姿（失败）",stage='失败信息',phase='error',radius_scale=p.get('size_scale', p['radial_scale']),
        primitive=p,has_hand_pose=False,model=str(model.resolve()),trajectory=str(trajectory.resolve()),
        summary=str(summary.resolve()),dt=.002)


def acquisition_preview(acquisition,model,assets,seed):
    """Native geometric contacts at each recorded optimization pose, not time."""
    source=acquisition/'acquisition_trajectory.npz';target=assets/'acquisition_trajectory.npz'
    if target.is_file() and target.stat().st_mtime_ns>=source.stat().st_mtime_ns:return target
    from retarget.env.grasp.spider_geometry import ContactRequirements
    with np.load(source,allow_pickle=False) as raw:values={key:raw[key].copy() for key in raw.files}
    with np.load(seed/'spider_seed.npz',allow_pickle=False) as raw:requirements=ContactRequirements(mujoco.MjModel.from_xml_path(str(model)),raw)
    compiled=requirements.model;data=mujoco.MjData(compiled);rows=[]
    for q in values['qpos']:
        data.qpos[:]=q;data.qvel[:]=0.;mujoco.mj_forward(compiled,data)
        rows.append(requirements.observe(data))
    counts=np.asarray([len(row['points']) for row in rows],int);capacity=max(1,int(counts.max()))
    points=np.zeros((len(rows),capacity,3));groups=np.full((len(rows),capacity),-1,int)
    for i,row in enumerate(rows):points[i,:counts[i]]=row['points'];groups[i,:counts[i]]=row['groups']
    np.savez_compressed(target,**values,contact_points=points,contact_counts=counts,contact_groups=groups)
    return target


def prepare(batch):
    batch = Path(batch).resolve()
    source = json.loads((batch/"summary.json").read_text())
    cases, unavailable, excluded, pending = [], [], [], []
    for row in source["jobs"]:
        if row["state"] not in ("completed", "ca_failed", "input_invalid", "solve_failed", "timeout"):
            pending.append(row["id"])
            continue
        p = row["primitive"]
        primary = CATALOG[row["type_id"]][0][0]
        if p["kind"] != primary:
            excluded.append(dict(id=row["id"], template=row["template"],
                shape=p["kind"], primary_shape=primary, reason="one primary primitive per grasp type"))
            continue
        work = Path(row["work"])
        stage = "MPPI轨迹" if row["state"] == "completed" else "CA失败预览"
        has_hand_pose = True
        assets = batch/"viewer_assets"/row["id"]
        assets.mkdir(parents=True, exist_ok=True)
        if row["state"] == "completed":
            output = work/"solve"
            manifest = json.loads((output/"manifest.json").read_text())["cases"][0]
            model = output/manifest["model"]
            trajectory = output/manifest["trajectory"]
            summary = output/manifest["summary"]
            method = "mppi"
            # The solver manifest points to the per-case directory; batch
            # runs contain one case, whether acquisition succeeded or failed.
            acquisition=summary.parent/'acquisition'
            report=acquisition/'acquisition.json'
            if report.is_file():
                details=json.loads(report.read_text())
                seed=Path(details['source'])
                preview=acquisition_preview(acquisition,model,assets,seed)
                stage='静止保持' if details['success'] else '接触建立'
                phase='hold' if details['success'] else 'acquisition'
                extra=dict(phase=phase,seed_source=str(seed),acquisition_report=str(report),
                    time_kind='seconds' if details['success'] else 'optimization_iteration',
                    initial_index=0 if details['success'] else -1)
                if details['success']:
                    cases.append(dict(task='grasp',method='mppi',hand='wuji2',template=row['template'],
                        case_id=row['id'],name=f"{row['id']} | 接触建立",stage='接触建立',phase='acquisition',
                        radius_scale=p.get('size_scale', p['radial_scale']),primitive=p,has_hand_pose=True,
                        model=str(model.resolve()),trajectory=str(preview.resolve()),
                        summary=str(report.resolve()),dt=1.,time_kind='optimization_iteration',initial_index=-1,
                        seed_source=str(seed),acquisition_report=str(report)))
                else:trajectory=preview
            else:extra={}
        else:
            extra={}
            refined = work/"ca_final/refinement_candidate.npz"
            initial = sorted((work/"ca_seed").glob("**/contactaware_reference.npz"))
            if (work/'spider_seed/spider_seed.npz').is_file():
                # This is a diagnostic preview, never solver input. Damaged
                # semantic masks must not prevent other cases from displaying.
                seed=work/'spider_seed'
                metadata=json.loads((seed/'spider_seed.json').read_text())
                model=seed/metadata['scene']
                with np.load(seed/'spider_seed.npz',allow_pickle=False) as raw:qpos=raw['qpos'].copy()
                if qpos.shape!=(33,) or not np.isfinite(qpos).all():
                    qpos=None
                if qpos is None:
                    # Do not substitute a fabricated hand configuration.
                    missing=dict(id=row['id'],state=row['state'],reason='非有限输入；无可展示的有限手姿')
                    unavailable.append(missing)
                    cases.append(object_only_case(row,work,assets,missing['reason']))
                    continue
                trajectory=assets/'initial_proposal.npz';np.savez_compressed(trajectory,qpos=qpos[None],time=np.zeros(1))
                failure=next((p for p in [work/'solve_failure.json',work/'prepare_failure.json',work/'failure.json'] if p.is_file()),None)
                summary=assets/'summary.json';write_json(summary,dict(state=row['state'],reason=row.get('reason'),
                    error=json.loads(failure.read_text()) if failure else None,ca_is_gate=False))
                cases.append(dict(task='grasp',method='mppi',hand='wuji2',template=row['template'],case_id=row['id'],
                    name=f"{row['id']} | 求解异常（保留初值）",stage='失败信息',phase='error',radius_scale=p.get('size_scale', p['radial_scale']),
                    primitive=p,has_hand_pose=True,model=str(model.resolve()),trajectory=str(trajectory.resolve()),
                    summary=str(summary.resolve()),dt=.002))
                continue
            if refined.is_file():
                model = work/"ca_final/sequence/scene/wuji_right/primitive_sdf.xml"
                with np.load(refined, allow_pickle=False) as raw:
                    hand = raw["hand_qpos"].copy()
                compiled = mujoco.MjModel.from_xml_path(str(model))
                qpos = compiled.qpos0.copy()
                qpos[:26] = hand
                diagnostics = work/"ca_final/contactaware_reference.json"
            elif initial:
                with np.load(initial[0], allow_pickle=False) as raw:
                    qpos = raw["proposed_hand_qpos"][0].copy()
                mesh = work/(row["id"]+".obj")
                _, _, xml, _ = build_scene(mesh, hand_model="wuji2")
                model = assets/"model.xml"
                model.write_text(xml)
                diagnostics = initial[0].parent/"contactaware.json"
            else:
                missing = dict(id=row["id"], state=row["state"], reason=row.get("reason"),
                    explanation="未保存手部姿态；仅展示该尺寸物体和失败原因")
                unavailable.append(missing)
                has_hand_pose = False
                stage = "失败信息"
                tree = ET.Element("mujoco")
                asset = ET.SubElement(tree, "asset")
                ET.SubElement(asset, "mesh", name="object_mesh", file=str(work/(row["id"]+".obj")))
                body = ET.SubElement(ET.SubElement(tree, "worldbody"), "body", name="grasp_object")
                ET.SubElement(body, "geom", type="mesh", mesh="object_mesh", rgba=".9 .55 .2 1")
                model = assets/"object_only.xml"
                ET.ElementTree(tree).write(model, encoding="utf-8")
                trajectory = assets/"unavailable_pose.npz"
                np.savez_compressed(trajectory, qpos=np.empty((1, 0)), time=np.zeros(1))
                summary = assets/"summary.json"
                diagnostics = work/"prepare_failure.json"
                write_json(summary, dict(**missing, accepted=False, task_success=False,
                    primitive=p, diagnostics=json.loads(diagnostics.read_text())))
                cases.append(dict(task="grasp", method="contactaware", hand="wuji2", template=row["template"],
                    case_id=row["id"], name=f"{row['id']} | r={p['radius_m']*1000:.2f} mm | 无保存手姿（失败）",
                    stage=stage, radius_scale=p.get("size_scale", p["radial_scale"]), primitive=p, has_hand_pose=False,
                    model=str(model.resolve()), trajectory=str(trajectory.resolve()), summary=str(summary.resolve()), dt=.002))
                continue
            if qpos.ndim != 1 or not np.isfinite(qpos).all():
                raise ValueError(f"Invalid saved CA candidate: {row['id']}")
            trajectory = assets/"candidate.npz"
            np.savez_compressed(trajectory, qpos=qpos[None], time=np.zeros(1))
            summary = assets/"summary.json"
            write_json(summary, dict(case=row["id"], template=row["template"], primitive=p,
                stage="failed ContactAware candidate only; no MPPI trajectory", accepted=False,
                task_success=False, reason=row.get("reason"), diagnostics=str(diagnostics),
                contactaware=json.loads(diagnostics.read_text())))
            method = "contactaware"
        outcome = "MPPI未通过" if row["state"] == "completed" and not row.get("task_success", False) else stage
        name = (f"{row['id']} | r={p['radius_m']*1000:.2f} mm, "
                f"h={p['height_m']*1000:.2f} mm | {outcome}")
        cases.append(dict(task="grasp", method=method, hand="wuji2", template=row["template"],
            case_id=row["id"], name=name, stage=stage, radius_scale=p.get("size_scale", p["radial_scale"]),
            primitive=p, has_hand_pose=has_hand_pose, model=str(model.resolve()), trajectory=str(trajectory.resolve()),
            summary=str(summary.resolve()), dt=.002,**extra))
    write_json(batch/"manifest.json", dict(version=1, task="grasp", method="mppi", cases=cases))
    result = dict(available=len(cases), mppi=sum(c["method"] == "mppi" for c in cases),
        unique_cases=len({c['case_id'] for c in cases}),available_phases=len(cases),
        ca_previews=sum(c["stage"] == "CA失败预览" for c in cases),
        object_only_failures=sum(not c["has_hand_pose"] for c in cases), unavailable=unavailable,
        primary_shape_only=True, source_cases=len(source["jobs"]),
        pending=pending,
        excluded_secondary=excluded,
        policy="One primary primitive/type; failed primary cases included; historical secondary results retained on disk")
    write_json(batch/"viewer_availability.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--watch", action="store_true", help="Refresh the manifest as background jobs finish")
    args = parser.parse_args()
    if not args.watch:
        print(json.dumps(prepare(args.batch), ensure_ascii=False))
    else:
        previous = None
        while True:
            summary = json.loads((args.batch/"summary.json").read_text())
            signature = tuple((r["id"],r["state"]) for r in summary["jobs"]
                              if r["state"] in ("completed","ca_failed","input_invalid","solve_failed","timeout"))
            if signature != previous:
                result = prepare(args.batch)
                print(json.dumps({k:result[k] for k in ("available","mppi","ca_previews","object_only_failures")}), flush=True)
                previous = signature
            if summary["state"] in ("completed","completed_with_failures"):
                from scripts.audit_grasp_size_consistency import audit
                report = audit(args.batch)
                print(json.dumps(dict(size_audit_cases=len(report["rows"]),flagged=len(report["flagged_cases"]))),flush=True)
                break
            time.sleep(5)

#!/usr/bin/env python3
"""Export a separate stage-by-stage, seed-zero gesture review (no promotion)."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json
from retarget.env.grasp.graspadmm_scene import from_upstream_qpos,canonical_body
from retarget.env.grasp.graspadmm_gesture import reference_diagnostics,digest
from retarget.env.grasp.inputs import build_scene
from retarget.task.grasp.dexonomy import load_template,anatomical_joints


def read(path):return json.loads(Path(path).read_text())


def physical_summary(path):
    if not path.is_file():return dict(state='not_run',six_direction_tested=False)
    r=read(path)
    return dict(protocol=r.get('protocol'),state=r.get('state'),success=r.get('success'),
        failure_stage=r.get('failure_stage'),six_direction_tested=r.get('six_direction_tested'),
        directions=[dict(direction=d['direction'],tested=d['tested'],success=d['success'],
            max_translation_m=max((c['translation_m'] for c in d.get('checks',[])),default=None),
            max_rotation_deg=max((c['rotation_deg'] for c in d.get('checks',[])),default=None)) for d in r.get('directions',[])])


def diagnostic(model,q,required,reference=None):
    data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_forward(model,data)
    object_id=model.body('grasp_object').id
    bodies=set();object_depth=0.;self_depth=0.;count=0
    for contact in data.contact:
        a,b=model.geom_bodyid[contact.geom]
        if object_id in (a,b):
            object_depth=max(object_depth,-float(contact.dist))
            if contact.dist<=.002:
                body=int(b if a==object_id else a)
                bodies.add(canonical_body(model.body(body).name));count+=1
        else:self_depth=max(self_depth,-float(contact.dist))
    result=dict(object_penetration_m=max(0.,object_depth),self_penetration_m=max(0.,self_depth),
        actual_contact_count=count,actual_contact_bodies=sorted(bodies),
        missing_required_groups=[g for g in required if not set(g)&bodies],
        semantic_status='pending_user_review',contact_threshold_m=.002)
    if reference is not None:result.update(reference_diagnostics(q,reference))
    return result


def render(model,q,path,*,shadow=False,hide_object=False,markers=None):
    from PIL import Image
    data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_forward(model,data)
    original=model.geom_rgba.copy()
    material_ids=model.geom_matid.copy()
    model.geom_matid[:]=-1
    palette=[(.85,.36,.22),(.25,.52,.85),(.25,.72,.75),(.34,.68,.36),(.65,.42,.78)]
    for geom in range(model.ngeom):
        name=canonical_body(model.body(int(model.geom_bodyid[geom])).name)
        color=(.64,.69,.75)
        for finger,prefix in enumerate(('th','ff','mf','rf','lf'),1):
            if name.startswith(f'right_finger{finger}_') or name.startswith('rh_'+prefix):color=palette[finger-1]
        if name=='grasp_object':color=(.95,.67,.23)
        model.geom_rgba[geom]=[*color,1.]
    model.vis.headlight.ambient[:]=.5
    model.vis.headlight.diffuse[:]=.8
    if hide_object:
        for geom in range(model.ngeom):
            if model.body(int(model.geom_bodyid[geom])).name=='grasp_object':model.geom_rgba[geom,3]=0.
    if shadow:
        joints=anatomical_joints(model,data)
        center=joints.mean(0)
        direction=np.array([.8,-2.,.35])
    else:
        rotation=Rotation.from_euler('XYZ',q[3:6])
        center=q[:3]+rotation.apply([0.,.015,-.075])
        direction=rotation.apply([.8,2.,-.35])
    renderer=mujoco.Renderer(model,height=400,width=400)
    views=[]
    for view in range(2):
        vector=direction if view==0 else (np.array([-2.,-.7,.3]) if shadow else rotation.apply([-2.,.7,-.3]))
        camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=.31
        camera.azimuth=np.degrees(np.arctan2(vector[1],vector[0]))
        camera.elevation=np.degrees(np.arctan2(vector[2],np.linalg.norm(vector[:2])))
        options=mujoco.MjvOption();options.geomgroup[1]=False
        renderer.update_scene(data,camera=camera,scene_option=options)
        direction_unit=vector/np.linalg.norm(vector)
        up=np.array([0.,0.,1.]) if shadow else rotation.apply([0.,0.,-1.])
        up=up-direction_unit*np.dot(up,direction_unit);up/=np.linalg.norm(up)
        # MjvCamera's world-up convention otherwise introduces an apparent
        # wrist roll between stages. Stabilize both GL eyes in the palm frame.
        for eye in renderer.scene.camera:
            eye.pos[:]=center+.31*direction_unit;eye.forward[:]=-direction_unit;eye.up[:]=up
        points=[] if markers is None else list(np.asarray(markers)[:,:3])
        if not shadow and not hide_object:
            object_id=model.body('grasp_object').id
            points.extend(c.pos.copy() for c in data.contact if c.dist<=.002 and object_id in model.geom_bodyid[c.geom])
        for point in points:
            if renderer.scene.ngeom>=renderer.scene.maxgeom:break
            geom=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0018]*3),np.asarray(point),np.eye(3).flatten(),np.array([1.,.15,.1,1.]))
            renderer.scene.ngeom+=1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        views.append(renderer.render().copy())
    renderer.close();model.geom_rgba[:]=original;model.geom_matid[:]=material_ids
    Image.fromarray(np.concatenate(views,axis=0)).save(path)


def missing(path,label):
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(400,800),'#e8ecf2');draw=ImageDraw.Draw(image)
    draw.multiline_text((35,370),label,fill='#8b2635',spacing=8)
    image.save(path)


def export(source,refined,output):
    from PIL import Image,ImageDraw
    plan=read(source/'plan.json')
    output.mkdir(parents=True,exist_ok=False)
    refinement={r['id']:r for r in read(refined/'summary.json')['jobs']}
    rows=[];manifest=[];source_hashes={};orientation=[]
    annotation_cache={}
    for index,job in enumerate(plan['jobs']):
        status=read(source/'jobs'/job['id']/'status.json')
        work=Path(status.get('work',source/'jobs'/job['id']))/'solve'
        summary_path=next(work.glob('000*/summary.json'))
        case=summary_path.parent;summary=read(summary_path)
        dest=output/'cases'/job['id'];dest.mkdir(parents=True)
        source_hashes[str(summary_path)]=digest(summary_path)
        for pattern in ('*.npy','native_candidate.npz','initialization.json','gesture_initialization.json'):
            for path in case.glob(pattern):source_hashes[str(path)]=digest(path)
        xml_path=case/'input_object.obj'
        model,_,xml,_=build_scene(xml_path,hand_model='wuji2')
        (dest/'scene.xml').write_text(xml)
        import xml.etree.ElementTree as ET
        template_tree=ET.fromstring(xml)
        world=template_tree.find('worldbody')
        for body in list(world):
            if body.get('name')=='grasp_object':world.remove(body)
        (dest/'template_scene.xml').write_text(ET.tostring(template_tree,encoding='unicode'))
        template=np.load(case/'fixed_wuji2_template.npy',allow_pickle=True).item()
        reference=template.get('gesture_reference')
        required=template['required_cbody']
        stages=[]
        name=job['template']
        if name not in annotation_cache:
            sm,sd,source_contacts,annotation=load_template(ROOT/'data/grasp/templates',name)
            markers=np.array([np.r_[sd.xpos[sm.body(body).id]+sd.xmat[sm.body(body).id].reshape(3,3)@contact[:3],
                                    sd.xmat[sm.body(body).id].reshape(3,3)@contact[3:]] for body,contact in source_contacts])
            # load_template intentionally strips visual assets. Load the full
            # official XML for rendering when available, with the same joints.
            try:
                visual=mujoco.MjModel.from_xml_path(str(ROOT/'data/grasp/templates/assets/hand/shadow/right.xml'))
                vd=mujoco.MjData(visual)
                for joint in range(sm.njnt):
                    dest_joint=visual.joint(sm.joint(joint).name).id
                    vd.qpos[visual.jnt_qposadr[dest_joint]]=sd.qpos[sm.jnt_qposadr[joint]]
                sm,sd=visual,vd
            except ValueError:
                # Skeleton geometry is generated from the original annotated
                # bones; it must never be presented as an empty render.
                import xml.etree.ElementTree as ET
                tree=ET.parse(ROOT/'data/grasp/templates/assets/hand/shadow/right.xml')
                for parent in tree.iter():
                    for child in list(parent):
                        if child.tag in ('asset','geom','site','actuator','tendon','contact','sensor','keyframe'):parent.remove(child)
                for body in tree.iter('body'):
                    child=next((c for c in body if c.tag=='body' and np.linalg.norm(np.fromstring(c.get('pos','0 0 0'),sep=' '))>1.e-4),None)
                    endpoint=child.get('pos') if child is not None else ('0 0 .027' if body.get('name','').endswith('distal') else None)
                    if endpoint is not None:ET.SubElement(body,'geom',type='capsule',fromto='0 0 0 '+endpoint,size='.004',rgba='.7 .75 .82 1')
                sm=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
                vd=mujoco.MjData(sm);vd.qpos[:]=sd.qpos;sd=vd
            target=output/f'annotation_{job["type_id"]}.png'
            render(sm,sd.qpos,target,shadow=True,markers=markers)
            annotation_cache[name]=target
            source_hashes[str(annotation)]=digest(annotation)
        annotation_path=annotation_cache[name]
        stages.append(dict(key='annotation',label='Original annotation',image=str(annotation_path.relative_to(output))))
        bone=np.load(case/'bone_only_template_qpos.npy')
        stages.append(dict(key='bone_only',label='Bone-only Wuji template',qpos=bone))
        fitted=from_upstream_qpos(template['grasp_qpos'][0])
        stages.append(dict(key='template',label='Whole-hand Wuji template',qpos=fitted))
        initial_path=case/'initial_grasp.npy'
        initial=np.load(initial_path,allow_pickle=True).item() if initial_path.is_file() else None
        if initial is not None:
            stages.append(dict(key='initial',label='Accepted initialization',qpos=from_upstream_qpos(initial['grasp_qpos'][0])))
            reference=initial.get('gesture_reference',reference)
        else:
            candidates=np.load(case/'gesture_candidates.npy',allow_pickle=True) if (case/'gesture_candidates.npy').is_file() else []
            info=read(case/'gesture_initialization.json') if (case/'gesture_initialization.json').is_file() else {}
            if len(candidates):
                best=min(range(len(candidates)),key=lambda i:info['candidates'][i]['fit']['final_cost'])
                stages.append(dict(key='rejected',label='REJECTED initialization candidate',qpos=from_upstream_qpos(candidates[best]['grasp_qpos'][0])))
            else:stages.append(dict(key='initial',label='Initialization unavailable'))
        final=case/'native_candidate.npz'
        q=None
        if final.is_file():
            with np.load(final) as values:q=from_upstream_qpos(values['grasp_qpos'])
        stages.append(dict(key='admm',label='ADMM output',**({} if q is None else dict(qpos=q))))
        mppi=refined/'cases'/job['id']/'grasp.npz'
        if mppi.is_file():
            with np.load(mppi) as values:q=values['qpos'].copy()
        else:q=None
        stages.append(dict(key='mppi',label='MPPI output',**({} if q is None else dict(qpos=q))))
        details=[]
        for stage in stages:
            if stage['key']=='annotation':details.append(stage);continue
            image=dest/f'{stage["key"]}.png'
            value=stage.get('qpos')
            if value is None:missing(image,'No output\nUpstream initialization failed')
            else:
                render(model,value,image,hide_object=stage['key'] in ('template','bone_only'))
                np.savez_compressed(dest/f'{stage["key"]}.npz',qpos=value[None],time=np.array([0.]))
                manifest.append(dict(task='grasp',hand='wuji2',method='graspadmm_mppi' if stage['key']=='mppi' else 'graspadmm',
                    review_only=True,name=job['id']+'_'+stage['key'],
                    case_id=job['id'],template=name,primitive=job['primitive'],size_scale=job['primitive']['size_scale'],
                    seed=0,phase=stage['key'],stage=stage['label'],time_kind='optimization_iteration',initial_index=-1,
                    model=str((dest/('template_scene.xml' if stage['key'] in ('template','bone_only') else 'scene.xml')).relative_to(output)),
                    trajectory=str((dest/f'{stage["key"]}.npz').relative_to(output)),summary=str((dest/'review.json').relative_to(output)),dt=1.))
            entry={k:v for k,v in stage.items() if k!='qpos'}
            entry['image']=str(image.relative_to(output))
            if value is not None:
                entry['qpos']=np.asarray(value).tolist()
                if stage['key'] not in ('template','bone_only'):entry['diagnostics']=diagnostic(model,value,required,reference)
            details.append(entry)
        combined=Image.new('RGB',(6*400,840),'white');draw=ImageDraw.Draw(combined)
        for column,entry in enumerate(details):
            combined.paste(Image.open(output/entry['image']),(column*400,40))
            draw.text((column*400+8,12),entry['label'],fill='black')
        combined.save(dest/'comparison.jpg',quality=90)
        mp=refinement[job['id']]
        record=dict(case_id=job['id'],type_id=job['type_id'],template=name,primitive=job['primitive'],seed=0,
            semantic_status='pending_user_review',initialization_valid=initial is not None,
            initialization=read(case/'gesture_initialization.json') if (case/'gesture_initialization.json').is_file() else None,
            native_admm_success=bool(summary.get('task_success')),native_mppi_success=bool(mp.get('task_success')),
            native_validated_fallback_success=bool(mp.get('validated_fallback_success')),
            native_admm_evaluation=physical_summary(case/'graspadmm_evaluation/evaluation.json'),
            native_mppi_evaluation=physical_summary(refined/'cases'/job['id']/'evaluation/evaluation.json'),
            failure_stage=mp.get('failure_stage'),source_summary=str(summary_path),refined_summary=str(refined/'cases'/job['id']/'summary.json'),
            stages=details,comparison=str((dest/'comparison.jpg').relative_to(output)),
            historical_initialization='not_saved_in_published_archive; not reconstructed from final poses')
        write_json(dest/'review.json',record);rows.append(record)
        if job['type_id']==9 and job['primitive']['size_scale']==1.5:
            seed=next((s['qpos'] for s in details if s['key'] in ('initial','rejected') and 'qpos' in s),None)
            if seed is not None:
                for degrees in (0,90,180,270):
                    value=np.asarray(seed).copy();root=Rotation.from_euler('XYZ',value[3:6])
                    palm_rotation=Rotation.from_euler('Y',degrees,degrees=True)
                    moved=(palm_rotation*root.inv()).inv()
                    value[:3]=(moved*root.inv()).apply(value[:3]);value[3:6]=moved.as_euler('XYZ')
                    path=output/f'type9_direction_{degrees}.png';render(model,value,path)
                    center=-root.inv().apply(np.asarray(seed)[:3])
                    transform=np.eye(4);transform[:3,:3]=palm_rotation.as_matrix();transform[:3,3]=center-palm_rotation.apply(center)
                    orientation.append(dict(object_rotation_palm_deg=degrees,axis_palm=[0,1,0],
                        source_case_id=job['id'],reference_qpos=seed,object_center_palm=center.tolist(),
                        object_rotation_about_center_palm=transform.tolist(),
                        image=path.name,qpos=value.tolist(),role='direction_review_only_not_an_accepted_grasp',
                        diagnostics=diagnostic(model,value,required),object_axis_palm=moved.inv().apply([0.,0.,1.]).tolist()))
        print(f'[{index+1}/{len(plan["jobs"])}] rendered {job["id"]}',flush=True)
    priority={'initial':0,'admm':1,'mppi':2,'template':3,'bone_only':4,'rejected':5}
    manifest.sort(key=lambda r:(r['case_id'],priority[r['phase']]))
    write_json(output/'manifest.json',dict(version=1,task='grasp',method='gesture_review',cases=manifest))
    audit=dict(planned=len(plan['jobs']),recorded=len(rows),seed_zero_only=all(r['seed']==0 for r in rows),
        type12_scales=[r['primitive']['size_scale'] for r in rows if r['type_id']==12],
        semantic_status='pending_user_review',formal_viewer_promoted=False,full_410_batch_started=False,
        approved_type12_reference=read(source/'gesture_references/type12.json'),
        source_hashes=source_hashes,
        source_unchanged=all(digest(p)==d for p,d in source_hashes.items()),
        original_filters_and_physical_protocol='unchanged_third_party_source',
        physical_results=dict(admm=sum(r['native_admm_success'] for r in rows),mppi=sum(r['native_mppi_success'] for r in rows),
                              validated_fallback=sum(r['native_validated_fallback_success'] for r in rows)),
        initialization_valid=sum(r['initialization_valid'] for r in rows))
    write_json(output/'audit.json',audit)
    write_json(output/'review.json',dict(audit=audit,cases=rows,type9_direction_candidates=orientation))
    encoded=json.dumps(dict(cases=rows,directions=orientation),ensure_ascii=False).replace('</','<\\/')
    document='''<!doctype html><meta charset="utf-8"><title>Grasp gesture review — seed 0</title>
<style>body{font:16px system-ui;background:#f3f6fa;color:#17212e;margin:24px}select{font:inherit;padding:8px}img{max-width:100%}.panel{background:white;padding:16px;border-radius:10px;margin:16px 0}.warning{color:#a63333}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd3dd}details{margin:12px 0}.directions{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}</style>
<h1>七类抓型逐阶段验收 · seed 0</h1><p>手势状态：待人工确认。物理通过单独记录。红色初始化候选已被拒绝，ADMM/MPPI 缺失不代表成功。</p>
<p>每列上、下各一个视角；原始标注与模板列不含物体。历史发布包未保存的初始化不以最终姿态冒充。</p>
<label>抓型 <select id="type"></select></label> <label>尺寸 <select id="size"></select></label>
<div id="case"></div><h2>Type 9 · 1.5× 参考方向候选</h2><p>仅用于方向确认，未固定朝向，也未计为合格抓取。</p><div class="directions" id="directions"></div>
<script>const data=DATA;const t=document.querySelector('#type'),s=document.querySelector('#size');
for(const n of [...new Set(data.cases.map(r=>r.type_id))])t.add(new Option(data.cases.find(r=>r.type_id===n).template,n));
function sizes(){s.replaceChildren();for(const r of data.cases.filter(r=>r.type_id===+t.value))s.add(new Option(r.primitive.size_scale+'×',r.case_id));show()}
function show(){const r=data.cases.find(r=>r.case_id===s.value);document.querySelector('#case').innerHTML=`<div class="panel"><h2>${r.case_id}</h2><p>初始化：${r.initialization_valid?'通过几何检查':'拒绝'} · 原生 ADMM：${r.native_admm_success?'通过':'未通过'} · 原生 MPPI：${r.native_mppi_success?'通过':'未通过'}</p><img src="${r.comparison}"><details><summary>分阶段接触、穿透与手势偏移</summary><pre></pre></details></div>`;document.querySelector('#case pre').textContent=JSON.stringify(r,null,2)}
t.onchange=sizes;s.onchange=show;sizes();for(const r of data.directions){const e=document.createElement('div');e.className='panel';const p=document.createElement('p');p.textContent='绕掌面 +Y '+r.object_rotation_palm_deg+'°';e.append(p);const i=document.createElement('img');i.src=r.image;e.append(i);document.querySelector('#directions').append(e)}
</script>'''.replace('DATA',encoded)
    (output/'index.html').write_text(document)
    lines=['# 七类抓型 seed 0 验收','',f"共 {len(rows)} 例；初始化几何检查通过 {audit['initialization_valid']} 例。手势均待人工确认。",'',
        '| 类型 | 例数 | 初始化合格 | ADMM 原生通过 | MPPI 原生通过 |', '|---|---:|---:|---:|---:|']
    for number in sorted({r['type_id'] for r in rows}):
        group=[r for r in rows if r['type_id']==number]
        lines.append(f"| {number} | {len(group)} | {sum(r['initialization_valid'] for r in group)} | {sum(r['native_admm_success'] for r in group)} | {sum(r['native_mppi_success'] for r in group)} |")
    lines.extend(['','独立审阅页：index.html。正式 viewer 未替换，410 例批跑未启动。'])
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--refined',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.source.resolve(),args.refined.resolve(),args.output.resolve())

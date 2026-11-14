"""Frozen standard-MANO regions, trajectory reference and train-only templates.

Virtual MCP regions never change the glove's 20-patch / 460-taxel ABI. Surface
anchors are face+barycentric coordinates, selected before candidate training.
"""
from __future__ import annotations
import copy,hashlib,json
import numpy as np
import torch
import trimesh
from .contact import FINGERS,PATCH_IDS
from .calibration_targets import target
from .contact_protocol import PROTOCOL,cluster_tree,region_finger
from scripts.calibrate_wuji_mano import CHAINS,TIPS

REVISION='standard_mano_contact_templates_v1'


def region_faces(model):
    topology=model.topology();faces=np.asarray(model.faces)
    sid_faces=np.asarray(topology['sid_face_ids']);patches=np.asarray(topology['sid_patch_indices'])
    regions={p:np.unique(sid_faces[patches==i]).tolist() for i,p in enumerate(PATCH_IDS)}
    rest=np.asarray(model.pose_reference_vertices);j=np.asarray(model.pose_reference_joints)
    center=rest[faces].mean(1);normal=np.cross(rest[faces[:,1]]-rest[faces[:,0]],rest[faces[:,2]]-rest[faces[:,0]])
    palm=np.asarray(model.anatomy.palmar_normal)
    palmar=normal@palm>0
    skin=model.model.backend.lbs_weights.detach().cpu().numpy()
    palm_skin=skin[faces,0].mean(1)>.25
    roots=j[[c[0] for c in CHAINS[1:]]]
    anchors=j[0]+.85*(roots-j[0])
    d=np.linalg.norm(center[:,None]-anchors[None],axis=-1)
    owner=d.argmin(1)
    for f,name in enumerate(FINGERS[1:]):
        ids=np.flatnonzero(palmar&palm_skin&(owner==f))
        if not len(ids):raise ValueError('标准 MANO 缺少 MCP 掌侧区域：'+name)
        ids=ids[np.argsort(d[ids,f])[:18]]
        regions['palm_mcp_'+name]=ids.tolist()
    return regions


def nearest_anchors(vertices,faces,regions,a,b):
    """Original triangles, vertex-to-triangle both ways; no collision proxy."""
    fa=np.asarray(regions[a],int);fb=np.asarray(regions[b],int)
    best=None
    for source,target_faces,reverse in ((fa,fb,False),(fb,fa,True)):
        vids=np.unique(faces[source]);points=vertices[vids]
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces[target_faces],process=False)
        closest,distance,which=trimesh.proximity.closest_point_naive(mesh,points)
        k=int(np.argmin(distance));v=int(vids[k]);f0=int(source[np.flatnonzero((faces[source]==v).any(1))[0]])
        bc0=np.eye(3)[int(np.flatnonzero(faces[f0]==v)[0])]
        f1=int(target_faces[which[k]])
        bc1=trimesh.triangles.points_to_barycentric(vertices[faces[[f1]]],closest[[k]])[0]
        anchors=[dict(face=f0,barycentric=bc0.tolist()),dict(face=f1,barycentric=bc1.tolist())]
        if reverse:anchors.reverse()
        if best is None or distance[k]<best[0]:best=(float(distance[k]),anchors)
    # Segment/triangle intersections can lie inside the other triangle, away
    # from all vertices and edge pairs. They are true zero surface distance.
    for source,target_faces,reverse in ((fa,fb,False),(fb,fa,True)):
        edge=np.asarray([(f,k,(k+1)%3) for f in source for k in range(3)])
        p=vertices[faces[edge[:,0],edge[:,1]]][:,None]
        direction=(vertices[faces[edge[:,0],edge[:,2]]]-p[:,0])[:,None]
        tri=vertices[faces[target_faces]][None];a=tri[:,:,0];e1=tri[:,:,1]-a;e2=tri[:,:,2]-a
        h=np.cross(direction,e2);det=(e1*h).sum(-1);safe=np.where(np.abs(det)>1e-15,det,1.)
        shift=p-a;u=(shift*h).sum(-1)/safe;q=np.cross(shift,e1)
        vv=(direction*q).sum(-1)/safe;t0=(e2*q).sum(-1)/safe
        hit=(np.abs(det)>1e-15)&(u>=0)&(vv>=0)&(u+vv<=1)&(t0>=0)&(t0<=1)
        if np.any(hit):
            i,j=np.argwhere(hit)[0];bc0=np.zeros(3);bc0[edge[i,1]]=1-t0[i,j];bc0[edge[i,2]]=t0[i,j]
            anchors=[dict(face=int(edge[i,0]),barycentric=bc0.tolist()),dict(face=int(target_faces[j]),barycentric=[float(1-u[i,j]-vv[i,j]),float(u[i,j]),float(vv[i,j])])]
            return 0.,anchors[::-1] if reverse else anchors
    # Include edge/edge witnesses: nearest points can lie in the interiors of
    # two edges, and vertex-only queries would overestimate the real skin gap.
    ea=np.asarray([(f,k,(k+1)%3) for f in fa for k in range(3)])
    eb=np.asarray([(f,k,(k+1)%3) for f in fb for k in range(3)])
    pa=vertices[faces[ea[:,0],ea[:,1]]];qa=vertices[faces[ea[:,0],ea[:,2]]]
    pb=vertices[faces[eb[:,0],eb[:,1]]];qb=vertices[faces[eb[:,0],eb[:,2]]]
    u=(qa-pa)[:,None];v=(qb-pb)[None];w=pa[:,None]-pb[None]
    aa=(u*u).sum(-1);bb=(u*v).sum(-1);cc=(v*v).sum(-1);dd=(u*w).sum(-1);ee=(v*w).sum(-1)
    den=aa*cc-bb*bb
    ss=np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-20)
    tt=np.divide(aa*ee-bb*dd,den,out=np.zeros_like(den),where=den>1e-20)
    candidates=[(ss,tt),(np.zeros_like(den),np.clip(ee/cc.clip(1e-20),0,1)),(np.ones_like(den),np.clip((ee+bb)/cc.clip(1e-20),0,1)),(np.clip(-dd/aa.clip(1e-20),0,1),np.zeros_like(den)),(np.clip((bb-dd)/aa.clip(1e-20),0,1),np.ones_like(den))]
    for ss,tt in candidates:
        valid=(ss>=0)&(ss<=1)&(tt>=0)&(tt<=1)
        distance=np.where(valid,np.linalg.norm(w+ss[...,None]*u-tt[...,None]*v,axis=-1),np.inf)
        aidx,bidx=np.unravel_index(distance.argmin(),distance.shape)
        gap=float(distance[aidx,bidx])
        if gap<best[0]:
            bc0=np.zeros(3);bc1=np.zeros(3)
            bc0[ea[aidx,1]]=1-ss[aidx,bidx];bc0[ea[aidx,2]]=ss[aidx,bidx]
            bc1[eb[bidx,1]]=1-tt[aidx,bidx];bc1[eb[bidx,2]]=tt[aidx,bidx]
            best=(gap,[dict(face=int(ea[aidx,0]),barycentric=bc0.tolist()),dict(face=int(eb[bidx,0]),barycentric=bc1.tolist())])
    return best


class FrozenContactReference:
    def __init__(self,model_path,side,mapping,initial,spec=None,device='cuda:0'):
        from .mano_runtime import ManoSurface
        from .official_mano import reference_from_frame
        self.model=ManoSurface(model_path,side=side,device=device);m=self.model
        m.enable_official_fk();m.build_binding(mapping)
        if spec and (spec.get('revision')!=REVISION or spec.get('weights_hash')!=m.weights_hash or spec.get('side')!=side):
            raise ValueError('冻结参考与当前 MANO 模型或手侧不匹配，请重新采集')
        if not spec:
            opened=next(r for r in initial['rows'] if r['stage']=='open' and r.get('skeleton'))
            m.install_flat_reference(reference_from_frame(opened,m.weights_hash,initial.get('serial'),'trajectory-reference'),allow_coarse=True)
        if spec:
            for k in ('raw_rotation','raw_translation','mount_pos','mount_rot'):setattr(m,k,np.asarray(spec[k]))
        self.regions=region_faces(m)
        self.spec=dict(revision=REVISION,standard_shape=True,weights_hash=m.weights_hash,side=side,
                       calibration_fingerprint=initial.get('calibration_fingerprint'),source_payload_sha256=initial.get('source_payload_sha256'),
                       skeleton_runtime_revision=(spec or {}).get('skeleton_runtime_revision',initial['rows'][0].get('skeleton_runtime_revision')),
                       **{k:getattr(m,k).tolist() for k in ('raw_rotation','raw_translation','mount_pos','mount_rot')})
        self.fingerprint=hashlib.sha256(json.dumps(self.spec,sort_keys=True).encode()).hexdigest()

    @torch.no_grad()
    def __call__(self,row):
        m=self.model;pose,root,*_=m.official_fk.parameters(m,row,CHAINS)
        v,_,_,_=m.decode(torch.as_tensor(pose[None],device=m.device,dtype=torch.float32),torch.as_tensor(root[None],device=m.device,dtype=torch.float32))
        # Region centroids are cheap, stable trajectory proxies; contact labels
        # do not depend on the subsequently optimized candidate or templates.
        vertices=v[0].cpu().numpy();spec=target(row['stage'])
        if spec['kind']=='noncontact_motion':
            tips=vertices[list(TIPS)]
            # Adjacent finger separation is a motion proxy, never a distal
            # contact equality. All five EMF observations remain supervised.
            spread=np.linalg.norm(tips[1:-1]-tips[2:],axis=-1)
            return dict(reference_gap_m=float(np.mean(spread)),reference_edges=[],contact_reference_id=self.fingerprint,
                        reference_alignment=self.spec['raw_rotation'],reference_mount_rotation=self.spec['mount_rot'])
        center={r:vertices[np.unique(m.faces[self.regions[r]])].mean(0) for r in spec['patches']}
        edges=spec['edges']
        if spec['kind']=='distal_cluster':
            regions=spec['patches'];d=np.linalg.norm(np.asarray([center[r] for r in regions])[:,None]-np.asarray([center[r] for r in regions])[None],axis=-1)
            edges=cluster_tree(regions,d)
        gaps=[np.linalg.norm(center[e['regions'][0]]-center[e['regions'][1]]) for e in edges if e['required']]
        return dict(reference_gap_m=float(np.sqrt(np.mean(np.square(gaps)))),reference_edges=copy.deepcopy(edges),
                    contact_reference_id=self.fingerprint,reference_alignment=self.spec['raw_rotation'],reference_mount_rotation=self.spec['mount_rot'])


def _standard_copy(model):
    m=copy.copy(model);m.model=copy.deepcopy(model.model);m.mapping=copy.deepcopy(model.mapping)
    m.official_fk=copy.deepcopy(model.official_fk);m.official_fk.version=None
    m.binding_overrides={};m.contact_registration=None;m.labels=None
    m.set_rest_vertex_residual();m.set_glove_geometry(np.zeros(43));m.set_shape(np.zeros(10));m.build_binding(m.mapping)
    return m


def reuse_templates(model,rows,artifact):
    """Reuse immutable train-only targets; never refit to a resumed geometry."""
    from .contact_protocol import STAGES
    if artifact.get('revision')!=REVISION or artifact.get('source')!='train_only' or artifact.get('validation_used') is not False:
        raise ValueError('续训接触模板版本或训练分区不兼容')
    actions={a['stage']:a['edges'] for a in artifact['actions']}
    expected={s['id'] for s in STAGES if target(s['id'])['patches']}
    if set(actions)!=expected:raise ValueError('续训接触模板缺少动作')
    for stage,edges in actions.items():
        if not edges:raise ValueError('续训接触模板缺少接触边')
        for edge in edges:
            if len(edge.get('regions',[]))!=2 or len(edge.get('anchors',[]))!=2:raise ValueError('续训模板接触边无效')
            for region,anchor in zip(edge['regions'],edge['anchors']):
                bc=np.asarray(anchor['barycentric'])
                if anchor['face'] not in artifact['regions'][region] or not 0<=anchor['face']<len(model.faces) or bc.shape!=(3,) or not np.isfinite(bc).all() or (bc<-1e-6).any() or abs(bc.sum()-1)>1e-5:
                    raise ValueError('续训模板曲面绑定无效')
    for row in rows:
        if row.get('capture_protocol')!=PROTOCOL or row.get('initialization'):continue
        row['contact_edges']=copy.deepcopy(actions.get(row['stage'],[])) if row.get('contact_confirmed') else []
    model.contact_template_artifact=copy.deepcopy(artifact)
    return model.contact_template_artifact


def prepare_templates(model,rows,cancel=None,steps=80):
    """Fit standard pose (not geometry), then freeze anchors using TRAIN only."""
    selected=[r for r in rows if r.get('capture_protocol')==PROTOCOL and r.get('partition')=='train' and r.get('contact_confirmed') and not r.get('initialization')]
    if not selected:raise ValueError('新协议缺少训练接触事件')
    m=_standard_copy(model);regions=region_faces(m);device=m.device
    from .card_math import exp_so3
    from .mano_gpu_objective import tip_emf_directions
    from .mano_contact_loss import vertex_normals,normal_residual
    from scipy.spatial.transform import Rotation
    seeds=[m.official_fk.parameters(m,r,CHAINS) for r in selected]
    t=lambda a:torch.as_tensor(np.asarray(a),device=device,dtype=torch.float32)
    seed=t([s[0] for s in seeds]);roots=t([s[1] for s in seeds]);delta=torch.nn.Parameter(torch.zeros_like(seed))
    # Use the very same physical pose chart as deployed delta inference. The
    # template cannot obtain zero gap by inventing an impossible joint rotation.
    from types import SimpleNamespace
    from .mano_joint_delta import PoseChart
    from .mano_objective import neutral_rotations,pose_angles
    from .thumb_chart import source_thumb_chart,SLOTS
    neutral=neutral_rotations(m);angles=t(pose_angles(m,seed.detach().cpu().numpy(),neutral))
    low=t(m.anatomy.lower);high=t(m.anatomy.upper);thumb=source_thumb_chart(m,selected)
    if thumb is not None:
        low=low.clone();high=high.clone();low[12:]=0;high[12:]=0;angles[:,12:]=0
        for k,(j,c) in enumerate(SLOTS):
            low[j,c]=float(thumb.bounds[k,0]);high[j,c]=float(thumb.bounds[k,1]);angles[:,j,c]=t(thumb.seeds[:,k])
    chart=PoseChart(m,selected,seed,objective=SimpleNamespace(basis=t(m.anatomy.basis),neutral=t(neutral),low=low,high=high,seed_angles=angles,thumb_chart=thumb))
    plans=[]
    with torch.no_grad():vv=m.decode(seed,roots)[0].cpu().numpy()
    for i,row in enumerate(selected):
        edges=copy.deepcopy(row.get('reference_edges') or target(row['stage'])['edges'])
        for edge in edges:
            gap,anchors=nearest_anchors(vv[i],m.faces,regions,*edge['regions'])
            plans.append(dict(frame=i,stage=row['stage'],regions=edge['regions'],required=edge.get('required',True),weight=edge.get('weight',1.),anchors=anchors,initial_gap_mm=gap*1000))
    frame=torch.tensor([e['frame'] for e in plans],device=device)
    face=torch.as_tensor(np.asarray([[m.faces[a['face']] for a in e['anchors']] for e in plans]),device=device)
    bc=t([[a['barycentric'] for a in e['anchors']] for e in plans])
    mask=t([[float(f in target(r['stage'])['active_fingers']) for f in FINGERS] for r in selected])
    jf=[next(f for f,c in enumerate(CHAINS) if j+1 in c) for j in range(15)]
    rawr=t([Rotation.from_quat([r.get('fingers',{}).get(f,{}).get('quaternion_xyzw',[0,0,0,1]) for f in FINGERS]).as_matrix() for r in selected])
    alignment=t([r.get('reference_alignment',m.raw_rotation) for r in selected]);mount=t([r.get('reference_mount_rotation',m.mount_rot) for r in selected])
    rest=m.rest;restj=m.rest_j;distal=torch.tensor([c[-1] for c in CHAINS],device=device)
    optimizer=torch.optim.Adam([delta],lr=.01)
    for step in range(steps):
        if cancel is not None and cancel.is_set():raise InterruptedError('接触模板构建已取消')
        optimizer.zero_grad();pose,_=chart.correct(seed,delta,mask)
        v,j,r,_=m.decode(pose,roots)
        p=(v[frame[:,None,None],face]*bc[...,None]).sum(-2)
        normal=vertex_normals(v,m.face_t)
        n=torch.nn.functional.normalize((normal[frame[:,None,None],face]*bc[...,None]).sum(-2),dim=-1)
        actual,desired=tip_emf_directions(v,j,rest,restj,distal,list(TIPS),rawr,alignment[:,None],mount)
        loss=((p[:,0]-p[:,1])/.002).square().mean()+.2*normal_residual(n[:,0],n[:,1]).square().mean()+((actual-desired)*mask[...,None]).square().sum()/mask.sum().clamp_min(1)*10+.02*delta.square().mean()
        loss.backward();optimizer.step()
    with torch.no_grad():
        pose,_=chart.correct(seed,delta,mask);final_vertices=m.decode(pose,roots)[0]
        vertices=final_vertices.cpu().numpy()
        from .mano_gpu_collision import BatchedSurfaceCollision
        witness=BatchedSurfaceCollision(m,len(selected)).query(final_vertices)
        depths=witness['maximum'].cpu().tolist()
    bank={};reports=[]
    for edge in plans:
        i=edge['frame'];gap,anchors=nearest_anchors(vertices[i],m.faces,regions,*edge['regions'])
        edge['anchors']=anchors;edge['template_gap_mm']=gap*1000
        tri=vertices[i][m.faces[[a['face'] for a in anchors]]];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=-1,keepdims=True).clip(1e-12)
        edge['template_normal_error_deg']=float(np.degrees(np.arccos(np.clip(-normal[0]@normal[1],-1,1))))
        edge['template_penetration_mm']=depths[i]*1000
        edge['template_source_event_id']=selected[i]['contact_event_id']
        edge['template_points_m']=[(vertices[i][m.faces[a['face']]]*np.asarray(a['barycentric'])[:,None]).sum(0).tolist() for a in anchors]
        edge['template_normals']=normal.tolist()
        edge['feasible']=bool(gap<=.005 and normal[0]@normal[1]<-.1 and depths[i]<=.002)
        bank.setdefault(edge['stage'],[]).append(edge)
    # Choose each action's lower total residual TRAIN repetition; never inspect
    # the validation repetition to select or move material coordinates.
    templates={}
    for stage in bank:
        candidates=[]
        for i,row in enumerate(selected):
            if row['stage']!=stage:continue
            start=sum(len(r.get('reference_edges') or target(r['stage'])['edges']) for r in selected[:i]);count=len(row.get('reference_edges') or target(stage)['edges'])
            candidates.append(plans[start:start+count])
        chosen=min(candidates,key=lambda es:sum(e['template_gap_mm'] for e in es if e['required']))
        templates[stage]=copy.deepcopy(chosen)
        reports.append(dict(stage=stage,edges=copy.deepcopy(chosen),source='train_only_standard_MANO',
                            pose_axis_angle=pose[chosen[0]['frame']].detach().cpu().tolist(),root_rotation=roots[chosen[0]['frame']].cpu().tolist(),
                            feasible=all(e['feasible'] for e in chosen if e['required'])))
    for row in rows:
        if row.get('capture_protocol')!=PROTOCOL or row.get('initialization'):continue
        row['contact_edges']=copy.deepcopy(templates[row['stage']]) if row.get('contact_confirmed') else []
    artifact=dict(revision=REVISION,source='train_only',shape='standard_MANO_beta_zero',regions=regions,actions=reports,steps=steps,
                  reference_ids=sorted({r['contact_reference_id'] for r in selected}),validation_used=False)
    model.contact_template_artifact=artifact
    return artifact


@torch.no_grad()
def region_geometry_report(model,rows,vertices):
    """Independent original-skin regional distance, never template distance only."""
    from .mano_contact_loss import normal_error_deg,vertex_normals
    artifact=getattr(model,'contact_template_artifact',None)
    if not artifact:raise ValueError('区域验收缺少冻结模板区域')
    regions=artifact['regions'];records=[];clusters=[]
    for i,row in enumerate(rows):
        if not row.get('contact_confirmed') or row.get('initialization'):continue
        vv=vertices[i].cpu().numpy()
        normals=vertex_normals(vertices[i],model.face_t).cpu().numpy()
        good_edges=[]
        for edge in row.get('contact_edges',[]):
            gap,anchors=nearest_anchors(vv,model.faces,regions,*edge['regions'])
            tri=vv[model.faces[[a['face'] for a in anchors]]]
            normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=-1,keepdims=True).clip(1e-12)
            smooth=np.array([(normals[model.faces[a['face']]]*np.asarray(a['barycentric'])[:,None]).sum(0) for a in anchors]);smooth/=np.linalg.norm(smooth,axis=-1,keepdims=True).clip(1e-12)
            dot=float(normal[0]@normal[1]);smooth_dot=float(smooth[0]@smooth[1])
            fixed=np.array([(vv[model.faces[a['face']]]*np.asarray(a['barycentric'])[:,None]).sum(0) for a in edge['anchors']])
            record=dict(stage=row['stage'],timestamp_ms=row['timestamp_ms'],event=row['contact_event_id'],regions=edge['regions'],required=edge['required'],
                        gap_m=gap,normal_dot=dot,normal_error_deg=float(np.degrees(np.arccos(np.clip(-dot,-1,1)))),
                        smooth_normal_error_deg=float(np.degrees(np.arccos(np.clip(-smooth_dot,-1,1)))),template_anchor_gap_m=float(np.linalg.norm(fixed[0]-fixed[1])),
                        evidence='trajectory_assumed_contact',query='original_region_triangles',weight=edge.get('weight',1.))
            records.append(record)
            if gap<=.005:good_edges.append(edge['regions'])
        if target(row['stage'])['kind']=='distal_cluster':
            nodes=set(target(row['stage'])['patches']);seen={next(iter(nodes))}
            while True:
                updated=seen|{b for a,b in good_edges if a in seen}|{a for a,b in good_edges if b in seen}
                if updated==seen:break
                seen=updated
            clusters.append(dict(stage=row['stage'],event=row['contact_event_id'],connected=seen==nodes,
                                 thumb_distances_mm={r['regions'][1]:r['gap_m']*1000 for r in records if r['event']==row['contact_event_id'] and r['regions'][0]=='thumb_distal'}))
    release_checks=[];actions={a['stage']:a['edges'] for a in artifact['actions']}
    windows={}
    for i,row in enumerate(rows):
        if row.get('initialization') or not row.get('training_eligible',True):continue
        windows.setdefault(row.get('window_id',row['stage']),[]).append(i)
    for key,ids in windows.items():
        if rows[ids[0]]['stage'] not in actions:continue
        measured=[i for i in ids if isinstance(rows[i].get('reference_gap_m'),(int,float))]
        events=[i for i in ids if rows[i].get('contact_confirmed')]
        if not measured or not events:continue
        low=min(rows[i]['reference_gap_m'] for i in measured);high=max(rows[i]['reference_gap_m'] for i in measured)
        far=[i for i in measured if not rows[i].get('contact_confirmed') and rows[i]['reference_gap_m']>=low+.6*(high-low)]
        # All EMF trajectory frames are already validated. Three temporally
        # spread original-skin witnesses additionally catch an always-closed
        # policy without a CPU triangle search on every sensor observation.
        chosen=sorted({far[0],far[len(far)//2],far[-1]}) if far else []
        stage=rows[ids[0]]['stage'];stage_records=[r for r in records if r['required'] and r['timestamp_ms'] in {rows[i]['timestamp_ms'] for i in events}]
        samples=[]
        for i in chosen:
            vv=vertices[i].cpu().numpy()
            for edge in actions[stage]:
                if not edge['required']:continue
                gap,_=nearest_anchors(vv,model.faces,regions,*edge['regions'])
                touch=[r['gap_m'] for r in stage_records if r['regions']==edge['regions']]
                increase=gap-min(touch) if touch else float('-inf')
                samples.append(dict(timestamp_ms=rows[i]['timestamp_ms'],regions=edge['regions'],gap_mm=gap*1000,opening_mm=increase*1000,valid=bool(increase>=.002)))
        release_checks.append(dict(stage=stage,window=key,observations=len(ids),reference_excursion_mm=(high-low)*1000,
                                   sampled_original_triangle_witnesses=samples,valid=bool(samples) and all(s['valid'] for s in samples)))
    required=[r for r in records if r['required']];stages=sorted({r['stage'] for r in required})
    averages=[np.mean([r['gap_m'] for r in required if r['stage']==stage]) for stage in stages]
    return dict(contact_protocol=PROTOCOL,approach_release_geometry=release_checks,
                approach_release_valid=bool(release_checks) and all(r['valid'] for r in release_checks),contact_gap_m=float(np.mean(averages)) if averages else float('inf'),
                contact_max_gap_m=max((r['gap_m'] for r in required),default=float('inf')),contact_observations=len(required),contact_stages=stages,
                contact_material_records=required,preferred_thumb_records=[r for r in records if not r['required']],contact_clusters=clusters,
                contact_clusters_valid=bool(clusters) and all(c['connected'] for c in clusters),
                contact_normal_error_deg=float(np.mean([r['normal_error_deg'] for r in required])) if required else None,
                contact_normal_max_error_deg=max((r['normal_error_deg'] for r in required),default=None),
                contact_smooth_normal_error_deg=float(np.mean([r['smooth_normal_error_deg'] for r in required])) if required else None)

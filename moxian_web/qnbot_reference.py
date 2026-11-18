"""Candidate-independent patch trajectories and anatomical return evidence."""
import copy,hashlib,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .qnbot_surface import QnbotSurface,FINGERS,SOURCE_FINGERS
from .calibration_targets import target


def observations(rows,device):
    p=[];r=[];valid=[]
    for row in rows:
        pp=[];rr=[];vv=[]
        for f in FINGERS:
            node=(row.get('fingers') or {}).get(f,{})
            xyz=np.asarray(node.get('position',[0,0,0]),float);quat=np.asarray(node.get('quaternion_xyzw',[0,0,0,1]),float)
            good=xyz.shape==(3,) and quat.shape==(4,) and np.isfinite(xyz).all() and np.isfinite(quat).all() and np.linalg.norm(quat)>.1 and node.get('valid') is not False and node.get('confidence',1)>0
            pp.append(xyz if good else np.zeros(3));rr.append(Rotation.from_quat(quat).as_matrix() if good else np.eye(3));vv.append(float(good))
        p.append(pp);r.append(rr);valid.append(vv)
    return tuple(torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32) for x in (p,r,valid))


@torch.no_grad()
def initialize_registration(model,rows):
    """Alternating rigid registration and constant per-distal installation."""
    q=model.angles(rows);rawp,rawr,mask=observations(rows,model.device)
    r,p=model.fk(q);wr=r[:,model.wrist_body];wp=p[:,model.wrist_body]
    dr=wr.transpose(-1,-2)[:,None]@r[:,model.distal_bodies];dp=(p[:,model.distal_bodies]-wp[:,None])@wr
    source=rawp.cpu().numpy();desired=dp.cpu().numpy();valid=mask.cpu().numpy().astype(bool)
    if not valid.any(0).all():raise ValueError('安装初始化缺少目标手指观测')
    align=np.eye(3);trans=np.zeros(3);mount=np.zeros((5,3))
    for _ in range(12):
        a=source[valid];b=desired[valid];ac=a-a.mean(0);bc=b-b.mean(0);u,_,vt=np.linalg.svd(ac.T@bc)
        fix=np.eye(3);fix[-1,-1]=np.linalg.det(vt.T@u.T);align=vt.T@fix@u.T;trans=b.mean(0)-align@a.mean(0)
        local=torch.einsum('nfji,nfj->nfi',dr,torch.as_tensor(source@align.T+trans,device=model.device,dtype=torch.float32)-dp).cpu().numpy()
        mount=np.array([np.median(local[valid[:,f],f],0) for f in range(5)])
        desired=(dp+torch.einsum('nfij,fj->nfi',dr,torch.as_tensor(mount,device=model.device,dtype=torch.float32))).cpu().numpy()
    mr=[];ar=torch.as_tensor(align,device=model.device,dtype=torch.float32)@rawr
    rotations=(dr.transpose(-1,-2)@ar).cpu().numpy()
    for f in range(5):mr.append(Rotation.from_matrix(rotations[valid[:,f],f]).mean().as_matrix())
    for key,value in dict(align0=align,trans0=trans,mount_p0=mount,mount_r0=mr).items():getattr(model,key).copy_(torch.as_tensor(np.asarray(value),device=model.device))


class QnbotReference:
    def __init__(self,side,mapping,initial,device='cuda:0',spec=None,material_binding=None):
        self.model=QnbotSurface(side,device);m=self.model;m.bind_mapping(mapping)
        if material_binding and not spec:m.install_binding(material_binding)
        if spec:
            m.load_artifact(spec['surface']);self.open_metrics=spec['open_metrics'];self.open_spread=spec['open_spread_m'];self.open_thumb=np.asarray(spec['open_thumb_rotation'])
        else:
            initialize_registration(m,initial['rows']);opened=[r for r in initial['rows'] if r['stage']=='open']
            if not opened:raise ValueError('初始化缺少展平观测')
            with torch.no_grad():
                q=m.angles(opened);b,p,thumb,spread=self.metrics(q)
                self.open_metrics=dict(bend_deg=np.percentile(b.cpu().numpy(),95,axis=0).tolist(),plane_deg=np.percentile(p.cpu().numpy(),95,axis=0).tolist())
                self.open_thumb=Rotation.from_matrix(thumb.cpu().numpy()).mean().as_matrix();self.open_spread=float(spread.median())
        self.spec=dict(revision='qnbot_reference_v1',surface=m.artifact(),open_metrics=self.open_metrics,open_thumb_rotation=self.open_thumb.tolist(),open_spread_m=self.open_spread,
                       calibration_fingerprint=initial.get('calibration_fingerprint'),skeleton_runtime_revision=initial['rows'][0].get('skeleton_runtime_revision'))
        self.fingerprint=hashlib.sha256(json.dumps(self.spec,sort_keys=True).encode()).hexdigest()
        m.fixed=True
        anchors=list(m.patch_anchors.values());names=list(m.patch_anchors)
        self.patch_names=names;self.support=torch.unique(m.faces[torch.as_tensor([a['face'] for a in anchors],device=m.device)].flatten(),sorted=True)
        self.anchor_tri=torch.searchsorted(self.support,m.faces[torch.as_tensor([a['face'] for a in anchors],device=m.device)])

    def metrics(self,q):
        m=self.model;r,p=m.fk(q);wr=r[:,m.wrist_body];wp=p[:,m.wrist_body];p=(p-wp[:,None])@wr;r=wr.transpose(-1,-2)[:,None]@r
        bends=[];plane=[]
        normal=torch.as_tensor(m.palm_normal,device=m.device,dtype=torch.float32)
        for finger in SOURCE_FINGERS:
            parts=('metacarpal','proximal','distal','tip') if finger=='thumb' else ('proximal','intermediate','distal','tip')
            points=p[:,[m.body_names.index(m.side+'_'+finger+'_'+part) for part in parts]]
            d=torch.nn.functional.normalize(points[:,1:]-points[:,:-1],dim=-1)
            angles=torch.rad2deg(torch.acos((d[:,1:]*d[:,:-1]).sum(-1).clamp(-1,1)))
            bends.append(angles[:,-1] if finger=='thumb' else angles.amax(1))
            plane.append(torch.rad2deg(torch.asin((d*normal).sum(-1).abs().clamp(0,1))).amax(1))
        spread=torch.linalg.vector_norm(p[:,m.tip_bodies[1:-1]]-p[:,m.tip_bodies[2:]],dim=-1).mean(1)
        return torch.stack(bends,1),torch.stack(plane,1),r[:,m.distal_bodies[0]],spread

    @torch.no_grad()
    def __call__(self,row):
        m=self.model;q=m.angles([row]);v=m(q,vertex_ids=self.support)[0];tri=v[:,self.anchor_tri]
        points=tri.mean(-2)[0].cpu().numpy();normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)[0].cpu().numpy()
        patch={name:dict(position_m=points[i].tolist(),normal=normal[i].tolist(),valid=True) for i,name in enumerate(self.patch_names)}
        spec=target(row['stage']);edges=copy.deepcopy(spec.get('edges',[]))
        if spec['kind']=='distal_cluster':
            from .contact_protocol import cluster_tree
            centers=np.array([patch[n]['position_m'] for n in spec['patches']]);edges=cluster_tree(spec['patches'],np.linalg.norm(centers[:,None]-centers[None],axis=-1))
        gaps=[np.linalg.norm(np.asarray(patch[e['regions'][0]]['position_m'])-patch[e['regions'][1]]['position_m']) for e in edges if e.get('required',True)]
        b,p,t,spread=self.metrics(q);bend=b[0].cpu().numpy();plane=p[0].cpu().numpy()
        thumb=float(Rotation.from_matrix(self.open_thumb.T@t[0].cpu().numpy()).magnitude()*180/np.pi)
        bb=np.asarray(self.open_metrics['bend_deg'])+8.;pp=np.asarray(self.open_metrics['plane_deg'])+12.
        _,_,valid=observations([row],m.device);all_valid=bool(valid[0].all())
        flat=all_valid and bool(np.all(bend<=bb)) and bool(np.all(plane[1:]<=pp[1:])) and thumb<=20.
        flat_exit=not all_valid or bool(np.any(bend>bb+3)) or bool(np.any(plane[1:]>pp[1:]+3)) or thumb>23.
        motion=spec['kind']=='noncontact_motion'
        if motion:flat=flat and float(spread[0])>=.75*self.open_spread
        return dict(reference_gap_m=float(np.sqrt(np.mean(np.square(gaps)))) if gaps else float(spread[0]),reference_edges=edges,
            contact_reference_id=self.fingerprint,patch_reference_source='frozen_qnbot_emf_proxy_not_surface_ground_truth',
            patch_trajectory={n:patch[n] for n in spec['patches']},flat_valid=flat,flat_exit=flat_exit,
            flat_evidence=dict(all_fingers_valid=all_valid,bend_deg=bend.tolist(),plane_deg=plane.tolist(),thumb_return_deg=thumb))

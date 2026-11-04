"""GPU MANO surface, native convex candidates, triangle verification and fitting.

This is a quasi-static normal approximation. ADC is relative evidence, never N.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
import trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scripts.calibrate_wuji_mano import PersonalizedMANO, CHAINS, TIPS, rotmat, rigid_alignment
from .contact import PATCHES, PATCH_IDS, FINGERS
from .hand_model import DEFAULT_MATERIAL, pair_mechanics, fusion_state, positive_gain
from .hand_anatomy import AnatomicalJoints,GEOMETRY_REVISION,closed_surface_parts
from .hand_dimensions import scale_segments,DIMENSION_MODE,DIMENSION_COUNT

@wp.kernel(enable_backward=False)
def inside_queries(solids:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),
                   inside:wp.array2d(dtype=float),distance:wp.array2d(dtype=float)):
    sid,part=wp.tid()
    query=wp.mesh_query_point_sign_winding_number(solids[part],points[sid],1.0)
    inside[sid,part]=0.0;distance[sid,part]=1.0
    if query.result:
        near=wp.mesh_eval_position(solids[part],query.face,query.u,query.v)
        inside[sid,part]=wp.where(query.sign<0.0,1.0,0.0)
        distance[sid,part]=wp.length(points[sid]-near)

@wp.kernel(enable_backward=False)
def triangle_queries(meshes:wp.array(dtype=wp.uint64), points:wp.array(dtype=wp.vec3),
                     groups:wp.array(dtype=int),inside:wp.array2d(dtype=float),solid_distance:wp.array2d(dtype=float),
                     normals:wp.array(dtype=wp.vec3), gap:wp.array2d(dtype=float),
                     facing:wp.array2d(dtype=float), nearest:wp.array2d(dtype=wp.vec3)):
    sid,p=wp.tid()
    hit=wp.mesh_query_point_no_sign(meshes[p],points[sid],.04)
    gap[sid,p]=1.0
    facing[sid,p]=1.0
    if hit.result:
        point=wp.mesh_eval_position(meshes[p],hit.face,hit.u,hit.v)
        normal=wp.mesh_eval_face_normal(meshes[p],hit.face)
        delta=points[sid]-point
        # An open patch normal cannot establish penetration. Only a point
        # inside the target's closed anatomical volume can have negative gap,
        # and only on the nearest surface (not a distant patch's back side).
        length=wp.length(delta);part=groups[p]
        penetrated=inside[sid,part]>0.5 and length<=solid_distance[sid,part]+0.0005
        gap[sid,p]=length*wp.where(penetrated,-1.0,1.0)
        facing[sid,p]=wp.dot(normals[sid],normal)
        nearest[sid,p]=point


def vertex_normals(v,faces):
    tri=v[faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);vn=np.zeros_like(v)
    for k in range(3):np.add.at(vn,faces[:,k],n)
    vn/=np.maximum(np.linalg.norm(vn,axis=1,keepdims=True),1e-12)
    return vn


def skeleton_targets(bone):
    if not bone:return None
    try:
        if bone['hand_side']!='left' or bone['coordinate_frame']!='wrist_local':return None
        nodes=dict(zip(bone['names'],bone['positions_local']))
        keys=['wrist']
        for f in ('thumb','index','middle','ring','little'):
            keys.extend([f+'_'+s for s in (('metacarpal','proximal','distal','tip') if f=='thumb' else ('proximal','intermediate','distal','tip'))])
        result=np.array([nodes[k] for k in keys],dtype=np.float32)
        return result if result.shape==(21,3) and np.isfinite(result).all() else None
    except (KeyError,ValueError,TypeError):return None


def aim_rotation(a,b):
    a=a/max(np.linalg.norm(a),1e-9);b=b/max(np.linalg.norm(b),1e-9)
    cross=np.cross(a,b);dot=np.clip(a@b,-1.,1.);length=np.linalg.norm(cross)
    if length<1e-7:
        if dot>0:return np.eye(3)
        axis=np.cross(a,np.eye(3)[np.argmin(np.abs(a))]);axis/=np.linalg.norm(axis)
        return Rotation.from_rotvec(axis*np.pi).as_matrix()
    return Rotation.from_rotvec(cross/length*math.atan2(length,dot)).as_matrix()


def blend_rotations(previous,current,alpha):
    """Shortest-path SO(3) interpolation, including the +/- pi rotvec boundary."""
    delta=Rotation.from_matrix(np.swapaxes(previous,-2,-1)@current).as_rotvec()
    return previous@Rotation.from_rotvec(delta*alpha).as_matrix()


class ManoSurface:
    def __init__(self,path,device='cuda:0'):
        self.device=torch.device(device);self.model=PersonalizedMANO(path,'left').to(device=self.device,dtype=torch.float32)
        self.weights_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.faces=self.model.faces.astype(np.int32);self.face_t=torch.tensor(self.faces.astype(np.int64),device=self.device)
        self.parents=self.model.parents
        depth=[0]*16
        for j in range(1,16):depth[j]=depth[self.parents[j]]+1
        self.levels=[[j for j in range(1,16) if depth[j]==d] for d in range(1,4)]
        self.ordered=[0]+[j for f,c in enumerate(CHAINS) for j in (*c,16+f)]
        self.shape=np.zeros(41,dtype=np.float64);self.labels=None;self.mapping=None;self.geometry=None
        self.dimension_parameters=None;self.bone_reference=None
        self.raw_rotation=np.eye(3);self.raw_translation=np.zeros(3);self.raw_aligned=False
        self.sensor_joint_position=None;self.mount_pos=np.zeros((5,3));self.mount_rot=np.tile(np.eye(3),(5,1,1))
        self.last_pose=np.zeros((15,3),np.float32);self.last_root=np.eye(3);self.last_frame=None
        self.tracking_time=None;self.tracking_version=None;self.tracking_seed=None
        self.tracking_q=None;self.tracking_optimizer=None
        self.anatomy=None
        self.thumb_quality=None
        self.mounting_calibrated=False
        self.has_wrist_targets=False;self.tracking_root=None;self.alignment_quality=None;self.tracking_tip_offset=np.zeros((5,3));self.tracking_bone_root=None;self.tracking_observed=None;self.tracking_stable_ms=0.;self.reference_translation=np.zeros(3);self.reference_root=None;self.flat_reference=None
        self.material=dict(DEFAULT_MATERIAL);self.response_gains={};self.binding_overrides={}
        self.compression=np.zeros((778,3),np.float32);self.last_time=None
        self.set_shape(self.shape)

    def set_dimensions(self,parameters):
        parameters=np.asarray(parameters,dtype=float)
        if parameters.shape!=(DIMENSION_COUNT,) or not np.isfinite(parameters).all():raise ValueError('手指尺寸参数无效')
        self.dimension_parameters=parameters.copy();self.set_shape(self.shape)

    def set_shape(self,value):
        value=np.array(value,dtype=np.float64)
        if value.shape!=(41,) or not np.isfinite(value).all():raise ValueError('无效的 MANO 尺寸参数')
        self.shape=value.copy()
        self.tracking_time=None
        t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            # Layout: beta[10], fixed global scale, five small log-length
            # corrections, five fingertip soft-tissue extensions (metres),
            # and twenty transverse half-thickness corrections (metres).
            neutral,nj=self.model.rest_shape(t(value[:10]),t(value[10]),t(np.zeros(5)),t(np.zeros(5)))
            nv=neutral.cpu().numpy();nj=nj.cpu().numpy()
            lengths=value[11:16].copy()
            if self.bone_reference is not None:
                base=np.array([sum(np.linalg.norm((nj[c[k+1]] if k<2 else nv[TIPS[f]])-nj[joint]) for k,joint in enumerate(c)) for f,c in enumerate(CHAINS)])
                # Only total length informs a soft starting scale. SDK joint
                # segments do not specify surface lengths or a new anatomy.
                lengths+=np.log(np.clip(np.asarray(self.bone_reference).sum(1)/base,.8,1.3))
            rest,j=self.model.rest_shape(t(value[:10]),t(value[10]),t(lengths),t(np.zeros(5)))
        v=rest.cpu().numpy();jn=j.cpu().numpy()
        if self.dimension_parameters is not None:
            v,jn=scale_segments(v,jn,self.model.weights.cpu().numpy(),CHAINS,TIPS,self.dimension_parameters)
            self.rest_j=torch.as_tensor(jn,dtype=torch.float32,device=self.device);self.rest_j_np=jn
            self.tip_anchors=v[list(TIPS)].copy()
            self.anatomy=AnatomicalJoints(jn,v,CHAINS,TIPS,self.model.palm_normal.cpu().numpy(),self.model.pose_mean.cpu().numpy())
            if self.labels is None:self._anatomy(v)
            self.rest=t(v);self.rest_np=v.astype(np.float32);self.rest_normals=vertex_normals(v,self.faces)
            self.shape_version=hashlib.sha256((GEOMETRY_REVISION+DIMENSION_MODE).encode()+np.round(value[:16],9).tobytes()+np.round(self.dimension_parameters,9).tobytes()+np.array(self.bone_reference if self.bone_reference is not None else [],dtype=np.float64).tobytes()).hexdigest()[:16]
            self.geometry=None
            if self.mapping:self.build_binding(self.mapping)
            return
        weights=self.model.finger_weights.cpu().numpy()
        for f,chain in enumerate(CHAINS):
            axis=v[TIPS[f]]-jn[chain[0]];axis/=np.linalg.norm(axis)
            u=(v-jn[chain[-1]])@axis;length=float((v[TIPS[f]]-jn[chain[-1]])@axis)
            blend=np.clip(u/max(length,1e-6),0,1);blend=blend*blend*(3-2*blend)
            v+=weights[:,f,None]*blend[:,None]*value[16+f]*axis
        self.rest_j=torch.as_tensor(jn,dtype=torch.float32,device=self.device);self.rest_j_np=jn
        self.tip_anchors=v[list(TIPS)].copy()
        self.anatomy=AnatomicalJoints(jn,v,CHAINS,TIPS,self.model.palm_normal.cpu().numpy(),self.model.pose_mean.cpu().numpy())
        if self.labels is None:self._anatomy(v)
        normals=vertex_normals(v,self.faces)
        offsets=value[21:][self.vertex_patches].copy()
        # Smooth scalar patch changes over the mesh, avoiding seams and lumps.
        edges=np.concatenate((self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]));edges=np.concatenate((edges,edges[:,::-1]))
        counts=np.bincount(edges[:,0],minlength=778)
        for _ in range(3):offsets=.5*offsets+.5*np.bincount(edges[:,0],weights=offsets[edges[:,1]],minlength=778)/counts
        # Thickness changes are transverse; the separate bounded length and
        # fingertip variables express uncertainty in the SDK skeleton prior.
        radial=normals.copy();fade=np.ones(778)
        for f,chain in enumerate(CHAINS):
            direction=v[TIPS[f]]-jn[chain[0]];direction/=np.linalg.norm(direction)
            mask=np.isin(self.vertex_patches,[i for i,p in enumerate(PATCH_IDS) if p.startswith(FINGERS[f]+'_')])
            radial[mask]-=(normals[mask]@direction)[:,None]*direction
            radial[mask]/=np.maximum(np.linalg.norm(radial[mask],axis=1,keepdims=True),1e-8)
            u=(v-jn[chain[0]])@direction;tip_u=(v[TIPS[f]]-jn[chain[0]])@direction
            fade[mask]=np.clip(u[mask]/.008,0,1)*np.clip((tip_u-u[mask])/.004,0,1)
        displacement=radial*offsets[:,None]*fade[:,None]
        self.rest=t(v+displacement);self.rest_np=self.rest.cpu().numpy()
        self.rest_normals=normals
        self.shape_version=hashlib.sha256(GEOMETRY_REVISION.encode()+np.round(value,9).tobytes()+np.array(self.bone_reference if self.bone_reference is not None else [],dtype=np.float64).tobytes()).hexdigest()[:16]
        self.geometry=None
        if self.mapping:self.build_binding(self.mapping)

    def _anatomy(self,v):
        weights=self.model.weights.cpu().numpy();self.labels=weights.argmax(-1)
        fw=np.stack([weights[:,list(c)].sum(-1) for c in CHAINS],axis=1)
        finger=fw.argmax(-1);palm=fw.max(-1)<.55
        self.vertex_patches=np.zeros(778,dtype=np.int32)
        for f,c in enumerate(CHAINS):
            joint=weights[:,list(c)].argmax(-1)
            for k in range(3):
                part=('proximal','proximal','distal')[k] if f==0 else ('proximal','intermediate','distal')[k]
                self.vertex_patches[(finger==f)&(joint==k)]=PATCH_IDS.index(FINGERS[f]+'_'+part)
        forward=self.rest_j_np[4]-self.rest_j_np[0];forward/=np.linalg.norm(forward)
        side=self.model.palm_lateral.cpu().numpy();self.palm_axes=(forward,side)
        u=(v-self.rest_j_np[0])@forward;w=v@side
        # Physical row 0 starts at the wrist. Initial correspondence is provisional.
        row=np.clip((u/max(float((self.rest_j_np[4]-self.rest_j_np[0])@forward),.01)*3).astype(int),0,2)
        col=(w<np.median(w[palm])).astype(int)
        self.vertex_patches[palm]=14+2*row[palm]+col[palm]
        fn=vertex_normals(v,self.faces);palm_normal=self.anatomy.palmar_normal
        self.face_patches=[]
        for face in self.faces:
            votes=np.bincount(self.vertex_patches[face],minlength=20);self.face_patches.append(int(votes.argmax()))
        self.face_patches=np.array(self.face_patches)
        self.patch_faces=[];self.binding_faces=[]
        for p in range(20):
            ids=np.flatnonzero(self.face_patches==p)
            self.patch_faces.append(ids.copy())
            palmar=(fn[self.faces[ids]].mean(1)@palm_normal)>.05
            if np.any(palmar):ids=ids[palmar]
            if not len(ids):raise ValueError('MANO 解剖分区缺失 '+PATCH_IDS[p])
            self.binding_faces.append(ids)
        self.patch_owners=[]
        for p in range(20):
            self.patch_owners.append(0 if p>=14 else int(np.bincount(self.labels[self.faces[self.patch_faces[p]].ravel()],minlength=16).argmax()))
        # Keep wrist edge fixed, not the entire palm.
        self.wrist_mask=u<np.percentile(u[palm],15)

    def build_binding(self,mapping):
        self.mapping=mapping
        assignments=np.array(mapping['assignments'],dtype=int)
        self.sid_faces=np.zeros(460,dtype=np.int32);self.sid_bary=np.zeros((460,3),np.float32)
        self.cell_area=np.zeros(460,np.float32)
        tri=self.rest_np[self.faces];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        for p in range(20):
            ids=np.flatnonzero(assignments==p);faces=self.binding_faces[p];centers=tri[faces].mean(1)
            if not len(ids):raise ValueError('每个 patch 须有 SID')
            if p<14:
                f=FINGERS.index(PATCH_IDS[p].split('_')[0]);axis=self.rest_np[TIPS[f]]-self.rest_j_np[CHAINS[f][0]];axis/=np.linalg.norm(axis)
                lateral=np.cross(axis,self.model.palm_normal.cpu().numpy());lateral/=max(np.linalg.norm(lateral),1e-9)
            else:axis,lateral=self.palm_axes
            uv=np.stack((centers@lateral,centers@axis),1)
            rows=ids//20;cols=ids%20
            u=(cols-cols.min()+.5)/(np.ptp(cols)+1);w=(rows-rows.min()+.5)/(np.ptp(rows)+1)
            lo=uv.min(0);span=np.maximum(np.ptp(uv,axis=0),1e-6);target=lo+np.stack((u,w),1)*span
            chosen=((target[:,None]-uv[None])**2).sum(-1).argmin(1)
            self.sid_faces[ids]=faces[chosen];self.sid_bary[ids]=1/3
            self.cell_area[ids]=max(float(areas[faces].sum())/len(ids),1e-8)
        for sid,record in self.binding_overrides.items():
            if 0<=int(sid)<460 and self.face_patches[int(record['face'])]==assignments[int(sid)]:
                self.sid_faces[int(sid)]=record['face'];self.sid_bary[int(sid)]=record['barycentric']
        self.binding_version=hashlib.sha256(json.dumps([mapping['version'],self.sid_faces.tolist(),self.sid_bary.tolist()]).encode()).hexdigest()[:16]

    def topology(self):
        if not self.mapping:return {'ready':False}
        return {'ready':True,'hand_side':'left','coordinate_frame':'wrist_local','units':'m',
                'weights_hash':self.weights_hash,'geometry_version':self.shape_version,'binding_version':self.binding_version,
                'faces':self.faces.tolist(),'face_patches':self.face_patches.tolist(),'patches':PATCHES,
                'sid_face_ids':self.sid_faces.tolist(),'sid_barycentric':self.sid_bary.tolist(),
                'mapping_confirmed':self.mapping.get('confirmed',[]),'mapping_note':'初始 SID 曲面位置为解剖插值；未逐点核验不是精确传感器定位。',
                'joint_chains':[[0]+list(range(1+4*f,5+4*f)) for f in range(5)]}

    def decode(self,pose,root):
        local=rotmat(pose.reshape(-1,15,3));batch=len(local);eye=torch.eye(3,device=self.device,dtype=local.dtype)
        rr=[None]*16;pp=[None]*16;rr[0]=eye.expand(batch,3,3);pp[0]=self.rest_j[0].expand(batch,3)
        for level in self.levels:
            pr=torch.stack([rr[self.parents[j]] for j in level],1);pos=torch.stack([pp[self.parents[j]] for j in level],1)
            relative=torch.stack([self.rest_j[j]-self.rest_j[self.parents[j]] for j in level])
            r=pr@local[:,np.array(level)-1];p=pos+(pr@relative[None,:,:,None]).squeeze(-1)
            for k,j in enumerate(level):rr[j]=r[:,k];pp[j]=p[:,k]
        rotations=torch.stack(rr,1);joints=torch.stack(pp,1)
        feature=(local-eye).reshape(batch,135)
        posed=self.rest[None].expand(batch,-1,-1) if self.dimension_parameters is not None else self.rest[None]+torch.einsum('bp,vcp->bvc',feature,self.model.posedirs)*math.exp(self.shape[10])
        offset=joints-(rotations@self.rest_j[None,:,:,None]).squeeze(-1)
        skin_r=torch.einsum('vj,bjkl->bvkl',self.model.weights,rotations)
        skin_t=torch.einsum('vj,bjk->bvk',self.model.weights,offset)
        v=(skin_r@posed[:,:,:,None]).squeeze(-1)+skin_t-self.rest_j[0]
        j=joints-self.rest_j[0]
        v=v@root.transpose(-1,-2);j=j@root.transpose(-1,-2);rotations=root[:,None]@rotations
        tip_offset=torch.as_tensor(self.tip_anchors-self.rest_j_np[[c[-1] for c in CHAINS]],device=self.device)
        tips=j[:,[c[-1] for c in CHAINS]]+(rotations[:,[c[-1] for c in CHAINS]]@tip_offset[None,:,:,None]).squeeze(-1)
        ordered=torch.cat((j,tips),1)[:,self.ordered]
        return v,j,rotations,ordered

    def pose_from_bones(self,target):
        rest=self.rest_j_np
        roots=[c[0] for c in CHAINS];dest=[target[1+4*f] for f in range(5)]
        root,_=rigid_alignment(rest[roots]-rest[0],np.array(dest))
        global_r=[np.eye(3)]*16;local=np.tile(np.eye(3),(15,1,1))
        for f,chain in enumerate(CHAINS):
            for k,j in enumerate(chain):
                end=rest[chain[k+1]] if k<2 else self.rest_np[TIPS[f]]
                delta=target[2+4*f+k]-target[1+4*f+k]
                parent=global_r[self.parents[j]]
                observed=parent.T@root.T@delta
                hinged=self.anatomy.fit_local_direction(j-1,end-rest[j],observed)
                local[j-1]=hinged if hinged is not None else self.anatomy.project_local_rotation(j-1,aim_rotation(end-rest[j],observed))
                global_r[j]=parent@local[j-1]
        pose=self.anatomy.project(Rotation.from_matrix(local).as_rotvec())
        return self.fit_thumb_directions(pose,root,target),root.astype(np.float32)

    def fit_thumb_directions(self,pose,root,target,tip_target=None,sensor_offset=None):
        """Solve the three thumb links together so CMC roll is observable.

        A single CMC link direction cannot specify opposition/twist. Jointly
        fitting the next two directions prevents the IP joint compensating
        for a wrongly seeded thumb base. This is kinematic IK, not dynamics.
        """
        angles=self.anatomy.angles(pose)[12:15]
        lower=self.anatomy.lower[12:15];upper=self.anatomy.upper[12:15];free=upper-lower>1e-8
        basis=self.anatomy.basis[12:15]
        chain=CHAINS[0];ends=np.concatenate((self.rest_j_np[list(chain[1:])],self.tip_anchors[:1]))
        rest=ends-self.rest_j_np[list(chain)];rest/=np.linalg.norm(rest,axis=1,keepdims=True)
        lengths=ends-self.rest_j_np[list(chain)]
        if sensor_offset is not None:lengths[-1]+=sensor_offset
        expected=np.diff(target[1:5],axis=0)@root;expected/=np.linalg.norm(expected,axis=1,keepdims=True)
        def evaluate(x):
            a=angles.copy();a[free]=x
            rotations=basis@Rotation.from_euler('XYZ',a).as_matrix()@basis.swapaxes(-2,-1)
            global_r=np.eye(3);out=[];end=self.rest_j_np[chain[0]]-self.rest_j_np[0]
            for r,d,length in zip(rotations,rest,lengths):
                global_r=global_r@r;out.append(global_r@d);end=end+global_r@length
            directional=np.asarray(out)-expected
            if tip_target is None:return directional.ravel()
            return np.concatenate(((root@end-tip_target)/.003,.03*directional.ravel()))
        initial=np.clip(angles[free],lower[free],upper[free])
        result=least_squares(lambda x:np.concatenate((evaluate(x),.03*(x-initial))),initial,
                             bounds=(lower[free],upper[free]),max_nfev=8,ftol=1e-4,xtol=1e-4,gtol=1e-4)
        if np.sum(evaluate(result.x)**2)<=np.sum(evaluate(initial)**2)+1e-10:
            a=angles.copy();a[free]=result.x
            pose=pose.copy();pose[12:15]=Rotation.from_matrix(basis@Rotation.from_euler('XYZ',a).as_matrix()@basis.swapaxes(-2,-1)).as_rotvec()
        return pose.astype(np.float32)

    def install_flat_reference(self,reference):
        """Initialize EMF extrinsics/orientation from a confirmed flat sample.

        Translation along the finger remains a PRIOR, not a measured length.
        Real sensor mounting is shared across all poses in multi-action fitting.
        """
        if reference.get('hand_side')!='left' or reference.get('pose')!='flat_open' or reference.get('observation_space')!='raw_emf_sensor':raise ValueError('参考必须明确确认为左手平摊原始 EMF 位姿')
        observed=np.asarray(reference['positions_local'],dtype=float);rotations=np.asarray(reference['rotations_local'],dtype=float).reshape(5,3,3)
        if observed.shape!=(5,3) or not np.isfinite(observed).all() or not np.isfinite(rotations).all():raise ValueError('平摊参考数据无效')
        if not np.allclose(rotations.swapaxes(-2,-1)@rotations,np.eye(3),atol=1e-3) or np.min(np.linalg.det(rotations))<.99:raise ValueError('平摊参考方向无效')
        # EMF origins are behind the anatomical surface tips. Eight millimetres
        # is only an explicit starting prior, never written as measured length.
        offsets=self.tip_anchors-self.rest_j_np[[c[-1] for c in CHAINS]]
        prior=-.008*offsets/np.linalg.norm(offsets,axis=1,keepdims=True)
        zeros=np.zeros((15,3));basis=self.anatomy.basis
        roots=np.array([c[0] for c in CHAINS]);local=self.tip_anchors-self.rest_j_np[0]+prior
        initial_r,initial_t=rigid_alignment(local[1:],observed[1:])
        x0=np.r_[Rotation.from_matrix(initial_r).as_rotvec(),initial_t,np.zeros(4)]
        def local_points(x):
            points=local.copy()
            for f in range(1,5):
                joint=CHAINS[f][0];rot=Rotation.from_rotvec(basis[joint-1,:,1]*x[5+f]).as_matrix()
                base=self.rest_j_np[joint]-self.rest_j_np[0];points[f]=base+rot@(local[f]-base)
            return points
        def residual(x):
            r=Rotation.from_rotvec(x[:3]).as_matrix();p=local_points(x)[1:]@r.T+x[3:6]
            return np.r_[((p-observed[1:])/.005).ravel(),.1*x[6:]]
        result=least_squares(residual,x0,bounds=(np.r_[[-np.inf]*6,[-.48]*4],np.r_[[np.inf]*6,[.48]*4]),max_nfev=35)
        r=Rotation.from_rotvec(result.x[:3]).as_matrix();translation=result.x[3:6]
        raw_rotation=r.T;raw_translation=-r.T@translation
        rms=float(np.sqrt(np.mean(np.sum((local_points(result.x)[1:]@r.T+translation-observed[1:])**2,axis=1))))
        if rms>.025 or np.linalg.norm(translation)>.25:raise ValueError('平摊参考与当前尺寸差异过大，保留旧参考')
        for f in range(1,5):zeros[CHAINS[f][0]-1]=basis[CHAINS[f][0]-1,:,1]*result.x[5+f]
        # Thumb abduction is unspecified in a flat-palm sample; retain its
        # positional seed, without forcing a straight or closed thumb.
        bones=skeleton_targets((self.last_frame or {}).get('skeleton'))
        if bones is not None:zeros=self.fit_thumb_directions(zeros,np.eye(3),bones,tip_target=observed[0]@raw_rotation.T+raw_translation,sensor_offset=prior[0])
        with torch.no_grad():_,_,rr,_=self.decode(torch.tensor(zeros[None],device=self.device,dtype=torch.float32),torch.eye(3,device=self.device)[None])
        distal=rr[0,[c[-1] for c in CHAINS]].cpu().numpy()
        self.raw_rotation=raw_rotation;self.raw_translation=raw_translation
        self.mount_pos=prior.astype(np.float32);self.sensor_joint_position=(offsets+prior).astype(np.float32);self.mount_rot=distal.swapaxes(-2,-1)@raw_rotation@rotations
        self.reference_translation=np.zeros(3);self.reference_root=np.eye(3,dtype=np.float32);self.flat_reference=reference.copy();self.flat_seed=zeros.astype(np.float32)
        self.flat_reference['flat_seed']=zeros.tolist();self.flat_reference['initial_sensor_prior_mm']=8.;self.flat_reference['registration_rmse_mm']=rms*1000
        self.tracking_time=None;self.raw_aligned=True;self.has_wrist_targets=True;self.mounting_calibrated=False
        return {'registration_rmse_mm':rms*1000,'sensor_position_status':'prior_requires_multi_pose_fit','changes_shape':False}

    def sensor_offsets(self):
        """Distal-joint to EMF origin, in the distal bone frame, metres."""
        if self.flat_reference is not None and self.sensor_joint_position is not None:return self.sensor_joint_position.copy()
        return self.tip_anchors-self.rest_j_np[[c[-1] for c in CHAINS]]+self.mount_pos

    def project_sensor_mount_(self,mount):
        """Known attachment: EMF is behind the tip, on the distal segment."""
        with torch.no_grad():
            vectors=self.tip_anchors-self.rest_j_np[[c[-1] for c in CHAINS]]
            lengths=np.linalg.norm(vectors,axis=1)
            axis=torch.as_tensor(vectors/lengths[:,None],device=mount.device,dtype=mount.dtype)
            along=(mount*axis).sum(-1,keepdim=True);cross=mount-along*axis
            cross*=torch.clamp(.015/torch.clamp(torch.linalg.norm(cross,dim=-1,keepdim=True),min=1e-9),max=1)
            lower=torch.as_tensor(-np.minimum(.035,lengths-.002)[:,None],device=mount.device,dtype=mount.dtype)
            mount.copy_(cross+torch.maximum(lower,torch.clamp(along,max=-.001))*axis)

    def project_sensor_joint_mount_(self,mount):
        reference=torch.tensor(self.tip_anchors-self.rest_j_np[[c[-1] for c in CHAINS]],device=mount.device,dtype=mount.dtype)
        with torch.no_grad():mount.sub_(reference)
        self.project_sensor_mount_(mount)
        with torch.no_grad():mount.add_(reference)

    def fit_fingertip_positions(self,pose,root,observed,previous=None,orientations=None):
        """Bounded current-frame IK for the four fingers, without velocity caps.

        The SDK pose can be a poor seed. One tiny Adam step per display frame
        otherwise makes the model slowly chase even an immediately available
        6DoF observation. A warm, small CPU solve initializes each finger from
        the current target; GPU skinning then refines the actual surface.
        """
        out=pose.copy();all_angles=self.anatomy.angles(pose)
        warm=self.anatomy.angles(previous) if previous is not None else all_angles
        chains=enumerate(CHAINS) if self.flat_reference is not None else enumerate(CHAINS[1:],1)
        for f,chain in chains:
            ids=np.array(chain)-1;basis=self.anatomy.basis[ids]
            lower=self.anatomy.lower[ids].copy();upper=self.anatomy.upper[ids]
            if self.flat_reference is not None and f:lower[0,0]=max(lower[0,0],np.deg2rad(-3))
            free=upper-lower>1e-8
            ends=np.concatenate((self.rest_j_np[list(chain[1:])],self.tip_anchors[f:f+1]))
            vectors=ends-self.rest_j_np[list(chain)]
            if self.flat_reference is not None:vectors[-1]=self.sensor_offsets()[f]
            else:vectors[-1]+=self.mount_pos[f]
            base=self.rest_j_np[chain[0]]-self.rest_j_np[0]
            destination=observed[f]@root
            desired_rotation=root.T@orientations[f]@self.mount_rot[f].T if self.flat_reference is not None and orientations is not None else None
            def endpoint(a):
                rotations=basis@Rotation.from_euler('XYZ',a).as_matrix()@basis.swapaxes(-2,-1)
                r=np.eye(3);point=base.copy()
                for rotation,vector in zip(rotations,vectors):r=r@rotation;point=point+r@vector
                return point+self.tracking_tip_offset[f],r
            options=[np.clip(all_angles[ids],lower,upper),np.clip(warm[ids],lower,upper)]
            initial=(options[1] if previous is not None else options[0]).copy()
            def residual(x):
                a=initial.copy();a[free]=x
                point,rotation=endpoint(a)
                direction=Rotation.from_matrix(rotation.T@desired_rotation).as_rotvec()/.12 if desired_rotation is not None else np.zeros(0)
                scale=.012 if self.flat_reference is not None and not self.mounting_calibrated else .003
                coupling=np.array([.4*(a[2,0]-.65*a[1,0])]) if self.flat_reference is not None and f else np.zeros(0)
                return np.concatenate(((point-destination)/scale,direction,coupling,.15*(x-initial[free])))
            if previous is not None and np.linalg.norm(endpoint(initial)[0]-destination)<.00015 and (desired_rotation is None or Rotation.from_matrix(endpoint(initial)[1].T@desired_rotation).magnitude()<.005):
                angles=initial
            else:
                solved=least_squares(residual,initial[free],bounds=(lower[free],upper[free]),max_nfev=8,ftol=1e-4,xtol=1e-4,gtol=1e-4)
                angles=initial.copy()
                if np.linalg.norm(residual(solved.x))<=np.linalg.norm(residual(initial[free])):angles[free]=solved.x
            out[ids]=Rotation.from_matrix(basis@Rotation.from_euler('XYZ',angles).as_matrix()@basis.swapaxes(-2,-1)).as_rotvec()
        return out.astype(np.float32)

    def fit_sensor_supported_pinch(self,pose,root,observed,orientations,frame):
        """Soft kinematic pad constraint; no forces, collision response or resizing.

        02 never consumes this result. Raw tactile agreement only supplies a
        self-contact hint to the independent pose solver.
        """
        self.retarget_contacts=[]
        contact=frame.get('contact') or {}
        if not contact.get('ready') or not self.mapping:return pose
        patches={p['id']:p for p in contact.get('patches',[])}
        if not patches.get('thumb_distal',{}).get('contact'):return pose
        pairs=[f for f in range(1,5) if patches.get(FINGERS[f]+'_distal',{}).get('contact') and np.linalg.norm(observed[f]-observed[0])<.045]
        if not pairs:return pose
        f=min(pairs,key=lambda f:np.linalg.norm(observed[f]-observed[0]));fingers=(0,f)
        all_angles=self.anatomy.angles(pose);ids=np.array([j-1 for finger in fingers for j in CHAINS[finger]])
        initial=all_angles[ids].copy();lower=self.anatomy.lower[ids].copy();upper=self.anatomy.upper[ids]
        lower[3,0]=max(lower[3,0],np.deg2rad(-3));free=upper-lower>1e-8
        initial=np.clip(initial,lower,upper);basis=self.anatomy.basis[ids]
        pad_offsets=[];pad_normals=[]
        for finger in fingers:
            chain=CHAINS[finger];axis=self.tip_anchors[finger]-self.rest_j_np[chain[-1]];axis/=np.linalg.norm(axis)
            target=self.tip_anchors[finger]-.005*axis
            faces=self.binding_faces[PATCH_IDS.index(FINGERS[finger]+'_distal')]
            centers=self.rest_np[self.faces[faces]].mean(1);face=faces[np.argmin(np.linalg.norm(centers-target,axis=1))]
            tri=self.rest_np[self.faces[face]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
            pad_offsets.append(tri.mean(0)-self.rest_j_np[chain[-1]]);pad_normals.append(normal)
        def forward(angles):
            local=basis@Rotation.from_euler('XYZ',angles).as_matrix()@basis.swapaxes(-2,-1);out=[]
            for k,finger in enumerate(fingers):
                chain=CHAINS[finger];r=np.eye(3);point=self.rest_j_np[chain[0]]-self.rest_j_np[0]
                for level,joint in enumerate(chain):
                    r=r@local[3*k+level]
                    if level<2:point=point+r@(self.rest_j_np[chain[level+1]]-self.rest_j_np[joint])
                out.append((root@(point+r@self.sensor_offsets()[finger]),root@r,root@(point+r@pad_offsets[k]),root@r@pad_normals[k]))
            return out
        start=forward(initial)
        if start[0][3]@start[1][3]>.5:return pose  # Same-facing flat pads are not a pinch.
        initial_gap=np.linalg.norm(start[0][2]-start[1][2])
        def residual(x):
            a=initial.copy();a[free]=x;out=forward(a);res=[]
            for k,finger in enumerate(fingers):
                sensor,r,pad,normal=out[k];res.extend((sensor-observed[finger])/(.006 if self.mounting_calibrated else .015))
                res.extend(Rotation.from_matrix((r@self.mount_rot[finger]).T@orientations[finger]).as_rotvec()/.25)
            res.extend((out[0][2]-out[1][2])/.004)
            res.extend(.15*(x-initial[free]));return np.asarray(res)
        result=least_squares(residual,initial[free],bounds=(lower[free],upper[free]),max_nfev=10,ftol=1e-4,xtol=1e-4,gtol=1e-4)
        solved=initial.copy();solved[free]=result.x;after=forward(solved);gap=np.linalg.norm(after[0][2]-after[1][2])
        accepted=np.sum(residual(result.x)**2)<np.sum(residual(initial[free])**2) and gap<initial_gap
        if accepted:
            pose=pose.copy();pose[ids]=Rotation.from_matrix(basis@Rotation.from_euler('XYZ',solved).as_matrix()@basis.swapaxes(-2,-1)).as_rotvec()
        self.retarget_contacts=[{'finger_a':'thumb','finger_b':FINGERS[f],'sensor_supported':True,'applied':bool(accepted),'pad_gap_before_mm':float(initial_gap*1000),'pad_gap_mm':float((gap if accepted else initial_gap)*1000),'force_feedback':False,'geometry_changed':False}]
        return pose.astype(np.float32)

    def fit_palm_correction(self,pose,root,observed):
        """Re-estimate a bad initial palm only after current finger IK.

        Do not explain one finger's transient solver error as a palm rotation.
        Multiple unresolved endpoints can indicate an incorrect shared frame.
        """
        local=Rotation.from_rotvec(pose).as_matrix();points=[]
        for f,chain in enumerate(CHAINS):
            ends=np.concatenate((self.rest_j_np[list(chain[1:])],self.tip_anchors[f:f+1]))
            vectors=ends-self.rest_j_np[list(chain)]
            if self.flat_reference is not None:vectors[-1]=self.sensor_offsets()[f]
            else:vectors[-1]+=self.mount_pos[f]
            r=np.eye(3);point=self.rest_j_np[chain[0]]-self.rest_j_np[0]
            for joint,vector in zip(chain,vectors):r=r@local[joint-1];point=point+r@vector
            points.append(point+self.tracking_tip_offset[f])
        points=np.asarray(points)
        if np.count_nonzero(np.linalg.norm(points@root.T-observed,axis=1)>.010)<3:return root
        u,_,vt=np.linalg.svd(points.T@observed);candidate=vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
        correction=Rotation.from_matrix(root.T@candidate).as_rotvec();angle=np.linalg.norm(correction)
        correction*=min(1.,np.deg2rad(10)/max(angle,1e-12))
        return (root@Rotation.from_rotvec(correction).as_matrix()).astype(np.float32)

    def constrain_pose_(self,pose):
        with torch.no_grad():pose.copy_(torch.as_tensor(self.anatomy.project(pose.detach().cpu().numpy()),device=self.device))

    def constrain_thumb_reference_(self,pose,seed):
        """Uncalibrated sensor offsets must not be absorbed as opposition roll."""
        limit=np.deg2rad(20. if self.mounting_calibrated else 8.)
        value=pose.detach().cpu().numpy();reference=np.asarray(seed)[12:15]
        for _ in range(3):
            current=Rotation.from_rotvec(value[0,12:15]).as_matrix()
            relative=Rotation.from_matrix(reference.swapaxes(-2,-1)@current).as_rotvec()
            relative*=np.minimum(1.,limit/np.maximum(np.linalg.norm(relative,axis=1,keepdims=True),1e-12))
            value[0,12:15]=Rotation.from_matrix(reference@Rotation.from_rotvec(relative).as_matrix()).as_rotvec()
            value=self.anatomy.project(value)
        with torch.no_grad():pose.copy_(torch.as_tensor(value,device=self.device))

    def observations(self,frame):
        p=np.array([frame['fingers'][f]['position'] for f in FINGERS],np.float32)
        r=Rotation.from_quat([frame['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
        target=skeleton_targets(frame.get('skeleton'))
        if self.bone_reference is None and target is not None:
            self.bone_reference=np.array([[np.linalg.norm(target[2+4*f+k]-target[1+4*f+k]) for k in range(3)] for f in range(5)])
            if np.any(self.bone_reference<.003) or np.any(self.bone_reference>.12):
                self.bone_reference=None
                raise ValueError('骨架参考长度无效，等待有效标定输出')
            self.set_shape(self.shape)
        if self.flat_reference is not None:
            return p@self.raw_rotation.T+self.raw_translation,self.raw_rotation@r,target
        mapped=frame.get('skeleton_targets')
        if mapped is not None:
            if (mapped.get('coordinate_frame')!='wrist_local' or mapped.get('hand_side')!='left' or
                abs(mapped.get('timestamp_ms',0)-frame['timestamp_ms'])>25):raise ValueError('官方指尖目标的坐标或时间不匹配')
            positions=np.array(mapped.get('positions_local'),dtype=np.float32)
            rotations=np.array(mapped.get('rotations_local'),dtype=np.float32)
            if positions.shape!=(5,3) or rotations.shape!=(5,9) or not np.isfinite(positions).all() or not np.isfinite(rotations).all():raise ValueError('官方指尖目标无效')
            rotations=rotations.reshape(5,3,3)
            if not np.allclose(rotations.swapaxes(-2,-1)@rotations,np.eye(3),atol=1e-3) or np.any(np.linalg.det(rotations)<.99):raise ValueError('官方指尖目标旋转不是 SO(3)')
            if not self.has_wrist_targets:
                self.tracking_time=None;self.raw_aligned=False;self.has_wrist_targets=True
                self.raw_rotation=np.eye(3);self.raw_translation=np.zeros(3)
            if not self.raw_aligned and target is not None:
                pose,root=self.pose_from_bones(target)
                with torch.no_grad():_,_,rr,_=self.decode(torch.tensor(pose[None],device=self.device),torch.tensor(root[None],device=self.device))
                self.mount_rot=rr[0,[c[-1] for c in CHAINS]].cpu().numpy().transpose(0,2,1)@rotations
                self.raw_aligned=True
            return positions-self.reference_translation,rotations,target
        if self.has_wrist_targets:raise ValueError('缺少官方腕部坐标指尖目标，暂停更新')
        if not self.raw_aligned and target is not None:
            self.raw_rotation,self.raw_translation=rigid_alignment(p,target[[4,8,12,16,20]])
            self.raw_aligned=True
            # Initial installation rotations are an offset, not anatomical axes.
            pose,root=self.pose_from_bones(target)
            with torch.no_grad():_,_,rr,_=self.decode(torch.tensor(pose[None],device=self.device),torch.tensor(root[None],device=self.device))
            dr=rr[0,[c[-1] for c in CHAINS]].cpu().numpy()
            self.mount_rot=dr.transpose(0,2,1)@self.raw_rotation@r
        return p@self.raw_rotation.T+self.raw_translation,self.raw_rotation@r,target

    def track(self,frame,steps=2):
        observed,orientations,target=self.observations(frame);self.retarget_contacts=[]
        if not self.raw_aligned:raise ValueError('首次显示手部模型需要新鲜官方骨架作为坐标初始化')
        stamp=frame['timestamp_ms'];dt=(stamp-self.tracking_time)/1000 if self.tracking_time is not None else 0.
        reset=self.tracking_time is None or dt<=0 or dt>.25 or self.tracking_version!=self.shape_version
        if self.flat_reference is not None:
            bone_pose=self.flat_seed.copy() if reset else self.last_pose.copy();root=np.eye(3,dtype=np.float32)
            seed=Rotation.from_rotvec(bone_pose).as_matrix()
        elif target is not None:
            bone_pose,root=self.pose_from_bones(target)
            seed=Rotation.from_rotvec(bone_pose).as_matrix()
        else:
            seed=self.tracking_seed if self.tracking_seed is not None else Rotation.from_rotvec(self.last_pose).as_matrix()
            root=self.last_root.copy()
        bone_root=root.copy()
        if self.has_wrist_targets:
            if not reset and self.tracking_observed is not None:
                tip_speed=np.max(np.linalg.norm(observed-self.tracking_observed,axis=1))/max(dt,.001)
                self.tracking_stable_ms=self.tracking_stable_ms+dt*1000 if tip_speed<.015 else 0.
            else:self.tracking_stable_ms=0.
            if reset:
                with torch.no_grad():initial_vertices,_,_,initial_points=self.decode(torch.tensor(Rotation.from_matrix(seed).as_rotvec()[None],dtype=torch.float32,device=self.device),torch.eye(3,device=self.device)[None])
                local=initial_points[0,[4,8,12,16,20]].cpu().numpy()
                u,_,vt=np.linalg.svd(local.T@observed)
                candidate=vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
                root=self.reference_root.copy() if self.flat_reference is not None else candidate.astype(np.float32)
                self.tracking_tip_offset=initial_vertices[0,list(TIPS)].cpu().numpy()-local
            else:
                # Keep the fitted palm frame across finger articulation. A new
                # Procrustes fit to curled tips mistakes flexion for palm roll.
                root=self.last_root.copy()
                if self.flat_reference is None and self.tracking_bone_root is not None:root=bone_root@self.tracking_bone_root.T@root
            if target is not None and self.flat_reference is None:
                thumb_pose=self.fit_thumb_directions(Rotation.from_matrix(seed).as_rotvec().astype(np.float32),root,target,tip_target=observed[0])
                seed=Rotation.from_rotvec(thumb_pose).as_matrix()
            current=self.fit_fingertip_positions(Rotation.from_matrix(seed).as_rotvec().astype(np.float32),root,observed,previous=None if reset else self.last_pose,orientations=orientations)
            if self.flat_reference is not None:current=self.fit_sensor_supported_pinch(current,root,observed,orientations,frame)
            if self.flat_reference is None and self.tracking_stable_ms>=120:root=self.fit_palm_correction(current,root,observed)
            seed=Rotation.from_rotvec(current).as_matrix()
        if not reset:
            # Current-frame IK follows motion; a short adaptive SO(3) filter
            # damps measurement noise. Legacy replay retains its old correction.
            speed=np.max(Rotation.from_matrix(np.swapaxes(self.tracking_seed,-2,-1)@seed).magnitude())/max(dt,.001)
            tau=(.012/(1+speed/2.)) if self.has_wrist_targets else (.045/(1+speed/3.))
            seed=blend_rotations(self.tracking_seed,seed,-math.expm1(-dt/tau))
            root_speed=Rotation.from_matrix(self.last_root.T@root).magnitude()/max(dt,.001)
            root_tau=.010/(1+root_speed/2.) if self.has_wrist_targets else .035
            root=blend_rotations(self.last_root,root,-math.expm1(-dt/root_tau))
            correction=np.tile(np.eye(3),(15,1,1)) if self.has_wrist_targets else np.swapaxes(self.tracking_seed,-2,-1)@Rotation.from_rotvec(self.last_pose).as_matrix()
            initial=self.anatomy.project(Rotation.from_matrix(seed@correction).as_rotvec())
            with torch.no_grad():self.tracking_q.copy_(torch.tensor(initial[None],device=self.device))
        else:
            initial=Rotation.from_matrix(seed).as_rotvec().astype(np.float32)
            self.tracking_q=torch.tensor(initial[None],device=self.device,requires_grad=True)
            self.tracking_root=torch.tensor(Rotation.from_matrix(root).as_rotvec()[None],device=self.device,dtype=torch.float32,requires_grad=True)
            parameters=[{'params':[self.tracking_q],'lr':.004 if self.has_wrist_targets else .012}]
            if self.has_wrist_targets:parameters.append({'params':[self.tracking_root],'lr':0. if self.flat_reference is not None else .002})
            self.tracking_optimizer=torch.optim.SGD(parameters) if self.has_wrist_targets else torch.optim.Adam(parameters)
        q=self.tracking_q;optimizer=self.tracking_optimizer
        if self.has_wrist_targets:
            with torch.no_grad():self.tracking_root.copy_(torch.tensor(Rotation.from_matrix(root).as_rotvec()[None],device=self.device,dtype=torch.float32))
        root_t=torch.tensor(root[None],device=self.device,dtype=torch.float32)
        p_t=torch.tensor(observed[None],device=self.device);r_t=torch.tensor(orientations[None],dtype=torch.float32,device=self.device)
        mount=torch.tensor(self.mount_pos,device=self.device,dtype=torch.float32);mr=torch.tensor(self.mount_rot,device=self.device,dtype=torch.float32)
        seed_t=torch.tensor(seed[None],device=self.device,dtype=torch.float32)
        for _ in range(steps):
            if self.has_wrist_targets:root_t=rotmat(self.tracking_root)
            optimizer.zero_grad();v,j,rr,ordered=self.decode(q,root_t)
            dr=rr[:,[c[-1] for c in CHAINS]]
            offsets=torch.tensor(self.sensor_offsets(),device=self.device,dtype=torch.float32)
            pred=j[:,[c[-1] for c in CHAINS]]+(dr@offsets[None,:,:,None]).squeeze(-1) if self.flat_reference is not None else v[:,list(TIPS)]+(dr@mount[None,:,:,None]).squeeze(-1)
            position_scale=.012 if self.flat_reference is not None and not self.mounting_calibrated else .003
            loss=torch.nn.functional.smooth_l1_loss((pred-p_t)/position_scale,torch.zeros_like(pred))
            prior_weight=.01 if self.has_wrist_targets else .2
            loss=loss+(.5 if self.flat_reference is not None else .012)*((dr@mr-r_t)**2).mean()+prior_weight*(((rotmat(q)-seed_t)/.3)**2).mean()
            if target is not None and self.flat_reference is None:
                direction=ordered[:,[4,8,12,16,20]]-ordered[:,[3,7,11,15,19]]
                expected=torch.tensor(target[[4,8,12,16,20]]-target[[3,7,11,15,19]],device=self.device)
                loss=loss+(.001 if self.has_wrist_targets else .03)*((torch.nn.functional.normalize(direction,dim=-1)-torch.nn.functional.normalize(expected,dim=-1))**2).mean()
            loss.backward();optimizer.step();self.constrain_pose_(q)
            if not self.has_wrist_targets:self.constrain_thumb_reference_(q,seed)
        with torch.no_grad():
            if self.has_wrist_targets:root_t=rotmat(self.tracking_root)
            v,j,rr,ordered=self.decode(q,root_t)
        root=root_t.detach().cpu().numpy()[0]
        self.tracking_tip_offset=np.zeros((5,3)) if self.flat_reference is not None else (v[0,list(TIPS)].cpu().numpy()-ordered[0,[4,8,12,16,20]].cpu().numpy())@root
        self.last_pose=q.detach().cpu().numpy()[0];self.last_root=root.astype(np.float32)
        self.tracking_seed=seed;self.tracking_time=stamp;self.tracking_version=self.shape_version;self.tracking_bone_root=bone_root;self.tracking_observed=observed.copy()
        self.last_frame=frame
        dr_np=rr[0,[c[-1] for c in CHAINS]].cpu().numpy()
        self.sensor_predictions=j[0,[c[-1] for c in CHAINS]].cpu().numpy()+(dr_np@self.sensor_offsets()[:,:,None]).squeeze(-1) if self.flat_reference is not None else v[0,list(TIPS)].cpu().numpy()+(dr_np@self.mount_pos[:,:,None]).squeeze(-1)
        errors=np.linalg.norm(self.sensor_predictions-observed,axis=1)*1000
        self.alignment_quality={'observation_space':'raw_emf_sensor' if self.flat_reference is not None else 'sdk_wrist_targets' if self.has_wrist_targets else 'legacy_rigid_fit',
                                'tip_errors_mm':dict(zip(FINGERS,map(float,errors))),
                                'sensor_errors_mm':dict(zip(FINGERS,map(float,errors))),'orientation_errors_deg':dict(zip(FINGERS,map(float,np.rad2deg(Rotation.from_matrix((dr_np@self.mount_rot).swapaxes(-2,-1)@orientations).magnitude())))),'calibration_fingerprint':frame.get('calibration_fingerprint'),
                                'tracking_mode':'responsive_current_frame_ik','joint_smoothing_max_ms':12 if self.has_wrist_targets else 45,'root_smoothing_max_ms':10 if self.has_wrist_targets else 35,'flat_reference_id':self.flat_reference.get('reference_id') if self.flat_reference else None,'retarget_contacts':self.retarget_contacts,'geometry_fixed':self.dimension_parameters is not None,'error_landmark':'emf_sensor_origin','sensor_mounting_status':'multi_pose_calibrated' if self.mounting_calibrated else 'orientation_reference_position_prior' if self.flat_reference else 'unregistered'}
        self.thumb_quality={'axes':'left_thumb_incoming_v1','source':'raw_emf_sensor' if self.flat_reference is not None else 'sdk_wrist_target' if self.has_wrist_targets else 'sdk_reference','mounting_calibrated':self.mounting_calibrated,'max_joint_correction_deg':None if self.has_wrist_targets else (20. if self.mounting_calibrated else 8.),'mapped_target_gap_mm':None if self.flat_reference is not None else float(errors[0]),'sensor_origin_gap_mm':float(errors[0]) if self.flat_reference is not None else None}
        if target is not None and self.flat_reference is None:
            points=ordered[0,1:5].detach().cpu().numpy()
            actual=np.diff(points,axis=0);expected=np.diff(target[1:5],axis=0)
            cosine=(actual*expected).sum(1)/np.maximum(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1),1e-12)
            self.thumb_quality.update(max_direction_gap_deg=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))).max()),tip_gap_mm=float(np.linalg.norm(points[-1]-target[4])*1000))
        return v[0].cpu().numpy()+self.reference_translation,j[0].cpu().numpy()+self.reference_translation,rr[0].cpu().numpy(),ordered[0].cpu().numpy()+self.reference_translation,float(np.sqrt(np.mean(errors**2)))


class CollisionSurface:
    """Native MuJoCo Warp candidate generation, actual MANO patch BVH verification."""
    def __init__(self,surface):
        self.surface=surface;self.version=surface.shape_version;self.device=str(surface.device)
        root=ET.Element('mujoco');opt=ET.SubElement(root,'option',gravity='0 0 0')
        ET.SubElement(opt,'flag',multiccd='disable')
        asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');contact=ET.SubElement(root,'contact')
        for joint in range(16):
            points=surface.rest_np[surface.labels==joint]-surface.rest_j_np[joint]
            hull=trimesh.convex.convex_hull(points)
            ET.SubElement(asset,'mesh',name=f'm{joint}',vertex=' '.join(map(str,hull.vertices.ravel())),face=' '.join(map(str,hull.faces.ravel())))
            body=ET.SubElement(world,'body',name=f'b{joint}');ET.SubElement(body,'freejoint')
            ET.SubElement(body,'geom',type='mesh',mesh=f'm{joint}',margin='.01',condim='1')
            parent=surface.parents[joint]
            if parent>=0:ET.SubElement(contact,'exclude',body1=f'b{parent}',body2=f'b{joint}')
        self.cpu=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        with wp.ScopedDevice(self.device):
            self.model=mjw.put_model(self.cpu);self.data=mjw.make_data(self.cpu,nworld=1,nconmax=512,nccdmax=512,njmax=2048)
            self.vertices=wp.array(surface.rest_np,dtype=wp.vec3,device=self.device)
            self.meshes=[wp.Mesh(points=self.vertices,indices=wp.array(surface.faces[ids].ravel(),dtype=int,device=self.device)) for ids in surface.patch_faces]
            self.mesh_ids=wp.array([mesh.id for mesh in self.meshes],dtype=wp.uint64,device=self.device)
            self.patch_groups=np.array([FINGERS.index(p.split('_')[0]) if i<14 else 5 for i,p in enumerate(PATCH_IDS)],np.int32)
            self.group_ids=wp.array(self.patch_groups,dtype=int,device=self.device)
            self.solids=[];self.solid_points=[];self.cap_loops=[]
            for points,faces,loops in closed_surface_parts(surface.rest_np,surface.faces,self.patch_groups[surface.face_patches]):
                buffer=wp.array(points,dtype=wp.vec3,device=self.device)
                self.solids.append(wp.Mesh(points=buffer,indices=wp.array(faces.ravel(),dtype=int,device=self.device),support_winding_number=True))
                self.solid_points.append(buffer);self.cap_loops.append(loops)
            self.solid_ids=wp.array([mesh.id for mesh in self.solids],dtype=wp.uint64,device=self.device)
            self.inside=wp.zeros((1238,6),dtype=float,device=self.device);self.solid_distance=wp.zeros((1238,6),dtype=float,device=self.device)
            self.points=wp.zeros(1238,dtype=wp.vec3,device=self.device);self.normals=wp.zeros(1238,dtype=wp.vec3,device=self.device)
            self.gap=wp.zeros((1238,20),dtype=float,device=self.device);self.facing=wp.zeros((1238,20),dtype=float,device=self.device);self.nearest=wp.zeros((1238,20),dtype=wp.vec3,device=self.device)

    def query(self,v,j,rotations):
        s=self.surface
        if s.shape_version!=self.version:raise ValueError('禁止使用旧尺寸的碰撞结构')
        quat=Rotation.from_matrix(rotations).as_quat()[:,[3,0,1,2]]
        q=np.concatenate((j,quat),axis=1).astype(np.float32).reshape(1,-1)
        wp.to_torch(self.data.qpos).copy_(torch.as_tensor(q,device=s.device));self.data.overflow.zero_()
        mjw.kinematics(self.model,self.data);mjw.collision(self.model,self.data)
        count=int(self.data.nacon.numpy()[0]);overflow=bool(np.any(self.data.overflow.numpy()))
        geoms=self.data.contact.geom.numpy()[:count];native_gap=self.data.contact.dist.numpy()[:count]
        pairs=set(tuple(sorted(map(int,p))) for p in geoms)
        wp.to_torch(self.vertices).copy_(torch.as_tensor(v,device=s.device))
        for mesh in self.meshes:mesh.refit()
        for mesh,buffer,loops in zip(self.solids,self.solid_points,self.cap_loops):
            closed_points=np.concatenate((v,np.array([v[loop].mean(0) for loop in loops]).reshape(-1,3)))
            wp.to_torch(buffer).copy_(torch.as_tensor(closed_points,device=s.device));mesh.refit()
        tris=v[s.faces[s.sid_faces]];points=(tris*s.sid_bary[:,:,None]).sum(1)
        normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        # Patch geometry is independent of a provisional SID interpolation.
        # Full-surface samples prevent initial palmar UV guesses from hiding a
        # real side/edge contact. Taxel positions remain separate observations.
        points=np.concatenate((points,v));normals=np.concatenate((normals,vertex_normals(v,s.faces)))
        wp.to_torch(self.points).copy_(torch.as_tensor(points,device=s.device));wp.to_torch(self.normals).copy_(torch.as_tensor(normals,device=s.device))
        wp.launch(inside_queries,(1238,6),inputs=[self.solid_ids,self.points,self.inside,self.solid_distance],device=self.device)
        wp.launch(triangle_queries,(1238,20),inputs=[self.mesh_ids,self.points,self.group_ids,self.inside,self.solid_distance,self.normals,self.gap,self.facing,self.nearest],device=self.device)
        return {'native_pairs':pairs,'native_count':count,'overflow':overflow,'native_min_gap':float(np.min(native_gap)) if count else 0.,
                'gap':self.gap.numpy(),'facing':self.facing.numpy(),'nearest':self.nearest.numpy(),'points':points,'normals':normals}


def fuse_contacts(s,frame,geometry,guided=None):
    ready=bool(frame.get('contact',{}).get('ready') and not geometry['overflow'])
    assignments=np.array(s.mapping['assignments']);patches=frame.get('contact',{}).get('patches',[])
    touch=[bool(p.get('contact')) for p in patches] if len(patches)==20 else [False]*20
    states=['unknown' if not ready else 'none']*20;records=[];confirmed=set(s.mapping.get('confirmed',[]))
    # Movement prior is inferred from joint geometry, independently of ADC.
    target=skeleton_targets(frame.get('skeleton'))
    folded=set()
    if target is not None:
        for f in range(1,5):
            a,b,c=target[1+4*f:4+4*f];x=b-a;y=c-b
            if np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-8)<.8:folded.add(f)
    rank={'unknown':0,'none':1,'conflict':2,'candidate':3,'confirmed':4}
    for a in range(20):
        for b in range(a+1,20):
            fa=FINGERS.index(PATCH_IDS[a].split('_')[0]) if a<14 else -1
            fb=FINGERS.index(PATCH_IDS[b].split('_')[0]) if b<14 else -1
            if fa==fb or fa==fb==-1:continue
            owner=tuple(sorted((s.patch_owners[a],s.patch_owners[b])))
            native=owner in geometry['native_pairs']
            cells_a=np.flatnonzero(assignments==a);cells_b=np.flatnonzero(assignments==b)
            ia=np.concatenate((cells_a,460+np.flatnonzero(s.vertex_patches==a)));ib=np.concatenate((cells_b,460+np.flatnonzero(s.vertex_patches==b)))
            # Check both surfaces, permitting inward skin crossings, not hull-only overlap.
            ga=geometry['gap'][ia,b];gb=geometry['gap'][ib,a]
            faces_a=geometry['facing'][ia,b];faces_b=geometry['facing'][ib,a]
            valid_a=faces_a<-.15;valid_b=faces_b<-.15
            candidates=[(float(abs(ga[k])),float(ga[k]),a,int(ia[k]),b) for k in np.flatnonzero(valid_a)]
            candidates += [(float(abs(gb[k])),float(gb[k]),b,int(ib[k]),a) for k in np.flatnonzero(valid_b)]
            if candidates:
                # Penetrating samples have priority over separated nearby samples.
                best=min(candidates,key=lambda z:(z[1]>=0,z[0]));_,gap,source,sid,dest=best
                facing=float(geometry['facing'][sid,dest]);point=geometry['points'][sid];other=geometry['nearest'][sid,dest]
            else:gap=1.;facing=1.;source=a;dest=b;sid=int(ia[0]);point=geometry['points'][sid];other=point
            directed_pinch=guided and guided.startswith('pinch_') and {fa,fb}=={0,FINGERS.index(guided[6:])}
            allowed=bool(directed_pinch or fa==0 or fb==0 or abs(fa-fb)==1 and min(fa,fb)>=0 or fa in folded and fb==-1 or fb in folded and fa==-1)
            if (fa==-1 or fb==-1) and guided in ('fist','hook'):allowed=True
            state=fusion_state(native,gap,facing,allowed,touch[a],touch[b],ready,bool(directed_pinch))
            # Only confirmed anatomical assignments can become hard fitting constraints.
            verified=PATCH_IDS[a] in confirmed and PATCH_IDS[b] in confirmed
            if state in ('confirmed','candidate'):
                rec={'patch_a':PATCH_IDS[a],'patch_b':PATCH_IDS[b],'state':state,'gap_m':gap,'verified':verified,
                     'point_a_m':point.tolist() if source==a else other.tolist(),'point_b_m':other.tolist() if source==a else point.tolist(),
                     'normal_a':geometry['normals'][sid].tolist() if source==a else (-geometry['normals'][sid]).tolist(),
                     'normal_b':(-geometry['normals'][sid]).tolist() if source==a else geometry['normals'][sid].tolist()}
                aa=float(s.cell_area[cells_a].sum())*max(2,int(patches[a].get('active_cells',0)) if len(patches)==20 else 2)/len(cells_a)
                ab=float(s.cell_area[cells_b].sum())*max(2,int(patches[b].get('active_cells',0)) if len(patches)==20 else 2)/len(cells_b)
                mechanical=pair_mechanics(gap,aa,ab,s.material)
                # Never expose N as if the uncalibrated ADC measured force.
                rec.update({k:v for k,v in mechanical.items() if k!='model_force_n'})
                rec['relative_model_load']=mechanical['model_force_n']
                records.append(rec)
                for p in (a,b):
                    if rank[state]>rank[states[p]]:states[p]=state
    # A tactile patch with no admissible pair is a conflict, not a geometric contact.
    for p in range(20):
        if ready and touch[p] and states[p]=='none':states[p]='conflict'
    return {'ready':ready,'status':'fused' if ready else 'unknown','pairs':records,
            'patches':[{**p,'state':states[i],'contact':states[i]=='confirmed' if ready else None,'value':int(states[i]=='confirmed') if ready else None,
                        'response':patches[i].get('response',0) if len(patches)==20 else 0} for i,p in enumerate(PATCHES)],
            'count':sum(x=='confirmed' for x in states) if ready else None,'mapping_version':s.mapping['version'],
            'note':'几何、动作和双侧触觉融合；单侧证据仅为候选。'}


def deform_surface(s,v,fusion,timestamp_ms,enabled=True):
    if not enabled:
        s.compression[:]=0;s.last_time=None
        return v.copy(),{'estimated':False,'enabled':False,'source':'disabled','max_m':0.,'force_calibrated':False,'affects_joints':False}
    desired=np.zeros_like(v)
    for r in fusion['pairs'] if fusion['ready'] else []:
        if r['state']!='confirmed':continue
        for side in ('a','b'):
            p=PATCH_IDS.index(r['patch_'+side]);center=np.array(r['point_'+side+'_m']);normal=np.array(r['normal_'+side])
            distance=np.linalg.norm(v-center,axis=1);weight=np.maximum(1-(distance/.008)**2,0.)**2
            weight*=np.isin(s.vertex_patches,np.unique(s.vertex_patches[s.faces[s.patch_faces[p]].ravel()]))
            disp=-normal[None]*r['compression_'+side+'_m']*weight[:,None]
            use=np.linalg.norm(disp,axis=1)>np.linalg.norm(desired,axis=1);desired[use]=disp[use]
    dt=.04 if s.last_time is None else max(0.,min(.25,(timestamp_ms-s.last_time)/1000))
    alpha=-math.expm1(-dt/.04)
    if s.last_time is not None and timestamp_ms-s.last_time>250:s.compression[:]=0
    s.compression+=(desired-s.compression)*alpha;s.last_time=timestamp_ms
    lengths=np.linalg.norm(s.compression,axis=1);s.compression*=np.minimum(1,s.material['max_compression_m']/np.maximum(lengths,1e-9))[:,None]
    s.compression[s.wrist_mask]=0
    return v+s.compression,{'estimated':True,'enabled':True,'source':'geometry_and_material_prior','max_m':float(np.linalg.norm(s.compression,axis=1).max()),'force_calibrated':False,'affects_joints':False}

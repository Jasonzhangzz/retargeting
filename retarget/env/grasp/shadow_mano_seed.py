"""Anatomical rigid-hand pose transfer for initialization only.

The two IK fits use corresponding point positions. No closure, collision, force,
normal, or posture penalty is added to GraspADMM or to this initialization fit.
"""
from dataclasses import dataclass
import numpy as np
import mujoco
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

FINGERS=('index','middle','ring','pinky','thumb')
SHADOW_CHAINS=tuple(tuple('rh_'+p+s for s in ('proximal','middle','distal')) for p in ('ff','mf','rf','lf','th'))
MANO_CHAINS=tuple((f+'1x',f+'2',f+'3') if f!='thumb' else ('thumb1z','thumb2','thumb3') for f in FINGERS)
BODY_MAP={a:b for ac,bc in zip(SHADOW_CHAINS,MANO_CHAINS) for a,b in zip(ac,bc)}
BODY_MAP.update(rh_palm='rh_palm',rh_lfmetacarpal='rh_palm')


def unit(v):
    length=np.linalg.norm(v,axis=-1,keepdims=True)
    if np.any(length<1e-10):raise ValueError('Degenerate anatomical axis')
    return v/length


def palm_frame(points,palm):
    x=unit(points[0]-points[3]);y=points[1]-palm;y=unit(y-x*np.dot(x,y))
    return np.stack((x,y,np.cross(x,y)),axis=1)


def bone_frame(axis,palm_x):
    z=unit(axis);x=unit(palm_x-z*np.dot(z,palm_x))
    return np.stack((x,np.cross(z,x),z),axis=1)


class RigidHand:
    def __init__(self,xml,chains):
        self.model=mujoco.MjModel.from_xml_path(str(xml));self.data=mujoco.MjData(self.model);self.chains=chains
        self.ids=np.array([[self.model.body(b).id for b in c] for c in chains]);self.palm=self.model.body('rh_palm').id
        self.low=np.full(self.model.nq,-np.inf);self.high=np.full(self.model.nq,np.inf)
        for j in range(self.model.njnt):
            if self.model.jnt_limited[j]:
                a=self.model.jnt_qposadr[j];self.low[a],self.high[a]=self.model.jnt_range[j]
        self.data.qpos[:]=np.clip(np.zeros(self.model.nq),self.low,self.high);mujoco.mj_kinematics(self.model,self.data)
        self.meshes={};self.tips=[]
        for chain,ids in zip(chains,self.ids):
            bid=ids[-1];mesh=self.mesh(chain[-1]);direction=unit((self.data.xpos[bid]-self.data.xpos[ids[-2]])@self.data.xmat[bid].reshape(3,3))
            scores=mesh.vertices@direction;selected=scores>=np.quantile(scores,.9)
            self.tips.append(mesh.vertices[selected].mean(0))
        self.tips=np.array(self.tips)
        self.neutral=self.points_local();self.neutral_palm=self.data.xpos[self.palm].copy()
        self.neutral_frame=palm_frame(self.neutral[:,0],self.neutral_palm)
        self.lengths=np.linalg.norm(np.diff(self.neutral,axis=1),axis=-1)

    def mesh(self,body):
        if body in self.meshes:return self.meshes[body]
        bid=self.model.body(body).id;parts=[]
        for g in np.flatnonzero(self.model.geom_bodyid==bid):
            if not self.model.geom_contype[g]:continue
            kind=int(self.model.geom_type[g]);size=self.model.geom_size[g]
            if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
                mid=self.model.geom_dataid[g];a,n=self.model.mesh_vertadr[mid],self.model.mesh_vertnum[mid];b,k=self.model.mesh_faceadr[mid],self.model.mesh_facenum[mid]
                part=trimesh.Trimesh(self.model.mesh_vert[a:a+n],self.model.mesh_face[b:b+k],process=False).convex_hull
            elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                part=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[24,24])
            elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):part=trimesh.creation.box(extents=2*size)
            elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):part=trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=48)
            elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):part=trimesh.creation.icosphere(subdivisions=3,radius=size[0])
            else:raise ValueError('Unsupported collision geometry for initialization: '+body)
            r=Rotation.from_quat(self.model.geom_quat[g],scalar_first=True).as_matrix()
            part.vertices=part.vertices@r.T+self.model.geom_pos[g];parts.append(part)
        if not parts:raise ValueError('No collision mesh for '+body)
        mesh=trimesh.util.concatenate(parts);self.meshes[body]=mesh
        return mesh

    def set(self,joints):
        self.data.qpos[:]=joints;mujoco.mj_kinematics(self.model,self.data)

    def points_local(self):
        points=np.empty((5,4,3));points[:,:3]=self.data.xpos[self.ids]
        for f,bid in enumerate(self.ids[:,-1]):points[f,3]=self.data.xpos[bid]+self.data.xmat[bid].reshape(3,3)@self.tips[f]
        return points

    def world_points(self,q):
        self.set(q[7:]);r=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
        return self.points_local()@r.T+q[:3]

    def positions(self,q,bodies,local):
        self.set(q[7:]);root=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix();points=[];normals=[]
        for body,p in zip(bodies,local):
            bid=self.model.body(body).id;rot=self.data.xmat[bid].reshape(3,3)
            points.append((p@rot.T+self.data.xpos[bid])@root.T+q[:3])
        return np.asarray(points)

    def contacts(self,q,bodies,local):
        points=self.positions(q,bodies,local);root=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix();normals=[]
        for body,p in zip(bodies,local):
            bid=self.model.body(body).id;rot=self.data.xmat[bid].reshape(3,3)
            _,_,n=self.project(body,np.asarray(p)[None]);normals.append(n[0]@rot.T@root.T)
        return np.c_[points,normals]

    def project(self,body,points):
        mesh=self.mesh(body);closest,distance,face=trimesh.proximity.closest_point(mesh,np.asarray(points))
        bary=trimesh.triangles.points_to_barycentric(mesh.triangles[face],closest)
        normal=(mesh.vertex_normals[mesh.faces[face]]*bary[...,None]).sum(1)
        return closest,distance,unit(normal)


@dataclass(frozen=True)
class TransferSettings:
    bone_max_evaluations:int=80
    contact_max_evaluations:int=80
    root_translation_bound_m:float=.1
    root_rotation_bound_rad:float=1.


class ShadowManoTransfer:
    def __init__(self,shadow_xml,mano_xml,settings=TransferSettings()):
        self.shadow=RigidHand(shadow_xml,SHADOW_CHAINS);self.mano=RigidHand(mano_xml,MANO_CHAINS);self.settings=settings

    def map_contacts(self,bodies,source_local):
        s,m=self.shadow,self.mano;out=[]
        for body,p in zip(bodies,source_local):
            if body not in BODY_MAP:raise ValueError('No anatomical contact correspondence for '+body)
            target=BODY_MAP[body]
            s.set(np.clip(np.zeros(s.model.nq),s.low,s.high));m.set(np.clip(np.zeros(m.model.nq),m.low,m.high))
            sbid=s.model.body(body).id;mbid=m.model.body(target).id
            sr=s.data.xmat[sbid].reshape(3,3);mr=m.data.xmat[mbid].reshape(3,3)
            if body in ('rh_palm','rh_lfmetacarpal'):
                sp=s.neutral_palm;mp=m.neutral_palm;sf=s.neutral_frame;mf=m.neutral_frame
                source_extent=s.mesh('rh_palm').vertices@s.data.xmat[s.palm].reshape(3,3).T+s.neutral_palm
                sv=(source_extent-sp)@sf
            else:
                f,k=next((i,j) for i,c in enumerate(s.chains) for j,b in enumerate(c) if b==body)
                sp=s.neutral[f,k];mp=m.neutral[f,k]
                sf=bone_frame(s.neutral[f,k+1]-sp,s.neutral_frame[:,0]);mf=bone_frame(m.neutral[f,k+1]-mp,m.neutral_frame[:,0])
                sv=(s.mesh(body).vertices@sr.T+s.data.xpos[sbid]-sp)@sf
            mv=(m.mesh(target).vertices@mr.T+m.data.xpos[mbid]-mp)@mf
            scale=np.maximum(np.ptp(mv,axis=0),1e-6)/np.maximum(np.ptp(sv,axis=0),1e-6)
            value=((np.asarray(p)@sr.T+s.data.xpos[sbid]-sp)@sf)*scale
            desired=(mp+mf@value-m.data.xpos[mbid])@mr
            projected,_,_=m.project(target,desired[None]);out.append(projected[0])
        return [BODY_MAP[b] for b in bodies],np.asarray(out)

    def fit(self,source_q,bodies,source_local):
        source_q=np.asarray(source_q,dtype=float);s,m=self.shadow,self.mano;settings=self.settings
        if not np.isfinite(source_q).all():raise ValueError('Nonfinite source pose')
        source_points=s.world_points(source_q)
        sr=Rotation.from_quat(source_q[3:7],scalar_first=True).as_matrix()
        source_palm=s.data.xpos[s.palm]@sr.T+source_q[:3]
        frame=palm_frame(source_points[:,0],source_palm)
        rotation=frame@m.neutral_frame.T;translation=source_palm-rotation@m.neutral_palm
        qroot=np.r_[translation,Rotation.from_matrix(rotation).as_quat(scalar_first=True)]
        base=m.model.key_qpos[0].copy() if m.model.nkey else np.zeros(m.model.nq)
        base=np.clip(base,m.low+1e-7,m.high-1e-7)
        m.set(np.clip(np.zeros(m.model.nq),m.low,m.high));anchors=m.neutral[:,0]@rotation.T+translation
        directions=unit(np.diff(source_points,axis=1));target=np.empty((5,4,3));target[:,0]=anchors
        for k in range(3):target[:,k+1]=target[:,k]+m.lengths[:,k,None]*directions[:,k]
        def bone_residual(joints):return (m.world_points(np.r_[qroot,joints])[:,1:]-target[:,1:]).ravel()
        initial_bone=bone_residual(base)
        shape=least_squares(bone_residual,base,bounds=(m.low+1e-8,m.high-1e-8),max_nfev=settings.bone_max_evaluations,x_scale=.3,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        mapped_bodies,local=self.map_contacts(bodies,source_local)
        # Re-evaluate the posed source after neutral anatomical mapping.
        source_contacts=s.contacts(source_q,bodies,source_local)
        def decode(x):
            rot=Rotation.from_rotvec(x[3:6])*Rotation.from_matrix(rotation)
            return np.r_[translation+x[:3],rot.as_quat(scalar_first=True),x[6:]]
        def residual(x):return (m.positions(decode(x),mapped_bodies,local)-source_contacts[:,:3]).ravel()
        start=np.r_[np.zeros(6),shape.x];low=np.r_[[-settings.root_translation_bound_m]*3,[-settings.root_rotation_bound_rad]*3,m.low+1e-8];high=np.r_[[settings.root_translation_bound_m]*3,[settings.root_rotation_bound_rad]*3,m.high-1e-8]
        before=residual(start)
        fitted=least_squares(residual,start,bounds=(low,high),max_nfev=settings.contact_max_evaluations,x_scale=np.r_[[.02]*3,[.3]*23],ftol=1e-7,xtol=1e-7,gtol=1e-7)
        q=decode(fitted.x);contacts=m.contacts(q,mapped_bodies,local);error=np.linalg.norm(contacts[:,:3]-source_contacts[:,:3],axis=1)
        if not np.isfinite(q).all() or not np.isfinite(contacts).all():raise FloatingPointError('Nonfinite mapped state')
        info=dict(bone_evaluations=shape.nfev,contact_evaluations=fitted.nfev,
            bone_initial_rms_m=float(np.sqrt(np.mean(initial_bone**2))),bone_final_rms_m=float(np.sqrt(np.mean(shape.fun**2))),
            contact_before_rms_m=float(np.sqrt(np.mean(before**2))),contact_after_rms_m=float(np.sqrt(np.mean(fitted.fun**2))),
            contact_errors_m=error.tolist(),max_contact_error_m=float(error.max()),contact_fit_within_2mm=bool(np.all(error<=.002)),
            mapping_finite=True,source_grasp_qpos=source_q.tolist(),mapped_grasp_qpos=q.tolist(),source_hand_cbody=bodies,mapped_hand_cbody=mapped_bodies,
            source_hand_cpn_w=source_contacts.tolist(),mapped_hand_cpn_b=np.c_[local,(contacts[:,3:]@Rotation.from_quat(q[3:7],scalar_first=True).as_matrix())].tolist(),
            note='Corresponding-point IK initialization only; source feasibility is not assumed to survive a change of hand model')
        # Local normals must use the current per-link frame, not just the wrist.
        m.set(q[7:]);rootrot=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
        info['mapped_hand_cpn_b']=np.array([np.r_[p,n@rootrot@m.data.xmat[m.model.body(b).id].reshape(3,3)] for p,n,b in zip(local,contacts[:,3:],mapped_bodies)]).tolist()
        return q,mapped_bodies,contacts,info

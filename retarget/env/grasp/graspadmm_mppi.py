"""Local MPPI pose refinement of an immutable GraspADMM output.

The sampler is the existing SPIDER MPPI implementation. This is a local
geometry objective, not a claim to reproduce GraspADMM's optimization cost.
Acceptance remains exclusively graspadmm_native_v1 in the pinned interpreter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from retarget.solver.base import Settings
from retarget.solver.spider_mppi import optimize
from retarget.env.geometry import ObjectMeshSurface
from .audit import GraspAudit
from .hand import HAND_BODIES, hand_tree, query_path
from .spider_physics import BatchedPhysics


@dataclass(frozen=True)
class RefinementSettings:
    population: int = 256
    worlds: int = 64
    iterations: int = 8
    stages: int = 2
    points_per_body: int = 128
    pos_noise: float = .0015
    rot_noise: float = .015
    joint_noise: float = .04
    final_stage_noise: float = .25
    max_pos_delta: float = .01
    max_rot_delta: float = .15
    max_joint_delta: float = .35
    target_gap_m: float = .0007
    contact_scale_m: float = .002
    penetration_scale_m: float = .001
    penetration_weight: float = 1000.
    collision_clearance_m: float = .00005
    collision_margin_m: float = .0001
    contact_weight: float = 1.
    normal_weight: float = .1
    pose_weight: float = .01
    wrench_weight: float = .1
    temperature: float = 1.
    nconmax: int = 256

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError(f"refinement.{key} must be finite and positive")
        for key in ("population","worlds","iterations","stages","points_per_body","nconmax"):
            if not isinstance(getattr(self,key),int):
                raise ValueError(f"refinement.{key} must be integer")
        if self.population<20 or self.points_per_body<8 or self.target_gap_m>=.002:
            raise ValueError("Need population>=20, points_per_body>=8 and target gap below native 2mm contact limit")
        if self.collision_margin_m<=self.collision_clearance_m:
            raise ValueError('Collision detection margin must exceed clearance guidance')


@wp.kernel(enable_backward=False)
def _depths(count:wp.array[int], worlds:wp.array[int], geoms:wp.array[wp.vec2i],
            distance:wp.array[float], object_geom:wp.array[int], clearance:float, depths:wp.array2d[float]):
    i=wp.tid()
    if i>=count[0]:return
    pair=geoms[i]
    if pair[0]<0 or pair[1]<0:return
    a=object_geom[pair[0]];b=object_geom[pair[1]]
    if a!=0 and b!=0:return
    kind=int(1)
    if a!=b:kind=0
    wp.atomic_max(depths,worlds[i],kind,wp.max(clearance-distance[i],0.))


def native_to_local(grasp):
    q=np.asarray(grasp,float).reshape(27)
    if not np.isfinite(q).all() or np.linalg.norm(q[3:7])<1e-12:
        raise ValueError("Need a finite native 27-coordinate grasp")
    return np.r_[q[:3],Rotation.from_quat(q[3:7],scalar_first=True).as_euler("XYZ"),q[7:]]


def local_to_native(q):
    q=np.asarray(q,float).reshape(26)
    return np.r_[q[:3],Rotation.from_euler("XYZ",q[3:6]).as_quat(scalar_first=True),q[6:]]


def refinement_model(mesh_path,settings=None):
    """Same official hand and actual object-local mesh; no recentering/scaling."""
    tree=hand_tree("wuji2")
    settings=settings or RefinementSettings()
    tree.find("option").set("gravity","0 0 0")
    # MJWarp does not support positive margins with MULTICCD. The single-CCD
    # distance is only optimization guidance; the pinned original evaluation
    # reconstructs its own model and retains its original collision settings.
    ET.SubElement(tree.find('option'),'flag',multiccd='disable')
    for geom in tree.iter("geom"):
        if geom.get("group")=="1":
            geom.set("contype","0");geom.set("conaffinity","0")
        else:
            geom.set("contype","1");geom.set("conaffinity","1");geom.set("margin",str(settings.collision_margin_m))
            geom.set("condim","1")
    ET.SubElement(tree.find("asset"),"mesh",name="refinement_object",file=str(Path(mesh_path).resolve()))
    obj=ET.SubElement(tree.find("worldbody"),"body",name="grasp_object")
    ET.SubElement(obj,"geom",type="mesh",mesh="refinement_object",contype="2",conaffinity="1",
                  margin=str(settings.collision_margin_m),condim="1",rgba=".25 .65 .85 1")
    return ET.tostring(tree,encoding="unicode")


class LocalPoseObjective:
    """Fixed object, native batched collisions, real surface distance guidance."""
    def __init__(self,mesh_path,settings,device="cuda:0"):
        self.settings=settings;self.device=device;self.W=settings.worlds
        self.xml=refinement_model(mesh_path,settings)
        self.cpu=mujoco.MjModel.from_xml_string(self.xml)
        if (self.cpu.nq,self.cpu.nv,self.cpu.nu)!=(26,26,26):
            raise ValueError("Expected fixed object and 26-coordinate WUJI2")
        self.physics=BatchedPhysics(self.cpu,self.W,device,nconmax=settings.nconmax,capture=False)
        self.audit=GraspAudit(ObjectMeshSurface.from_model(self.cpu,self.cpu.body('grasp_object').id),device)
        bodies,local,normals=[],[],[]
        self.slices={}
        for name in HAND_BODIES:
            with np.load(query_path(self.cpu,name),allow_pickle=False) as raw:
                # Deterministic subsampling plus coordinate extrema preserves
                # pads/edges without changing the actual native collision test.
                p=raw['local_pos'];n=raw['local_normal']
                ids=np.unique(np.r_[np.linspace(0,len(p)-1,min(settings.points_per_body,len(p))).astype(int),
                                    p.argmin(0),p.argmax(0)])
                start=len(bodies);bodies.extend([self.cpu.body(name).id]*len(ids))
                local.extend(p[ids]);normals.extend(n[ids]);self.slices[name]=slice(start,len(bodies))
        self.bodies=torch.tensor(bodies,device=device,dtype=torch.long)
        self.local=torch.tensor(np.asarray(local),device=device,dtype=torch.float32)
        self.normals=torch.tensor(np.asarray(normals),device=device,dtype=torch.float32)
        joint_ids=[np.flatnonzero(self.cpu.jnt_qposadr==i)[0] for i in range(26)]
        low,high=self.cpu.jnt_range[joint_ids].T
        # Float32 roundoff at an exact float64 native limit must not introduce
        # artificial limit violations when the candidate is exported again.
        low=low.copy();high=high.copy();low[6:]+=1.e-6;high[6:]-=1.e-6
        self.low=torch.tensor(low,device=device,dtype=torch.float32)
        self.high=torch.tensor(high,device=device,dtype=torch.float32)
        self.radius=max(self.audit.radius,.005)
        from retarget.task.dex.cost import grasp_wrench_directions
        self.directions=torch.tensor(grasp_wrench_directions(),device=device)
        self.object_flags=wp.array((self.cpu.geom_bodyid==self.cpu.body('grasp_object').id).astype(np.int32),dtype=int,device=device)
        self.depth=wp.zeros((self.W,2),dtype=float,device=device)
        self.zeros=torch.zeros((self.W,26),device=device)
        self.trace=[];self.parts={};self.reference=None
        # Kinematics + collision is sufficient: refinement does not integrate
        # dynamics or solve fictitious object controls during scoring.
        neutral=torch.tensor(self.cpu.qpos0,device=device,dtype=torch.float32)
        self.physics.reset(neutral,self.zeros,control=neutral)
        mjw.kinematics(self.physics.model,self.physics.data)
        mjw.collision(self.physics.model,self.physics.data)
        wp.synchronize_device(device)

    def bind(self,grasp,required_cbody):
        reference=native_to_local(grasp)
        self.reference=torch.tensor(reference,device=self.device,dtype=torch.float32)
        s=self.settings
        radius=torch.tensor([s.max_pos_delta]*3+[s.max_rot_delta]*3+[s.max_joint_delta]*20,device=self.device)
        self.qlo=torch.maximum(self.low,self.reference-radius)
        self.qhi=torch.minimum(self.high,self.reference+radius)
        if bool((self.qlo>self.qhi).any()):raise ValueError('ADMM candidate too far beyond physical joint limits')
        self.groups=[]
        for group in required_cbody:
            indices=[]
            for name in group:
                if name not in self.slices:raise ValueError(f'Unknown required body {name}')
                sl=self.slices[name];indices.extend(range(sl.start,sl.stop))
            if not indices:raise ValueError('Empty required contact group')
            self.groups.append(torch.tensor(indices,device=self.device,dtype=torch.long))
        if not self.groups:raise ValueError('Refinement requires explicit original contact groups')
        self.trace=[];self.parts={}
        return reference

    def project(self,q):return torch.maximum(torch.minimum(q,self.qhi),self.qlo)

    def evaluate(self,controls):
        values=[]
        with wp.ScopedDevice(self.device),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            for start in range(0,len(controls),self.W):
                q=self.project(controls[start:start+self.W,0]);n=len(q)
                if n<self.W:q=torch.cat((q,q[:1].expand(self.W-n,-1)))
                self.physics.reset(q,self.zeros,control=q)
                mjw.kinematics(self.physics.model,self.physics.data)
                mjw.collision(self.physics.model,self.physics.data)
                self.depth.zero_()
                wp.launch(_depths,self.physics.data.naconmax,inputs=[self.physics.data.nacon,
                    self.physics.data.contact.worldid,self.physics.data.contact.geom,self.physics.data.contact.dist,
                    self.object_flags,self.settings.collision_clearance_m,self.depth],device=self.device)
                world=self.physics.xpos[:,self.bodies]+torch.einsum('bcij,cj->bci',self.physics.xmat[:,self.bodies],self.local)
                normal_hand=torch.einsum('bcij,cj->bci',self.physics.xmat[:,self.bodies],self.normals)
                projected,normal_object,phi=self.audit.query(world.flatten(0,1))
                phi=phi.reshape(self.W,-1);projected=projected.reshape_as(world);normal_object=normal_object.reshape_as(world)
                facing=-(normal_hand*normal_object).sum(-1)
                s=self.settings
                score=((phi-s.target_gap_m)/s.contact_scale_m).square()+s.normal_weight*(1-facing).square()
                contact=torch.zeros(self.W,device=self.device)
                for ids in self.groups:contact+=score[:,ids].amin(1)
                # One candidate contact per rigid body is a smooth guidance
                # proxy. Native filtering/QP and all six directions decide pass.
                selected=[]
                for sl in self.slices.values():selected.append(score[:,sl].argmin(1)+sl.start)
                selected=torch.stack(selected,1)
                gather=selected[:,:,None].expand(-1,-1,3)
                point=projected.gather(1,gather);inward=-normal_object.gather(1,gather)
                gap=phi.gather(1,selected);face=facing.gather(1,selected)
                basis=torch.nn.functional.one_hot(inward.abs().argmin(-1),3).to(inward.dtype)
                tangent=torch.nn.functional.normalize(torch.cross(inward,basis,dim=-1),dim=-1)
                second=torch.cross(inward,tangent,dim=-1)
                tangents=torch.stack((tangent,second,-tangent,-second),2)
                force=(inward[:,:,None]+.6*tangents)/(1+.6**2)**.5
                torque=torch.cross(point[:,:,None].expand_as(force),force,dim=-1)/self.radius
                rays=torch.cat((force,torque),-1).flatten(1,2)
                active=torch.sigmoid((.002-gap.abs())/.0004)*torch.sigmoid((face-.25)*8)
                support=(rays@self.directions.T).clamp_min(0)*active.repeat_interleave(4,1)[:,:,None]
                deficit=torch.relu(.05-support.amax(1)).square()/(.05**2)
                wrench=deficit.amax(1)+.25*deficit.mean(1)
                penetration=(wp.to_torch(self.depth)/s.penetration_scale_m).square().sum(1)
                scale=q.new_tensor([s.max_pos_delta]*3+[s.max_rot_delta]*3+[s.max_joint_delta]*20)
                pose=((q-self.reference)/scale).square().mean(1)
                cost=s.contact_weight*contact+s.penetration_weight*penetration+s.pose_weight*pose+s.wrench_weight*wrench
                valid=(wp.to_torch(self.physics.data.overflow)==0)&torch.isfinite(cost)
                cost=torch.where(valid,cost,torch.inf)
                self.parts=dict(contact=contact[:n].clone(),penetration=penetration[:n].clone(),pose=pose[:n].clone(),wrench=wrench[:n].clone())
                values.append(cost[:n].clone())
        return torch.cat(values)

    def refine(self,grasp,required_cbody,seed):
        started=time.perf_counter();self.bind(grasp,required_cbody)
        mean=self.project(self.reference)[None].clone()
        initial_cost=float(self.evaluate(mean[None])[0])
        initial_parts={k:float(v[0]) for k,v in self.parts.items()}
        trace=[mean[0].cpu().numpy().copy()];history=[]
        s=self.settings
        solver=Settings(type='mppi',population=s.population,worlds=s.worlds,iterations=s.iterations,
            temperature=s.temperature,final_noise=.1,improvement_threshold=0.,improvement_check_steps=s.iterations+1)
        noise=mean.new_tensor([s.pos_noise]*3+[s.rot_noise]*3+[s.joint_noise]*20)
        for stage in range(s.stages):
            factor=s.final_stage_noise**stage
            mean,report=optimize(mean,self.evaluate,solver,knots=1,knot_steps=1,
                channel_noise=noise*factor,seed=int(seed)+104729*stage,
                first_noise=1.,last_noise=1.,return_best=True)
            mean=self.project(mean)
            history.append(dict(stage=stage,**report));trace.append(mean[0].cpu().numpy().copy())
        final_cost=float(self.evaluate(mean[None])[0])
        if final_cost>initial_cost+1.e-6:
            raise RuntimeError('MPPI best-candidate policy increased its own geometry objective')
        final_parts={k:float(v[0]) for k,v in self.parts.items()}
        return mean[0].cpu().numpy().astype(float),dict(
            initial_cost=initial_cost,final_cost=final_cost,initial_parts=initial_parts,final_parts=final_parts,
            optimization_seconds=time.perf_counter()-started,history=history,
            settings=asdict(s),seed=int(seed),sampled=sum(x['sampled'] for x in history),
            objective='local geometry clearance, required-body contact, sampled wrench support and ADMM pose proximity',
            sampler='existing retarget.solver.spider_mppi.optimize; top-decile weighting; no gradients',
            trajectory=np.asarray(trace),acceptance_during_optimization=False)

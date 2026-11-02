"""Fixed-object, 26-coordinate MPPI grasp acquisition before free-body MPC."""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
import time
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from retarget.common.results import write_json
from retarget.env.geometry import ObjectMeshSurface
from retarget.solver.spider_mppi import optimize
from .spider_seed import load_seed,GROUPS,finger_group
from .spider_geometry import NativeContactGeometry,ContactRequirements,exact_closure
from .spider_physics import BatchedPhysics
from .hand import query_path,HAND_BODIES
from .audit import GraspAudit


@dataclass(frozen=True)
class AcquisitionSettings:
    coarse_population:int=256
    fine_population:int=128
    iterations:int=8
    worlds:int=128
    coarse_pos_noise:float=.02
    coarse_rot_noise:float=.15
    coarse_joint_noise:float=.25
    fine_pos_noise:float=.003
    fine_rot_noise:float=.03
    fine_joint_noise:float=.08
    contact_scale_m:float=.005
    target_phi:float=-.0002
    penetration_limit_m:float=.0003
    contact_weight:float=1.
    palm_weight:float=8.
    penetration_weight:float=10.
    self_collision_weight:float=10.
    bone_weight:float=.002
    native_contact_weight:float=2.
    pose_weight:float=.02
    closure_weight:float=1.
    max_elites:int=8
    block_sweeps:int=2
    block_iterations:int=4

    def __post_init__(self):
        for name,value in asdict(self).items():
            if not np.isfinite(value) or (value<=0 and name!='target_phi'):
                raise ValueError(f'Invalid acquisition.{name}')
        if min(self.coarse_population,self.fine_population)<20 or not 1<=self.max_elites<=8:
            raise ValueError('Acquisition needs population >=20 and at most eight elites')


class PoseObjective:
    def __init__(self,source,settings,device='cuda:0'):
        self.source,self.metadata,self.seed,self.cpu=load_seed(source)
        self.settings=settings;self.device=device;self.W=settings.worlds
        self.physics=BatchedPhysics(self.cpu,self.W,device,capture=False)
        self.geometry=NativeContactGeometry(self.physics,self.seed)
        self.audit=GraspAudit(ObjectMeshSurface.from_model(self.cpu,self.cpu.body('grasp_object').id),device)
        tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,device=device,dtype=dtype)
        self.tensor=tensor
        self.initial=tensor(self.seed['qpos']);self.qzero=torch.zeros((self.W,32),device=device)
        ids=[np.flatnonzero(self.cpu.jnt_qposadr==i)[0] for i in range(26)]
        low,high=self.cpu.jnt_range[ids].T;self.low=tensor(low);self.high=tensor(high)
        bodies,local,normals,groups=[],[],[],[]
        self.pool_slices=[]
        for i,body in enumerate(self.seed['bodies']):
            group=finger_group(self.cpu,body)
            if group==5:continue
            if group==2 and 'cylinder_middle_slots' in self.seed and i not in self.seed['cylinder_middle_slots']:continue
            with np.load(query_path(self.cpu,int(body))) as raw:
                mask=self.seed['masks'][self.seed['mask_offsets'][i]:self.seed['mask_offsets'][i+1]]
                indices=np.flatnonzero(mask)
                basal=self.geometry.region['bodies']==body
                if basal.any():
                    from scipy.spatial import cKDTree
                    distance,_=cKDTree(self.geometry.region['local'][basal]).query(raw['local_pos'][indices])
                    indices=indices[distance>1e-7]
                if not len(indices):continue
                indices=indices[np.linspace(0,len(indices)-1,min(len(indices),128)).astype(int)]
                bodies.extend([body]*len(indices));local.extend(raw['local_pos'][indices]);normals.extend(raw['local_normal'][indices]);groups.extend([group]*len(indices))
        ids=np.linspace(0,len(self.seed['palm_bodies'])-1,min(len(self.seed['palm_bodies']),512)).astype(int)
        bodies.extend(self.seed['palm_bodies'][ids]);local.extend(self.seed['palm_local'][ids]);normals.extend(self.seed['palm_normals'][ids]);groups.extend([5]*len(ids))
        self.bodies=tensor(bodies,torch.long);self.local=tensor(np.asarray(local));self.normals=tensor(np.asarray(normals));self.groups=tensor(groups,torch.long)
        self.required=tensor(self.seed['required_groups'],torch.bool)
        self.group_ids={g:torch.nonzero(self.groups==g,as_tuple=True)[0] for g in range(6) if self.seed['required_groups'][g]}
        if any(len(ids)==0 for ids in self.group_ids.values()):raise ValueError('input_invalid: required contact pool is empty')
        all_bodies,all_local=[],[]
        for name in HAND_BODIES:
            body=self.cpu.body(name).id
            with np.load(query_path(self.cpu,body)) as raw:
                p=raw['local_pos'];ids=np.arange(len(p))
                all_bodies.extend([body]*len(ids));all_local.extend(p[ids])
        self.all_bodies=tensor(all_bodies,torch.long);self.all_local=tensor(np.asarray(all_local))
        from .input_geometry import WujiSelfCollision
        from .template import BoneReference
        self.capsules=WujiSelfCollision(self.cpu,device)
        self.bones=BoneReference(self.cpu,self.seed['gesture_qpos'],device)
        if self.metadata['power']:
            required=self.tensor(self.seed['required_groups'][:5],torch.bool)
            self.bones.weights=torch.where(required,self.bones.weights.clamp_max(10.),self.bones.weights)
        self.self_depth=torch.vmap(self.capsules.depths)
        self.bone_residual=torch.vmap(self.bones.style_residual)
        self.elite_q=None;self.elite_cost=None;self.trace=[];self.last_parts={}
        self.physics.reset(self.initial,self.qzero,control=self.initial[:26]);self.physics.warmup()

    def project(self,q):return torch.maximum(torch.minimum(q,self.high),self.low)

    def points(self,bodies,local):
        return self.physics.xpos[:,bodies]+torch.einsum('bcij,cj->bci',self.physics.xmat[:,bodies],local)

    def query(self,points):
        obj=self.cpu.body('grasp_object').id
        R=self.physics.xmat[:,obj];origin=self.physics.xpos[:,obj]
        local=torch.einsum('bij,bcj->bci',R.transpose(-1,-2),points-origin[:,None])
        projected,normal,phi=self.audit.query(local.flatten(0,1))
        phi=phi.reshape(local.shape[:2]);projected=projected.reshape_as(local);normal=normal.reshape_as(local)
        delta=local-projected
        gradient=torch.sign(phi)[:,:,None]*torch.nn.functional.normalize(delta,dim=-1)
        gradient=torch.where((phi.abs()<1e-7)[:,:,None],normal,gradient)
        return phi,torch.einsum('bij,bcj->bci',R,gradient),local

    def evaluate(self,controls):
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            return self._evaluate(controls)

    def _evaluate(self,controls):
        values=[]
        for start in range(0,len(controls),self.W):
            q=self.project(controls[start:start+self.W,0]);n=len(q)
            if n<self.W:q=torch.cat((q,q[:1].expand(self.W-n,-1)))
            full=self.initial.expand(self.W,-1).clone();full[:,:26]=q
            self.physics.reset(full,self.qzero,control=q)
            # Forward is a contact/FK query only: never integrate or move object.
            mjw.forward(self.physics.model,self.physics.data)
            points=self.points(self.bodies,self.local)
            phi,outward,local_obj=self.query(points)
            normals=torch.einsum('bcij,cj->bci',self.physics.xmat[:,self.bodies],self.normals)
            facing=-(normals*outward).sum(-1)
            score=((phi-self.settings.target_phi)/self.settings.contact_scale_m).square()+.25*(1-facing).square()
            # Palm samples are material-local and follow the root joints.
            if 'cylinder_middle_half_band' in self.seed:
                band=float(self.seed['cylinder_middle_half_band'])
                middle=(self.groups==2)
                score=score+middle[None]*(torch.relu(local_obj[:,:,2].abs()-band)/.005).square()*5
            contact=torch.zeros(self.W,device=self.device)
            for group,ids in self.group_ids.items():
                contact+=score[:,ids].amin(1)*(self.settings.palm_weight if group==5 else 1.)
            whole_phi,_,_=self.query(self.points(self.all_bodies,self.all_local))
            penetration=(torch.relu(-whole_phi.amin(1)-self.settings.penetration_limit_m)/.001).square()
            self_pen=(torch.relu(self.self_depth(q.double()).amax(1)-.0001)/.001).square().float()
            bones=self.bone_residual(q.double()).square().mean(1).float()
            reference=self.tensor(self.seed.get('pose_intent_qpos',self.seed['qpos'][:26]))
            pose=((q[:,:3]-reference[:3])/.015).square().mean(1)+((q[:,3:6]-reference[3:6])/.26).square().mean(1)
            self.geometry.sample(forces=False)
            actual=self.geometry.slots<wp.to_torch(self.geometry.count)[:,None]
            closure=self.geometry.wrench_cost(valid_mask=actual)
            native=torch.zeros(self.W,device=self.device)
            for group in self.group_ids:
                present=(actual&(wp.to_torch(self.geometry.groups)==group)).any(1)
                native+=(~present)*(5. if group==5 else 1.)
            s=self.settings
            cost=(s.contact_weight*contact+s.penetration_weight*penetration+s.self_collision_weight*self_pen+
                  s.bone_weight*bones+s.pose_weight*pose+s.closure_weight*closure+s.native_contact_weight*native)
            overflow=wp.to_torch(self.physics.data.overflow)
            cost=torch.where(torch.isfinite(cost)&(overflow==0),cost,torch.inf)
            self.last_parts=dict(contact=contact,penetration=penetration,self_collision=self_pen,bones=bones,pose=pose,closure=closure,native_missing=native)
            values.append(cost[:n].clone())
            candidate_q=q[:n].clone();candidate_cost=cost[:n].clone()
            if self.elite_q is not None:
                candidate_q=torch.cat((self.elite_q,candidate_q));candidate_cost=torch.cat((self.elite_cost,candidate_cost))
            ids=torch.argsort(candidate_cost)[:s.max_elites]
            self.elite_q=candidate_q[ids];self.elite_cost=candidate_cost[ids]
            self.trace.append(self.elite_q[0].clone())
        return torch.cat(values)

    def audit_candidate(self,q):
        from .hand import HAND_BODIES
        data=mujoco.MjData(self.cpu);data.qpos[:]=self.seed['qpos'];data.qpos[:26]=q
        data.ctrl[:]=q;mujoco.mj_forward(self.cpu,data)
        requirements=ContactRequirements(self.cpu,self.seed)
        observed=requirements.observe(data)
        present=np.zeros(6,bool);present[observed['groups']]=True
        points=[]
        for name in HAND_BODIES:
            body=self.cpu.body(name).id
            with np.load(query_path(self.cpu,body)) as raw:
                points.append(raw['local_pos']@data.xmat[body].reshape(3,3).T+data.xpos[body])
        obj=self.cpu.body('grasp_object').id;R=data.xmat[obj].reshape(3,3)
        local=(np.concatenate(points)-data.xpos[obj])@R
        _,_,phi=self.audit.query(self.tensor(local))
        pen=max(0.,-float(phi.min()))
        self_pen=float(self.capsules.depths(self.capsules.fk.tensor(q)).max())
        closure=exact_closure(observed,center=data.xpos[obj],radius=self.audit.radius)
        middle=True
        if 'cylinder_middle_half_band' in self.seed:
            ids=observed['groups']==2
            z=((observed['points']-data.xpos[obj])@R)[:,2]
            middle=bool(np.any(ids&(abs(z)<=float(self.seed['cylinder_middle_half_band'])+.00001)))
        contacts=bool(present[self.seed['required_groups']].all())
        passed=contacts and pen<=.0005 and self_pen<=.0002 and middle and closure['success']
        return dict(success=bool(passed),contact_established=contacts,group_names=GROUPS,
            required_groups=self.seed['required_groups'],present_groups=present,
            penetration_mm=pen*1000,self_penetration_mm=self_pen*1000,
            middle_in_center_band=middle,force_closure=closure),observed,data

    def wrist_sampling_transform(self,anchor):
        """Correlate finger noise with wrist noise to preserve current gaps.

        This changes the MPPI sampling covariance, not the accepted solution.
        Every transformed candidate still receives the full physical cost.
        Tangential sliding is free; only each guided surface's normal gap is
        compensated. No CPU optimization is invoked in the rollout evaluator.
        """
        self.evaluate(anchor[None])
        points=self.points(self.bodies,self.local)
        phi,outward,_=self.query(points)
        q=anchor[0].cpu().numpy();data=mujoco.MjData(self.cpu)
        data.qpos[:]=self.seed['qpos'];data.qpos[:26]=q;mujoco.mj_forward(self.cpu,data)
        transform=np.eye(26)
        for group,ids in self.group_ids.items():
            if group==5:continue
            chosen=int(ids[(phi[0,ids]-self.settings.target_phi).abs().argmin()])
            body=int(self.bodies[chosen]);local=self.local[chosen].cpu().numpy()
            point=data.xpos[body]+data.xmat[body].reshape(3,3)@local
            normal=outward[0,chosen].cpu().numpy()
            jp,jr=np.zeros((3,self.cpu.nv)),np.zeros((3,self.cpu.nv))
            mujoco.mj_jac(self.cpu,data,jp,jr,point,body)
            row=normal@jp[:,:26];start=6+group*4
            finger=row[start:start+4]
            denominator=float(finger@finger)+1e-7
            transform[start:start+4,:6]=-finger[:,None]*row[None,:6]/denominator
        directed=torch.zeros(26,device=self.device)
        if self.seed['required_groups'][5]:
            ids=self.group_ids[5]
            palm=self.cpu.body('right_palm_link').id;r=self.geometry.region
            inv=self.physics.xmat[0,palm].T
            chart=(points[0,ids]-self.physics.xpos[0,palm])@inv.T
            handn=torch.einsum('cij,cj->ci',self.physics.xmat[0,self.bodies[ids]],self.normals[ids])
            nn=handn@inv.T
            up=chart@self.tensor(r['up']);side=chart@self.tensor(r['side'])
            valid=(up>=-.003)&(up<=r['longitudinal'])&(side>=r['lateral'][0])&(side<=r['lateral'][1])&(nn@self.tensor(r['normal'])>.5)
            facing=-(handn*outward[0,ids]).sum(-1)
            score=(phi[0,ids]-self.settings.target_phi).abs()+.005*(1-facing)
            score=score.masked_fill(facing<.25,torch.inf)
            chosen=ids[score.argmin()]
            directed[:3]=(self.settings.target_phi-phi[0,chosen])*outward[0,chosen]
        return self.tensor(transform),directed

    def guided_contact_delta(self,anchor,group):
        """A geometry-conditioned MPPI proposal, never an accepted IK update."""
        self.evaluate(anchor[None]);points=self.points(self.bodies,self.local)
        phi,outward,_=self.query(points);ids=self.group_ids[group]
        nh=torch.einsum('cij,cj->ci',self.physics.xmat[0,self.bodies[ids]],self.normals[ids])
        score=((phi[0,ids]-self.settings.target_phi)/.005).square()+.25*(1+(nh*outward[0,ids]).sum(-1)).square()
        chosen=int(ids[score.argmin()]);body=int(self.bodies[chosen]);local=self.local[chosen].cpu().numpy()
        data=mujoco.MjData(self.cpu);data.qpos[:]=self.seed['qpos'];data.qpos[:26]=anchor[0].cpu().numpy();mujoco.mj_forward(self.cpu,data)
        point=data.xpos[body]+data.xmat[body].reshape(3,3)@local
        normal_hand=data.xmat[body].reshape(3,3)@self.normals[chosen].cpu().numpy()
        normal_object=outward[0,chosen].cpu().numpy()
        jp,jr=np.zeros((3,self.cpu.nv)),np.zeros((3,self.cpu.nv));mujoco.mj_jac(self.cpu,data,jp,jr,point,body)
        from retarget.env.load_support import skew
        start,end=(0,6) if group==5 else (6+4*group,10+4*group)
        J=np.r_[(normal_object@jp[:,:26])[None]/.005,(-skew(normal_hand)@jr[:,:26])*.5][:,start:end]
        residual=np.r_[(float(phi[0,chosen])-self.settings.target_phi)/.005,(normal_hand+normal_object)*.5]
        step=-np.linalg.pinv(J,rcond=1e-4)@residual
        limit=np.asarray([.02]*3+[.2]*3) if group==5 else np.full(4,.35)
        delta=np.zeros(26);delta[start:end]=np.clip(step,-limit,limit)
        return self.tensor(delta)


def _acquire_impl(source,destination,solver,settings=None,device='cuda:0'):
    settings=AcquisitionSettings(**(settings or {}));started=time.perf_counter()
    objective=PoseObjective(source,settings,device);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    mean=objective.initial[:26].reshape(1,26).clone();history=[]
    for label,population,scales in [('coarse',settings.coarse_population,(settings.coarse_pos_noise,settings.coarse_rot_noise,settings.coarse_joint_noise)),('fine',settings.fine_population,(settings.fine_pos_noise,settings.fine_rot_noise,settings.fine_joint_noise))]:
        noise=objective.tensor([scales[0]]*3+[scales[1]]*3+[scales[2]]*20)
        sampling=replace(solver,population=population,worlds=settings.worlds,iterations=settings.iterations,improvement_check_steps=3,improvement_threshold=1e-5)
        mean,report=optimize(mean,objective.evaluate,sampling,knots=1,knot_steps=1,channel_noise=noise,seed=0 if label=='coarse' else 1,first_noise=1.,last_noise=1.)
        mean=objective.project(mean);history.append(dict(stage=label,**report))
        mean=objective.elite_q[:1].clone()
    # Separate wrist/finger perturbations preserve contacts already formed.
    # A full 26-D perturbation otherwise breaks several sub-mm contacts while
    # trying to close just one detached finger. The optimizer is still MPPI.
    for sweep in range(settings.block_sweeps):
        for block in range(6):
            noise=torch.zeros(26,device=device)
            if block==0:
                noise[:3]=settings.fine_pos_noise;noise[3:6]=settings.fine_rot_noise
            else:noise[6+4*(block-1):6+4*block]=settings.fine_joint_noise
            noise*=.3**sweep
            sampling=replace(solver,population=settings.fine_population,worlds=settings.worlds,
                iterations=settings.block_iterations,improvement_check_steps=3,improvement_threshold=1e-5)
            evaluator=objective.evaluate
            if block==0:
                anchor=mean[0].clone();transform,directed=objective.wrist_sampling_transform(mean)
                def evaluator(controls):
                    latent=controls[:,0].clone()
                    if len(latent)>8:
                        factors=objective.tensor([.25,.5,.75,1.,1.25,1.5,-.5,-1.])
                        latent[1:9]=anchor+factors[:,None]*directed
                    physical=anchor+(latent-anchor)@transform.T
                    return objective.evaluate(physical[:,None])
            elif block-1 in objective.group_ids:
                anchor=mean[0].clone();directed=objective.guided_contact_delta(mean,block-1)
                def evaluator(controls):
                    physical=controls[:,0].clone()
                    if len(physical)>8:
                        factors=objective.tensor([.25,.5,.75,1.,1.25,1.5,-.5,-1.])
                        physical[1:9]=anchor+factors[:,None]*directed
                    return objective.evaluate(physical[:,None])
            mean,report=optimize(mean,evaluator,sampling,knots=1,knot_steps=1,channel_noise=noise,
                seed=10+sweep*6+block,first_noise=1.,last_noise=1.)
            mean=objective.elite_q[:1].clone();history.append(dict(stage=f'block_{sweep}_{block}',**report))
    candidates=torch.cat((mean,objective.elite_q)).cpu().numpy()
    audits=[];selected=None
    for index,q in enumerate(candidates):
        report,observed,data=objective.audit_candidate(q);audits.append(dict(index=index,**report))
        if report['success'] and selected is None:selected=(q,observed,data,report)
    best=selected[0] if selected is not None else candidates[0]
    best_report,best_contacts,best_data=objective.audit_candidate(best)
    body_local=[]
    for body,point in zip(best_contacts['bodies'],best_contacts['hand_points']):
        body_local.append(best_data.xmat[body].reshape(3,3).T@(point-best_data.xpos[body]))
    np.savez_compressed(destination/'contacts.npz',bodies=best_contacts['bodies'],local=np.asarray(body_local).reshape(-1,3),
        points=best_contacts['points'],groups=best_contacts['groups'])
    traces=torch.stack([objective.initial[:26],*objective.trace]).cpu().numpy()
    full=np.repeat(objective.seed['qpos'][None],len(traces),axis=0);full[:,:26]=traces
    full=np.concatenate((full,np.r_[best,objective.seed['qpos'][26:]][None]))
    np.savez_compressed(destination/'acquisition_trajectory.npz',qpos=full,time=np.arange(len(full),dtype=float))
    result=dict(success=selected is not None,ca_success=objective.metadata['ca_success'],
        ca_is_gate=False,object_pose_fixed=True,object_reference_qpos=objective.seed['qpos'][26:],
        settings=asdict(settings),iterations=history,candidates=audits,seconds=time.perf_counter()-started,
        selected_diagnostics=best_report,
        hand_qpos=best,scene=str(objective.source/objective.metadata['scene']),
        source=str(objective.source),trajectory='acquisition_trajectory.npz',time_kind='optimization_iteration')
    if selected is not None:
        from .spider_reference import save_acquired_reference
        result['reference']=str(save_acquired_reference(destination/'reference',objective,best,selected[1],selected[2]))
    write_json(destination/'acquisition.json',result)
    return result


def acquire(source,destination,solver,settings=None,device='cuda:0'):
    with torch.no_grad(),wp.ScopedDevice(device),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))):
        return _acquire_impl(source,destination,solver,settings,device)

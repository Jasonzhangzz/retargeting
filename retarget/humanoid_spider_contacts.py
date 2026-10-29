"""Humanoid adapter for run.py's sampled contact sets + DeviceLambdaLM.

SPIDER's actual box pose error replaces the hand task's motion metric. The
force witnesses stay prospective; only physical simulation moves the box.
"""
import mujoco
import numpy as np
import warp as wp

from retarget.humanoid_terms import Stage, contact_point
from retarget.spider_pose import quat_log
from retarget.humanoid_contacts import bounded_project


@wp.struct
class Batch:
    c: wp.array3d(dtype=wp.vec3)
    x: wp.array3d(dtype=wp.vec3)
    dual: wp.array3d(dtype=wp.vec3)
    weights: wp.array2d(dtype=float)
    friction: wp.array3d(dtype=wp.vec2)
    rotation: wp.array2d(dtype=wp.mat33)
    position: wp.array2d(dtype=wp.vec3)
    velocity: wp.array2d(dtype=wp.spatial_vector)
    goal: wp.array2d(dtype=wp.spatial_vector)


@wp.kernel(enable_backward=False)
def gather(job: Stage, batch: Batch, last: int):
    w,role=wp.tid();m,t,k=job.model,job.target,job.index
    ob=m.object_body;r=job.rotation[w,ob];origin=job.position[w,ob]
    batch.c[k,w,role]=t.z[k,w,role]
    batch.x[k,w,role]=wp.transpose(r)@(contact_point(job,w,role)-origin)
    batch.dual[k,w,role]=t.dual[k,w,role]
    if role==0:
        batch.position[k,w]=origin;batch.rotation[k,w]=r
        v=m.object_dof
        linear=wp.transpose(r)@wp.vec3(job.qvel[w,v],job.qvel[w,v+1],job.qvel[w,v+2])
        batch.velocity[k,w]=wp.spatial_vector(linear[0],linear[1],linear[2],job.qvel[w,v+3],job.qvel[w,v+4],job.qvel[w,v+5])
        next_k=wp.min(k+1,last);adr=m.object_qpos
        target=wp.vec3(t.qpos[next_k,adr],t.qpos[next_k,adr+1],t.qpos[next_k,adr+2])
        delta=wp.transpose(r)@(target-origin)
        desired=wp.quat(t.qpos[next_k,adr+4],t.qpos[next_k,adr+5],t.qpos[next_k,adr+6],t.qpos[next_k,adr+3])
        angular=quat_log(wp.quat_inverse(wp.quat_from_matrix(r))*desired)
        batch.goal[k,w]=wp.spatial_vector(delta[0],delta[1],delta[2],angular[0],angular[1],angular[2])


class SpiderContacts:
    def __init__(self,backend):
        self.backend=backend
        p,s=backend.problem,backend.settings
        self.lm=None
        if not p.surface or s.contact_radius<=0:return
        from retarget.contact_exploration import ObjectMeshSurface
        from retarget.contact_solvers import SampledComFreeLM,skew
        from retarget.contact_lm_warp import DeviceLambdaLM
        from retarget.humanoid_model import box_mesh
        ob=backend.model.object_body;geom=p.object_geom
        vertices,faces=box_mesh(p.model.geom_size[geom])
        rot=np.empty(9);mujoco.mju_quat2Mat(rot,p.model.geom_quat[geom]);rot=rot.reshape(3,3)
        vertices=vertices@rot.T+p.model.geom_pos[geom]
        self.explorer=SampledComFreeLM(ObjectMeshSurface(vertices,faces),
            radius=s.contact_radius,rho=s.rho,samples=s.contact_samples,
            lm_iterations=s.contact_lm_iterations,sweeps=s.contact_sweeps,
            seed=s.seed,device=backend.device,mass=float(p.model.body_mass[ob]),
            dt=backend.model.dt,wrench_weight=1.)
        inertia_rot=np.empty(9);mujoco.mju_quat2Mat(inertia_rot,p.model.body_iquat[ob]);inertia_rot=inertia_rot.reshape(3,3)
        inertia=inertia_rot@np.diag(p.model.body_inertia[ob])@inertia_rot.T
        mass=p.model.body_mass[ob];com=p.model.body_ipos[ob];cross=skew(com)
        spatial=np.block([[mass*np.eye(3),-mass*cross],[mass*cross,inertia-mass*cross@cross]])
        n,w=backend.dense_steps+1,backend.worlds
        self.batch=batch=Batch()
        for name in ('c','x','dual'):setattr(batch,name,wp.zeros((n,w,2),dtype=wp.vec3,device=backend.device))
        batch.weights=wp.zeros((n,2),device=backend.device)
        batch.friction=wp.zeros((n,w,2),dtype=wp.vec2,device=backend.device)
        batch.position=wp.zeros((n,w),dtype=wp.vec3,device=backend.device)
        batch.rotation=wp.zeros((n,w),dtype=wp.mat33,device=backend.device)
        for name in ('velocity','goal'):setattr(batch,name,wp.zeros((n,w),dtype=wp.spatial_vector,device=backend.device))
        self.lm=DeviceLambdaLM(self.explorer,groups=n*w,contacts=2,worlds=w,
            physics=dict(inverse=np.linalg.inv(spatial),com=com,gravity=p.model.opt.gravity,
                         floor=(np.array([0.,0.,p.ground_height]),np.array([0.,0.,1.])),
                         support_vertices=vertices,spider_pose_weights=[1.,1.,1.,.3,.3,.3]),device=backend.device)
        self.object_radius=float(np.linalg.norm(vertices-com,axis=1).max())

    def update(self,snapshot,z,dual):
        b,p,s=self.backend,self.backend.problem,self.backend.settings
        radii=np.array([s.contact_radius]*2+[s.foot_contact_radius]*2)
        anchor=b.anchors[:,None]
        prox=(s.rho*(snapshot['x']+dual)+s.anchor_weight*anchor)/(s.rho+s.anchor_weight)
        candidate=bounded_project(p,prox,anchor,radii)
        statistics=[]
        if self.lm is not None:
            n,w=b.dense_steps+1,b.worlds
            weights=p.mask[b.reference_indices,:2].astype(np.float32)
            self.batch.weights.assign(weights)
            mu=np.broadcast_to(np.array(p.contact_friction[:2])[None,None,:,None],(n,w,2,2))
            self.batch.friction.assign(mu.astype(np.float32))
            templates=np.broadcast_to(b.anchors[:,None,:2],(n,w,2,3)).reshape(-1,2,3)
            repeated=np.broadcast_to(weights[:,None],(n,w,2)).reshape(-1,2)
            self.lm.bind(self.batch,templates,repeated,self.object_radius)
            for stage in b.stages:wp.launch(gather,(w,2),inputs=[stage.job,self.batch,b.dense_steps],device=b.device)
            updated=self.lm.solve().numpy().reshape(n,w,2,3)
            candidate[:,:,:2]=updated
            statistics=self.lm.job.statistics.numpy()
        else:
            candidate[:,:,:2]=anchor[:,:,:2]
        candidate=np.where(p.mask[b.reference_indices,None,:,None],candidate,anchor)
        shift=np.linalg.norm(candidate-b.anchors[:,None],axis=-1)
        diag=dict(method='sampled_comfree_batched_lm' if self.lm else 'bounded_contact_projection',
                  solver_backend='retarget.contact_lm_warp.DeviceLambdaLM',
                  task_cost='SPIDER weighted object position + quaternion-log rotation norm',
                  candidate_count=s.contact_samples,lm_iterations=s.contact_lm_iterations,
                  max_hand_shift_mm=float(shift[:,:,:2].max()*1000),
                  max_foot_shift_mm=float(shift[:,:,2:].max()*1000),
                  forces_injected_into_simulator=False,task_force_witness_is_prospective=True,
                  initialized_from_current_mjwarp_object_pose_and_velocity=True)
        if len(statistics):
            active=statistics[:,4]>0
            diag.update(searched_groups=int(statistics[:,8].sum()),active_groups=int(active.sum()),
                        changed_contact_groups=int(((statistics[:,2]>0)&active).sum()),
                        mean_weighted_pose_error=float(statistics[active,9].mean()/1000) if active.any() else 0.)
        return candidate,np.zeros_like(z),np.zeros(z.shape[:2]+(3,)),np.zeros(z.shape[:2]+(3,)),diag

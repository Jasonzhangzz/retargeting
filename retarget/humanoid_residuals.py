"""Elementwise square residuals of the existing MJWarp whole-body objective.

Used only by the opt-in condensed Gauss--Newton solver. The scalar objective
and execution remain in humanoid_mjwarp; parity is independently tested.
"""
import warp as wp
from retarget.humanoid_terms import Stage, contact_point, realized_wrench, orientation_error


@wp.func
def put3(out:wp.array3d(dtype=float),k:int,w:int,i:int,v:wp.vec3,scale:float):
    for j in range(3):out[k,w,i+j]=scale*v[j]
    return i+3


@wp.kernel(enable_backward=False)
def residuals(job:Stage,out:wp.array3d(dtype=float)):
    w=wp.tid();m,t,k=job.model,job.target,job.index
    out[k,w,0]=0. # Every column below is written, including inactive contacts.
    n=wp.sqrt(job.normalizer);i=int(0)
    root=job.position[w,m.root]
    i=put3(out,k,w,i,root-t.position[k,m.root],n*wp.sqrt(m.weights[0]))
    for a in range(3):
        for b in range(3):
            out[k,w,i]=n*wp.sqrt(.5*m.weights[1])*(job.rotation[w,m.root][a,b]-t.rotation[k,m.root][a,b]);i+=1
    for j in range(m.joint_qpos.shape[0]):
        adr=m.joint_qpos[j]
        out[k,w,i]=n*wp.sqrt(m.weights[2])*(job.qpos[w,adr]-t.qpos[k,adr]);i+=1
    for j in range(m.robot_dofs.shape[0]):
        adr=m.robot_dofs[j]
        out[k,w,i]=n*wp.sqrt(m.weights[3])*(job.qvel[w,adr]-t.qvel[k,adr]);i+=1
    for j in range(m.bodies.shape[0]):
        b=m.bodies[j]
        i=put3(out,k,w,i,(job.position[w,b]-root)-(t.position[k,b]-t.position[k,m.root]),n*wp.sqrt(m.weights[4]))
        for a in range(3):
            for c in range(3):
                out[k,w,i]=n*wp.sqrt(.5*m.weights[5])*(job.rotation[w,b][a,c]-t.rotation[k,b][a,c]);i+=1
    R,Rref=job.rotation[w,m.torso],t.rotation[k,m.torso]
    i=put3(out,k,w,i,wp.vec3(R[0,2]-Rref[0,2],R[1,2]-Rref[1,2],R[2,2]-Rref[2,2]),n*wp.sqrt(m.weights[6])/m.upright_scale)
    error=job.com[w]-t.com[k];dv=wp.vec3(0.)
    if k>0:dv=(job.com[w]-job.previous_com[w]-t.com[k]+t.com[k-1])/m.dt
    dcm=error+dv/wp.sqrt(9.81/wp.max(t.com[k][2]-m.ground,.2))
    scale=n*wp.sqrt(m.weights[7])
    out[k,w,i]=scale*error[0]/m.balance_xy_scale;i+=1
    out[k,w,i]=scale*error[1]/m.balance_xy_scale;i+=1
    out[k,w,i]=scale*error[2]/m.balance_z_scale;i+=1
    out[k,w,i]=scale*wp.sqrt(.5)*dcm[0]/m.balance_xy_scale;i+=1
    out[k,w,i]=scale*wp.sqrt(.5)*dcm[1]/m.balance_xy_scale;i+=1
    out[k,w,i]=scale*.5*dv[2]/m.balance_velocity_scale;i+=1
    out[k,w,i]=scale*wp.max(t.position[k,m.root][2]-root[2]-m.balance_height_margin,0.)/m.balance_height_scale;i+=1
    # Same material-point foot slip, independent of projected z movement.
    for role in range(2,4):
        b=m.role_bodies[role];point=contact_point(job,w,role)
        local=wp.transpose(job.rotation[w,b])@(point-job.position[w,b])
        old=job.previous_position[w,b]+job.previous_rotation[w,b]@local
        slip=(point-old)/m.dt
        for j in range(2):
            out[k,w,i]=n*wp.sqrt(m.weights[8]*t.mask[k,role])*slip[j];i+=1
    for j in range(job.ctrl.shape[1]):
        out[k,w,i]=n*wp.sqrt(m.weights[9])*(job.ctrl[w,j]-t.ctrl[k,j]);i+=1
    for j in range(job.ctrl.shape[1]):
        out[k,w,i]=n*wp.sqrt(m.weights[10])*(job.ctrl[w,j]-job.previous_ctrl[w,j]);i+=1
    for j in range(job.distance.shape[1]):
        d=float(0.)
        if job.active[w,j]>0:d=wp.min(job.distance[w,j],0.)
        out[k,w,i]=n*wp.sqrt(m.weights[11])*d;i+=1
    f,tau,of,ot=wp.vec3(0.),wp.vec3(0.),wp.vec3(0.),wp.vec3(0.)
    op,orr,oc=wp.vec3(0.),wp.identity(n=3,dtype=float),wp.vec3(0.)
    if m.object_body>=0:
        op=job.position[w,m.object_body];orr=job.rotation[w,m.object_body];oc=op+orr@m.ipos[m.object_body]
    # Wrench uses exactly the fixed contact ADMM variables of the action block.
    for role in range(4):
        z=t.z[k,w,role];force=t.forces[k,w,role]
        if role<2 and m.object_body>=0:
            z=op+orr@z;of+=force;ot+=wp.cross(z-oc,force);force=-force
        f+=force;tau+=wp.cross(z-job.com[w],force)
    desired=t.wrench[k,0]+m.robot_mass*20.*(t.com[k]-job.com[w])
    desired+=m.robot_mass*4.*wp.vec3(t.qvel[k,0]-job.qvel[w,0],t.qvel[k,1]-job.qvel[w,1],t.qvel[k,2]-job.qvel[w,2])
    desired_tau=t.wrench[k,1]+20.*orientation_error(R,Rref)
    scale=wp.max(m.robot_mass*9.81,1.);oscale=wp.max(m.object_mass*9.81,1.);weight=n*wp.sqrt(m.weights[12]);wrench_start=i
    i=put3(out,k,w,i,(f-desired)/scale,weight)
    i=put3(out,k,w,i,(tau-desired_tau)/(scale*.5),weight)
    if m.object_body>=0:
        support=t.floor_force[k,w];of+=support;ot+=t.floor_torque[k,w]-wp.cross(oc,support)
        i=put3(out,k,w,i,(of-t.object_wrench[k,0])/oscale,weight)
        i=put3(out,k,w,i,(ot-t.object_wrench[k,1])/(oscale*.3),weight)
    else:
        i=put3(out,k,w,i,wp.vec3(0.),weight);i=put3(out,k,w,i,wp.vec3(0.),weight)
    actual=realized_wrench(job,w);rw=weight*wp.sqrt(m.realization_weight)*float(job.physical_contact)
    i=put3(out,k,w,i,f/scale-wp.vec3(actual[0],actual[1],actual[2]),rw)
    i=put3(out,k,w,i,tau/(scale*.5)-wp.vec3(actual[3],actual[4],actual[5]),rw)
    if m.object_body>=0:
        i=put3(out,k,w,i,of/oscale-wp.vec3(actual[6],actual[7],actual[8]),rw)
        i=put3(out,k,w,i,ot/(oscale*.3)-wp.vec3(actual[9],actual[10],actual[11]),rw)
    else:
        i=put3(out,k,w,i,wp.vec3(0.),rw);i=put3(out,k,w,i,wp.vec3(0.),rw)
    raw=float(0.)
    if m.weights[12]>0.:
        for j in range(wrench_start,i):raw+=out[k,w,j]*out[k,w,j]/(job.normalizer*m.weights[12])
    robust_factor=float(1.)
    if m.object_body>=0:robust_factor=wp.sqrt(2./(wp.sqrt(1.+raw/(m.wrench_robust_scale*m.wrench_robust_scale))+1.))
    for j in range(wrench_start,i):out[k,w,j]*=robust_factor
    for role in range(4):
        point=contact_point(job,w,role)
        if role<2 and m.object_body>=0:point=wp.transpose(orr)@(point-op)
        weight=n*wp.sqrt(t.penalty[0]*t.mask[k,role])*float(k>0)/m.scale
        i=put3(out,k,w,i,point-t.z[k,w,role]+t.dual[k,w,role],weight)
    for role in range(4):
        i=put3(out,k,w,i,t.z[k,w,role]-t.anchor[k,role],n*wp.sqrt(t.penalty[1]*t.mask[k,role])/m.scale)
    for role in range(2):
        b=m.role_bodies[role]
        center=job.position[w,b]+job.rotation[w,b]@m.hand_centers[role]
        center_ref=t.hand_position[k,role]
        i=put3(out,k,w,i,center-center_ref,n*wp.sqrt(m.weights[13]*t.mask[k,role])*float(m.object_body>=0))


def residual_width(backend):
    m=backend.model
    return (12+len(m.joint_qpos)+len(m.robot_dofs)+12*len(m.bodies)+3+7+4
            +2*backend.problem.model.nu+backend.stages[0].job.distance.shape[1]+24+24+6)


class Residuals:
    def __init__(self,backend):
        if backend.settings.cost_mode=='spider':
            raise ValueError('Legacy least-squares residual solvers do not represent the unsquared SPIDER norm; use AdamW')
        self.backend=backend
        self.values=wp.zeros((backend.dense_steps+1,backend.worlds,residual_width(backend)),device=backend.device)
        self.graph=None

    def evaluate(self):
        import torch
        b=self.backend
        def launch():
            for stage in b.stages:wp.launch(residuals,b.worlds,inputs=[stage.job,self.values],device=b.device)
        caller=torch.cuda.current_stream(str(b.device));b.torch_stream.wait_stream(caller)
        with torch.cuda.stream(b.torch_stream):
            if self.graph is None:
                launch()
                with wp.ScopedCapture(device=b.device) as capture:launch()
                self.graph=capture.graph
            wp.capture_launch(self.graph)
        caller.wait_stream(b.torch_stream)
        return wp.to_torch(self.values)

"""GPU Kelvin--Voigt contacts with tangential memory and Coulomb return mapping.

This is a new compliant-contact adapter, not the upstream ComFree formula.
Contact forces are independent constitutive laws; only the free object's six
velocities are solved together by backward Euler. No contact QP is executed.
"""
from dataclasses import dataclass,asdict
import math
import torch
from retarget.solver.cem import DEFAULTS
from .comfree_rollout import RolloutSettings
from scripts.calibrate_wuji_mano import rotmat


from retarget.solver.cem import SpringSettings as SpringHistorySettings


def skew(x):
    a,b,c=x.unbind(-1);z=torch.zeros_like(a)
    return torch.stack((z,-c,b,c,z,-a,-b,a,z),-1).reshape(*x.shape[:-1],3,3)


def spring_force(velocity,normal,compression,history,dt,settings,friction,*,jacobian=False):
    """Force on the object; positive normal velocity increases compression.

    history is world-frame tangential displacement of the object relative to
    the fixed hand. A clipped tangential force is returned to the elastic state
    after a step, so sliding does not accumulate an unbounded hidden spring.
    """
    s=settings;eye=torch.eye(3,device=velocity.device,dtype=velocity.dtype)
    nn=normal[..., :,None]*normal[...,None,:];projector=eye-nn
    vt=velocity-(velocity*normal).sum(-1,keepdim=True)*normal
    xi=history-(history*normal).sum(-1,keepdim=True)*normal
    an=s.normal_stiffness_n_m*dt+s.normal_damping_ns_m
    at=s.tangent_stiffness_n_m*dt+s.tangent_damping_ns_m
    fn=(s.normal_stiffness_n_m*compression+an*(velocity*normal).sum(-1)).clamp_min(0.)
    trial=-s.tangent_stiffness_n_m*xi-at*vt
    length=trial.norm(dim=-1).clamp_min(1.e-12);limit=friction*fn
    ratio=(limit/length).clamp_max(1.)
    ft=ratio[...,None]*trial
    force=-fn[...,None]*normal+ft
    if not jacobian:return force,fn,ft
    direction=trial/length[...,None]
    sticking=length<=limit
    jt_stick=-at*projector
    tangent_jac=(eye-direction[..., :,None]*direction[...,None,:])@projector
    jt_slide=friction*an*direction[..., :,None]*normal[...,None,:]-(limit/length)[...,None,None]*at*tangent_jac
    derivative=-an*nn+torch.where(sticking[...,None,None],jt_stick,jt_slide)
    derivative=torch.where((fn>0)[...,None,None],derivative,torch.zeros_like(derivative))
    return force,fn,ft,derivative


class SpringHistoryRollout:
    def __init__(self,worlds,contacts=64,*,device='cuda:0',settings=None,spring=None,backend='warp'):
        if not str(device).startswith('cuda'):raise ValueError('Spring rollout requires CUDA; no CPU fallback')
        self.worlds,self.contacts,self.device=worlds,contacts,str(device)
        self.settings=settings or DEFAULTS.rollout
        self.spring=spring or DEFAULTS.spring
        if backend not in ('warp','torch'):raise ValueError('Unknown spring execution backend')
        self.backend=backend
        if backend=='warp':
            from .spring_history_warp import integrate_fused
            self.integrate_fused=integrate_fused
        self.gravity=torch.tensor(self.settings.gravity,device=device,dtype=torch.float32)
        self.initial_angular=torch.tensor(self.settings.initial_angular_velocity,device=device,dtype=torch.float32)
        self.factors=torch.tensor([1.,.5,.25,.125,0.],device=device,dtype=torch.float32)
        self.metadata=dict(protocol='mano_kelvin_voigt_tangent_history_h5_v5',settings=asdict(self.settings),spring=asdict(self.spring),
            physical_units=True,history='Reset each candidate; integrate within rollout; transport to tangent plane; Coulomb return; clear on detachment',
            integration='Backward Euler, six-dimensional object momentum, GPU damped Newton',qp=False,full_mjwarp_dynamics=False,
            original_comfree_formula=False,strict_force_closure=False)
        self.metadata['execution_backend']=backend
        self.metadata['newton_early_exit_relative_tolerance']=.01 if backend=='warp' else 0.

    def _step(self,linear,momentum,com,rotation,local_points,local_normals,hand,history,inertia,inv_inertia,eligible):
        s=self.settings;c=self.spring;dt=s.timestep/c.substeps;b=len(com)
        p=(rotation[:,None]@local_points[...,None]).squeeze(-1)+com[:,None]
        n=(rotation[:,None]@local_normals[...,None]).squeeze(-1)
        history=history-(history*n).sum(-1,keepdim=True)*n
        compression=((p-hand)*n).sum(-1)
        arm=p-com[:,None]
        eye=torch.eye(3,device=com.device,dtype=com.dtype).expand(b,self.contacts,3,3)
        B=torch.cat((eye,-skew(arm)),-1)
        iw=rotation@inertia@rotation.transpose(-1,-2)
        ii=rotation@inv_inertia@rotation.transpose(-1,-2)
        M=com.new_zeros((b,6,6));M[:,:3,:3]=torch.eye(3,device=com.device)*s.mass;M[:,3:,3:]=iw
        rhs=torch.cat((s.mass*(linear+dt*self.gravity),momentum),-1)
        u=torch.cat((linear+dt*self.gravity,(ii@momentum[...,None]).squeeze(-1)),-1)
        factors=self.factors

        def force(u,jacobian=False):
            vp=(B@u[:,None,:,None]).squeeze(-1)
            result=spring_force(vp,n,compression,history,dt,c,s.friction,jacobian=jacobian)
            return tuple(v*eligible[...,None,None] if v.ndim==4 else v*eligible[...,None] if v.ndim==3 else v*eligible for v in result)

        def error(residual):
            return torch.cat((residual[...,:3]/s.mass,
                torch.einsum('bij,b...j->b...i',ii,residual[...,3:])),dim=-1)

        for _ in range(c.newton_iterations):
            f,fn,ft,j=force(u,True)
            wrench=(B.transpose(-1,-2)@f[...,None]).squeeze(-1).sum(1)
            residual=(M@u[...,None]).squeeze(-1)-rhs-dt*wrench
            A=M-dt*(B.transpose(-1,-2)@j@B).sum(1)
            delta,info=torch.linalg.solve_ex(A,-residual)
            delta=torch.where(((info==0)&torch.isfinite(delta).all(-1))[:,None],delta,torch.zeros_like(delta))
            trial=u[:,None]+factors[None,:,None]*delta[:,None]
            vp=torch.einsum('bcij,blj->blci',B,trial)
            ftrial,_,_=spring_force(vp,n[:,None],compression[:,None],history[:,None],dt,c,s.friction)
            ftrial*=eligible[:,None,:,None]
            wtrial=torch.einsum('bcji,blcj->bli',B,ftrial)
            rtrial=torch.einsum('bij,blj->bli',M,trial)-rhs[:,None]-dt*wtrial
            metric=error(rtrial)
            metric=torch.cat((metric[...,:3]/c.linear_residual_m_s,metric[...,3:]/c.angular_residual_rad_s),-1).square().sum(-1)
            metric=torch.where(torch.isfinite(metric),metric,torch.inf)
            choice=metric.argmin(-1)
            u=trial.gather(1,choice[:,None,None].expand(-1,1,6)).squeeze(1)
        f,fn,ft=force(u)
        wrench=(B.transpose(-1,-2)@f[...,None]).squeeze(-1).sum(1)
        residual=error((M@u[...,None]).squeeze(-1)-rhs-dt*wrench)
        converged=(residual[:,:3].norm(dim=-1)<=c.linear_residual_m_s)&(residual[:,3:].norm(dim=-1)<=c.angular_residual_rad_s)
        finite=torch.isfinite(u).all(-1)&torch.isfinite(f).all((-1,-2))
        safe=torch.nan_to_num(u,nan=0.,posinf=0.,neginf=0.)
        vp=(B@safe[:,None,:,None]).squeeze(-1);vt=vp-(vp*n).sum(-1,keepdim=True)*n
        # Includes the sticking update and the Coulomb sliding return map.
        history=-(ft+c.tangent_damping_ns_m*vt)/c.tangent_stiffness_n_m
        linear=safe[:,:3];momentum=(iw@safe[:,3:,None]).squeeze(-1)
        com=com+dt*linear
        rotation=rotmat(safe[:,3:]*dt)@rotation
        new_p=(rotation[:,None]@local_points[...,None]).squeeze(-1)+com[:,None]
        new_n=(rotation[:,None]@local_normals[...,None]).squeeze(-1)
        gap=((hand-new_p)*new_n).sum(-1)
        attached=eligible&(fn>0)&(gap<=0.)
        history=history-(history*new_n).sum(-1,keepdim=True)*new_n
        history=torch.where(attached[...,None],history,torch.zeros_like(history))
        return linear,momentum,com,rotation,history,converged&finite,fn,ft

    @torch.no_grad()
    def solve(self,hand_points,projected_points,outward_normals,signed_distance,center_world,rotation,inertia_body,*,eligible=None):
        s=self.settings;c=self.spring;b=self.worlds
        if hand_points.shape!=(b,self.contacts,3):raise ValueError('Contact packet shape mismatch')
        finite=torch.isfinite(hand_points).all((-1,-2))&torch.isfinite(projected_points).all((-1,-2))&torch.isfinite(outward_normals).all((-1,-2))
        finite &= torch.isfinite(signed_distance).all(-1)&torch.isfinite(center_world).all(-1)&torch.isfinite(rotation).all((-1,-2))&torch.isfinite(inertia_body).all((-1,-2))
        inv,info=torch.linalg.inv_ex(inertia_body)
        _,chol=torch.linalg.cholesky_ex(inertia_body)
        valid=finite&(info==0)&(chol==0)
        identity=torch.eye(3,device=center_world.device,dtype=center_world.dtype).expand(b,-1,-1)
        inertia=torch.where(valid[:,None,None],inertia_body,identity)
        inv=torch.where(valid[:,None,None],inv,identity)
        r=torch.where(valid[:,None,None],rotation,identity)
        initial_rotation=r.clone();com=torch.nan_to_num(center_world,nan=0.,posinf=0.,neginf=0.);initial_center=com.clone()
        hand=torch.nan_to_num(hand_points,nan=0.,posinf=0.,neginf=0.)
        points=torch.nan_to_num(projected_points,nan=0.,posinf=0.,neginf=0.)
        normals=torch.nn.functional.normalize(torch.nan_to_num(outward_normals,nan=0.,posinf=0.,neginf=0.),dim=-1)
        allowed=(signed_distance<=s.contact_capture_m)&(normals.square().sum(-1)>.5)
        if eligible is not None:allowed &= eligible
        allowed &= valid[:,None]
        local_points=torch.einsum('bij,bcj->bci',r.transpose(-1,-2),points-com[:,None])
        local_normals=torch.einsum('bij,bcj->bci',r.transpose(-1,-2),normals)
        history=torch.zeros_like(hand)
        linear=torch.zeros_like(com);momentum=(r@inertia@r.transpose(-1,-2)@self.initial_angular.expand(b,-1)[...,None]).squeeze(-1)
        free_r=r.clone();free_momentum=momentum.clone();max_force=com.new_zeros(b)
        initial_contacts=(allowed&(signed_distance<0.)).sum(-1)
        if self.backend=='warp':
            linear,momentum,com,r,history,good,fn,ft,max_force,free_r,newton_steps=self.integrate_fused(
                self,linear,momentum,com,r,local_points,local_normals,hand,history,inertia,inv,allowed)
            valid &= good
        else:
            for _ in range(s.horizon):
                for _ in range(c.substeps):
                    linear,momentum,com,r,history,good,fn,ft=self._step(linear,momentum,com,r,local_points,local_normals,hand,history,inertia,inv,allowed)
                    valid &= good
                    max_force=torch.maximum(max_force,fn.amax(-1))
                    free_omega=(free_r@inv@free_r.transpose(-1,-2)@free_momentum[...,None]).squeeze(-1)
                    free_r=rotmat(free_omega*(s.timestep/c.substeps))@free_r
            newton_steps=torch.full((b,),s.horizon*c.substeps*c.newton_iterations,device=com.device,dtype=torch.int32)
        angular=(r@inv@r.transpose(-1,-2)@momentum[...,None]).squeeze(-1)
        free_angular=(free_r@inv@free_r.transpose(-1,-2)@free_momentum[...,None]).squeeze(-1)
        def angle(mat):
            sine=.5*torch.stack((mat[:,2,1]-mat[:,1,2],mat[:,0,2]-mat[:,2,0],mat[:,1,0]-mat[:,0,1]),-1).norm(dim=-1)
            cosine=.5*(mat.diagonal(dim1=-2,dim2=-1).sum(-1)-1)
            return torch.atan2(sine,cosine)
        displacement=(com-initial_center).norm(dim=-1);theta=angle(initial_rotation.transpose(-1,-2)@r)
        speed=linear.norm(dim=-1);omega=angular.norm(dim=-1)
        pose_cost=(displacement/s.translation_scale_m).square()+(theta/math.radians(s.rotation_scale_deg)).square()
        velocity_cost=(speed/s.linear_speed_threshold_m_s).square()+(omega/s.angular_speed_threshold_rad_s).square()
        duration=s.timestep*s.horizon;dt=s.timestep/c.substeps;count=s.horizon*c.substeps
        free_displacement=self.gravity.norm()*dt**2*count*(count+1)/2
        free_pose=(free_displacement/s.translation_scale_m).square()+(angle(initial_rotation.transpose(-1,-2)@free_r)/math.radians(s.rotation_scale_deg)).square()
        free_velocity=(self.gravity.norm()*duration/s.linear_speed_threshold_m_s).square()+(free_angular.norm(dim=-1)/s.angular_speed_threshold_rad_s).square()
        cost=velocity_cost;free=free_velocity
        valid &= torch.isfinite(cost)&torch.isfinite(omega)&torch.isfinite(history).all((-1,-2))
        threshold=(speed<=s.linear_speed_threshold_m_s)&(omega<=s.angular_speed_threshold_rad_s)
        stable=valid&allowed.any(-1)&threshold&(cost<=free*(1-s.minimum_improvement))
        final_point=(r[:,None]@local_points[...,None]).squeeze(-1)+com[:,None]
        final_n=(r[:,None]@local_normals[...,None]).squeeze(-1)
        final_gap=((hand-final_point)*final_n).sum(-1)
        return dict(cost=torch.where(valid,cost,torch.inf),valid=valid,stable=stable,pose_cost=pose_cost,velocity_cost=velocity_cost,
            linear_speed_m_s=speed,angular_speed_rad_s=omega,free_pose_cost=free_pose,free_velocity_cost=free_velocity,
            displacement_m=displacement,rotation_rad=theta,center_world=com,rotation=r,linear_velocity=linear,angular_velocity=angular,
            free_cost=free,initial_contact_count=initial_contacts,final_contact_count=(allowed&(final_gap<0.)).sum(-1),
            max_row_force_n=max_force,eligible=allowed,tangential_displacement_m=history,
            normal_force_n=fn,tangent_force_n=ft,implicit_converged=valid,newton_linear_solves=newton_steps)

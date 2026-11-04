"""Batched GPU solver for the two original native GraspADMM QPs."""
import math
import importlib
import torch
from .native_admm import ContactQP,Settings
from .graspadmm_source import require_upstream


class NativeBatchedQP(ContactQP):
    def __init__(self,contacts,device,*,mode='optimization',miu_coef=None,
                 tolerance=1e-8,max_steps=10000):
        if mode not in ('optimization','filter'):raise ValueError(mode)
        friction=tuple(miu_coef or ((.1,0.) if mode=='optimization' else (.6,0.)))
        settings=Settings(opt_miu_coef=friction,qp_tolerance=tolerance,qp_max_steps=max_steps)
        self.source=require_upstream();self.settings=settings;self.mode=mode
        self.backend=importlib.import_module('dexonomy.qp.qp_batched');self.solvers={}
        nx=6*contacts;nc=nx+contacts+1
        self.G=torch.zeros((nc,nx),device=device,dtype=torch.float64)
        self.h=torch.zeros(nc,device=device,dtype=torch.float64)
        index=torch.arange(nx,device=device);self.G[index,index]=-1.
        self.h[:nx]=-.01 if mode=='optimization' else 0.
        for i in range(contacts):self.G[nx+i,6*i:6*i+6]=1.
        self.h[nx:nx+contacts]=1.;self.G[-1]=-1.;self.h[-1]=-1.
        self.E64=torch.zeros((contacts,6,6),device=device,dtype=torch.float64)
        self.E64[:,0,:]=1.
        self.E64[:,1,0]=self.E64[:,2,2]=friction[0]
        self.E64[:,1,1]=self.E64[:,2,3]=-friction[0]
        self.E64[:,3,4]=friction[1];self.E64[:,3,5]=-friction[1]
        self.E=self.E64.float()

    def solve(self,points,inward,external,center):
        if not points.is_cuda:raise ValueError('Native production QP is CUDA-only; CPU fallback is disabled')
        nx=self.G.shape[1];nc=len(self.h)
        # Eight rho choices plus identity, with room for the factorization scratch.
        bytes_per_pose=20*(nx+2*nc)**2*8
        chunk=max(1,(512*1024*1024)//bytes_per_pose)
        if len(points)>chunk:
            parts=[self.solve(points[i:i+chunk],inward[i:i+chunk],external[i:i+chunk],center[i:i+chunk]) for i in range(0,len(points),chunk)]
            return {key:torch.cat([p[key] for p in parts],0) for key in parts[0]}
        # The original CPU QP forms and solves its matrices in double precision.
        # Preserve that precision on CUDA, especially for rank-deficient wrench maps.
        with torch.no_grad():
            H,g,matrix=self.matrices(points,inward,external,center)
            batch,nx=g.shape;nc=len(self.h)
            if batch not in self.solvers:
                self.solvers.clear()
                solver=self.backend._RELUQP(batch,nx,nc,points.device)
                for name,value in vars(solver).items():
                    if isinstance(value,torch.Tensor) and value.is_floating_point():setattr(solver,name,value.double())
                solver.eps_abs=self.settings.qp_tolerance/math.sqrt(max(nx,nc))
                solver.max_step=self.settings.qp_max_steps;self.solvers[batch]=solver
            solver=self.solvers[batch]
            solver.rho_ind.copy_((solver.rhos-.1).abs().argmin().expand(batch))
            G=self.G.expand(batch,-1,-1);h=self.h.expand(batch,-1)
            alpha=solver.solve(H,g,G,torch.full_like(h,-torch.inf),h).clone()
            dual=solver.output[:,nx+nc:,0].clone();slack=h-(G@alpha[...,None]).squeeze(-1)
            primal=(-slack).clamp_min(0).amax(-1)
            stationarity=((H@alpha[...,None]).squeeze(-1)+g+(G.transpose(-1,-2)@dual[...,None]).squeeze(-1)).abs().amax(-1)
            complementarity=(dual*slack).abs().amax(-1);dual_violation=(-dual).clamp_min(0).amax(-1)
            valid=torch.isfinite(alpha).all(-1)&torch.isfinite(dual).all(-1)&(primal<=1e-6)&(stationarity<=1e-6)&(complementarity<=1e-6)&(dual_violation<=1e-6)
        return dict(alpha=alpha.to(points.dtype),valid=valid,primal=primal,stationarity=stationarity,
                    complementarity=complementarity,dual_violation=dual_violation)

    def matrices(self,points,inward,external,center):
        grasp=self.grasp_matrix(points,inward,center)
        transform=grasp.double()@self.E64
        matrix=transform.transpose(1,2).reshape(len(points),6,-1)
        return matrix.transpose(-1,-2)@matrix,(external.double()[:,None]@matrix).squeeze(1),matrix

    def grasp_matrix(self,points,inward,center):
        if self.mode=='optimization':
            b,n,_=points.shape
            rot=self.source.qp.torch_normal_to_rot(inward.reshape(-1,3).double()).reshape(b,n,3,3)
            grasp=points.new_zeros((b,n,6,6),dtype=torch.float32)
            grasp[...,:3,:3]=rot;grasp[...,3:,3:]=rot
            grasp[...,3:,:3]=torch.linalg.cross((points-center[:,None]).double()[...,None].expand_as(rot),rot,dim=-2)
            return grasp
        # Exact torch transcription of native np_normal_to_rot, whose tangent
        # construction differs from the optimization QP's torch_normal_to_rot.
        a0=inward;guess=torch.zeros_like(a0);guess[...,1]=1.
        alternate=torch.zeros_like(a0);alternate[...,2]=1.
        guess=torch.where((a0[...,1].abs()>.99)[...,None],alternate,guess)
        a1=guess-(guess*a0).sum(-1,keepdim=True)*a0
        a1=a1/a1.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        a2=torch.linalg.cross(a0,a1,dim=-1);rot=torch.stack((a0,a1,a2),-1)
        b,n,_=points.shape;grasp=points.new_zeros((b,n,6,6))
        grasp[...,:3,:3]=rot;grasp[...,3:,3:]=rot
        grasp[...,3:,:3]=torch.linalg.cross((points-center[:,None])[...,None].expand_as(rot),rot,dim=-2)
        return grasp

    def error(self,points,inward,external,center,alpha):
        return self.contact_wrenches(points,inward,external,center,alpha)[1]

    def contact_wrenches(self,points,inward,external,center,alpha):
        grasp=self.grasp_matrix(points,inward,center);n=points.shape[1]
        if self.mode=='optimization':per_contact=grasp.float()@self.E
        else:per_contact=grasp.double()@self.E64
        wrench=(per_contact@alpha.detach().to(per_contact.dtype).reshape(len(points),n,6,1)).squeeze(-1)
        return wrench,(wrench.sum(1)+external.to(wrench.dtype)).norm(dim=-1)

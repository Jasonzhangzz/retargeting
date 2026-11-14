"""Explicit sparse shooting KKT correctness reference (CPU, not a GPU backend).

Production eliminates shooting states and uses compact CUDA block PCR. This
independent sparse formulation verifies that elimination preserves the Newton
step, including segment resets, damping, and a fixed active joint set.
"""
import torch


@torch.no_grad()
def solve_shooting_kkt_reference(objective,q,lower=None,upper=None,damping=.01):
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    if not np.isfinite(damping) or damping<0:raise ValueError('Invalid damping')
    r,j,delta=objective.residuals(q)
    r,j,delta=[v.detach().cpu().double().numpy() for v in (r,j,delta)]
    n,d=q.shape;nd=n*d;weight=objective.weight
    gu=(j*r[...,None]).sum(1)+weight*delta;gx=-weight*delta
    time=sp.diags(objective.link.detach().cpu().numpy()[1:].astype(float),-1,shape=(n,n)) if n>1 else sp.csc_matrix((1,1))
    previous=sp.kron(time,sp.eye(d),format='csc')
    joint_g=gu.reshape(-1)+previous.T@gx.reshape(-1)
    free=np.ones(nd)
    if (lower is None)!=(upper is None):raise ValueError('Provide both bounds')
    if lower is not None:
        values=q.detach().cpu().double().numpy().reshape(-1)
        lo=np.broadcast_to(lower.detach().cpu().double().numpy(),q.shape).reshape(-1)
        hi=np.broadcast_to(upper.detach().cpu().double().numpy(),q.shape).reshape(-1)
        free=~(((values<=lo+1e-7)&(joint_g>0))|((values>=hi-1e-7)&(joint_g<0)))
    free=free.astype(float)
    project=sp.diags(free,format='csc');eye=sp.eye(nd,format='csc')
    blocks=np.swapaxes(j,-1,-2)@j+(weight+damping)*np.eye(d)
    huu=project@sp.block_diag([sp.csc_matrix(block) for block in blocks],format='csc')@project+eye-project
    hux=-weight*project;hxx=weight*eye;constraint_u=-previous@project
    kkt=sp.bmat([[huu,hux,constraint_u.T],[hux.T,hxx,eye],[constraint_u,eye,None]],format='csc')
    rhs=np.concatenate((-free*gu.reshape(-1),-gx.reshape(-1),np.zeros(nd)))
    solution=spsolve(kkt,rhs);residual=kkt@solution-rhs
    step=torch.from_numpy(solution[:nd].reshape(n,d)).to(q)
    return step,dict(backend='CPU scipy sparse shooting KKT reference',unknowns=3*nd,nonzeros=kkt.nnz,linear_residual_inf=float(abs(residual).max()),states='x_t=q_(t-1), u_t=q_t',production_backend='CUDA compact state-eliminated block PCR')

"""GPU global-memory L-BFGS recursion for long trajectory vectors.

Robokit's per-thread history matrices spill for T*D-wide vectors. Keep its
optimizer lifecycle, projected variables and line search; stream the same ten
curvature pairs through GPU reductions, with O(batch * history * T * D) storage.
No time-step loop or host evaluation of tensor values is used.
"""
import torch
import warp as wp


def search_direction(solver):
    to=wp.to_torch;g=to(solver.gradient);previous=to(solver.gradient_prev);step=to(solver.last_step_dq)
    sh,yh,rh=to(solver.s_history),to(solver.y_history),to(solver.rho_history)
    count,start=to(solver.history_count),to(solver.history_start);b,m,d=sh.shape;batch=torch.arange(b,device=g.device)
    change=g-previous;ys=(step*change).sum(-1);new=(ys>1e-8)&torch.isfinite(ys);slot=((start+count)%m).long()
    sh[batch,slot]=torch.where(new[:,None],step,sh[batch,slot]);yh[batch,slot]=torch.where(new[:,None],change,yh[batch,slot]);rh[batch,slot]=torch.where(new,ys.clamp_min(1e-30).reciprocal(),rh[batch,slot])
    start.copy_(torch.where(new&(count==m),(start+1)%m,start));count.copy_(torch.where(new,(count+1).clamp_max(m),count))
    indices=(start[:,None]+count[:,None]-1-torch.arange(m,device=g.device)[None])%m
    s=sh[batch[:,None],indices.long()];y=yh[batch[:,None],indices.long()];rho=rh[batch[:,None],indices.long()]
    valid=torch.arange(m,device=g.device)[None]<count[:,None];rho=torch.where(valid,rho,0.)
    q=g.clone();alphas=[]
    for i in range(m):
        alpha=rho[:,i]*(s[:,i]*q).sum(-1);alphas.append(alpha);q=q-alpha[:,None]*y[:,i]
    yy=y[:,0].square().sum(-1);sy=(s[:,0]*y[:,0]).sum(-1)
    gamma=torch.where((count>0)&(yy>1e-12),sy/yy.clamp_min(1e-30),float(solver.config.h0_scale))
    if solver.config.normalize_direction:gamma=torch.full_like(gamma,float(solver.config.h0_scale))
    q=q*gamma[:,None]
    for i in reversed(range(m)):
        beta=rho[:,i]*(y[:,i]*q).sum(-1);q=q+(alphas[i]-beta)[:,None]*s[:,i]
    direction=-q
    if solver.config.normalize_direction:direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    to(solver.search_direction).copy_(direction);to(solver.initial_slope).copy_((g*direction).sum(-1))
    candidates=direction[:,None]*to(solver.line_search_alphas)[None,:,None]*to(solver._mask)[None,None]
    to(solver._candidate_velocities).copy_(candidates.reshape(b*solver.n_line_search,d))


def select_best(solver):
    to=wp.to_torch;cost=to(solver.costs);b=solver.batch_size;k=solver.n_line_search;trials=to(solver._expanded_costs).reshape(b,k)
    sufficient=trials<=cost[:,None]+float(solver.config.wolfe_c1)*to(solver.line_search_alphas)[None]*to(solver.initial_slope)[:,None]
    passing=sufficient&(to(solver.initial_slope)[:,None]<0);indices=torch.arange(k,device=cost.device)[None].expand(b,-1)
    winner=torch.where(passing,indices,-1).amax(-1);batch=torch.arange(b,device=cost.device);clamped=winner.clamp_min(0)
    new_cost=torch.where(winner>=0,trials[batch,clamped],cost)
    velocities=to(solver._candidate_velocities).reshape(b,k,-1);move=torch.where((winner>=0)[:,None],velocities[batch,clamped],0.)
    to(solver.last_step_dq).copy_(move);to(solver._winner_expanded_idx).copy_(torch.where(winner>=0,batch*k+winner,-1).int())
    best=to(solver.best_costs);to(solver._best_improved).copy_((new_cost<best).int());best.copy_(torch.minimum(best,new_cost));cost.copy_(new_cost)

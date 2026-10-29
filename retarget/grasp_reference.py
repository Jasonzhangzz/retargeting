"""Closing hand reference built from unchanged Dexonomy surface contacts.

As in run.py, a position servo needs a command beyond the touching posture
to develop contact load. This constructs that reference, not a new ADMM cost.
The object pose, surface anchors, contact queries and initial state stay fixed.
"""
from retarget.grasp_capture import capture_guard
import numpy as np
import torch
import time

from retarget.grasp_template import WujiContactKinematics


def closing_posture(model, bodies, local, normals, initial, points, outward, *, depth, device):
    begun = time.perf_counter()
    fk = WujiContactKinematics(model,bodies,device)
    root, start = fk.tensor(initial[:6]), fk.tensor(initial[6:])
    lp, ln = fk.tensor(local), fk.tensor(normals)
    target = fk.tensor(points-depth*outward)
    ids = [np.flatnonzero(model.jnt_qposadr==i)[0] for i in range(6,26)]
    low,high = (fk.tensor(v) for v in model.jnt_range[ids].T)
    def residual(joints):
        p,_ = fk.contacts(torch.cat((root,joints)),lp,ln)
        return torch.cat(((p-target).flatten()/.01,.02*(joints-start)))
    q = start.clamp(low,high)
    identity = torch.eye(20,device=fk.device,dtype=fk.dtype)
    damping = fk.tensor(.01)
    def step():
        r = residual(q)
        jac = torch.func.jacfwd(residual)(q)
        delta = torch.linalg.solve_ex(jac.T@jac+damping*identity,-jac.T@r,check_errors=False)[0]
        trial = (q+delta.clamp(-.15,.15)).clamp(low,high)
        accept = residual(trial).square().sum()<r.square().sum()
        q.copy_(torch.where(accept,trial,q))
        damping.copy_(torch.where(accept,damping*.5,damping*4).clamp(1e-5,1e6))
    original = q.clone()
    stream = torch.cuda.Stream(device=fk.device)
    stream.wait_stream(torch.cuda.current_stream(fk.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_guard(), torch.cuda.graph(graph, stream=stream):
        q.copy_(original)
        damping.fill_(.01)
        for _ in range(60):
            step()
    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.current_stream(fk.device).wait_stream(stream)
    value = np.r_[initial[:6],q.cpu().numpy()]
    p,_ = fk.contacts(fk.tensor(value),lp,ln)
    return value,dict(depth_mm=1000*depth,root_fixed=True,seconds=time.perf_counter()-begun,
        target_error_mm=(1000*torch.linalg.vector_norm(p-target,dim=-1)).cpu().tolist(),
        joint_delta_rad=(value[6:]-initial[6:]).tolist())


def closing_sequence(initial, closed, *, duration, ref_dt):
    count = max(2,int(np.ceil(duration/ref_dt))+1)
    alpha = np.minimum(np.arange(count)*ref_dt/duration,1.)
    return initial[None]+alpha[:,None]*(closed-initial)[None]

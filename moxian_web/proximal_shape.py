"""Proximal-only cross-section diagnostics for fingers-together calibration."""
import numpy as np


def refine_proximal_widths(model,rows,cancel=None,max_nfev=15):
    """Small static breadth fit; runtime articulation is never optimized here."""
    import torch
    from scipy.optimize import least_squares
    from scripts.calibrate_wuji_mano import CHAINS
    selected=[r for r in rows if r.get('stage')=='together']
    if not selected or model.official_fk is None:return {'state':'unobserved','applied':False}
    selected=[selected[i] for i in np.unique(np.linspace(0,len(selected)-1,min(3,len(selected))).astype(int))]
    q,r,o,c=model.official_parameters(selected);weights=model.model.weights.cpu().numpy()
    original=model.proximal_lateral_scales.copy();base=model.rest_np.copy();evaluations=0
    # The small finger must retain its original link geometry. Local breadth
    # expansion plus an MCP offset produced a visible bulge, so neither is
    # estimated for that finger. Its ordinary whole-finger radial size remains
    # part of the established anthropometric calibration.
    free=np.array([0,1,2])
    def residual(active):
        nonlocal evaluations
        if cancel is not None and cancel.is_set():raise InterruptedError('近端宽度拟合已取消')
        x=original.copy();x[free]=active
        evaluations+=1;model.proximal_lateral_scales=np.array(x);model.set_shape(model.shape)
        with torch.no_grad():v,j,_,_=model.decode(q,r,o,c)
        gaps=np.array([[s['gap_m'] for s in proximal_pair_gaps(a,b,model.faces,weights,CHAINS)] for a,b in zip(v.cpu().numpy(),j.cpu().numpy())])
        return np.r_[((gaps-.0005)/.001).ravel(),(x-original)/.3]
    def jac(x):
        return np.stack([(residual(x+np.eye(3)[k]*.002)-residual(x-np.eye(3)[k]*.002))/.004 for k in range(3)],1)
    before=residual(original[free])
    try:
        solved=least_squares(residual,original[free],jac=jac,bounds=(-np.log(1.35),np.log(1.35)),max_nfev=max_nfev,ftol=1e-4,xtol=1e-4,gtol=1e-4)
        after=residual(solved.x);change=float(np.linalg.norm(model.rest_np-base,axis=1).max())
        accepted=bool(after@after<before@before and change<.0035)
        result={'state':'fitted','applied':accepted,'evaluations':evaluations,'scales':np.exp(model.proximal_lateral_scales).tolist(),'pinky_local_shape_frozen':True,'max_vertex_change_mm':change*1000}
    except BaseException:
        model.proximal_lateral_scales=original;model.set_shape(model.shape);raise
    if not accepted:model.proximal_lateral_scales=original;model.set_shape(model.shape)
    return result


def proximal_pair_gaps(vertices,joints,faces,weights,chains):
    """Signed lateral gaps at three common planes along each proximal pair.

    Unlike a minimum over the whole finger, a fingertip contact cannot hide a
    proximal gap. Intended for the explicitly recorded straight/together guide.
    """
    v=np.asarray(vertices);j=np.asarray(joints);faces=np.asarray(faces)
    masks=[(weights[:,list(c)].sum(1)[faces]>.45).sum(1)>=2 for c in chains]
    result=[]
    for f in range(1,4):
        a,b=chains[f],chains[f+1]
        axis=(j[a[1]]-j[a[0]])+(j[b[1]]-j[b[0]]);axis/=np.linalg.norm(axis)
        side=j[b[0]]-j[a[0]];side-=axis*(side@axis);side/=np.linalg.norm(side)
        lo=max(j[a[0]]@axis,j[b[0]]@axis);hi=min(j[a[1]]@axis,j[b[1]]@axis)
        if hi-lo<.004:raise ValueError('并拢动作的近端节段没有共同截面')
        for t in (.2,.5,.8):
            level=lo+t*(hi-lo);bounds=[]
            for finger in (f,f+1):
                tri=v[faces[masks[finger]]];points=[]
                for u,w in ((0,1),(1,2),(2,0)):
                    start,end=tri[:,u],tri[:,w];ds=start@axis-level;de=end@axis-level
                    mask=(ds*de<=0)&(abs(de-ds)>1e-12)
                    blend=-ds[mask]/(de-ds)[mask]
                    points.extend((start[mask]+blend[:,None]*(end[mask]-start[mask]))@side)
                if not points:raise ValueError('近端截面缺少网格交点')
                bounds.append((min(points),max(points)))
            result.append({'pair':f'{f}_{f+1}','level':t,'gap_m':float(bounds[1][0]-bounds[0][1])})
    return result

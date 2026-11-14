"""Proximal-only cross-section diagnostics for fingers-together calibration."""
import numpy as np


def fit_rigid_finger_spacing(model,rows,cancel=None,max_nfev=35):
    """Calibrate fixed whole-finger anchors/axis zeros, without local inflation.

    Runtime uses only official IK rotations followed by fixed-length FK. These
    eight constants never depend on live touch, proximity or gesture labels.
    """
    import torch
    from scipy.optimize import least_squares
    from scripts.calibrate_wuji_mano import CHAINS
    selected=[r for r in rows if r.get('stage')=='together']
    if not selected or not model.rigid_finger_fk:return {'state':'unobserved','applied':False}
    selected=[selected[i] for i in np.unique(np.linspace(0,len(selected)-1,min(3,len(selected))).astype(int))]
    original=np.r_[model.finger_base_offsets_m,model.finger_axis_yaw_rad];weights=model.model.weights.cpu().numpy();evaluations=0;distances=None
    def gaps(x):
        nonlocal evaluations,distances
        if cancel is not None and cancel.is_set():raise InterruptedError('固定指根标定已取消')
        evaluations+=1;model.finger_base_offsets_m=np.array(x[:4]);model.finger_axis_yaw_rad=np.array(x[4:]);model.set_shape(model.shape)
        q,r,o,c=model.official_parameters(selected)
        with torch.no_grad():v,j,_,_=model.decode(q,r,o,c)
        sections=[proximal_pair_gaps(a,b,model.faces,weights,CHAINS,True) for a,b in zip(v.cpu().numpy(),j.cpu().numpy())]
        distances=np.array([[s['surface_distance_m'] for s in group] for group in sections])
        return np.array([[s['gap_m'] for s in group] for group in sections])
    def residual(x):
        g=gaps(x)
        return np.r_[(distances/.0002).ravel(),(np.maximum(-g-.002,0)/.0005).ravel(),(np.maximum(g,0)/.001).ravel(),(x[:4]-original[:4])/.005,(x[4:]-original[4:])/.15,np.mean(x[:4])/.001,np.mean(x[4:])/.03]
    def jac(x):
        return np.stack([(residual(x+np.eye(8)[k]*step)-residual(x-np.eye(8)[k]*step))/(2*step) for k,step in enumerate([.0001]*4+[.001]*4)],1)
    before=gaps(original)
    try:
        solved=least_squares(residual,original,jac=jac,bounds=(np.r_[[-.006]*4,[-.21]*4],np.r_[[.006]*4,[.21]*4]),max_nfev=max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        after=gaps(solved.x);accepted=bool(distances.max()<.0005 and after.min()>-.003)
    except BaseException:
        gaps(original);raise
    if not accepted:gaps(original)
    return {'state':'fitted','applied':accepted,'evaluations':evaluations,'before_gaps_mm':(before*1000).tolist(),'after_gaps_mm':(after*1000).tolist(),'surface_distances_mm':(distances*1000).tolist(),'finger_base_offsets_m':model.finger_base_offsets_m.tolist(),'finger_axis_yaw_rad':model.finger_axis_yaw_rad.tolist(),'local_shape_changes':False,'runtime_contact_feedback':False}


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


def section_distance(a,b):
    """Exact distance between two coplanar polyline segment sets."""
    a=np.asarray(a);b=np.asarray(b);cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    u=a[:,1]-a[:,0];v=b[:,1]-b[:,0];w=b[None,:,0]-a[:,None,0];det=cross(u[:,None],v[None]);safe=np.where(abs(det)>1e-15,det,1.)
    t=cross(w,v[None])/safe;s=cross(w,u[:,None])/safe
    if np.any((abs(det)>1e-15)&(t>=0)&(t<=1)&(s>=0)&(s<=1)):return 0.
    def distance(points,segments):
        start=segments[:,0];edge=segments[:,1]-start;d=points[:,None]-start
        blend=np.clip((d*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-20),0,1)
        return np.linalg.norm(d-blend[...,None]*edge,axis=-1).min()
    return float(min(distance(a.reshape(-1,2),b),distance(b.reshape(-1,2),a)))


def proximal_pair_gaps(vertices,joints,faces,weights,chains,measure_distance=False,include_thumb=False):
    """Signed lateral gaps at three common planes along each proximal pair.

    Unlike a minimum over the whole finger, a fingertip contact cannot hide a
    proximal gap. Intended for the explicitly recorded straight/together guide.
    """
    v=np.asarray(vertices);j=np.asarray(joints);faces=np.asarray(faces)
    masks=[(weights[:,list(c[1:] if f==0 and include_thumb else c)].sum(1)[faces]>.45).sum(1)>=2 for f,c in enumerate(chains)]
    result=[]
    for f in range(0 if include_thumb else 1,4):
        from .side_contact import side_pair_frame
        axis,side,lo,hi=side_pair_frame(v,j,chains,f)
        for t in (.2,.5,.8):
            level=lo+t*(hi-lo);bounds=[];curves=[]
            for finger in (f,f+1):
                tri=v[faces[masks[finger]]];points=[]
                for u,w in ((0,1),(1,2),(2,0)):
                    start,end=tri[:,u],tri[:,w];ds=start@axis-level;de=end@axis-level
                    mask=(ds*de<=0)&(abs(de-ds)>1e-12)
                    blend=-ds[mask]/(de-ds)[mask]
                    points.extend((start[mask]+blend[:,None]*(end[mask]-start[mask]))@side)
                if not points:raise ValueError('近端截面缺少网格交点')
                bounds.append((min(points),max(points)))
                if measure_distance:
                    normal=np.cross(axis,side);ds=tri@axis-level;cut=[];valid=[]
                    for u,w in ((0,1),(1,2),(2,0)):
                        den=ds[:,w]-ds[:,u];mask=(ds[:,u]*ds[:,w]<=0)&(abs(den)>1e-12);blend=np.where(mask,-ds[:,u]/np.where(mask,den,1),0)
                        cut.append(tri[:,u]+blend[:,None]*(tri[:,w]-tri[:,u]));valid.append(mask)
                    cut=np.stack(cut,1);valid=np.stack(valid,1);ids=np.flatnonzero(valid.sum(1)>=2);order=np.argsort(~valid[ids],axis=1)[:,:2];seg=cut[ids[:,None],order]
                    curves.append(np.stack((seg@side,seg@normal),-1))
            record={'pair':f'{f}_{f+1}','level':t,'gap_m':float(bounds[1][0]-bounds[0][1])}
            if measure_distance:record['surface_distance_m']=section_distance(*curves)
            result.append(record)
    return result

"""Material witnesses at the same proximal cross-sections used for acceptance."""
import numpy as np
from scripts.calibrate_wuji_mano import CHAINS,TIPS


def side_pair_frame(vertices,joints,chains,f):
    a=chains[f][1:] if f==0 else chains[f];b=chains[f+1]
    start_a=joints[a[0]];end_a=vertices[TIPS[0]] if f==0 else joints[a[1]]
    start_b=joints[b[0]];end_b=joints[b[1]]
    axis=(end_a-start_a)+(end_b-start_b);axis/=max(np.linalg.norm(axis),1e-12)
    side=start_b-start_a;side-=axis*(side@axis);side/=max(np.linalg.norm(side),1e-12)
    lo=max(start_a@axis,start_b@axis);hi=min(end_a@axis,end_b@axis)
    if hi-lo<(.001 if f==0 else .004):raise ValueError('侧面节段没有共同截面')
    return axis,side,lo,hi


def cuts(vertices,faces,ids,axis,level):
    tri=vertices[faces[ids]];d=tri@axis-level;points=[];barys=[];valid=[]
    for a,b in ((0,1),(1,2),(2,0)):
        den=d[:,b]-d[:,a];ok=(d[:,a]*d[:,b]<=0)&(abs(den)>1e-12)
        t=np.clip(-d[:,a]/np.where(ok,den,1),0,1)
        bc=np.zeros((len(ids),3));bc[:,a]=1-t;bc[:,b]=t
        points.append((tri*bc[...,None]).sum(1));barys.append(bc);valid.append(ok)
    points=np.stack(points,1);barys=np.stack(barys,1);valid=np.stack(valid,1)
    keep=np.flatnonzero(valid.sum(1)>=2);order=np.argsort(~valid[keep],axis=1)[:,:2]
    return ids[keep],points[keep[:,None],order],barys[keep[:,None],order]


def closest_segments(a,b,axis,side,exclude):
    """Closest coplanar segment pair, including interior intersections."""
    start_a=a[:,0,None];da=a[:,1,None]-start_a
    start_b=b[None,:,0];db=b[None,:,1]-start_b
    shape=(len(a),len(b));options=[]
    for t in (0.,1.):
        p=start_a+t*da;s=np.clip(((p-start_b)*db).sum(-1)/np.maximum((db*db).sum(-1),1e-20),0,1)
        options.append((np.full(shape,t),s))
    for s in (0.,1.):
        p=start_b+s*db;t=np.clip(((p-start_a)*da).sum(-1)/np.maximum((da*da).sum(-1),1e-20),0,1)
        options.append((t,np.full(shape,s)))
    plane=np.stack((side,np.cross(axis,side)),1);u=da@plane;v=db@plane;w=(start_b-start_a)@plane
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    det=cross(u,v);safe=np.where(abs(det)>1e-14,det,1.)
    t=cross(w,v)/safe;s=cross(w,u)/safe
    intersection=(abs(det)>1e-14)&(t>=0)&(t<=1)&(s>=0)&(s<=1)
    options.append((np.clip(t,0,1),np.clip(s,0,1)))
    scores=[]
    for k,(t,s) in enumerate(options):
        distance=np.linalg.norm(start_a+t[...,None]*da-start_b-s[...,None]*db,axis=-1)
        if k==4:distance=np.where(intersection,distance,np.inf)
        scores.append(np.where(exclude,np.inf,distance))
    scores=np.stack(scores)
    if not np.isfinite(scores).any():raise ValueError('侧面截面仅包含拓扑相连曲面')
    k,i,j=np.unravel_index(scores.argmin(),scores.shape)
    return i,j,float(options[k][0][i,j]),float(options[k][1][i,j])


def side_material_pairs(model,vertices,joints,include_thumb=False,finger_pairs=None,skin=None):
    """Three actual plane intersections per adjacent proximal phalanx pair.

    User-confirmed thumb/index side closure is optional. Nothing here asserts
    ADC evidence or supplies runtime gesture labels.
    """
    v=np.asarray(vertices);j=np.asarray(joints);faces=model.faces
    skin=model.model.weights.detach().cpu().numpy() if skin is None else skin;tri=v[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    masks=[(skin[:,list(c[1:] if f==0 else c)].sum(1)[faces]>.45).sum(1)>=2 for f,c in enumerate(CHAINS)]
    result=[]
    for f in (range(0 if include_thumb else 1,4) if finger_pairs is None else finger_pairs):
        axis,side,lo,hi=side_pair_frame(v,j,CHAINS,f)
        for fraction in (.2,.5,.8):
            level=lo+fraction*(hi-lo)
            ia,pa,ba=cuts(v,faces,np.flatnonzero(masks[f]&(normal@side>.05)),axis,level)
            ib,pb,bb=cuts(v,faces,np.flatnonzero(masks[f+1]&(normal@side<-.05)),axis,level)
            if not len(ia) or not len(ib):raise ValueError('侧面截面缺少相向材料曲面')
            connected=(faces[ia][:,None,:,None]==faces[ib][None,:,None,:]).any((2,3))
            ai,bi,t,s=closest_segments(pa,pb,axis,side,connected)
            wa=(1-t)*ba[ai,0]+t*ba[ai,1];wb=(1-s)*bb[bi,0]+s*bb[bi,1]
            point_a=wa@tri[ia[ai]];point_b=wb@tri[ib[bi]]
            result.append(dict(pair=f'{f}_{f+1}',level=fraction,
                               face_a=int(ia[ai]),face_b=int(ib[bi]),barycentric_a=wa.tolist(),barycentric_b=wb.tolist(),
                               gap_m=float(np.linalg.norm(point_a-point_b)),normal_dot=float(normal[ia[ai]]@normal[ib[bi]])))
    return result

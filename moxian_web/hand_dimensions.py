"""Fixed anthropometric scales; no vertex displacements or contact deformation."""
from __future__ import annotations
import numpy as np
DIMENSION_MODE='fixed_segment_scales_v1'
DIMENSION_COUNT=20  # 5 fingers x 3 axial scales, then 5 radial scales
PENETRATION_TOLERANCE_M=.005
PROXIMAL_WIDTH_POLICY='fixed_proximal_lateral_scales_v1'

def scale_proximal_width(vertices,joints,weights,chains,palm_normal,log_scales):
    """Independent proximal breadth, without inflating depth or moving joints.

    A smooth static rest-shape scale; never depends on live pose or touch.
    """
    x=np.asarray(log_scales,dtype=float)
    if x.shape!=(4,) or not np.isfinite(x).all() or np.max(abs(x))>np.log(1.8)+1e-8:
        raise ValueError('四指近端宽度比例须在 1/1.8–1.8 内')
    v=np.asarray(vertices);out=v.copy()
    smooth=lambda t:np.clip(t,0,1)**2*(3-2*np.clip(t,0,1))
    for f,c in enumerate(chains[1:]):
        axis=joints[c[1]]-joints[c[0]];length=np.linalg.norm(axis);axis/=length
        side=np.cross(axis,palm_normal);side/=np.linalg.norm(side)
        offset=v-joints[c[0]];u=offset@axis/length
        envelope=smooth((u+.2)/.3)*(1-smooth((u-.7)/.65))
        membership=weights[:,list(c)].sum(1)
        out+=side[None]*(offset@side*np.expm1(x[f])*envelope*membership)[:,None]
    return out

def scale_segments(vertices,joints,weights,chains,tips,parameters):
    x=np.asarray(parameters,dtype=float)
    if x.shape!=(DIMENSION_COUNT,) or not np.isfinite(x).all():raise ValueError('手指尺寸需要 15 个节段比例和 5 个粗细比例')
    if np.max(np.abs(x[:15]))>np.log(1.5) or np.max(np.abs(x[15:]))>np.log(1.6):raise ValueError('手指尺寸比例超出安全范围')
    lengths=np.exp(x[:15].reshape(5,3));radial=np.exp(x[15:]);new_j=np.array(joints,copy=True);out=weights[:,0,None]*vertices
    for f,chain in enumerate(chains):
        ends=np.concatenate((joints[list(chain[1:])],vertices[[tips[f]]]))
        vectors=ends-joints[list(chain)]
        for k,joint in enumerate(chain):
            if k:new_j[joint]=new_j[chain[k-1]]+vectors[k-1]*lengths[f,k-1]
            axis=vectors[k]/np.linalg.norm(vectors[k]);A=radial[f]*np.eye(3)+(lengths[f,k]-radial[f])*np.outer(axis,axis)
            transformed=(vertices-joints[joint])@A.T+new_j[joint]
            out+=weights[:,joint,None]*transformed
    return out,new_j

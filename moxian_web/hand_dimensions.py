"""Fixed anthropometric scales; no vertex displacements or contact deformation."""
from __future__ import annotations
import numpy as np
DIMENSION_MODE='fixed_segment_scales_v1'
DIMENSION_COUNT=20  # 5 fingers x 3 axial scales, then 5 radial scales
PENETRATION_TOLERANCE_M=.005

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

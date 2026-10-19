"""Interpolate the 30 Hz demonstration object pose."""
import numpy as np


def reference(poses, times):
    frame=np.clip(times*30,0,len(poses)-1)
    left=np.floor(frame).astype(int)
    right=np.minimum(left+1,len(poses)-1)
    alpha=(frame-left)[:,None]
    result=(1-alpha)*poses[left]+alpha*poses[right]
    qa=poses[left,3:]
    qb=poses[right,3:]*np.where(np.sum(qa*poses[right,3:],axis=1,keepdims=True)<0,-1,1)
    result[:,3:]=(1-alpha)*qa+alpha*qb
    result[:,3:]/=np.linalg.norm(result[:,3:],axis=1,keepdims=True)
    return result

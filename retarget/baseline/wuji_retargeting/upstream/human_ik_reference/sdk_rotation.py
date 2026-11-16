"""Independent port of SDK 2026.8.31 SO(3) branches, including their roundoff.

This compatibility module intentionally differs from the stable mathematical
functions in kinematics.py near zero and pi. It imports no SDK/native helpers.
Addresses are original ELF VAs for SHA a179dcb7...ac7fd5.
"""
import math
import numpy as np
from .kinematics import skew

PI_BRANCH=3.141591653589793  # rodata 0x245000


def _ordered_matmul(left, right):
    """ELF 0xda5422 scalar-column order; keep each multiply/add separate."""
    result=left[:,0,None]*right[None,0,:]
    for k in range(1,left.shape[1]):
        result=left[:,k,None]*right[None,k,:]+result
    return result


def _norm3(v):
    return math.sqrt((v[0]*v[0]+v[1]*v[1])+v[2]*v[2])


def sdk_so3_log(rotation):
    """VA 0xda54ef: acos(trace), skew formula, or longest column of R+I."""
    r=np.asarray(rotation,dtype=float)
    trace=((0.+r[0,0])+r[1,1])+r[2,2]
    theta=math.acos(float(np.clip((trace-1.)*.5,-1.,1.)))
    if abs(theta)<1e-10:return np.zeros(3)
    if theta>PI_BRANCH:
        plus=r+np.eye(3)
        squares=[(plus[0,k]*plus[0,k]+plus[1,k]*plus[1,k])+plus[2,k]*plus[2,k] for k in range(3)]
        k=int(np.argmax(squares));length=math.sqrt(squares[k])
        if length<1e-12:return np.zeros(3)
        return (plus[:,k]/length)*theta
    factor=theta/(math.sin(theta)+math.sin(theta))
    return np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])*factor


def sdk_inverse_rotation_vector(rotation,log_vector=None):
    """VA 0xdaae1c: vector for inverse rotation and its magnitude (not axis)."""
    r=np.asarray(rotation,dtype=float)
    v=sdk_so3_log(r) if log_vector is None else np.asarray(log_vector,dtype=float)
    theta=_norm3(v)
    if theta==0.:return np.zeros(3),0.
    if theta>PI_BRANCH:
        inverse=sdk_so3_log(r.T)
        return inverse,_norm3(inverse)
    return -v,theta


def sdk_rotation_jacobian_block(inverse_vector,theta):
    """VA 0xdaaee0 bottom-right 3x3, independent of translation input.

    theta >= 1e-8 uses the literal closed form and can suffer cancellation.
    The theta < 1e-8 branch has the opposite half-skew sign in the artifact.
    It is normally only reached with zero after the acos-based log.
    """
    v=np.asarray(inverse_vector,dtype=float);k=skew(v)
    if theta<1e-8:return np.eye(3)-.5*k+(k@k)/12.
    sine=math.sin(theta);one_minus_cosine=1.-math.cos(theta)
    with np.errstate(divide='ignore',invalid='ignore'):
        alpha=1./(theta*theta)-np.divide(sine,(theta+theta)*one_minus_cosine)
        beta=np.divide((theta*.5)*sine,one_minus_cosine)
        # Native scales the vector first, then takes its outer product.
        return (alpha*v)[:,None]*v[None,:]+beta*np.eye(3)+.5*k


def sdk_pose_residual_jacobian(transform,target,j_position,j_angular):
    """Compatibility residual/J used by SDK, before positional/rotational weights."""
    rc=transform[:3,:3];relative=_ordered_matmul(rc.T,target[:3,:3])
    residual_rotation=sdk_so3_log(relative)
    inverse,theta=sdk_inverse_rotation_vector(relative,residual_rotation)
    block=sdk_rotation_jacobian_block(inverse,theta)
    residual=np.r_[transform[:3,3]-target[:3,3],residual_rotation]
    # Native da9f21 first rotates each geometric angular column to the local
    # frame, then applies the log-Jacobian block (da9fbe..da9fe7). Associating
    # the first two matrices instead can change an IK line-search decision.
    angular_local=_ordered_matmul(rc.T,j_angular)
    jac=np.vstack((j_position,-_ordered_matmul(block,angular_local)))
    return residual,jac

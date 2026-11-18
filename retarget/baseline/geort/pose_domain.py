"""Wuji2 sampling/output bounds: no negative finger flexion.

Glove extension allowances are source metadata, never transferred to robot
training poses. Keep this module usable by the web process without torch.
"""
import numpy as np

DEFAULT_EXTENSION = dict(mcp_extension_deg=15.,pip_extension_deg=10.,
                         dip_extension_deg=20.,thumb_extension_deg=10.)
ROBOT_EXTENSION = dict.fromkeys(DEFAULT_EXTENSION,0.)
SAMPLING_POLICY='nonhyperextended_flexion_uniform_v2'


def extension_domain(native_lower,native_upper,glove_policy=None):
    policy={**DEFAULT_EXTENSION,**(glove_policy or {})}
    low=np.asarray(native_lower,dtype=np.float32).copy()
    high=np.asarray(native_upper,dtype=np.float32).copy()
    if (low.shape!=(20,) or high.shape!=(20,) or not np.isfinite([low,high]).all()
            or np.any(low>=high)):
        raise ValueError('无效机器人关节范围')
    if any(not np.isfinite(policy[k]) or not 0<=policy[k]<=30 for k in DEFAULT_EXTENSION):
        raise ValueError('无效的标定反伸范围')
    # Native Wuji2 finger flexion zero is straight. Thumb CMC opposition and
    # finger abduction are different axes and retain signed native ranges.
    low[[2,3]]=np.maximum(low[[2,3]],0.)
    for offset in (0,2,3):
        ids=[4*f+offset for f in range(1,5)]
        low[ids]=np.maximum(low[ids],0.)
    if np.any(low>=high):raise ValueError('反伸范围与机器人关节范围无交集')
    return dict(lower=low.tolist(),upper=high.tolist(),glove_extension_policy=policy,
                robot_extension_policy=ROBOT_EXTENSION.copy(),
                policy='wuji2_nonhyperextended_flexion_v2')


def sample_poses(domain,count,rng):
    """Draw inside the allowed pose domain; never clip an invalid native pose."""
    lower=np.asarray(domain['lower'],float);upper=np.asarray(domain['upper'],float)
    if (type(count) is not int or count<=0 or lower.shape!=(20,) or upper.shape!=(20,)
            or not np.isfinite([lower,upper]).all() or np.any(lower>=upper)):
        raise ValueError('无效采样姿态域')
    return (rng.uniform(0,1,(count,20))*(upper-lower)+lower).astype(np.float32)

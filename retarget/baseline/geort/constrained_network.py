"""Bounded IK output for both GeoRT objectives, with legacy model loading."""
import numpy as np
import torch
from .training import reference

MODE='contact_v1'


def domain(robot,glove_policy=None):
    from .pose_domain import extension_domain
    return extension_domain(robot.lower,robot.upper,glove_policy)


class DomainIK(torch.nn.Module):
    """Return native-normalized angles; restriction is part of the network.

    No post-inference clipping, optimizer, or filtering is necessary. Native
    normalization is retained for the native-angle neural FK interface.
    """
    def __init__(self,robot,spec):
        super().__init__();self.base=reference('model').IKModel(robot.spec['keypoint_joints'])
        low=np.asarray(spec['lower'],np.float32);high=np.asarray(spec['upper'],np.float32)
        if low.shape!=(20,) or high.shape!=(20,) or not np.isfinite([low,high]).all() or np.any(low>=high):
            raise ValueError('无效的 GeoRT 允许姿态域')
        if np.any(low<robot.lower) or np.any(high>robot.upper):raise ValueError('允许姿态域必须位于原生关节范围内')
        self.register_buffer('offset',torch.tensor((low-robot.lower)/(robot.upper-robot.lower)*2-1))
        self.register_buffer('span',torch.tensor((high-low)/(robot.upper-robot.lower)))

    def forward(self,x):return (self.base(x)+1)*self.span+self.offset


def create_ik(robot,config):
    mode=config.get('training_mode','official')
    if mode not in ('official',MODE):raise ValueError('未知的 GeoRT 训练模式')
    if config.get('output_domain'):return DomainIK(robot,config['output_domain'])
    if mode=='official':return reference('model').IKModel(robot.spec['keypoint_joints'])
    raise ValueError('缺少 GeoRT 输出姿态域')

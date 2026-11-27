"""Rigid MCP registration and bounded wrist residuals in the observed glove frame.

Only robot poses move. Observed EMF, taxels and contact anchors are unchanged.
Geometry networks continue to receive native q20 / inverse-FK link coordinates.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .cart_wrist import CartWristNetwork,so3_exp,relative_rotation,smooth_bound_unit
from .cart_network import CartNetwork

VERSION='fixed_mcp_registration_residual_wrist_v1'
MCP=(5,9,13,17)  # index, middle, ring, little; exclude articulated thumb
LIMITS=(.02,.02,.02,.2,.2,.2)
STEPS=(.001,.001,.001,.01,.01,.01)
FREE_WRIST_STEPS=(.02,.02,.02,.2,.2,.2)


def robot_palm_landmarks(hand,q):
    """Use fixed first-joint anchors, not downstream link origins moved by flexion."""
    p,r=hand.frames(q)[:2]
    points=[p[:,0]]
    for finger in range(5):
        owner=1+4*finger;body=hand.ids[owner]
        pivot=hand.tensor(hand.robot.model.jnt_pos[hand.robot.model.body_jntadr[body]])
        anchor=p[:,owner]+(r[:,owner]@pivot[...,None]).squeeze(-1)
        points.extend((anchor,anchor,anchor,anchor))
    return torch.stack(points,1)


def palm_registration(robot,glove):
    from retarget.baseline.geort.adapter import anatomical_basis
    # Longitudinal metacarpal direction fixes the poorly conditioned rotation
    # around the nearly collinear MCP row. Wrist position is not fitted.
    rotation=anatomical_basis(glove)@anatomical_basis(robot).T
    translation=glove[list(MCP)].mean(0)-rotation@robot[list(MCP)].mean(0)
    return rotation,translation


def alignment_from_topology(hand,path):
    from retarget.baseline.geort.adapter import SourceAdapter
    from scipy.spatial.transform import Rotation
    topology=json.loads(Path(path).read_text());adapter=SourceAdapter(topology)
    joints=np.asarray(topology['rest_joints_m'])[adapter.indices]
    source=(joints-joints[0])@adapter.transform
    with torch.no_grad():robot=robot_palm_landmarks(hand,torch.zeros((1,20),device=hand.device))[0].cpu().numpy()
    rotation,translation=palm_registration(robot,source)
    aligned=robot@rotation.T+translation
    return dict(version=VERSION,landmark_indices=list(MCP),fit='metacarpal_longitudinal_transverse_axes_and_MCP_centroid_no_scale_train_calibration_only',
        translation_m=translation.tolist(),rotation_vector_rad=Rotation.from_matrix(rotation).as_rotvec().tolist(),
        rotation_matrix=rotation.tolist(),robot_mcp_m=robot[list(MCP)].tolist(),glove_mcp_m=source[list(MCP)].tolist(),
        mcp_before_mm=(np.linalg.norm(robot[list(MCP)]-source[list(MCP)],axis=-1)*1000).tolist(),
        mcp_after_mm=(np.linalg.norm(aligned[list(MCP)]-source[list(MCP)],axis=-1)*1000).tolist(),
        residual_component_limits=list(LIMITS),residual_step_scales=list(STEPS),prior_weight=0.,
        target_frame='fixed_glove_wrist_canonical_axes',targets_transformed_by_robot=False)


def rotation_vector(matrix):
    sine=torch.stack((matrix[...,2,1]-matrix[...,1,2],matrix[...,0,2]-matrix[...,2,0],
                      matrix[...,1,0]-matrix[...,0,1]),-1)*.5
    cosine=(matrix.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    angle=torch.atan2(sine.norm(dim=-1),cosine.clamp(-1,1))
    return sine/torch.sinc(angle/torch.pi).clamp_min(1.e-4)[...,None]


def wrist_residual(state,base):
    return torch.cat((state[...,20:23]-base[:3],relative_rotation(state[...,23:26],base[3:])), -1)


def unbounded_wrist_alignment(alignment):
    """Keep MCP calibration, replace wrist hard limits with numerical scales."""
    return {**alignment, 'wrist_mode':'unbounded',
            'residual_component_limits':None, 'normalization_scales':list(LIMITS),
            'residual_step_scales':list(FREE_WRIST_STEPS),
            'wrist_transition':'unbounded_translation_and_SO3_increment',
            'prior_weight':0.}


class PalmAlignedNetwork(CartWristNetwork):
    """Public q26 states are absolute; the policy's wrist coordinates are residuals."""
    def __init__(self,*args,alignment,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('palm_base',torch.tensor(alignment['translation_m']+alignment['rotation_vector_rad'],dtype=torch.float32))
        self.wrist_limits.copy_(torch.tensor(LIMITS));self.wrist_step.copy_(torch.tensor(STEPS))
        self.wrist_mode=alignment.get('wrist_mode','bounded')
        if self.wrist_mode not in ('bounded','unbounded'):raise ValueError('Unknown wrist mode')
        if self.wrist_mode=='unbounded':
            # This buffer is now an input normalization scale, not a hard limit.
            self.wrist_step.copy_(torch.tensor(FREE_WRIST_STEPS))
            self.wrist_decoder[-1]=torch.nn.Identity()

    def initial_state(self,q):
        if q.shape[-1]==26:return q
        if q.shape[-1]!=20:raise ValueError('Expected q20 initial state')
        return torch.cat((q,self.palm_base.expand(*q.shape[:-1],6)),-1)

    def normalize(self,q):
        return torch.cat((CartNetwork.normalize(self,q[...,:20]),wrist_residual(q,self.palm_base)/self.wrist_limits),-1)

    def unnormalize(self,u):
        residual=u[...,20:]*self.wrist_limits
        rotation=rotation_vector(so3_exp(residual[...,3:])@so3_exp(self.palm_base[3:]))
        return torch.cat((CartNetwork.unnormalize(self,u[...,:20]),residual[...,:3]+self.palm_base[:3],rotation),-1)

    def apply_delta(self,q,delta,*,normalized=False):
        current=self.normalize(q)
        finger=smooth_bound_unit(current[...,:20]+delta[...,:20])
        if self.wrist_mode=='unbounded':
            fingers=CartNetwork.unnormalize(self,finger)
            translation=q[...,20:23]+delta[...,20:23]*self.wrist_step[:3]
            rotation=rotation_vector(so3_exp(delta[...,23:26]*self.wrist_step[3:])@so3_exp(q[...,23:26]))
            result=torch.cat((fingers,translation,rotation),-1)
            return self.normalize(result) if normalized else result
        wrist=smooth_bound_unit(current[...,20:]+delta[...,20:]*self.wrist_step/self.wrist_limits)
        u=torch.cat((finger,wrist),-1)
        return u if normalized else self.unnormalize(u)

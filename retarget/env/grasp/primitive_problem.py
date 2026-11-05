"""Anatomical CEM coordinates and an immutable wrist for the primitive atlas."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .personalized_primitive import FrozenPersonalizedDecoder
from .mano_solver import CHAINS,TIPS
from moxian_web.hand_anatomy import AnatomicalJoints

class AnatomicalPrimitiveDecoder(FrozenPersonalizedDecoder):
    def __init__(self,snapshot,device='cuda:0'):
        super().__init__(snapshot,device,45)
        a=np.load(str(snapshot)+'/geometry.npz')
        # Construct in the original left anatomy, then reflect rotation axes.
        axial=np.array([1.,-1.,-1.]);mean=self.base.pose_mean.detach().cpu().numpy().reshape(15,3)*axial
        anatomy=AnatomicalJoints(a['rest_joints'],a['rest_vertices'],CHAINS,TIPS,self.base.palm_normal.cpu().numpy(),mean)
        anatomy.basis=axial[None,:,None]*anatomy.basis
        self.anatomy=anatomy
        active=np.flatnonzero((anatomy.upper-anatomy.lower).ravel()>1e-8)
        self.register_buffer('active_anatomical',torch.tensor(active,device=device))
        self.register_buffer('anatomical_basis',torch.tensor(anatomy.basis,dtype=torch.float32,device=device))
        self.register_buffer('anatomical_low',torch.tensor(anatomy.lower.ravel()[active],dtype=torch.float32,device=device))
        self.register_buffer('anatomical_high',torch.tensor(anatomy.upper.ravel()[active],dtype=torch.float32,device=device))
        self.components=len(active)

    def angles(self,variables):
        shape=variables.shape[:-1];theta=variables.new_zeros((*shape,45));theta[...,self.active_anatomical]=variables[...,6:];theta=theta.reshape(*shape,15,3)
        x,y,z=(theta*.5).unbind(-1);cx,sx=x.cos(),x.sin();cy,sy=y.cos(),y.sin();cz,sz=z.cos(),z.sin()
        # Quaternion of Rx @ Ry @ Rz, then stable quaternion logarithm.
        v=torch.stack((sx*cy*cz+cx*sy*sz,cx*sy*cz-sx*cy*sz,cx*cy*sz+sx*sy*cz),-1);w=cx*cy*cz-sx*sy*sz
        n=v.norm(dim=-1);factor=torch.where(n>1e-6,2*torch.atan2(n,w)/n.clamp_min(1e-8),2+n.square()/3)
        local=v*factor[...,None];return (self.anatomical_basis@local[...,None]).squeeze(-1).flatten(-2)

    def from_pose(self,pose):
        theta=self.anatomy.angles(np.asarray(pose).reshape(-1,15,3));theta=np.clip(theta,self.anatomy.lower,self.anatomy.upper)
        return theta.reshape(-1,45)[:,self.active_anatomical.cpu().numpy()]


def frozen_wrist_problem(mean,decoder,reference=None,*,continuity_radius=.12):
    """Every candidate has exactly the declared wrist; no six-DOF escape."""
    from retarget.solver.base import Problem
    low=mean.clone();high=mean.clone();low[:,6:]=decoder.anatomical_low;high[:,6:]=decoder.anatomical_high
    if reference is not None:
        low[:,6:]=torch.maximum(low[:,6:],reference[:,6:]-continuity_radius)
        high[:,6:]=torch.minimum(high[:,6:],reference[:,6:]+continuity_radius)
    sigma=mean.new_tensor([.005]*3+[.05]*3+[.12]*decoder.components).expand_as(mean)
    def project(v):return torch.maximum(low[:,None],torch.minimum(high[:,None],v))
    return Problem(mean,sigma,None,project)

def canonicalize_wuji_wrist(qpos, reference_index):
    """Fix the wrist using a world-frame change, preserving hand/object geometry.

    The cached object is at identity. Transform it with the same rigid transform
    as the hand; freezing the six wrist coordinates alone would change a grasp.
    Returned object quaternions use MuJoCo WXYZ and metre translations.
    """
    qpos=np.asarray(qpos,dtype=np.float64)
    rotations=Rotation.from_euler('XYZ',qpos[:,3:6]).as_matrix()
    relative=rotations[reference_index]@np.swapaxes(rotations,1,2)
    origin=qpos[reference_index,:3]-np.einsum('nij,nj->ni',relative,qpos[:,:3])
    pose=np.concatenate((origin,Rotation.from_matrix(relative).as_quat(scalar_first=True)),axis=1)
    fixed=qpos.copy();fixed[:,:6]=qpos[reference_index,:6]
    return fixed,pose


def reference_wrist_problem(mean,decoder):
    """Calibrate the reference wrist once, then freeze its accepted result."""
    from retarget.solver.base import Problem
    radius=mean.new_tensor([.025]*3+[.35]*3+[0.]*decoder.components)
    low=mean-radius;high=mean+radius
    low[:,6:]=decoder.anatomical_low;high[:,6:]=decoder.anatomical_high
    sigma=mean.new_tensor([.008]*3+[.1]*3+[.12]*decoder.components).expand_as(mean)
    def project(v):return torch.maximum(low[:,None],torch.minimum(high[:,None],v))
    return Problem(mean,sigma,None,project)

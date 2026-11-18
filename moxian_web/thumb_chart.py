"""Differentiable thumb coordinates from the actual bounded source skeleton.

Fixed SDK-to-MANO link binding rotations are not anatomical joint angles.
In particular, clipping their roll as thumb twist destroys right-hand control.
Use the six source joint coordinates and their calibrated local axes instead.
"""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from retarget.baseline.card.math import exp_so3
from scripts.calibrate_wuji_mano import CHAINS

SUFFIXES=('cmc_abduction','cmc_flexion','cmc_opposition','mcp_abduction','mcp_flexion','ip_flexion')
SLOTS=((12,0),(12,1),(12,2),(13,1),(13,0),(14,0))


class ThumbChart:
    def __init__(self, model, rows):
        documents=[r['skeleton']['thumb_kinematics'] for r in rows]
        doc=documents[0]
        self.geometry_applied=bool(doc.get('geometry_applied'))
        if any(json.dumps(d,sort_keys=True)!=json.dumps(doc,sort_keys=True) for d in documents):
            raise ValueError('Cannot combine different thumb kinematic models')
        if doc.get('revision')!='bounded_urdf_thumb_v1' or doc.get('side')!=model.side:
            raise ValueError('Invalid thumb kinematic model/side')
        joints=doc['joints']
        names=[model.side+'_thumb_'+s for s in SUFFIXES]
        if [j['name'] for j in joints]!=names:raise ValueError('Invalid thumb joint order')
        rotations=np.array([j['rotation'] for j in joints]);axes=np.array([j['axis'] for j in joints]);bounds=np.array([j['bounds'] for j in joints])
        if rotations.shape!=(6,3,3) or axes.shape!=(6,3) or bounds.shape!=(6,2) or not all(np.isfinite(a).all() for a in (rotations,axes,bounds)):
            raise ValueError('Invalid thumb axes/bounds')
        if not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-6) or np.any(np.linalg.det(rotations)<.999) or not np.allclose(np.linalg.norm(axes,axis=1),1) or np.any(bounds[:,0]>bounds[:,1]):
            raise ValueError('Invalid thumb rotations/bounds')
        model.official_fk.bind(model,CHAINS)
        self.t=lambda a:torch.as_tensor(a,dtype=torch.float32,device=model.device)
        self.origins=self.t(rotations);self.axes=self.t(axes)
        self.axis_references=self.t(np.eye(3)[np.abs(axes).argmin(-1)])
        self.alignment=self.t(model.official_fk.sdk_to_mano)
        self.corrections=self.t(model.official_fk.corrections[list(CHAINS[0])])
        self.bounds=bounds
        self.seeds=np.array([[dict(zip(r['skeleton']['joint_names'],r['skeleton']['joint_positions_rad']))[n] for n in names] for r in rows])
        if not np.isfinite(self.seeds).all() or np.any(self.seeds<bounds[:,0]-1e-5) or np.any(self.seeds>bounds[:,1]+1e-5):
            raise ValueError('Source thumb joint coordinates exceed bounds')
        # A new source model must agree with the same-frame source rotations;
        # never silently use a packaged default with a different zero pose.
        expected=self.rotations(self.t(self.seeds))
        self.seed_rotations=expected

    def rotations(self,q,geometry=None):
        r=torch.eye(3,dtype=q.dtype,device=q.device).expand(*q.shape[:-1],3,3)
        outputs=[]
        for i in range(6):
            axis=self.axes[i]
            if geometry is not None and i<3 and not self.geometry_applied:
                reference=self.axis_references[i]
                tangent=torch.nn.functional.normalize(torch.linalg.cross(axis,reference),dim=0)
                tilt=geometry[37+2*i]*tangent+geometry[38+2*i]*torch.linalg.cross(axis,tangent)
                axis=exp_so3(tilt)@axis
            r=r@self.origins[i]@exp_so3(q[...,i,None]*axis)
            if i in (2,4,5):outputs.append(self.alignment@r@self.corrections[len(outputs)])
        return torch.stack((outputs[0],outputs[0].transpose(-1,-2)@outputs[1],outputs[1].transpose(-1,-2)@outputs[2]),-3)

    def decode(self,angles,geometry=None):
        return self.rotations(torch.stack([angles[...,j,k] for j,k in SLOTS],-1),geometry)


def source_thumb_chart(model,rows):
    available=[bool(r.get('skeleton',{}).get('thumb_kinematics')) for r in rows]
    if any(available) and not all(available):raise ValueError('Missing thumb kinematics in part of the sequence')
    if not any(available):return None
    if model.official_fk is None:raise ValueError('Source thumb chart requires the official MANO binding')
    return ThumbChart(model,rows)

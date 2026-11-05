"""Frozen glove-personalized MANO geometry for offline primitive CEM mapping.

The live left hand is immutable. A reflection creates its right-handed copy;
axial rotation vectors and triangle winding are reflected along with positions.
"""
from pathlib import Path
import numpy as np
import torch
from .mano_solver import ManoDecoder, CHAINS, TIPS
from .pose_gpu import WujiPoseAdapter
from retarget.env.geometry import closest_triangles
from scripts.inspect_shadow_mano_mapping import palm, bone_frame, barycentric

FINGERS=('thumb','index','middle','ring','pinky')
PREFIX=('th','ff','mf','rf','lf')

class FrozenPersonalizedDecoder(ManoDecoder):
    def __init__(self, snapshot, device='cuda:0', components=24):
        snapshot=Path(snapshot)
        super().__init__(snapshot/'MANO_LEFT.pkl',device,components)
        a=np.load(snapshot/'geometry.npz',allow_pickle=False)
        reflect=self.rest.new_tensor([-1.,1.,1.]);axial=self.rest.new_tensor([1.,-1.,-1.])
        self.rest=torch.as_tensor(a['rest_vertices'],device=device)*reflect
        self.joints_rest=torch.as_tensor(a['rest_joints'],device=device)*reflect
        self.relative_joints=torch.stack([self.joints_rest[j] if p<0 else self.joints_rest[j]-self.joints_rest[p] for j,p in enumerate(self.base.parents)])
        self.basis=(self.basis.reshape(components,15,3)*axial).reshape(components,45)
        self.base.pose_mean=(self.base.pose_mean.reshape(15,3)*axial).flatten()
        self.base.faces=self.base.faces[:,[0,2,1]].copy()
        self.base.posedirs.zero_()  # Same fixed local geometry as the saved WebUI.
        self.register_buffer('skin_tip_anchors',torch.as_tensor(a['tip_anchors'],device=device)*reflect)

    def rest_keypoints(self):
        result=[self.joints_rest[0]]
        for f,c in enumerate(CHAINS):
            result.extend(self.joints_rest[list(c)]);result.append(self.skin_tip_anchors[f])
        return torch.stack(result).cpu().numpy()


def anatomical_contacts(decoder, model, record):
    """Object-independent, same-finger/segment mapping of cached WUJI slots."""
    import mujoco
    data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
    native=WujiPoseAdapter(model,'cpu');source=native.keypoints(torch.zeros(26,dtype=torch.float64)).numpy()
    dest=decoder.rest_keypoints();sp=palm(source);mp=palm(dest)
    vertices=decoder.rest.cpu().numpy();faces=np.asarray(decoder.base.faces);weights=decoder.base.weights.cpu().numpy();out=[]
    for i,(body,local,ln) in enumerate(zip(record['body_names'],record['local_points'],record['local_normals'])):
        bid=model.body(body).id;r=data.xmat[bid].reshape(3,3);p=data.xpos[bid]+r@local;n=r@ln;n/=np.linalg.norm(n)
        if body=='right_palm_link':
            f=5;j=0;ratio=np.linalg.norm(dest[5]-dest[17])/np.linalg.norm(source[5]-source[17]);desired=dest[0]+mp@(sp.T@(p-source[0])*ratio);normal=mp@sp.T@n;allowed=weights[:,0]>.35;shadow='rh_palm';finger='palm'
        else:
            f=int(body.split('finger')[1].split('_')[0])-1;k=max(0,int(body[-1])-2);j=CHAINS[f][k]
            start,end=source[1+4*f+k:3+4*f+k];ds,de=dest[1+4*f+k:3+4*f+k]
            sb=bone_frame(start,end,sp[:,0]);mb=bone_frame(ds,de,mp[:,0]);ratio=np.linalg.norm(de-ds)/np.linalg.norm(end-start)
            desired=ds+mb@(sb.T@(p-start)*ratio);normal=mb@sb.T@n
            allowed=(weights[:,j]>.25)&(weights[:,list(CHAINS[f])].sum(-1)>.65)
            shadow='rh_'+PREFIX[f]+('proximal','middle','distal')[k];finger=FINGERS[f]
        ids=np.flatnonzero(allowed[faces].sum(-1)>=2)
        if not len(ids):raise ValueError('No anatomical surface for '+body)
        tri=vertices[faces[ids]];points=closest_triangles(desired[None],tri)[0]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        score=((points-desired)**2).sum(-1)+.006**2*(1-normals@normal);k=int(score.argmin());face=int(ids[k]);bary=barycentric(points[k],vertices[faces[face]])
        out.append(dict(id=i+1,shadow_body=shadow,finger=finger,mano_joint_id=int(j),mano_face_id=face,mano_barycentric=bary.tolist(),wuji_body=body,wuji_local_point=local,wuji_local_normal=ln,anatomical_patch_restricted=True,neutral_mapping_error_mm=float(np.linalg.norm(points[k]-desired)*1000)))
    return out

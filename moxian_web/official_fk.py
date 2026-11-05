"""Fixed link-frame binding from official QnBot skeleton poses to MANO FK.

No inverse kinematics, joint projection, temporal filter or tactile input here.
The packaged left URDF supplies the zero-DOF link frames (URDF rpy = Rz Ry Rx).
SDK links drive LBS through fixed SE3 bindings. Three personalized MCP offsets
preserve palm breadth; thumb, small finger and distal bindings retain their origins.
There is no second skeleton solver or contact-dependent pose correction.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .skeleton import JOINT_NAMES,validate_pose

TRACKING_SOURCE='official_qnbot_skeleton_fk_v1'
BINDING_POLICY='official_links_personalized_mcp_se3_v4'
LINKS=[['thumb_metacarpal','thumb_proximal','thumb_distal'],*[ [f'{f}_{s}' for s in ('proximal','intermediate','distal')] for f in ('index','middle','ring','little')]]

def official_urdf():
    root=Path(__file__).resolve().parents[1]/'.venv-moxian-ik/lib'
    paths=list(root.glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*/share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio/left_skeleton.urdf'))
    if len(paths)!=1:raise ValueError('无法唯一定位官方左手骨架 URDF；停止 MANO 更新')
    return paths[0]

def urdf_reference(path):
    path=Path(path);tree=ET.parse(path).getroot();transforms={}
    children={j.find('child').get('link') for j in tree.findall('joint')}
    roots={l.get('name') for l in tree.findall('link')}-children
    if len(roots)!=1:raise ValueError('官方 URDF 根节点不唯一')
    transforms[roots.pop()]=np.eye(4);pending=list(tree.findall('joint'))
    while pending:
        progressed=False
        for joint in list(pending):
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in transforms:continue
            origin=joint.find('origin');tf=np.eye(4)
            if origin is not None:
                tf[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            transforms[child]=transforms[parent]@tf;pending.remove(joint);progressed=True
        if not progressed:raise ValueError('官方 URDF 关节树无效')
    wrist=np.linalg.inv(transforms['left_wrist']);poses=[wrist@transforms['left_'+n] for n in JOINT_NAMES]
    return {'timestamp_ms':0.,'hand_side':'left','coordinate_frame':'wrist_local','names':JOINT_NAMES.copy(),'positions_local':[p[:3,3].tolist() for p in poses],'orientations_local_xyzw':[Rotation.from_matrix(p[:3,:3]).as_quat().tolist() for p in poses]}

def basis(direction,normal):
    z=np.asarray(direction,dtype=float);z=z/np.linalg.norm(z)
    y=np.asarray(normal,dtype=float)-z*(normal@z);n=np.linalg.norm(y)
    if n<1e-7:raise ValueError('骨段与掌面参考退化，无法建立固定 TF')
    y/=n;return np.stack((np.cross(y,z),y,z),axis=1)

def palm_basis(points):
    forward=points[2]-points[0];forward/=np.linalg.norm(forward)
    side=points[1]-points[4];side-=forward*(side@forward);side/=np.linalg.norm(side)
    return np.stack((side,forward,np.cross(side,forward)),axis=1)

class OfficialSkeletonFK:
    def __init__(self,path=None):
        self.path=Path(path) if path else official_urdf();self.reference=urdf_reference(self.path)
        self.asset_hash=hashlib.sha256(self.path.read_bytes()).hexdigest();self.version=None

    def bind(self,model,chains):
        if self.version==model.shape_version:return
        ref=self.reference;index={n:i for i,n in enumerate(ref['names'])};p=np.array(ref['positions_local']);r=Rotation.from_quat(ref['orientations_local_xyzw']).as_matrix()
        sdk_roots=p[[index[n[0]] for n in LINKS]];mano_roots=model.rest_j_np[[c[0] for c in chains]]-model.rest_j_np[0]
        sb=palm_basis(np.vstack((np.zeros(3),sdk_roots[1:])));mb=palm_basis(np.vstack((np.zeros(3),mano_roots[1:])))
        self.mano_to_sdk=sb@mb.T;self.sdk_to_mano=self.mano_to_sdk.T
        self.sdk_bases=sdk_roots;self.corrections=np.tile(np.eye(3),(16,1,1));self.indices={}
        self.translations=np.zeros((16,3))
        for f,chain in enumerate(chains):
            names=LINKS[f]+[('thumb' if f==0 else ('index','middle','ring','little')[f-1])+'_tip']
            for k,j in enumerate(chain):
                i=index[names[k]];end=p[index[names[k+1]]]
                mend=model.rest_j_np[chain[k+1]] if k<2 else model.tip_anchors[f]
                align=basis(end-p[i],sb[:,2])@basis(mend-model.rest_j_np[j],mb[:,2]).T
                self.corrections[j]=r[i].T@align;self.indices[j]=i
                # A fixed TF includes translation. Copying the SDK origin here
                # silently replaces the personalized MCP spacing and lengths.
                # Only the four MCP surface anchors need the personalized
                # palm breadth. Preserve the existing thumb and distal sensor
                # bindings, so a proximal repair cannot move a fitted pinch.
                if 0<f<4 and k==0:
                    self.translations[j]=r[i].T@(self.mano_to_sdk@(model.rest_j_np[j]-model.rest_j_np[0])-p[i])
        self.version=model.shape_version
        self.binding_hash=hashlib.sha256(BINDING_POLICY.encode()+self.asset_hash.encode()+model.shape_version.encode()+np.round(self.corrections,10).tobytes()+np.round(self.translations,10).tobytes()).hexdigest()[:16]

    def parameters(self,model,frame,chains):
        self.bind(model,chains);s=validate_pose(frame['skeleton'])
        if abs(s['timestamp_ms']-frame['timestamp_ms'])>25:raise ValueError('官方骨架与输入时间差超过 25 ms')
        index={n:i for i,n in enumerate(s['names'])};p=np.array(s['positions_local']);r=Rotation.from_quat(s['orientations_local_xyzw']).as_matrix();rw=r[index['wrist']]
        root=self.sdk_to_mano@rw@self.mano_to_sdk;global_r=[root]+[None]*15;local=np.tile(np.eye(3),(15,1,1));centres=np.zeros((16,3))
        ref_names=self.reference['names']
        for chain in chains:
            for j in chain:
                current=index[ref_names[self.indices[j]]];global_r[j]=self.sdk_to_mano@r[current]@self.corrections[j]
                local[j-1]=global_r[model.parents[j]].T@global_r[j]
                centres[j]=self.sdk_to_mano@(p[current]-p[index['wrist']]+r[current]@self.translations[j])
        bases=p[[index[n[0]] for n in LINKS]]
        offsets=(bases@rw-self.sdk_bases)@self.mano_to_sdk
        pose=Rotation.from_matrix(local).as_rotvec().astype(np.float32)
        # decode() applies root after LBS, so centres here are root-local.
        centres=centres@root
        return pose,root.astype(np.float32),offsets.astype(np.float32),np.array(global_r),centres.astype(np.float32)

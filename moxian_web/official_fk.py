"""Fixed link-frame binding from official QnBot skeleton poses to MANO FK.

No inverse kinematics, joint projection, temporal filter or tactile input here.
The packaged left URDF supplies the zero-DOF link frames (URDF rpy = Rz Ry Rx).
SDK rotations drive calibrated, fixed-length finger FK and LBS. Thumb pad roll
uses its own link frame; a palm normal cannot stand in for the thumb pad axis.
There is no second skeleton solver or contact-dependent pose correction.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .skeleton import JOINT_NAMES,validate_pose

TRACKING_SOURCE='official_qnbot_skeleton_fk_v1'
BINDING_POLICY='official_rotations_calibrated_rigid_finger_fk_thumb_pad_v8'
LINKS=[['thumb_metacarpal','thumb_proximal','thumb_distal'],*[ [f'{f}_{s}' for s in ('proximal','intermediate','distal')] for f in ('index','middle','ring','little')]]

def official_urdf(side="left"):
    root=Path(__file__).resolve().parents[1]/'.venv-moxian-ik/lib'
    paths=list(root.glob(f'python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*/share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio/{side}_skeleton.urdf'))
    if len(paths)!=1:raise ValueError('无法唯一定位官方左手骨架 URDF；停止 MANO 更新')
    return paths[0]

def urdf_reference(path,side="left"):
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
    wrist=np.linalg.inv(transforms[side+'_wrist']);poses=[wrist@transforms[side+'_'+n] for n in JOINT_NAMES]
    return {'timestamp_ms':0.,'hand_side':side,'coordinate_frame':'wrist_local','names':JOINT_NAMES.copy(),'positions_local':[p[:3,3].tolist() for p in poses],'orientations_local_xyzw':[Rotation.from_matrix(p[:3,:3]).as_quat().tolist() for p in poses]}

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
    def __init__(self,path=None,side="left",fixed_binding=None):
        from .hand_side import check_side
        self.side=check_side(side)
        self.path=Path(path) if path else official_urdf(side);self.reference=urdf_reference(self.path,side)
        self.asset_hash=hashlib.sha256(self.path.read_bytes()).hexdigest();self.version=None
        self.fixed_binding=fixed_binding
        self.source_reference_hash=None

    def bind(self,model,chains):
        if self.version==model.shape_version:return
        ref=self.reference;index={n:i for i,n in enumerate(ref['names'])};p=np.array(ref['positions_local']);r=Rotation.from_quat(ref['orientations_local_xyzw']).as_matrix()
        reference_joints=getattr(model,'pose_reference_joints',model.rest_j_np)
        reference_vertices=getattr(model,'pose_reference_vertices',model.rest_np)
        sdk_roots=p[[index[n[0]] for n in LINKS]];basis_joints=getattr(model,'palm_basis_joints',reference_joints);mano_roots=basis_joints[[c[0] for c in chains]]-basis_joints[0]
        sb=palm_basis(np.vstack((np.zeros(3),sdk_roots[1:])));mb=palm_basis(np.vstack((np.zeros(3),mano_roots[1:])))
        self.mano_to_sdk=sb@mb.T;self.sdk_to_mano=self.mano_to_sdk.T
        self.sdk_bases=sdk_roots;self.corrections=np.tile(np.eye(3),(16,1,1));self.indices={}
        self.translations=np.zeros((16,3))
        self.palm_anchors={}
        # Both sides declare a -y pad axis in the tip frame. Only the left
        # asset has a fixed Rz(pi) between distal and tip. Derive the distal
        # pad axis from that transform instead of assuming left-hand +y.
        thumb_pad=r[index['thumb_distal']].T@r[index['thumb_tip']]@np.array([0.,-1.,0.])
        for f,chain in enumerate(chains):
            names=LINKS[f]+[('thumb' if f==0 else ('index','middle','ring','little')[f-1])+'_tip']
            for k,j in enumerate(chain):
                i=index[names[k]];end=p[index[names[k+1]]]
                mend=reference_joints[chain[k+1]] if k<2 else reference_vertices[[744,320,443,554,671][f]]
                # Thumb pad axes differ between the left/right tip assets.
                # Keep each side's actual local pad axis through the binding.
                sdk_normal=r[i]@thumb_pad if f==0 else sb[:,2]
                mano_normal=model.anatomy.palmar_normal if f==0 else mb[:,2]
                align=basis(end-p[i],sdk_normal)@basis(mend-reference_joints[j],mano_normal).T
                self.corrections[j]=r[i].T@align;self.indices[j]=i
                if model.rigid_finger_fk and f>0:
                    yaw=Rotation.from_rotvec(model.anatomy.palmar_normal*model.finger_axis_yaw_rad[f-1]).as_matrix()
                    self.corrections[j]=self.corrections[j]@yaw
                # A fixed TF includes translation. Copying the SDK origin here
                # silently replaces the personalized MCP spacing and lengths.
                # Only the four MCP surface anchors need the personalized
                # palm breadth. Preserve the existing thumb and distal sensor
                # bindings, so a proximal repair cannot move a fitted pinch.
                if not model.rigid_finger_fk and 0<f<4 and k==0:
                    self.translations[j]=r[i].T@(self.mano_to_sdk@(model.rest_j_np[j]-model.rest_j_np[0])-p[i])
            if model.rigid_finger_fk and f>0:
                meta=index[('index','middle','ring','little')[f-1]+'_metacarpal']
                anchor=r[meta].T@(self.mano_to_sdk@(model.rest_j_np[chain[0]]-model.rest_j_np[0])-p[meta])
                self.palm_anchors[f]=(meta,anchor)
        self.version=model.shape_version
        if self.fixed_binding:
            self.mano_to_sdk=np.asarray(self.fixed_binding['mano_to_wrist_rotation'])
            self.sdk_to_mano=self.mano_to_sdk.T
            self.corrections=np.asarray(self.fixed_binding['binding_corrections'])
            self.translations[:]=0
        self.binding_hash=hashlib.sha256(BINDING_POLICY.encode()+self.asset_hash.encode()+model.shape_version.encode()+np.round(self.corrections,10).tobytes()+np.round(self.translations,10).tobytes()).hexdigest()[:16]

    def parameters(self,model,frame,chains):
        personalized=frame.get('skeleton',{}).get('personalized_model')
        source_path=frame.get('skeleton',{}).get('source_urdf_path')
        if not personalized and self.fixed_binding:
            updated=OfficialSkeletonFK(source_path,side=self.side)
            self.__dict__.update(updated.__dict__)
        elif not personalized and source_path and str(self.path)!=source_path:
            updated=OfficialSkeletonFK(source_path,side=self.side)
            self.__dict__.update(updated.__dict__)
        if personalized and str(self.path)!=personalized['urdf_path']:
            updated=OfficialSkeletonFK(personalized['urdf_path'],self.side,personalized)
            self.__dict__.update(updated.__dict__)
        source_reference=frame.get('skeleton',{}).get('source_reference')
        source_hash=frame.get('skeleton',{}).get('source_reference_hash')
        if not personalized and source_reference and source_hash!=self.source_reference_hash:
            validate_pose(source_reference,self.side)
            self.reference=source_reference;self.source_reference_hash=source_hash;self.version=None
        self.bind(model,chains);s=validate_pose(frame['skeleton'],self.side)
        if abs(s['timestamp_ms']-frame['timestamp_ms'])>25:raise ValueError('官方骨架与输入时间差超过 25 ms')
        index={n:i for i,n in enumerate(s['names'])};p=np.array(s['positions_local']);r=Rotation.from_quat(s['orientations_local_xyzw']).as_matrix();rw=r[index['wrist']]
        root=self.sdk_to_mano@rw@self.mano_to_sdk;global_r=[root]+[None]*15;local=np.tile(np.eye(3),(15,1,1));centres=np.zeros((16,3))
        ref_names=self.reference['names']
        for chain in chains:
            for j in chain:
                current=index[ref_names[self.indices[j]]];global_r[j]=self.sdk_to_mano@r[current]@self.corrections[j]
                local[j-1]=global_r[model.parents[j]].T@global_r[j]
                centres[j]=self.sdk_to_mano@(p[current]-p[index['wrist']]+r[current]@self.translations[j])
        if model.rigid_finger_fk:
            # Official orientations, calibrated rigid segment lengths. A base
            # correction moves the entire finger chain, never just its MCP.
            for f,chain in enumerate(chains):
                if f==0:
                    # CMC attaches to the personalized palm. Reusing SDK MCP/IP
                    # origins with different MANO rest lengths stretches skin
                    # between its rigid links and changes the effective length.
                    centres[chain[0]]=root@(model.rest_j_np[chain[0]]-model.rest_j_np[0])
                else:
                    meta,anchor=self.palm_anchors[f];current=index[ref_names[meta]]
                    centres[chain[0]]=self.sdk_to_mano@(p[current]-p[index['wrist']]+r[current]@anchor)
                for k,joint in enumerate(chain[1:],1):
                    parent=chain[k-1]
                    centres[joint]=centres[parent]+global_r[parent]@(model.rest_j_np[joint]-model.rest_j_np[parent])
        bases=p[[index[n[0]] for n in LINKS]]
        offsets=(bases@rw-self.sdk_bases)@self.mano_to_sdk
        pose=Rotation.from_matrix(local).as_rotvec().astype(np.float32)
        # decode() applies root after LBS, so centres here are root-local.
        centres=centres@root
        return pose,root.astype(np.float32),offsets.astype(np.float32),np.array(global_r),centres.astype(np.float32)

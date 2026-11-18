"""Differentiable Qnbot/OpenXR skin. No MANO assets or coordinate conversion.

SKN source coordinates and weights are immutable. The bone_extrinsics policy
derives longitudinal rest-skin stretch and inverse binds from 15 lengths only;
it never fits width, depth or vertex displacements. All output is metres
in the official wrist frame; MuJoCo is only used to read the delivered model.
"""
from __future__ import annotations
import hashlib,json,struct,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation

REVISION='qnbot_skin_v1'
FINGERS=('thumb','index','middle','ring','pinky')
SOURCE_FINGERS=('thumb','index','middle','ring','little')


def resource_root():
    explicit=os.environ.get('MOXIAN_QNBOT_RESOURCES')
    if explicit:
        root=Path(explicit)
        if (root/'resources/mujoco/right.xml').is_file():return root
        raise ValueError('MOXIAN_QNBOT_RESOURCES 缺少官方模型资源')
    root=Path(__file__).resolve().parents[1]
    paths=sorted((root/'.venv-moxian-ik/lib').glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*/share/qnbot_hand/hands/dynamic_openxr_hand'))
    if len(paths)!=1:raise ValueError('无法唯一定位 Qnbot 官方 mesh，请设置 MOXIAN_QNBOT_RESOURCES')
    return paths[0]


def read_skin(path):
    data=Path(path).read_bytes();offset=16
    nv,nt,nf,nb=struct.unpack_from('<4i',data)
    if min(nv,nf,nb)<=0 or nt not in (0,nv):raise ValueError('无效 SKN 头')
    def array(dtype,count,shape):
        nonlocal offset
        value=np.frombuffer(data,dtype=dtype,count=count,offset=offset).copy().reshape(shape)
        offset+=value.nbytes;return value
    vertices=array('<f4',nv*3,(nv,3));uv=array('<f4',nt*2,(nt,2));faces=array('<i4',nf*3,(nf,3))
    weights=np.zeros((nv,nb),np.float32);names=[];positions=[];quaternions=[]
    for bone in range(nb):
        names.append(data[offset:offset+40].split(b'\0',1)[0].decode());offset+=40
        positions.append(array('<f4',3,(3,)));quaternions.append(array('<f4',4,(4,)))
        n=struct.unpack_from('<i',data,offset)[0];offset+=4
        ids=array('<i4',n,(n,));w=array('<f4',n,(n,))
        if np.any(ids<0) or np.any(ids>=nv) or np.any(w<0):raise ValueError('无效 SKN 权重')
        weights[ids,bone]=w
    if offset!=len(data) or not np.isfinite(vertices).all() or not np.isfinite(weights).all() or not np.allclose(weights.sum(1),1,atol=1e-5):raise ValueError('SKN 长度／蒙皮权重不匹配')
    if faces.min()<0 or faces.max()>=nv:raise ValueError('SKN 面片索引越界')
    # SKN and MuJoCo quaternions are WXYZ; scipy is XYZW.
    rotation=Rotation.from_quat(np.asarray(quaternions)[:,[1,2,3,0]]).as_matrix().astype(np.float32)
    return dict(vertices=vertices,faces=faces,uv=uv,weights=weights,bone_names=names,
                bind_position=np.asarray(positions),bind_rotation=rotation,sha256=hashlib.sha256(data).hexdigest())


def exp_so3(v):
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    k=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)
    squared=v.square().sum(-1);theta=squared.clamp_min(1e-12).sqrt()
    # CUDA sinc JVPs promote tangents to float64 in PyTorch 2.10. Polynomial
    # branches also preserve finite derivatives at the neutral pose.
    a=torch.where(squared<1e-4,1-squared/6+squared.square()/120,torch.sin(theta)/theta)[...,None,None]
    b=torch.where(squared<1e-4,.5-squared/24+squared.square()/720,(1-torch.cos(theta))/squared.clamp_min(1e-12))[...,None,None]
    return torch.eye(3,device=v.device,dtype=v.dtype)+a*k+b*(k@k)


def normals(vertices,faces):
    tri=vertices[...,faces,:];area=torch.linalg.cross(tri[...,1,:]-tri[...,0,:],tri[...,2,:]-tri[...,0,:],dim=-1)
    out=torch.zeros_like(vertices)
    for k in range(3):out=out.index_add(-2,faces[:,k],area)
    return torch.nn.functional.normalize(out,dim=-1)


class QnbotSurface(nn.Module):
    def __init__(self,side='right',device='cpu',root=None,finger_twist=False):
        super().__init__()
        if side not in ('left','right'):raise ValueError('无效手侧')
        import mujoco
        self.side=side;self.root=Path(root) if root else resource_root();self.device=torch.device(device)
        self.xml=self.root/f'resources/mujoco/{side}.xml'
        skin=read_skin(self.root/f'resources/meshes/{side}/visual/{side}_hand_hd.skn')
        mj=mujoco.MjModel.from_xml_path(str(self.xml));self.mj_model=mj
        self.bone_names=skin['bone_names'];self.joint_names=[mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(mj.njnt)]
        if mj.nq!=25 or mj.nv!=25:raise ValueError('官方模型须为25自由度')
        self.body_names=[mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(mj.nbody)]
        self.bone_body=[self.body_names.index(n) for n in self.bone_names]
        self.parents=list(map(int,mj.body_parentid));self.body_joints=[list(range(int(mj.body_jntadr[i]),int(mj.body_jntadr[i]+mj.body_jntnum[i]))) for i in range(mj.nbody)]
        self.wrist_body=self.body_names.index(side+'_wrist')
        self.finger_owner=np.array([next((f for f,n in enumerate(SOURCE_FINGERS) if '_'+n+'_' in name),-1) for name in self.body_names])
        self.bone_finger=np.array([self.finger_owner[i] for i in self.bone_body]);self.joint_finger=np.asarray([self.finger_owner[i] for i in mj.jnt_bodyid])
        self.root_bodies=[self.body_names.index(side+'_'+f+'_metacarpal') for f in SOURCE_FINGERS]
        self.distal_bodies=[self.body_names.index(side+'_'+f+'_distal') for f in SOURCE_FINGERS]
        self.tip_bodies=[self.body_names.index(side+'_'+f+'_tip') for f in SOURCE_FINGERS]
        self.region_revision='qnbot_material_regions_v1';self.mapping=None;self._topology=None
        def buf(name,value,dtype=torch.float32):self.register_buffer(name,torch.as_tensor(value,device=self.device,dtype=dtype))
        buf('vertices0',skin['vertices']);buf('faces',skin['faces'],torch.long);buf('weights',skin['weights']);buf('bind_p0',skin['bind_position']);buf('bind_r0',skin['bind_rotation'])
        buf('joint_finger_ids',self.joint_finger,torch.long);buf('bone_ids',self.bone_body,torch.long);buf('distal_ids',self.distal_bodies,torch.long);buf('tip_ids',self.tip_bodies,torch.long)
        buf('body_p0',mj.body_pos);buf('body_r0',Rotation.from_quat(mj.body_quat[:,[1,2,3,0]]).as_matrix());buf('joint_axis0',mj.jnt_axis);buf('joint_p0',mj.jnt_pos);buf('limits',mj.jnt_range);buf('q0',mj.qpos0)
        from .thumb_limits import native_thumb_straight,extension_bounds,THUMB_EXTENSION_DEG
        self.thumb_straight=native_thumb_straight(self.root/f'resources/pinocchio/{side}.urdf',side)
        for name,straight in self.thumb_straight.items():
            i=self.joint_names.index(name);self.limits[i]=torch.tensor(extension_bounds(name,float(self.limits[i,0]),float(self.limits[i,1]),straight),device=self.device)
        self.thumb_extension_deg=THUMB_EXTENSION_DEG
        from .finger_limits import is_finger_dip,straight_angle,extension_bounds as finger_bounds,DIP_EXTENSION_DEG,PIP_EXTENSION_DEG,REVISION as FINGER_REVISION
        self.finger_straight={}
        for i,name in enumerate(self.joint_names):
            if not (is_finger_dip(name) or name.endswith('_pip_flexion')):continue
            body=next(b for b,js in enumerate(self.body_joints) if i in js)
            if not torch.allclose(self.joint_axis0[i],torch.tensor([0.,0.,1.],device=self.device)):raise ValueError('末节反折只支持原生Z屈伸轴')
            straight=straight_angle(self.body_r0[body].cpu().tolist());self.finger_straight[name]=straight
            self.limits[i]=torch.tensor(finger_bounds(name,float(self.limits[i,0]),float(self.limits[i,1]),straight),device=self.device)
        self.joint_limit_policy=dict(revision=FINGER_REVISION,dip_extension_deg=DIP_EXTENSION_DEG,pip_extension_deg=PIP_EXTENSION_DEG,thumb_extension_deg=THUMB_EXTENSION_DEG,straight_angles_rad=dict(self.finger_straight))
        buf('default_limits',self.limits.clone())
        self.geometry=nn.ParameterDict({k:nn.Parameter(torch.zeros(shape,device=self.device)) for k,shape in dict(length=(20,),width=(20,),depth=(20,),root=(5,3),axis=(25,3),residual=(len(skin['vertices']),3),mount_pos=(5,3),mount_rot=(5,3),alignment=(3,),translation=(3,)).items()})
        buf('align0',np.eye(3));buf('trans0',np.zeros(3));buf('mount_p0',np.zeros((5,3)));buf('mount_r0',np.tile(np.eye(3),(5,1,1)))
        # Every non-bone helper body inherits its nearest anatomical parent.
        owner=[]
        for i in range(mj.nbody):
            j=i
            while j not in self.bone_body and j!=0:j=self.parents[j]
            owner.append(self.bone_body.index(j) if j in self.bone_body else 0)
        self.body_bone=owner
        edges=np.unique(np.sort(np.concatenate([skin['faces'][:,[0,1]],skin['faces'][:,[1,2]],skin['faces'][:,[2,0]]]),1),axis=0)
        buf('edges',edges,torch.long)
        buf('vertex_finger',np.where(np.max(skin['weights'][:,1:],1)>.2,self.bone_finger[skin['weights'].argmax(1)],-1),torch.long)
        self.asset_hash=hashlib.sha256((REVISION+skin['sha256']).encode()+self.xml.read_bytes()+(self.root/f'resources/pinocchio/{side}_skeleton.urdf').read_bytes()).hexdigest()
        self.topology_hash=hashlib.sha256(skin['faces'].tobytes()).hexdigest()
        with torch.no_grad():self.register_buffer('base_r',self.fk(self.q0[None],personalized=False)[0][0]);self.register_buffer('base_p',self.fk(self.q0[None],personalized=False)[1][0])
        self._build_regions()
        import copy
        self._native_layout=dict(names=list(self.joint_names),body_joints=copy.deepcopy(self.body_joints),finger=self.joint_finger.copy(),
            tensors={name:getattr(self,name).detach().clone() for name in ('joint_axis0','joint_p0','limits','q0','default_limits')},policy=copy.deepcopy(self.joint_limit_policy))
        self.finger_twist_enabled=False
        if finger_twist:self.configure_finger_twist(True)

    def configure_finger_twist(self,enabled):
        """Add four virtual axial joints after MCP flexion, without asset edits."""
        import copy
        enabled=bool(enabled)
        if enabled==self.finger_twist_enabled:return
        base=self._native_layout;self.joint_names=list(base['names']);self.body_joints=copy.deepcopy(base['body_joints'])
        data={k:v.to(device=self.device,dtype=self.q0.dtype).clone() for k,v in base['tensors'].items()}
        owners=base['finger'].copy();self.joint_limit_policy=copy.deepcopy(base['policy'])
        if enabled:
            from .finger_limits import ROOT_TWIST_DEG
            axes=[]
            for i,finger in enumerate(('index','middle','ring','little')):
                body=self.body_names.index(self.side+'_'+finger+'_proximal');end=self.body_names.index(self.side+'_'+finger+'_intermediate')
                axis=self.body_p0[end];axes.append(axis/axis.norm())
                self.body_joints[body].append(len(self.joint_names));self.joint_names.append(self.side+'_'+finger+'_mcp_twist')
            data['joint_axis0']=torch.cat((data['joint_axis0'],torch.stack(axes)))
            data['joint_p0']=torch.cat((data['joint_p0'],torch.zeros((4,3),device=self.device,dtype=self.q0.dtype)))
            data['q0']=torch.cat((data['q0'],torch.zeros(4,device=self.device,dtype=self.q0.dtype)))
            bounds=torch.tensor([[-np.deg2rad(ROOT_TWIST_DEG),np.deg2rad(ROOT_TWIST_DEG)]]*4,device=self.device,dtype=self.q0.dtype)
            for key in ('limits','default_limits'):data[key]=torch.cat((data[key],bounds))
            owners=np.r_[owners,np.arange(1,5)]
            self.joint_limit_policy['finger_axial_twist']=dict(revision='mcp_post_flexion_bounded_axial_v2',allowance_deg=ROOT_TWIST_DEG,
                joint_names=self.joint_names[25:],axis_convention='proximal_to_pip',placement='after_MCP_abduction_and_flexion',
                periodic=False,online_angle_representation='bounded_radians')
        for key,value in data.items():setattr(self,key,value)
        self.joint_finger=owners;self.joint_finger_ids=torch.as_tensor(owners,device=self.device,dtype=torch.long)
        old=self.geometry['axis'];self.geometry['axis']=nn.Parameter(torch.zeros((len(self.joint_names),3),device=self.device,dtype=old.dtype),requires_grad=old.requires_grad)
        self.finger_twist_enabled=enabled;self._rest_cache=None;self._topology=None

    def fk(self,q,personalized=True,geometry=None):
        batch=len(q);g=self.geometry if geometry is None else geometry;rot=[torch.eye(3,device=q.device,dtype=q.dtype).expand(batch,3,3)];pos=[torch.zeros(batch,3,device=q.device,dtype=q.dtype)]
        axes=exp_so3(g['axis'])@self.joint_axis0[...,None] if personalized else self.joint_axis0[...,None]
        axes=axes.squeeze(-1)
        length=g['length'].exp() if personalized else torch.ones(20,device=q.device)
        for i in range(1,len(self.parents)):
            parent=self.parents[i];p=self.body_p0[i]*length[self.body_bone[parent]]
            if personalized and i in self.root_bodies:p=p+.01*g['root'][self.root_bodies.index(i)]
            r=self.body_r0[i].expand(batch,3,3);p=p.expand(batch,3)
            for j in self.body_joints[i]:
                jr=exp_so3(axes[j]*q[:,j,None]);jp=self.joint_p0[j]
                p=p+(r@(jp-(jr@jp[:,None]).squeeze(-1))[...,None]).squeeze(-1);r=r@jr
            pos.append(pos[parent]+(rot[parent]@p[...,None]).squeeze(-1));rot.append(rot[parent]@r)
        return torch.stack(rot,1),torch.stack(pos,1)

    def rest(self,geometry=None,vertex_ids=None,kinematics=None):
        """Derive rest skin/inverse binds together; native bone axes use local X.

        In the 51-coordinate policy width/depth/root/residual are exactly zero,
        so terminal lengths also move the skin despite SKN having no tip bones.
        Legacy artifacts retain their explicit historical geometry for preview.
        """
        cached=geometry is None and vertex_ids is None
        if cached and not torch.is_grad_enabled() and getattr(self,'fixed',False) and getattr(self,'_rest_cache',None) is not None:return self._rest_cache
        g=self.geometry if geometry is None else geometry
        r,p=self.fk(self.q0[None],geometry=g) if kinematics is None else kinematics
        r=r[0,self.bone_ids];p=p[0,self.bone_ids]
        change=r@self.base_r[self.bone_ids].transpose(-1,-2)
        bind_r=change@self.bind_r0
        bind_p=p+(change@(self.bind_p0-self.base_p[self.bone_ids])[...,None]).squeeze(-1)
        scale=torch.stack((g['length'].exp(),g['depth'].exp(),g['width'].exp()),-1)
        vertices=self.vertices0 if vertex_ids is None else self.vertices0[vertex_ids]
        weights=self.weights if vertex_ids is None else self.weights[vertex_ids]
        local=torch.einsum('kji,vkj->vki',self.bind_r0,vertices[:,None]-self.bind_p0[None])*scale[None]
        shaped=(torch.einsum('kij,vkj->vki',bind_r,local)+bind_p[None])*weights[...,None]
        residual=g['residual'] if vertex_ids is None else g['residual'][vertex_ids]
        result=shaped.sum(1)+residual,bind_r,bind_p
        if cached and not torch.is_grad_enabled() and getattr(self,'fixed',False):self._rest_cache=result
        return result

    def forward(self,q,vertex_ids=None,geometry=None,kinematics=None,rest_kinematics=None):
        rest,br,bp=self.rest(geometry=geometry,vertex_ids=vertex_ids,kinematics=rest_kinematics)
        r,p=self.fk(q,geometry=geometry) if kinematics is None else kinematics
        rr=r[:,self.bone_ids]@br.transpose(-1,-2)
        if vertex_ids is not None:w=self.weights[vertex_ids]
        else:w=self.weights
        # Sparse support evaluation avoids a B x V x 20 x 3 intermediate.
        if vertex_ids is not None:
            # Small contact support: blend affine transforms in one operation.
            blended=torch.einsum('vk,bkij->bvij',w,rr)
            translation=p[:,self.bone_ids]-(rr@bp[...,None]).squeeze(-1)
            out=(blended@rest[None,...,None]).squeeze(-1)+torch.einsum('vk,bki->bvi',w,translation)
        else:
            out=torch.zeros((len(q),len(rest),3),device=q.device,dtype=q.dtype)
            for k in range(20):
                out=out+((rest-bp[k])@rr[:,k].transpose(-1,-2)+p[:,self.bone_body[k],None])*w[:,k,None]
        wr=r[:,self.wrist_body];wp=p[:,self.wrist_body]
        return (out-wp[:,None])@wr, (p-wp[:,None])@wr, wr.transpose(-1,-2)[:,None]@r

    def sensor_prediction(self,q,geometry=None,kinematics=None):
        r,p=self.fk(q,geometry=geometry) if kinematics is None else kinematics
        g=self.geometry if geometry is None else geometry;wr=r[:,self.wrist_body];wp=p[:,self.wrist_body]
        dr=wr.transpose(-1,-2)[:,None]@r[:,self.distal_ids]
        dp=(p[:,self.distal_ids]-wp[:,None])@wr
        mp=self.mount_p0+.01*g['mount_pos'];mr=self.mount_r0@exp_so3(g['mount_rot'])
        return dp+(dr@mp[...,None]).squeeze(-1),dr@mr

    def alignment(self,geometry=None):
        g=self.geometry if geometry is None else geometry
        return exp_so3(g['alignment'])@self.align0,self.trans0+.01*g['translation']

    def angles(self,rows):
        result=[]
        for row in rows:
            s=row.get('skeleton') or {};names=s.get('joint_names',[]);values=s.get('joint_positions_rad',[])
            virtual=set(self.joint_names[25:])
            if len(names)!=len(values) or any(n not in names and n not in virtual for n in self.joint_names):raise ValueError('记录缺少官方25关节角，需重新预处理原始EMF')
            q=np.array([values[names.index(n)] if n in names else 0. for n in self.joint_names],np.float32)
            if not np.isfinite(q).all():raise ValueError('官方关节角无效')
            result.append(q)
        return torch.as_tensor(np.asarray(result),device=self.device)

    def _build_regions(self):
        from .contact import PATCH_IDS
        with torch.no_grad():v,j,r=self(self.q0[None]);v=v[0].cpu().numpy();j=j[0].cpu().numpy();r=r[0].cpu().numpy()
        f=self.faces.cpu().numpy();tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
        center=tri.mean(1);w=self.weights.cpu().numpy()[f].mean(1)
        pad=np.mean([-r[b][:,1] for b in self.distal_bodies[1:]],0);pad/=np.linalg.norm(pad);self.palm_normal=pad
        roots=j[[self.body_names.index(self.side+'_'+n+'_proximal') for n in SOURCE_FINGERS[1:]]]
        forward=roots[1]-j[self.wrist_body];forward-=pad*np.dot(forward,pad);forward/=np.linalg.norm(forward)
        lateral=np.cross(forward,pad);self.palm_forward=forward;self.palm_lateral=lateral
        regions={}
        for patch in PATCH_IDS:
            if patch.startswith('palm_'):continue
            finger,part=patch.split('_');source=SOURCE_FINGERS[FINGERS.index(finger)];name=self.side+'_'+source+'_'+part
            # The delivered LEFT thumb has a fixed distal->tip roll of pi.
            # Its pad axis must come from the anatomical tip, not distal -Y.
            k=self.bone_names.index(name);axis=-r[self.tip_bodies[FINGERS.index(finger)]][:,1]
            ids=np.flatnonzero((w[:,k]>.15)&(normal@axis>.15))
            if len(ids)<3:raise ValueError('官方 mesh 缺少掌侧区域 '+patch)
            regions[patch]=ids.tolist()
        palm_weight=w[:,[0]+[i for i,n in enumerate(self.bone_names) if n.endswith('_metacarpal')]].sum(1)
        palm=np.flatnonzero((palm_weight>.55)&(normal@pad>.25))
        xy=center@np.stack((forward,lateral),1);palm_xy=xy[palm]
        lo=palm_xy.min(0);hi=palm_xy.max(0)
        for row in range(3):
            for col in range(2):
                a=lo+(hi-lo)*[row/3,col/2];b=lo+(hi-lo)*[(row+1)/3,(col+1)/2]
                ids=palm[np.all((palm_xy>=a)&(palm_xy<=b+1e-9),axis=1)]
                if not len(ids):raise ValueError('官方 mesh 掌面区域为空')
                regions[f'palm_{row}_{col}']=ids.tolist()
        for k,finger in enumerate(FINGERS[1:]):
            anchor=.85*roots[k]+.15*j[self.wrist_body]
            regions['palm_mcp_'+finger]=palm[np.argsort(np.linalg.norm(center[palm]-anchor,axis=-1))[:24]].tolist()
        self.regions=regions
        self.patch_anchors={name:dict(face=int(ids[np.argmin(np.linalg.norm(center[ids]-center[ids].mean(0),axis=-1))]),barycentric=[1/3]*3) for name,ids in regions.items()}

    def bind_mapping(self,mapping):
        from .contact import PATCH_IDS,validate_mapping
        from .tactile_surface import bind_grid
        assignments,_=validate_mapping(mapping['assignments'],mapping.get('confirmed',[]));f=self.faces.cpu().numpy()
        with torch.no_grad():
            vv,jj,rr=self(self.q0[None]);v=vv[0].cpu().numpy();jj=jj[0].cpu().numpy();rr=rr[0].cpu().numpy()
        face_ids=np.zeros(460,int);bc=np.zeros((460,3),np.float32)
        for i,patch in enumerate(PATCH_IDS):
            sids=np.flatnonzero(np.asarray(assignments)==i);ids=np.asarray(self.regions[patch])
            if patch.startswith('palm_'):axis=self.palm_forward;lateral=self.palm_lateral
            else:
                finger=FINGERS.index(patch.split('_')[0]);source=SOURCE_FINGERS[finger];b=self.body_names.index(self.side+'_'+source+'_'+patch.split('_')[1])
                part=patch.split('_')[1];next_part='tip' if part=='distal' else 'distal' if source=='thumb' or part=='intermediate' else 'intermediate'
                end=self.body_names.index(self.side+'_'+source+'_'+next_part);axis=jj[end]-jj[b];axis/=np.linalg.norm(axis)
                pad=-rr[self.tip_bodies[finger]][:,1];lateral=np.cross(axis,pad);lateral/=np.linalg.norm(lateral)
            ii,bb=bind_grid(v[f[ids]],sids,axis,lateral);face_ids[sids]=ids[ii];bc[sids]=bb
        self.mapping=dict(mapping);self.sid_faces=face_ids;self.sid_bc=bc;self.sid_normal_offset=np.zeros(460,np.float32)
        self.binding_source='anatomical_initialization'
        self.binding_hash=hashlib.sha256(json.dumps([self.region_revision,assignments,face_ids.tolist(),bc.tolist(),self.sid_normal_offset.tolist()],sort_keys=True).encode()).hexdigest()
        self._topology=None
        self._rest_cache=None

    def anchors(self,vertices,anchors):
        ids=torch.as_tensor([a['face'] for a in anchors],device=self.device,dtype=torch.long);bc=torch.as_tensor([a['barycentric'] for a in anchors],device=self.device)
        tri=vertices[:,self.faces[ids]]
        return (tri*bc[None,...,None]).sum(-2),torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)

    def topology(self):
        if self.mapping is None:raise ValueError('等待SID分区')
        if self._topology is None:
            with torch.no_grad():v,j,_=self(self.q0[None])
            chains=[[self.wrist_body]+[self.body_names.index(self.side+'_'+f+'_'+part) for part in (('metacarpal','proximal','distal','tip') if f=='thumb' else ('metacarpal','proximal','intermediate','distal','tip'))] for f in SOURCE_FINGERS]
            self._topology=dict(ready=True,surface_backend='qnbot',surface_policy='qnbot_original_triangles',geometry_version=self.version(),binding_version=self.binding_hash,
                coordinate_frame='wrist_local',hand_side=self.side,faces=self.faces.cpu().tolist(),rest_vertices_m=v[0].cpu().tolist(),rest_joints_m=j[0].cpu().tolist(),joint_chains=chains,joint_names=self.body_names,
                sid_face_ids=self.sid_faces.tolist(),sid_barycentric=self.sid_bc.tolist(),sid_normal_offset_m=self.sid_normal_offset.tolist(),sid_patch_indices=self.mapping['assignments'],region_revision=self.region_revision,
                region_binding_source=getattr(self,'binding_source','anatomical_initialization'),palm_view=dict(normal=self.palm_normal.tolist(),forward=self.palm_forward.tolist()),regions=self.regions)
        return self._topology

    def version(self):
        fixed=b''.join(getattr(self,k).detach().cpu().numpy().tobytes() for k in ('align0','trans0','mount_p0','mount_r0'))
        return hashlib.sha256(self.asset_hash.encode()+fixed+b''.join(p.detach().cpu().numpy().tobytes() for p in self.geometry.values())).hexdigest()[:20]

    def artifact(self):
        result=dict(surface_backend='qnbot',runtime_backend=REVISION,asset_hash=self.asset_hash,topology_hash=self.topology_hash,region_revision=self.region_revision,side=self.side,
                    geometry={k:p.detach().cpu().tolist() for k,p in self.geometry.items()},registration={k:getattr(self,k).cpu().tolist() for k in ('align0','trans0','mount_p0','mount_r0')},
                    joint_limits_rad=self.limits.cpu().tolist(),joint_limit_policy=__import__('copy').deepcopy(self.joint_limit_policy),finger_twist_enabled=self.finger_twist_enabled)
        if self.mapping is not None:result['material_binding']=dict(assignments=self.mapping['assignments'],face_ids=self.sid_faces.tolist(),barycentric=self.sid_bc.tolist(),normal_offset_m=self.sid_normal_offset.tolist(),source=getattr(self,'binding_source','anatomical_initialization'))
        return result

    def install_binding(self,binding):
        from .contact import PATCH_IDS
        ids=np.asarray(binding['face_ids']);bc=np.asarray(binding['barycentric'],np.float32)
        if self.mapping is None or binding.get('assignments')!=self.mapping['assignments'] or ids.shape!=(460,) or bc.shape!=(460,3) or not np.issubdtype(ids.dtype,np.integer) or not np.isfinite(bc).all() or bc.min()<0 or not np.allclose(bc.sum(1),1,atol=1e-5):raise ValueError('官方材料绑定与当前SID分区不兼容')
        for sid,face in enumerate(ids):
            if face not in self.regions[PATCH_IDS[self.mapping['assignments'][sid]]]:raise ValueError('SID面片不在指定解剖区域')
        offsets=np.asarray(binding.get('normal_offset_m',np.zeros(460)),np.float32)
        if offsets.shape!=(460,) or not np.isfinite(offsets).all() or np.max(np.abs(offsets))>.005:raise ValueError('无效SID表面法向偏移')
        self.sid_faces=ids;self.sid_bc=bc;self.sid_normal_offset=offsets;self.binding_source=binding.get('source','saved_material_binding');self.binding_hash=hashlib.sha256(json.dumps([self.region_revision,self.mapping['assignments'],ids.tolist(),bc.tolist(),offsets.tolist()],sort_keys=True).encode()).hexdigest();self._topology=None

    def load_artifact(self,artifact):
        if artifact.get('bone_extrinsics'):
            from .bone_extrinsics import validate_candidate
            validate_candidate(artifact)
        for name,value in dict(surface_backend='qnbot',runtime_backend=REVISION,asset_hash=self.asset_hash,topology_hash=self.topology_hash,region_revision=self.region_revision,side=self.side).items():
            if artifact.get(name)!=value:raise ValueError('官方 mesh 权重不兼容：'+name)
        self.configure_finger_twist(len(artifact['geometry']['axis'])==29)
        rig=(artifact.get('bone_extrinsics') or {}).get('rig')
        bounds=rig['limits'] if rig else artifact.get('joint_limits_rad',self.default_limits)
        bounds=torch.as_tensor(bounds,device=self.device,dtype=self.limits.dtype)
        if bounds.shape!=self.limits.shape or not torch.isfinite(bounds).all() or (bounds[:,0]>bounds[:,1]).any():raise ValueError('无效关节限位')
        if rig and rig['joint_names']!=self.joint_names:raise ValueError('关节限位名称不一致')
        values={};registration={}
        for k,p in self.geometry.items():
            v=torch.as_tensor(artifact['geometry'][k],device=self.device,dtype=p.dtype)
            if v.shape!=p.shape or not torch.isfinite(v).all():raise ValueError('无效官方 mesh 参数 '+k)
            if k in ('length','width','depth') and (not torch.isfinite(v.exp()).all() or (v.exp()<=0).any()):raise ValueError('官方mesh尺寸不能用有限正数表示')
            values[k]=v
        for k in ('align0','trans0','mount_p0','mount_r0'):
            v=torch.as_tensor(artifact['registration'][k],device=self.device,dtype=getattr(self,k).dtype)
            if v.shape!=getattr(self,k).shape or not torch.isfinite(v).all():raise ValueError('无效安装配准 '+k)
            registration[k]=v
        if artifact.get('material_binding'):
            if self.mapping is None:self.bind_mapping(dict(assignments=artifact['material_binding']['assignments'],confirmed=[]))
            self.install_binding(artifact['material_binding'])
        with torch.no_grad():
            self.limits.copy_(bounds)
            for k,p in self.geometry.items():p.copy_(values[k])
            for k,v in registration.items():getattr(self,k).copy_(v)
        self.joint_limit_policy=__import__('copy').deepcopy((artifact.get('bone_extrinsics') or {}).get('joint_limit_policy',artifact.get('joint_limit_policy',dict(revision='stored_legacy_joint_limits'))))
        self._rest_cache=None;self._topology=None

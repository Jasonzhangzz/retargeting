"""CUDA anatomical initialization with the existing two position-only IK costs."""
import numpy as np
import torch
from .pose_gpu import ModelFK
from .shadow_mano_seed import ShadowManoTransfer
from .pose_ik import bounded_lm,SizeIKSettings
from retarget.initialization.dexonomy.util.torch_util import torch_quaternion_to_matrix,torch_matrix_to_quaternion
from scripts.calibrate_wuji_mano import rotmat


def palm_basis(points,palm):
    x=torch.nn.functional.normalize(points[...,0,:]-points[...,3,:],dim=-1)
    y=points[...,1,:]-palm
    y=torch.nn.functional.normalize(y-x*(x*y).sum(-1,keepdim=True),dim=-1)
    return torch.stack((x,y,torch.linalg.cross(x,y,dim=-1)),-1)


class CudaShadowManoSeed:
    def __init__(self,shadow_xml,mano_xml,bodies,source_local,device='cuda:0'):
        self.device=torch.device(device);self.dtype=torch.float64
        cpu=ShadowManoTransfer(shadow_xml,mano_xml);self.source=cpu.shadow;self.target=cpu.mano
        self.source_fk=ModelFK(cpu.shadow.model,self.device,self.dtype)
        self.target_fk=ModelFK(cpu.mano.model,self.device,self.dtype)
        self.source_bodies=list(bodies)
        self.bodies,mapped=cpu.map_contacts(bodies,np.asarray(source_local))
        t=lambda a:torch.as_tensor(a,device=self.device,dtype=self.dtype)
        self.source_ids=[cpu.shadow.model.body(b).id for b in bodies]
        self.target_ids=[cpu.mano.model.body(b).id for b in self.bodies]
        self.source_local=t(source_local);self.target_local=t(mapped)
        self.source_normal=t(np.array([cpu.shadow.project(b,p[None])[2][0] for b,p in zip(bodies,np.asarray(source_local))]))
        self.target_normal=t(np.array([cpu.mano.project(b,p[None])[2][0] for b,p in zip(self.bodies,mapped)]))
        self.low=t(cpu.mano.low+1e-8);self.high=t(cpu.mano.high-1e-8)
        self.neutral_frame=t(cpu.mano.neutral_frame);self.neutral_palm=t(cpu.mano.neutral_palm)
        self.neutral_points=t(cpu.mano.neutral);self.lengths=t(cpu.mano.lengths)
        base=cpu.mano.model.key_qpos[0] if cpu.mano.model.nkey else np.zeros(cpu.mano.model.nq)
        self.base=t(np.clip(base,cpu.mano.low+1e-7,cpu.mano.high-1e-7))
        self.source_tips=t(cpu.shadow.tips);self.target_tips=t(cpu.mano.tips)
        self.scale=t([.02]*3+[.3]*23)

    def points(self,joints,*,source=False):
        hand=self.source if source else self.target
        fk=self.source_fk if source else self.target_fk
        tips=self.source_tips if source else self.target_tips
        p,r=fk.frames(joints);ids=hand.ids.tolist()
        fingers=[]
        for f,chain in enumerate(ids):
            tip=p[...,chain[-1],:]+(r[...,chain[-1],:,:]@tips[f,:,None]).squeeze(-1)
            fingers.append(torch.cat((p[...,chain,:],tip[...,None,:]),-2))
        return torch.stack(fingers,-3),p,r

    def fit(self,source_q):
        q=torch.as_tensor(source_q,device=self.device,dtype=self.dtype)
        source_points,sp,sr=self.points(q[:,7:],source=True)
        root=torch_quaternion_to_matrix(q[:,3:7])
        source_points=source_points@root[:,None].transpose(-1,-2)+q[:,None,None,:3]
        palm=(root@sp[:,self.source.palm,:,None]).squeeze(-1)+q[:,:3]
        basis=palm_basis(source_points[:,:,0],palm)
        rotation=basis@self.neutral_frame.T;translation=palm-(rotation@self.neutral_palm[:,None]).squeeze(-1)
        anchors=self.neutral_points[:,0]@rotation.transpose(-1,-2)+translation[:,None]
        directions=torch.nn.functional.normalize(source_points[:,:,1:]-source_points[:,:,:-1],dim=-1)
        targets=[anchors]
        for k in range(3):targets.append(targets[-1]+self.lengths[None,:,k,None]*directions[:,:,k])
        target=torch.stack(targets,2)
        context=dict(rotation=rotation,translation=translation,target=target)
        def bone_residual(z,data):
            points=self.points(z*.3)[0]
            world=points@data['rotation'].T+data['translation']
            return (world[:,1:]-data['target'][:,1:]).reshape(-1)
        settings=SizeIKSettings(iterations=79,starts=1,damping=.01)
        joints,cost,_=bounded_lm(bone_residual,self.base.expand(len(q),-1)/.3,self.low/.3,self.high/.3,settings,oracle=lambda _:context)
        joints=joints*.3
        source_contact=(sr[:,self.source_ids]@self.source_local[None,...,None]).squeeze(-1)+sp[:,self.source_ids]
        source_contact=(root[:,None]@source_contact[...,None]).squeeze(-1)+q[:,None,:3]
        contact_context=dict(rotation=rotation,translation=translation,target=source_contact)
        def contact_residual(z,data):
            x=z*self.scale;p,r=self.target_fk.frames(x[6:])
            local=(r[self.target_ids]@self.target_local[...,None]).squeeze(-1)+p[self.target_ids]
            turn=rotmat(x[3:6][None])[0]@data['rotation']
            world=local@turn.T+data['translation']+x[:3]
            return (world-data['target']).reshape(-1)
        seed=torch.cat((torch.zeros((len(q),6),device=q.device,dtype=q.dtype),joints),-1)/self.scale
        low=torch.cat((q.new_full((3,),-.1),q.new_full((3,),-1.),self.low))/self.scale
        high=torch.cat((q.new_full((3,),.1),q.new_full((3,),1.),self.high))/self.scale
        before=torch.vmap(contact_residual)(seed,contact_context).reshape(len(q),-1,3)
        z,contact_cost,_=bounded_lm(contact_residual,seed,low,high,settings,oracle=lambda _:contact_context)
        x=z*self.scale;turn=rotmat(x[:,3:6])@rotation
        mapped=torch.cat((translation+x[:,:3],torch_matrix_to_quaternion(turn),x[:,6:]),-1)
        p,r=self.target_fk.frames(x[:,6:]);contact=(r[:,self.target_ids]@self.target_local[None,...,None]).squeeze(-1)+p[:,self.target_ids]
        contact=(turn[:,None]@contact[...,None]).squeeze(-1)+mapped[:,None,:3]
        normal=(turn[:,None]@r[:,self.target_ids]@self.target_normal[None,...,None]).squeeze(-1)
        error=(contact-source_contact).norm(dim=-1)
        finite=torch.isfinite(mapped).all(-1)&torch.isfinite(error).all(-1)&(q[:,3:7].norm(dim=-1)>1e-8)
        return dict(grasp_qpos=mapped,hand_cpn_w=torch.cat((contact,normal),-1),
                    hand_cpn_b=torch.cat((self.target_local,self.target_normal),-1)[None].expand(len(q),-1,-1),
                    source_hand_points=source_contact,mapping_error_m=error,finite=finite,
                    bone_cost=cost,contact_cost=contact_cost,initial_contact_cost=before.square().sum((1,2)),
                    initialization_solver='GPU bounded LM; the same two position residuals and bounds; ADMM hand step remains native rigid-body simulation')

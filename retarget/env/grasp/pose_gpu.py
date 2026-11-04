"""Device pose transforms and native-model hinge/slide FK adapters."""
import torch
from retarget.initialization.dexonomy.util.torch_util import torch_quaternion_to_matrix


class ModelFK:
    def __init__(self,model,device,dtype=torch.float32):
        self.model=model;self.device=device;self.dtype=dtype
        def t(x):return torch.as_tensor(x,device=device,dtype=dtype)
        self.position=t(model.body_pos)
        self.rotation=torch_quaternion_to_matrix(t(model.body_quat))
        self.axis=t(model.jnt_axis);self.pivot=t(model.jnt_pos);self.reference=t(model.qpos0)
        self.parents=model.body_parentid.tolist()
        self.joints=[list(range(int(model.body_jntadr[b]),int(model.body_jntadr[b]+model.body_jntnum[b]))) if model.body_jntnum[b] else [] for b in range(model.nbody)]
        self.kinds=model.jnt_type.tolist();self.addresses=model.jnt_qposadr.tolist()
        if any(k not in (2,3) for k in self.kinds):raise ValueError('ModelFK supports native slide/hinge joints only')

    def frames(self,q):
        shape=q.shape[:-1];eye=torch.eye(3,device=q.device,dtype=q.dtype).expand(*shape,3,3)
        positions=[q.new_zeros((*shape,3))];rotations=[eye]
        for body in range(1,len(self.parents)):
            p=self.position[body].expand(*shape,3);r=self.rotation[body].expand(*shape,3,3)
            for joint in self.joints[body]:
                x=q[...,self.addresses[joint]]-self.reference[self.addresses[joint]]
                axis=self.axis[joint]
                if self.kinds[joint]==2:p=p+(r@axis[...,None]).squeeze(-1)*x[...,None]
                else:
                    a,b,c=axis.unbind(-1);z=torch.zeros_like(a)
                    cross=torch.stack((z,-c,b,c,z,-a,-b,a,z)).reshape(3,3)
                    turn=eye+x.sin()[...,None,None]*cross+(1-x.cos())[...,None,None]*(cross@cross)
                    pivot=self.pivot[joint]
                    p=p+(r@(pivot-(turn@pivot[...,None]).squeeze(-1))[...,None]).squeeze(-1)
                    r=r@turn
            parent=self.parents[body]
            positions.append(positions[parent]+(rotations[parent]@p[...,None]).squeeze(-1))
            rotations.append(rotations[parent]@r)
        return torch.stack(positions,-2),torch.stack(rotations,-3)


def vertex_normals(vertices,faces):
    triangles=vertices[...,faces,:]
    normal=torch.linalg.cross(triangles[...,1,:]-triangles[...,0,:],triangles[...,2,:]-triangles[...,0,:])
    output=torch.zeros_like(vertices)
    for j in range(3):output.index_add_(-2,faces[:,j],normal)
    return torch.nn.functional.normalize(output,dim=-1,eps=1.e-12)


def matrix_to_euler_xyz(matrix):
    # Native serial root Rx @ Ry @ Rz, matching scipy intrinsic XYZ.
    y=matrix[...,0,2].clamp(-1,1).asin()
    x=torch.atan2(-matrix[...,1,2],matrix[...,2,2])
    z=torch.atan2(-matrix[...,0,1],matrix[...,0,0])
    singular=y.cos().abs()<1.e-6
    x=torch.where(singular,torch.atan2(matrix[...,2,1],matrix[...,1,1]),x)
    z=torch.where(singular,torch.zeros_like(z),z)
    return torch.stack((x,y,z),-1)


def euler_xyz_to_matrix(angles):
    x,y,z=angles.unbind(-1);cx,sx=x.cos(),x.sin();cy,sy=y.cos(),y.sin();cz,sz=z.cos(),z.sin()
    return torch.stack((cy*cz,-cy*sz,sy,cx*sz+sx*sy*cz,cx*cz-sx*sy*sz,-sx*cy,sx*sz-cx*sy*cz,sx*cz+cx*sy*sz,cx*cy),-1).reshape(*angles.shape[:-1],3,3)


def palm_frame(wrist,roots):
    origin=torch.cat((wrist[...,None,:],roots),-2).mean(-2)
    x=torch.nn.functional.normalize(roots[...,1,:]-roots[...,4,:],dim=-1)
    forward=roots[...,1:,:].mean(-2)-wrist
    z=torch.nn.functional.normalize(torch.linalg.cross(x,forward),dim=-1)
    y=torch.linalg.cross(z,x)
    return origin,torch.stack((x,y,z),-1)


class WujiPoseAdapter:
    """Original palm/bone direction IK residual and existing bounded GPU LM."""
    def __init__(self,model,device):
        from retarget.initialization.ik import FINGERS
        self.model=model;self.fk=ModelFK(model,device,torch.float64)
        self.device=device;self.names=list(FINGERS)
        bodies=[model.body('right_palm_link').id];sites=[]
        self.mano=[]
        for name,(_,robot,mano) in FINGERS.items():
            bodies.extend(model.body(f'right_finger{robot}_link{k}').id for k in (2,3,4))
            site=model.site(f'right_{name}_tip').id;sites.append((int(model.site_bodyid[site]),model.site_pos[site]))
            self.mano.extend(range(mano,mano+4))
        self.bodies=bodies;self.sites=sites
        self.site_ids=[v[0] for v in sites]
        self.site_local=torch.as_tensor([v[1].tolist() for v in sites],device=device,dtype=torch.float64)
        ids=[next(j for j,a in enumerate(model.jnt_qposadr) if a==i) for i in range(26)]
        self.low=torch.as_tensor(model.jnt_range[ids,0],device=device,dtype=torch.float64)
        self.high=torch.as_tensor(model.jnt_range[ids,1],device=device,dtype=torch.float64)
        self.neutral=torch.as_tensor(model.qpos0[:26],device=device,dtype=torch.float64).clamp(self.low+1.e-8,self.high-1.e-8)
        points=self.keypoints(self.neutral)
        self.robot_origin,self.robot_basis=palm_frame(torch.zeros(3,device=device,dtype=torch.float64),points[[1,5,9,13,17]])

    def keypoints(self,q):
        p,r=self.fk.frames(q)
        tips=p[...,self.site_ids,:]+(r[...,self.site_ids,:,:]@self.site_local[...,None]).squeeze(-1)
        values=[p[...,self.bodies[0],:]]
        for i in range(5):
            values.extend(p[...,body,:] for body in self.bodies[1+3*i:4+3*i]);values.append(tips[...,i,:])
        return torch.stack(values,-2)

    def initialize(self,human):
        from .pose_ik import bounded_lm,SizeIKSettings
        human=human.to(device=self.device,dtype=torch.float64)
        if human.ndim==2:human=human[None]
        origin,basis=palm_frame(human[:,0],human[:,[1,5,9,13,17]])
        rotation=basis@self.robot_basis.T
        base=self.neutral.expand(len(human),-1).clone().clamp(self.low+.1,self.high-.1)
        base[:,:3]=origin-(rotation@self.robot_origin[...,None]).squeeze(-1)
        base[:,3:6]=matrix_to_euler_xyz(rotation)
        # Each row keeps its fixed palm transform. Residuals exactly match ik.py.
        target=human[:,self.mano].reshape(-1,5,4,3)
        desired=torch.nn.functional.normalize(target[:,:,1:]-target[:,:,:-1],dim=-1)
        def residual(values,context):
            q=torch.cat((context['base'],values))
            points=self.keypoints(q)[1:].reshape(5,4,3)
            unit=torch.nn.functional.normalize(points[:,1:]-points[:,:-1],dim=-1)
            return (unit-context['desired']).flatten()
        def context(_):return dict(base=base[:,:6],desired=desired)
        result,_,_=bounded_lm(residual,base[:,6:],self.low[6:],self.high[6:],SizeIKSettings(iterations=64),oracle=context)
        return torch.cat((base[:,:6],result),1)

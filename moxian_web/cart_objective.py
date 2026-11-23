"""Shared hand kinematics and native-mesh supervision/audit geometry."""
import numpy as np
import torch
import warp as wp
from retarget.baseline.geort.adapter import RobotAdapter
from retarget.baseline.geort.pose_domain import extension_domain
from retarget.env.grasp.pose_gpu import ModelFK
from retarget.baseline.card.collision import WujiMeshCollision
from .wuji_surface import WujiSurface
from .cart_mesh_audit import _signed_witness
from .cart_contact_witness import contact_witness, DEEP_PENETRATION_M


@wp.kernel(enable_backward=False)
def _nearest(mesh: wp.uint64, query: wp.array(dtype=wp.vec3),
             hits: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3), valid: wp.array(dtype=int)):
    i = wp.tid(); q = wp.mesh_query_point_no_sign(mesh, query[i], 2.)
    if q.result:
        hits[i] = wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        normals[i] = wp.mesh_eval_face_normal(mesh,q.face); valid[i] = 1


@wp.kernel(enable_backward=False)
def _nearest_contact(skin: wp.uint64, solid: wp.uint64, query: wp.array(dtype=wp.vec3),
                     escape: wp.array(dtype=wp.vec3), threshold: float,
                     hits: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                     valid: wp.array(dtype=int), retracted: wp.array(dtype=int)):
    i=wp.tid()
    witness=contact_witness(skin,solid,query[i],escape[i],threshold)
    hits[i]=witness.point;normals[i]=witness.normal
    valid[i]=witness.valid;retracted[i]=witness.retracted


class CartHand:
    def __init__(self, device='cpu'):
        self.device = str(device); self.robot = RobotAdapter(); self.surface = WujiSurface()
        self.domain = extension_domain(self.robot.lower, self.robot.upper)
        self.low = self.tensor(self.domain['lower']); self.high = self.tensor(self.domain['upper'])
        self.fk = ModelFK(self.robot.model, self.device); self.ids = self.robot.body_ids
        from .cart_kinematics import FixedWristFK
        self.fixed_fk=FixedWristFK(self.fk,self.ids)
        self.basis = self.tensor(self.robot.basis)
        self.collision = WujiMeshCollision(self.surface, self.device)
        self.full_meshes = self.collision.meshes; self.pad_meshes = {}; self.samples_cache = {}
        self.tip_owners = [self.ids.index(int(self.robot.model.site_bodyid[i])) for i in self.robot.tip_ids]
        self.tip_local = self.tensor(self.robot.model.site_pos[self.robot.tip_ids])
        self.centres=self.tensor([mesh.bounds.mean(0) for mesh in self.surface.meshes])

    def tensor(self, value): return torch.as_tensor(np.asarray(value), device=self.device, dtype=torch.float32)

    def frames(self, q):
        full_p, full_r = self.fixed_fk.frames(q)
        p, r = full_p[:,self.ids], full_r[:,self.ids]
        cp = (p-p[:,:1])@self.basis; cr = self.basis.T@r
        tips = cp[:,self.tip_owners]+(cr[:,self.tip_owners]@self.tip_local[...,None]).squeeze(-1)
        landmarks = [cp[:,0]]
        for f in range(5):
            landmarks.extend(cp[:,o] for o in (2+4*f,3+4*f,4+4*f)); landmarks.append(tips[:,f])
        return cp,cr,torch.stack(landmarks,1),full_p,full_r

    def nearest(self, query, owner, palmar=False, *, escape=None, return_retracted=False):
        if palmar and owner not in self.pad_meshes:
            tri = np.asarray(self.surface.contact_surface(owner)['triangles'], np.float32)
            if not len(tri): raise ValueError('Wuji2 缺少接触曲面')
            self.pad_meshes[owner] = wp.Mesh(points=wp.array(tri.reshape(-1,3),dtype=wp.vec3,device=self.device),
                indices=wp.array(np.arange(tri.size//3),dtype=int,device=self.device))
        mesh = self.pad_meshes[owner] if palmar else self.full_meshes[owner]
        shape = query.shape; flat = query.detach().reshape(-1,3).contiguous()
        hits = wp.zeros(len(flat),dtype=wp.vec3,device=self.device)
        normals = wp.zeros(len(flat),dtype=wp.vec3,device=self.device); valid = wp.zeros(len(flat),dtype=int,device=self.device)
        retracted=wp.zeros(len(flat),dtype=int,device=self.device)
        if escape is None:
            wp.launch(_nearest,len(flat),inputs=[mesh.id,wp.from_torch(flat,dtype=wp.vec3),hits,normals,valid],device=self.device)
        else:
            directions=escape.detach().reshape(-1,3).contiguous()
            wp.launch(_nearest_contact,len(flat),inputs=[mesh.id,self.full_meshes[owner].id,
                wp.from_torch(flat,dtype=wp.vec3),wp.from_torch(directions,dtype=wp.vec3),DEEP_PENETRATION_M,
                hits,normals,valid,retracted],device=self.device)
        values=(wp.to_torch(hits).reshape(shape),wp.to_torch(normals).reshape(shape),wp.to_torch(valid).bool().reshape(shape[:-1]))
        return values+(wp.to_torch(retracted).bool().reshape(shape[:-1]),) if return_retracted else values

    def samples(self, owner, dense):
        key = (owner,dense)
        if key not in self.samples_cache:
            mesh = self.surface.meshes[owner]
            p = np.vstack((mesh.vertices,mesh.triangles_center)); n = np.vstack((mesh.vertex_normals,mesh.face_normals))
            if not dense:
                ids = np.unique(np.r_[np.linspace(0,len(p)-1,64).astype(int),np.argmin(p,0),np.argmax(p,0)])
                p,n = p[ids],n[ids]
            self.samples_cache[key] = (self.tensor(p),self.tensor(n))
        return self.samples_cache[key]

    def pair(self,p,r,a,b,dense=False):
        choices = []
        for source,target in ((a,b),(b,a)):
            local,ln = self.samples(source,dense)
            points = (r[:,source,None]@local[None,:,:,None]).squeeze(-1)+p[:,source,None]
            normals = (r[:,source,None]@ln[None,:,:,None]).squeeze(-1)
            query = ((points-p[:,target,None])[...,None,:]@r[:,target,None]).squeeze(-2)
            escape=(-(normals[...,None,:]@r[:,target,None])).squeeze(-2)
            hit,hn,valid,retracted = self.nearest(query,target,escape=escape,return_retracted=True)
            centres=[]
            for owner in (source,target):
                centre=self.centres[owner] if hasattr(self,'centres') else self.tensor(self.surface.meshes[owner].bounds.mean(0))
                centres.append((r[:,owner]@centre[:,None]).squeeze(-1)+p[:,owner])
            facing=(normals*(centres[1]-centres[0])[:,None]).sum(-1)>1.e-7
            valid=valid&(~retracted|facing)
            other = (r[:,target,None]@hit[...,None]).squeeze(-1)+p[:,target,None]
            on = (r[:,target,None]@hn[...,None]).squeeze(-1)
            distance = (points-other).square().sum(-1)
            angle = (1+(normals*on).sum(-1).clamp(-1,1)).clamp_min(0)
            score = distance.detach()/.001**2 + angle.detach()/(1-np.cos(np.deg2rad(30)))
            score = torch.where(valid,score,torch.inf); index = score.argmin(-1); batch = torch.arange(len(p),device=p.device)
            one = tuple(v[batch,index] for v in (distance,angle,points,other,normals,on,valid))
            if source == b: one = (one[0],one[1],one[3],one[2],one[5],one[4],one[6])
            choices.append(one)
        use = choices[0][6] & (~choices[1][6] | (choices[0][0].detach() <= choices[1][0].detach()))
        return tuple(torch.where(use.reshape((-1,)+(1,)*(x.ndim-1)),x,y) for x,y in zip(*choices))

    def penetration(self,q,pfull,rfull,dense=False):
        if not q.requires_grad:
            depth,valid = self.collision.query(pfull,rfull,dense=dense)
            maximum = depth.amax((1,2))/1000
            return maximum,valid
        # Freeze nearest witnesses, then differentiate BOTH moving links.
        p,r = pfull[:,self.ids],rfull[:,self.ids]
        points,owners = self.collision.samples[dense]; source = wp.to_torch(points); owner = wp.to_torch(owners).long()
        shape = (len(q),len(source),21)
        hits = wp.zeros(shape,dtype=wp.vec3,device=self.device); normals = wp.zeros(shape,dtype=wp.vec3,device=self.device)
        valid = wp.zeros(shape,dtype=int,device=self.device)
        wp.launch(_signed_witness,shape,inputs=[self.collision.mesh_ids,points,owners,self.collision.allowed,self.collision.collar,
            wp.from_torch(p.detach().contiguous(),dtype=wp.vec3),wp.from_torch(r.detach().contiguous(),dtype=wp.mat33),hits,normals,valid],device=self.device)
        world = (r[:,owner]@source[...,None]).squeeze(-1)+p[:,owner]
        hit = (r[:,None]@wp.to_torch(hits)[...,None]).squeeze(-1)+p[:,None]
        normal = (r[:,None]@wp.to_torch(normals)[...,None]).squeeze(-1)
        depths = torch.where(wp.to_torch(valid).bool(),(-((world[:,:,None]-hit)*normal).sum(-1)).clamp_min(0),0.)
        return depths.amax((1,2)),torch.ones(len(q),device=q.device,dtype=torch.bool)

"""Offline MANO LBS and exact triangle distance. Never used by the predictor."""
from pathlib import Path
import numpy as np
import torch
from .rotations import axis_angle_to_matrix


class ManoGeometry(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        from scripts.calibrate_wuji_mano import _ManoReader, as_array
        with open(path, 'rb') as f:
            d = _ManoReader(f, encoding='latin1').load()
        for key in ('v_template','shapedirs','posedirs','weights','J_regressor','hands_mean','hands_components'):
            self.register_buffer(key, torch.as_tensor(as_array(d[key]).copy(), dtype=torch.float32))
        self.faces = as_array(d['f']).astype(np.int64)
        if self.v_template.shape!=(778,3) or self.weights.shape!=(778,16) or self.posedirs.shape!=(778,3,135):
            raise ValueError('Expected standard 778-vertex / 16-joint MANO topology')
        extent=float(torch.linalg.vector_norm(self.v_template.max(0).values-self.v_template.min(0).values))
        if not .08<extent<.5:raise ValueError('MANO template must be expressed in metres')
        tree = as_array(d['kintree_table']); lookup = {int(v): i for i,v in enumerate(tree[1])}
        self.parents = [-1]+[lookup[int(v)] for v in tree[0,1:]]
        self.register_buffer('calibrated_joints', None if 'retarget_rest_joints' not in d else torch.tensor(d['retarget_rest_joints'], dtype=torch.float32))

    @torch.no_grad()
    def forward(self, pose, betas=None, convention='mano'):
        q = torch.as_tensor(pose, device=self.v_template.device, dtype=torch.float32)
        b = len(q)
        beta = q.new_zeros(b, self.shapedirs.shape[-1]) if betas is None else torch.as_tensor(betas, device=q.device, dtype=q.dtype).expand(b,-1)
        rest = self.v_template + torch.einsum('vck,bk->bvc', self.shapedirs, beta)
        joints = torch.einsum('jv,bvc->bjc', self.J_regressor, rest)
        if self.calibrated_joints is not None:
            joints = self.calibrated_joints.expand(b,-1,-1)
        local = axis_angle_to_matrix(torch.cat((q.new_zeros(b,1,3),q[:,6:].reshape(b,15,3)),1))
        feature = (local[:,1:]-torch.eye(3,device=q.device)).reshape(b,135)
        posed = rest+torch.einsum('bp,vcp->bvc',feature,self.posedirs)
        rotations=[]; positions=[]
        for j,parent in enumerate(self.parents):
            if parent<0:
                rotations.append(local[:,j]); positions.append(joints[:,j])
            else:
                rotations.append(rotations[parent]@local[:,j])
                positions.append(positions[parent]+(rotations[parent]@(joints[:,j]-joints[:,parent])[...,None]).squeeze(-1))
        r=torch.stack(rotations,1); p=torch.stack(positions,1)
        t=p-(r@joints[...,None]).squeeze(-1)
        skin_r=torch.einsum('vj,bjkl->bvkl',self.weights,r)
        skin_t=torch.einsum('vj,bjk->bvk',self.weights,t)
        vertices=(skin_r@posed[...,None]).squeeze(-1)+skin_t
        root=axis_angle_to_matrix(q[:,3:6]); origin=joints[:,0]
        if convention=='mano':
            world=(root[:,None]@(vertices-origin[:,None])[...,None]).squeeze(-1)+origin[:,None]+q[:,None,:3]
            wrist=origin+q[:,:3]
        elif convention=='origin':
            world=(root[:,None]@vertices[...,None]).squeeze(-1)+q[:,None,:3]
            wrist=(root@origin[...,None]).squeeze(-1)+q[:,:3]
        else:
            raise ValueError('Unknown root convention')
        physical=q.clone(); physical[:,:3]=wrist
        return world.cpu().numpy(), physical.cpu().numpy()


def mirror_pose(pose):
    out=np.asarray(pose).copy()
    out[:,:3]*=np.array([-1,1,1])
    out[:,3:6]*=np.array([1,-1,-1])
    out[:,6:]= (out[:,6:].reshape(-1,15,3)*np.array([1,-1,-1])).reshape(-1,45)
    return out


def prepare_mano_assets(assets):
    p=Path(assets); model=ManoGeometry(p/'MANO_RIGHT.pkl')
    names=['Wrist','Index1','Index2','Index3','Middle1','Middle2','Middle3','Pinky1','Pinky2','Pinky3','Ring1','Ring2','Ring3','Thumb1','Thumb2','Thumb3']
    order=['Wrist','Thumb1','Thumb2','Thumb3','Thumb4','Index1','Index2','Index3','Index4','Middle1','Middle2','Middle3','Middle4','Ring1','Ring2','Ring3','Ring4','Pinky1','Pinky2','Pinky3','Pinky4']
    j=np.zeros((21,778),np.float32)
    for i,n in enumerate(names):j[order.index(n)]=model.J_regressor[i].numpy()
    for n,v in [('Thumb4',745),('Index4',317),('Middle4',445),('Ring4',556),('Pinky4',673)]:j[order.index(n),v]=1
    np.save(p/'J_regressor.npy',j);np.save(p/'faces.npy',model.faces)


def _warp_kernel():
    # Kept in a separate file so Warp can inspect the kernel's source.
    from .mesh_distance import triangle_distance
    return triangle_distance


class MeshDistance:
    def __init__(self, vertices, faces):
        import warp as wp
        self.wp=wp
        self.vertices=wp.array(np.asarray(vertices,np.float32),dtype=wp.vec3,device='cpu')
        self.faces=wp.array(np.asarray(faces,np.int32).ravel(),dtype=wp.int32,device='cpu')
        self.mesh=wp.Mesh(points=self.vertices,indices=self.faces)

    def __call__(self, points):
        wp=self.wp; shape=np.asarray(points).shape[:-1]
        query=wp.array(np.asarray(points,np.float32).reshape(-1,3),dtype=wp.vec3,device='cpu')
        distance=wp.empty(len(query),dtype=wp.float32,device='cpu')
        wp.launch(_warp_kernel(),dim=len(query),inputs=[self.mesh.id,query,distance],device='cpu')
        return distance.numpy().reshape(shape)

    def within(self,points,radius):
        """Exact nearest distances inside radius; 2*radius sentinel outside.

        This preserves threshold labels while avoiding ray-based sign queries
        and distant BVH traversal. The sentinel is not a distance measurement.
        """
        from .mesh_distance import bounded_triangle_distance
        wp=self.wp;shape=np.asarray(points).shape[:-1]
        query=wp.array(np.asarray(points,np.float32).reshape(-1,3),dtype=wp.vec3,device='cpu')
        radii=np.broadcast_to(np.asarray(radius,np.float32),shape).copy().ravel()
        if not np.isfinite(radii).all() or np.any(radii<=0):raise ValueError('Invalid query radius')
        r=wp.array(radii,dtype=wp.float32,device='cpu');distance=wp.empty(len(query),dtype=wp.float32,device='cpu')
        wp.launch(bounded_triangle_distance,dim=len(query),inputs=[self.mesh.id,query,r,distance],device='cpu')
        return distance.numpy().reshape(shape)

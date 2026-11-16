"""Dense bidirectional vertex/face-centre distances to original region triangles."""
import numpy as np
import torch
import warp as wp
from moxian_web.warp_runtime import serialized
from retarget.env.grasp.pose_gpu import vertex_normals


@wp.kernel(enable_backward=False)
def _closest(vertices:wp.array(dtype=wp.vec3),sources:wp.array(dtype=wp.vec3i),
             bary:wp.array(dtype=wp.vec3),meshes:wp.array(dtype=wp.uint64),
             distances:wp.array(dtype=float),targets:wp.array(dtype=wp.vec3i),target_bary:wp.array(dtype=wp.vec3)):
    i=wp.tid();s=sources[i];bc=bary[i]
    p=vertices[s[0]]*bc[0]+vertices[s[1]]*bc[1]+vertices[s[2]]*bc[2]
    q=wp.mesh_query_point_no_sign(meshes[i],p,1.)
    distances[i]=1.e12
    if q.result:
        delta=p-wp.mesh_eval_position(meshes[i],q.face,q.u,q.v)
        distances[i]=wp.dot(delta,delta)
        targets[i]=wp.vec3i(wp.mesh_get_index(meshes[i],3*q.face),wp.mesh_get_index(meshes[i],3*q.face+1),wp.mesh_get_index(meshes[i],3*q.face+2))
        target_bary[i]=wp.vec3(q.u,q.v,1.-q.u-q.v)


class SurfacePairs:
    @serialized
    def __init__(self,vertices,faces,regions,edges):
        self.device=str(vertices.device);self.faces=faces;self.count=len(edges)
        self.vertex=wp.array(vertices.detach().cpu().numpy(),dtype=wp.vec3,device=self.device)
        f=faces.cpu().numpy();self.meshes=[];self.indices=[];cache={};sources=[];bary=[];meshes=[];owners=[]
        for i,edge in enumerate(edges):
            a,b=edge['regions']
            for source,target in ((a,b),(b,a)):
                if target not in cache:
                    idx=wp.array(f[np.asarray(regions[target])].ravel(),dtype=int,device=self.device);self.indices.append(idx)
                    mesh=wp.Mesh(points=self.vertex,indices=idx);self.meshes.append(mesh);cache[target]=mesh.id
                triangles=f[np.asarray(regions[source])];ids=np.unique(triangles)
                sample=np.concatenate((np.repeat(ids[:,None],3,axis=1),triangles))
                bc=np.concatenate((np.tile([1.,0.,0.],(len(ids),1)),np.full((len(triangles),3),1/3)))
                sources.extend(sample);bary.extend(bc);meshes.extend([cache[target]]*len(sample));owners.extend([i]*len(sample))
        self.source=torch.as_tensor(np.asarray(sources),device=self.device,dtype=torch.long)
        self.bc=torch.as_tensor(np.asarray(bary),device=self.device,dtype=torch.float32)
        self.samples=len(sources);self.owners=torch.as_tensor(owners,device=self.device)
        self.wp_source=wp.array(np.asarray(sources),dtype=wp.vec3i,device=self.device)
        self.wp_bc=wp.array(np.asarray(bary),dtype=wp.vec3,device=self.device)
        self.wp_mesh=wp.array(meshes,dtype=wp.uint64,device=self.device)
        self.distance=wp.empty(self.samples,dtype=float,device=self.device)
        self.target=wp.empty(self.samples,dtype=wp.vec3i,device=self.device)
        self.target_bc=wp.empty(self.samples,dtype=wp.vec3,device=self.device)

    @torch.no_grad()
    @serialized
    def evaluate(self,vertices):
        wp.to_torch(self.vertex).copy_(vertices)
        for mesh in self.meshes:mesh.refit()
        wp.launch(_closest,self.samples,inputs=[self.vertex,self.wp_source,self.wp_bc,self.wp_mesh,self.distance,self.target,self.target_bc],device=self.device)
        distances=wp.to_torch(self.distance)
        selected=[]
        for i in range(self.count):
            ids=torch.where(self.owners==i)[0];selected.append(ids[distances[ids].argmin()])
        selected=torch.stack(selected)
        if (distances[selected]>=1.e11).any():raise ValueError('表面接触查询失败')
        normals=vertex_normals(vertices,self.faces)
        source_normal=torch.nn.functional.normalize((normals[self.source[selected]]*self.bc[selected,:,None]).sum(-2),dim=-1)
        tri=wp.to_torch(self.target)[selected].long();bc=wp.to_torch(self.target_bc)[selected]
        target_normal=torch.nn.functional.normalize((normals[tri]*bc[...,None]).sum(-2),dim=-1)
        gap=distances[selected].clamp_min(0).sqrt()*1000
        error=torch.rad2deg(torch.acos((-(source_normal*target_normal).sum(-1)).clamp(-1,1)))
        return gap.cpu().tolist(),error.cpu().tolist()

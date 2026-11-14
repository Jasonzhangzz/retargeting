"""Original-skin signed witnesses, independent of tactile and templates.

Caps close anatomical partitions for sign classification only. Distances and
gradients always use original Qnbot triangles. MuJoCo proxy geoms are not used.
"""
import numpy as np
import torch
import warp as wp
from .warp_runtime import serialized
from .hand_anatomy import closed_surface_parts


@wp.kernel(enable_backward=False)
def _query(meshes:wp.array(dtype=wp.uint64),solids:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),source:wp.array(dtype=int),
           face:wp.array2d(dtype=int),bary:wp.array2d(dtype=wp.vec2),distance:wp.array2d(dtype=float)):
    i,k=wp.tid();face[i,k]=-1;distance[i,k]=1.
    if source[i]!=k:
        q=wp.mesh_query_point_no_sign(meshes[k],points[i],.025)
        if q.result:
            p=wp.mesh_eval_position(meshes[k],q.face,q.u,q.v)
            solid=wp.mesh_query_point_sign_winding_number(solids[k],points[i],.025)
            d=wp.length(points[i]-p);inside=False
            if solid.result:
                cap=wp.mesh_eval_position(solids[k],solid.face,solid.u,solid.v)
                inside=solid.sign<0. and d<=wp.length(points[i]-cap)+.0005
            face[i,k]=q.face;bary[i,k]=wp.vec2(q.u,q.v);distance[i,k]=wp.where(inside,-d,d)


class QnbotCollision:
    @serialized
    def __init__(self,model,dense=False):
        self.device=str(model.device);self.faces=model.faces.cpu().numpy().astype(np.int32);self.tf=model.faces
        with torch.no_grad():vertices=model(model.q0[None])[0][0].cpu().numpy()
        labels=model.weights.cpu().numpy().argmax(1)
        self.parts=np.array([np.bincount(labels[f],minlength=20).argmax() for f in self.faces]);self.part_ids=[np.flatnonzero(self.parts==p) for p in range(20)]
        if any(not len(x) for x in self.part_ids):raise ValueError('官方碰撞分区为空')
        self.closed=closed_surface_parts(vertices,self.faces,self.parts,20)
        self.vertex_buffer=wp.array(vertices,dtype=wp.vec3,device=self.device);self.meshes=[];self.solids=[];self.buffers=[]
        for p,(v,f,loops) in enumerate(self.closed):
            buf=wp.array(v,dtype=wp.vec3,device=self.device);self.buffers.append(buf)
            self.meshes.append(wp.Mesh(points=self.vertex_buffer,indices=wp.array(self.faces[self.part_ids[p]].ravel(),dtype=int,device=self.device)))
            self.solids.append(wp.Mesh(points=buf,indices=wp.array(f.ravel(),dtype=int,device=self.device),support_winding_number=True))
        self.mesh_ids=wp.array([x.id for x in self.meshes],dtype=wp.uint64,device=self.device);self.solid_ids=wp.array([x.id for x in self.solids],dtype=wp.uint64,device=self.device)
        sources=np.concatenate((np.repeat(np.arange(len(vertices))[:,None],3,axis=1),self.faces));bc=np.concatenate((np.tile([1.,0.,0.],(len(vertices),1)),np.full((len(self.faces),3),1/3)))
        source_part=np.concatenate((labels,self.parts))
        if not dense:
            ids=np.unique(np.linspace(0,len(sources)-1,min(1536,len(sources))).astype(int));sources=sources[ids];bc=bc[ids];source_part=source_part[ids]
        self.source=torch.as_tensor(sources,device=model.device,dtype=torch.long);self.bc=torch.as_tensor(bc,device=model.device,dtype=torch.float32)
        self.source_part=wp.array(source_part.astype(np.int32),dtype=int,device=self.device)
        self.source_part_t=torch.as_tensor(source_part,device=model.device,dtype=torch.long)
        n=len(sources);self.points=wp.zeros(n,dtype=wp.vec3,device=self.device);self.face=wp.zeros((n,20),dtype=int,device=self.device);self.bary=wp.zeros((n,20),dtype=wp.vec2,device=self.device);self.distance=wp.zeros((n,20),dtype=float,device=self.device)
        self.part_t=[torch.as_tensor(x,device=model.device) for x in self.part_ids]
        self.dense=dense

    @torch.no_grad()
    @serialized
    def query(self,vertices):
        wp.to_torch(self.vertex_buffer).copy_(vertices.detach())
        for mesh in self.meshes:mesh.refit()
        for solid,buf,(_,_,loops) in zip(self.solids,self.buffers,self.closed):
            out=wp.to_torch(buf);out[:len(vertices)].copy_(vertices.detach())
            for k,loop in enumerate(loops):out[len(vertices)+k].copy_(vertices[torch.as_tensor(loop,device=vertices.device)].mean(0))
            solid.refit()
        points=(vertices[self.source]*self.bc[...,None]).sum(1);wp.to_torch(self.points).copy_(points)
        wp.launch(_query,(len(points),20),inputs=[self.mesh_ids,self.solid_ids,self.points,self.source_part,self.face,self.bary,self.distance],device=self.device)
        face=wp.to_torch(self.face).long();distance=wp.to_torch(self.distance)
        target=torch.stack([ids[face[:,k].clamp_min(0)] for k,ids in enumerate(self.part_t)],1)
        # An incident triangle cannot be an independent self-collision witness.
        near=(self.faces_tensor(vertices.device)[target][...,None]==self.source[:,None,None,:]).any(-1).any(-1)
        valid=(face>=0)&~near&torch.isfinite(distance)
        chosen=(valid&(distance<.001)).nonzero()
        if len(chosen)>8192:
            ids=torch.topk(torch.where(valid,distance,torch.full_like(distance,1.)).flatten(),8192,largest=False).indices
            chosen=torch.stack((ids//20,ids%20),1)
        a,b=chosen.unbind(1) if len(chosen) else (torch.empty(0,device=vertices.device,dtype=torch.long),)*2
        uv=wp.to_torch(self.bary)[a,b].clone();signed=distance[a,b].clone()
        return dict(source=self.source[a],source_bc=self.bc[a],source_part=self.source_part_t[a],target_part=b,target=self.faces_tensor(vertices.device)[target[a,b]],target_bc=torch.stack((uv[:,0],uv[:,1],1-uv.sum(-1)),1),
                    sign=torch.where(signed<0,-torch.ones_like(signed),torch.ones_like(signed)),max_penetration_mm=float(torch.where(valid,(-distance).clamp_min(0),torch.zeros_like(distance)).max())*1000,
                    samples=len(points),query='qnbot_original_triangles_vertices_and_face_centroids' if self.dense else 'qnbot_original_triangles_stratified_witnesses')

    def faces_tensor(self,device):return self.tf

    @staticmethod
    def residual(vertices,witness,tolerance_m=0.):
        p=(vertices[witness['source']]*witness['source_bc'][...,None]).sum(-2);q=(vertices[witness['target']]*witness['target_bc'][...,None]).sum(-2)
        signed=(p-q).square().sum(-1).clamp_min(1e-16).sqrt()*witness['sign']
        return ((-signed-tolerance_m)/.002).clamp_min(0)


@wp.kernel(enable_backward=False)
def _dense_batch(meshes:wp.array(dtype=wp.uint64),solids:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),samples:int,
                 source_part:wp.array(dtype=int),source_ids:wp.array(dtype=wp.vec3i),part_faces:wp.array2d(dtype=wp.vec3i),face_counts:wp.array(dtype=int),
                 lower:wp.array(dtype=wp.vec3),upper:wp.array(dtype=wp.vec3),maximum:wp.array(dtype=float),bad:wp.array(dtype=int)):
    i,k=wp.tid();world=i//samples;s=i%samples
    point=points[i];lo=lower[world*20+k];hi=upper[world*20+k]
    inside_box=point[0]>=lo[0]-.000001 and point[1]>=lo[1]-.000001 and point[2]>=lo[2]-.000001 and point[0]<=hi[0]+.000001 and point[1]<=hi[1]+.000001 and point[2]<=hi[2]+.000001
    if source_part[s]!=k and inside_box:
        q=wp.mesh_query_point_no_sign(meshes[world*20+k],points[i],.025)
        if q.result:
            if q.face>=face_counts[k]:wp.atomic_add(bad,world,1)
            else:
                target=part_faces[k,q.face%face_counts[k]];source=source_ids[s];near=False
                for a in range(3):
                    for b in range(3):
                        if target[a]==source[b]:near=True
                if not near:
                    solid=wp.mesh_query_point_sign_winding_number(solids[world*20+k],points[i],.025)
                    if solid.result and solid.sign<0.:
                        p=wp.mesh_eval_position(meshes[world*20+k],q.face,q.u,q.v);cap=wp.mesh_eval_position(solids[world*20+k],solid.face,solid.u,solid.v)
                        distance=wp.length(points[i]-p)
                        if distance<=wp.length(points[i]-cap)+.0005:wp.atomic_max(maximum,world,distance)


class QnbotBatchCollision:
    """Independent per-world BVHs; one captured batched query dispatch."""
    @serialized
    def __init__(self,model,batch=16):
        self.device=str(model.device);self.batch=batch;self.nv=len(model.vertices0);self.faces=model.faces.cpu().numpy().astype(np.int32)
        with torch.no_grad():v=model(model.q0[None])[0][0].cpu().numpy()
        labels=model.weights.cpu().numpy().argmax(1);parts=np.array([np.bincount(labels[f],minlength=20).argmax() for f in self.faces]);self.part_ids=[np.flatnonzero(parts==p) for p in range(20)]
        closed=closed_surface_parts(v,self.faces,parts,20);self.loops=[loops for _,_,loops in closed]
        ids=np.arange(batch);width=int(np.ceil(np.sqrt(batch)));offset=np.stack(((ids%width)*2.,(ids//width)*2.,np.zeros(batch)),1).astype(np.float32)
        self.offset=torch.tensor(offset,device=model.device);initial=v[None]+offset[:,None];self.vertices=wp.array(initial.reshape(-1,3),dtype=wp.vec3,device=self.device)
        self.meshes=[None]*(20*batch);self.solids=[None]*(20*batch);self.buffers=[]
        maxfaces=max(map(len,self.part_ids));packed=np.zeros((20,maxfaces,3),np.int32)
        for k,((points,closed_faces,loops),skin_ids) in enumerate(zip(closed,self.part_ids)):
            packed[k,:len(skin_ids)]=self.faces[skin_ids]
            stride=len(points);buffer=wp.array((points[None]+offset[:,None]).reshape(-1,3),dtype=wp.vec3,device=self.device);self.buffers.append(buffer)
            for i in range(batch):
                triangles=self.faces[skin_ids]+i*self.nv
                self.meshes[i*20+k]=wp.Mesh(points=self.vertices,indices=wp.array(triangles.ravel(),dtype=int,device=self.device))
                cf=closed_faces+i*stride;self.solids[i*20+k]=wp.Mesh(points=buffer,indices=wp.array(cf.ravel(),dtype=int,device=self.device),support_winding_number=True)
        source=np.concatenate((np.repeat(np.arange(self.nv)[:,None],3,1),self.faces));self.samples=len(source)
        self.source=torch.tensor(source,device=model.device);self.bc=torch.tensor(np.concatenate((np.tile([1.,0.,0.],(self.nv,1)),np.full((len(self.faces),3),1/3))),device=model.device,dtype=torch.float32)
        self.source_wp=wp.array(source,dtype=wp.vec3i,device=self.device);self.source_part=wp.array(np.concatenate((labels,parts)).astype(np.int32),dtype=int,device=self.device)
        self.part_faces=wp.array(packed,dtype=wp.vec3i,device=self.device);self.counts=wp.array([len(x) for x in self.part_ids],dtype=int,device=self.device)
        self.mesh_ids=wp.array([x.id for x in self.meshes],dtype=wp.uint64,device=self.device);self.solid_ids=wp.array([x.id for x in self.solids],dtype=wp.uint64,device=self.device)
        self.points=wp.zeros(batch*self.samples,dtype=wp.vec3,device=self.device);self.maximum=wp.zeros(batch,dtype=float,device=self.device);self.bad=wp.zeros(batch,dtype=int,device=self.device)
        self.part_vertices=[torch.tensor(np.unique(self.faces[ids]),device=model.device) for ids in self.part_ids]
        self.lower=wp.zeros(batch*20,dtype=wp.vec3,device=self.device);self.upper=wp.zeros(batch*20,dtype=wp.vec3,device=self.device)
        self.query_torch_stream=torch.cuda.Stream(device=model.device);self.query_stream=wp.stream_from_torch(self.query_torch_stream);self.query_graph=None;self.graph_attempted=False;self.graph_error=None

    @torch.no_grad()
    @serialized
    def query(self,vertices):
        count=len(vertices)
        if count>self.batch or count<1:raise ValueError('碰撞批次大小无效')
        if not torch.isfinite(vertices).all() or vertices.abs().max()>.6:raise ValueError('官方曲面出现非有限坐标或单位异常')
        if count<self.batch:vertices=torch.cat((vertices,vertices[-1:].expand(self.batch-count,-1,-1)))
        shifted=vertices+self.offset[:,None];wp.to_torch(self.vertices).copy_(shifted.reshape(-1,3))
        for buffer,loops in zip(self.buffers,self.loops):
            value=wp.to_torch(buffer).reshape(self.batch,self.nv+len(loops),3);value[:,:self.nv].copy_(shifted)
            for k,loop in enumerate(loops):value[:,self.nv+k].copy_(vertices[:,torch.as_tensor(loop,device=vertices.device)].mean(1)+self.offset)
        points=(vertices[:,self.source]*self.bc[None,...,None]).sum(-2)+self.offset[:,None];wp.to_torch(self.points).copy_(points.reshape(-1,3))
        wp.to_torch(self.lower).copy_((torch.stack([vertices[:,ids].amin(1) for ids in self.part_vertices],1)+self.offset[:,None]).reshape(-1,3))
        wp.to_torch(self.upper).copy_((torch.stack([vertices[:,ids].amax(1) for ids in self.part_vertices],1)+self.offset[:,None]).reshape(-1,3))
        wp.to_torch(self.maximum).zero_();wp.to_torch(self.bad).zero_()
        def queries():
            for mesh in self.meshes:mesh.refit()
            for solid in self.solids:solid.refit()
            wp.launch(_dense_batch,(self.batch*self.samples,20),inputs=[self.mesh_ids,self.solid_ids,self.points,self.samples,self.source_part,self.source_wp,self.part_faces,self.counts,self.lower,self.upper,self.maximum,self.bad],device=self.device)
        current=torch.cuda.current_stream(vertices.device);self.query_torch_stream.wait_stream(current)
        with wp.ScopedStream(self.query_stream):
            if not self.graph_attempted:
                queries();self.graph_attempted=True
                try:
                    with wp.ScopedCapture(stream=self.query_stream) as captured:queries()
                    self.query_graph=captured.graph
                except RuntimeError as exc:self.graph_error=str(exc)
            if self.query_graph is not None:wp.capture_launch(self.query_graph)
            elif self.graph_attempted:queries()
        current.wait_stream(self.query_torch_stream)
        result=wp.to_torch(self.maximum)[:count].clone()*1000
        if bool(wp.to_torch(self.bad).any()):raise ValueError('分离世界碰撞查询发生跨序列命中')
        return result

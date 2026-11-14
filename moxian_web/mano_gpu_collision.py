"""GPU-resident batched original-triangle witnesses; no convex proxies.

Worlds occupy a separated grid inside sixteen shared anatomical BVHs. Anatomical
closed volumes establish signs; artificial seam caps never supply distances.
"""
import math
import numpy as np
import torch
import warp as wp

from .mano_collision import SurfaceCollision, COLLISION_MARGIN_M
from .warp_runtime import serialized


@wp.func
def triangle_distance_squared(point:wp.vec3,a:wp.vec3,b:wp.vec3,c:wp.vec3):
    ab=b-a;ac=c-a;ap=point-a
    normal=wp.cross(ab,ac);nn=wp.dot(normal,normal)
    projected=point-normal*(wp.dot(ap,normal)/wp.max(nn,1.e-20))
    v0=ab;v1=ac;v2=projected-a
    d00=wp.dot(v0,v0);d01=wp.dot(v0,v1);d11=wp.dot(v1,v1)
    d20=wp.dot(v2,v0);d21=wp.dot(v2,v1)
    den=d00*d11-d01*d01
    v=(d11*d20-d01*d21)/wp.max(den,1.e-20)
    w=(d00*d21-d01*d20)/wp.max(den,1.e-20)
    result=float(1.e6)
    if den>1.e-20 and v>=0. and w>=0. and v+w<=1.:result=wp.length_sq(point-projected)
    for edge in range(3):
        start=a;end=b
        if edge==1:start=b;end=c
        if edge==2:start=c;end=a
        delta=end-start;t=wp.clamp(wp.dot(point-start,delta)/wp.max(wp.dot(delta,delta),1.e-20),0.,1.)
        result=wp.min(result,wp.length_sq(point-start-t*delta))
    return result


@wp.kernel
def batched_queries(meshes:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),local_points:wp.array(dtype=wp.vec3),samples:int,
                    closed_points:wp.array(dtype=wp.vec3),closed_faces:wp.array2d(dtype=wp.vec3i),counts:wp.array(dtype=int),skin_counts:wp.array(dtype=int),stride:int,
                    lower:wp.array(dtype=wp.vec3),upper:wp.array(dtype=wp.vec3),source_part:wp.array(dtype=int),
                    face:wp.array2d(dtype=int),uv:wp.array2d(dtype=wp.vec2),distance:wp.array2d(dtype=float)):
    sample,part=wp.tid();world=sample//samples
    pnt=points[sample];lo=lower[world*16+part];hi=upper[world*16+part]
    far=(pnt[0]<lo[0]-.001 or pnt[1]<lo[1]-.001 or pnt[2]<lo[2]-.001 or
         pnt[0]>hi[0]+.001 or pnt[1]>hi[1]+.001 or pnt[2]>hi[2]+.001)
    distance[sample,part]=1.;face[sample,part]=-2;uv[sample,part]=wp.vec2(0.,0.)
    if not far and source_part[sample%samples]!=part:
        face[sample,part]=-1
        hit=wp.mesh_query_point_no_sign(meshes[part],pnt,1.)
        if hit.result:
            point=local_points[sample];direction=wp.vec3(.827813,.419317,.372113)
            winding=int(0);cap_distance=float(1.e6)
            base=(world*16+part)*stride
            for f in range(counts[part]):
                ids=closed_faces[part,f]
                a=closed_points[base+ids[0]];b=closed_points[base+ids[1]];c=closed_points[base+ids[2]]
                e1=b-a;e2=c-a;p=wp.cross(direction,e2);det=wp.dot(e1,p)
                if wp.abs(det)>1.e-10:
                    inv=1./det;t=point-a;u=wp.dot(t,p)*inv;q=wp.cross(t,e1)
                    v=wp.dot(direction,q)*inv;travel=wp.dot(e2,q)*inv
                    if u>=0. and v>=0. and u+v<=1. and travel>1.e-7:
                        if det>0.:winding+=1
                        else:winding-=1
                if f>=skin_counts[part]:cap_distance=wp.min(cap_distance,triangle_distance_squared(point,a,b,c))
            closest=wp.mesh_eval_position(meshes[part],hit.face,hit.u,hit.v)
            d=wp.length(pnt-closest)
            inside=winding!=0 and d<=wp.sqrt(cap_distance)+.0005
            distance[sample,part]=wp.where(inside,-d,d);face[sample,part]=hit.face;uv[sample,part]=wp.vec2(hit.u,hit.v)



class BatchedSurfaceCollision:
    @serialized
    def __init__(self, model, batch):
        self.reference = SurfaceCollision(model)
        c = self.reference
        self.batch, self.device = batch, model.device
        self.vertices_count = len(model.rest_np)
        self.t = lambda x,dtype=torch.float32: torch.as_tensor(x,device=self.device,dtype=dtype)
        self.faces = self.t(c.faces,torch.long)
        self.source_ids = self.t(c.source_ids,torch.long)
        self.source_bc = self.t(c.source_bc)
        self.source_part = self.t(c.source_part,torch.long)
        self.excluded = self.t(c.excluded,torch.bool)
        ids=torch.arange(batch,device=self.device);width=math.ceil(math.sqrt(batch))
        self.offsets=torch.stack(((ids%width)*2.,(ids//width)*2.,torch.zeros_like(ids)),1)
        self.meshes=[];self.solids=[];self.solid_buffers=[]
        with wp.ScopedDevice(str(self.device)):
            # Build spatial BVH topology from separated real rest meshes.
            # Refitting a BVH constructed from all-zero vertices is quadratic
            # in world count because every branch spans unrelated worlds.
            initial=np.asarray(model.rest_np,np.float32)[None]+self.offsets.cpu().numpy()[:,None]
            self.vertex_buffer=wp.array(initial.reshape(-1,3),dtype=wp.vec3)
            closed=[m.indices.numpy().reshape(-1,3) for m in c.solids]
            maxfaces=max(map(len,closed));packed=np.zeros((16,maxfaces,3),np.int32)
            for part,face in enumerate(closed):packed[part,:len(face)]=face
            self.closed_faces=wp.array(packed,dtype=wp.vec3i)
            self.closed_counts=wp.array([len(x) for x in closed],dtype=int)
            self.skin_counts=wp.array([len(x) for x in c.part_faces],dtype=int)
            self.solid_stride=self.vertices_count+max(map(len,c.loops))
            self.closed_points=wp.zeros(batch*16*self.solid_stride,dtype=wp.vec3)
            self.local_points=wp.zeros(batch*len(c.source_ids),dtype=wp.vec3)
            for part,ids in enumerate(c.part_faces):
                face=c.faces[ids];indices=np.concatenate([face+i*self.vertices_count for i in range(batch)])
                self.meshes.append(wp.Mesh(points=self.vertex_buffer,indices=wp.array(indices.ravel(),dtype=int)))
            self.mesh_ids=wp.array([m.id for m in self.meshes],dtype=wp.uint64)
            self.solid_ids=wp.array([m.id for m in self.solids],dtype=wp.uint64)
            count=batch*len(c.source_ids)
            self.points=wp.zeros(count,dtype=wp.vec3)
            self.face=wp.zeros((count,16),dtype=int);self.uv=wp.zeros((count,16),dtype=wp.vec2)
            self.distance=wp.zeros((count,16),dtype=float)
            self.lower=wp.zeros(batch*16,dtype=wp.vec3);self.upper=wp.zeros(batch*16,dtype=wp.vec3)
            self.source_parts_wp=wp.array(c.source_part,dtype=int)
        self.part_face_ids=[self.t(x,torch.long) for x in c.part_faces]
        self.loop_indices=[[self.t(loop,torch.long) for loop in loops] for loops in c.loops]
        self.part_vertex_ids=[self.t(np.unique(c.faces[x]),torch.long) for x in c.part_faces]
        self.valid=None;self.query_graph=None;self.graph_attempted=False;self.graph_error=None
        # PyTorch streams are nonblocking: capturing this stream must not make
        # the display thread's legacy/default stream depend on a capture.
        self.query_torch_stream=torch.cuda.Stream(device=self.device)
        self.query_stream=wp.stream_from_torch(self.query_torch_stream)
        wp.synchronize_device(str(self.device))

    @torch.no_grad()
    @serialized
    def query(self, vertices):
        b,s=self.batch,len(self.source_ids)
        # The grid separation is valid only for wrist-local hands, not scenes.
        self.valid=torch.isfinite(vertices).all() & (vertices.abs().amax()<.5)
        shifted=vertices+self.offsets[:,None]
        wp.to_torch(self.vertex_buffer).copy_(shifted.reshape(-1,3))
        closed=wp.to_torch(self.closed_points).reshape(b,16,self.solid_stride,3)
        closed[:,:,:self.vertices_count].copy_(vertices[:,None])
        for part,loops in enumerate(self.loop_indices):
            for k,loop in enumerate(loops):closed[:,part,self.vertices_count+k].copy_(vertices[:,loop].mean(1))
        local=(vertices[:,self.source_ids]*self.source_bc[None,:,:,None]).sum(-2)
        wp.to_torch(self.local_points).copy_(local.reshape(-1,3))
        wp.to_torch(self.points).copy_((local+self.offsets[:,None]).reshape(-1,3))
        wp.to_torch(self.lower).copy_((torch.stack([vertices[:,ids].amin(1) for ids in self.part_vertex_ids],1)+self.offsets[:,None]).reshape(-1,3))
        wp.to_torch(self.upper).copy_((torch.stack([vertices[:,ids].amax(1) for ids in self.part_vertex_ids],1)+self.offsets[:,None]).reshape(-1,3))
        def run_queries():
            for mesh in self.meshes:mesh.refit()
            for mesh in self.solids:mesh.refit()
            wp.launch(batched_queries,(b*s,16),inputs=[self.mesh_ids,self.points,self.local_points,s,self.closed_points,self.closed_faces,self.closed_counts,self.skin_counts,self.solid_stride,self.lower,self.upper,self.source_parts_wp,self.face,self.uv,self.distance],device=str(self.device))
        stream=self.query_stream
        current=torch.cuda.current_stream(self.device)
        self.query_torch_stream.wait_stream(current)
        with wp.ScopedStream(stream):
            if not self.graph_attempted:
                run_queries();self.graph_attempted=True
                try:
                    with wp.ScopedCapture(stream=stream) as captured:run_queries()
                    self.query_graph=captured.graph
                except RuntimeError as exc:self.graph_error=str(exc)
            if self.query_graph is not None:wp.capture_launch(self.query_graph)
            else:run_queries()
        current.wait_stream(self.query_torch_stream)
        face=wp.to_torch(self.face).reshape(b,s,16).long()
        uv=wp.to_torch(self.uv).reshape(b,s,16,2)
        distance=wp.to_torch(self.distance).reshape(b,s,16)
        actual=[];owners=[]
        for p,ids in enumerate(self.part_face_ids):
            actual.append(ids[face[:,:,p].clamp_min(0)%len(ids)])
            owners.append(face[:,:,p].clamp_min(0)//len(ids))
        target_faces=torch.stack(actual,-1);owners=torch.stack(owners,-1)
        complete=(face==-2)|((face>=0)&(owners==torch.arange(b,device=self.device)[:,None,None]))
        self.valid=self.valid & complete.all() & torch.isfinite(distance).all()
        source=torch.arange(s,device=self.device)[None,:,None]
        mask=(face>=0) & (self.source_part[None,:,None]!=torch.arange(16,device=self.device)[None,None]) & ~self.excluded[source,target_faces]
        capacity=min(2048,s*16)
        flat_mask=mask.flatten(1)&(distance.flatten(1)<COLLISION_MARGIN_M)
        overflow=flat_mask.sum(1)>capacity
        score=torch.where(flat_mask,distance.flatten(1),torch.full_like(distance.flatten(1),1000.))
        selected=score.topk(capacity,dim=1,largest=False,sorted=False).indices
        samples=selected//16
        target_faces=target_faces.flatten(1).gather(1,selected)
        selected_uv=uv.reshape(b,s*16,2).gather(1,selected[:,:,None].expand(-1,-1,2))
        bc=torch.cat((selected_uv,1-selected_uv.sum(-1,keepdim=True)),-1)
        selected_distance=distance.flatten(1).gather(1,selected)
        target_ids=self.faces[target_faces]
        source_ids=self.source_ids[samples];source_bc=self.source_bc[samples]
        worlds=torch.arange(b,device=self.device)[:,None,None]
        tri=vertices[worlds,target_ids]
        target=(tri*bc[...,None]).sum(-2)
        source=(vertices[worlds,source_ids]*source_bc[...,None]).sum(-2)
        delta=source-target
        normal=torch.nn.functional.normalize(delta,dim=-1)*torch.where(selected_distance<0,-1.,1.)[...,None]
        fallback=torch.nn.functional.normalize(torch.linalg.cross(tri[...,1,:]-tri[...,0,:],tri[...,2,:]-tri[...,0,:]),dim=-1)
        normal=torch.where(delta.norm(dim=-1,keepdim=True)<1e-8,fallback,normal)
        return dict(source_ids=source_ids,source_bc=source_bc,target_ids=target_ids,target_bc=bc,normal=normal,
                    mask=flat_mask.gather(1,selected),witness_overflow=overflow,
                    maximum=torch.where(mask,(-distance).clamp_min(0),0).amax((1,2)),valid=self.valid)

    def residual(self,vertices,witness):
        worlds=torch.arange(self.batch,device=self.device)[:,None,None]
        a=(vertices[worlds,witness['source_ids']]*witness['source_bc'][...,None]).sum(-2)
        b=(vertices[worlds,witness['target_ids']]*witness['target_bc'][...,None]).sum(-2)
        gap=((a-b)*witness['normal']).sum(-1)
        return torch.relu(COLLISION_MARGIN_M-gap)/.001*witness['mask']

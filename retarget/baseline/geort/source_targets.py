"""Batched original-skin contact labels, preserving the legacy witness rules."""
import numpy as np
import torch
import warp as wp
from moxian_web.warp_runtime import serialized
from moxian_web.hand_anatomy import closed_surface_parts
from .adapter import FINGERS


@wp.kernel(enable_backward=False)
def _gap(vertices:wp.array2d(dtype=wp.vec3),source:wp.array(dtype=wp.vec3i),bc:wp.array(dtype=wp.vec3),
         owner:wp.array(dtype=int),meshes:wp.array(dtype=wp.uint64),result:wp.array3d(dtype=float)):
    b,s,t=wp.tid();a=owner[s]
    if a==t:return
    tri=source[s];w=bc[s]
    point=vertices[b,tri[0]]*w[0]+vertices[b,tri[1]]*w[1]+vertices[b,tri[2]]*w[2]
    q=wp.mesh_query_point_no_sign(meshes[b*5+t],point,1.)
    if q.result:
        delta=point-wp.mesh_eval_position(meshes[b*5+t],q.face,q.u,q.v)
        wp.atomic_min(result,b,a,t,wp.dot(delta,delta))


@wp.kernel(enable_backward=False)
def _depth(points:wp.array2d(dtype=wp.vec3),source:wp.array(dtype=wp.vec3i),owner:wp.array(dtype=int),
           meshes:wp.array(dtype=wp.uint64),solids:wp.array(dtype=wp.uint64),
           faces:wp.array2d(dtype=wp.vec3i),result:wp.array3d(dtype=float)):
    b,s,t=wp.tid();a=owner[s]
    if a==t:return
    point=points[b,s];q=wp.mesh_query_point_no_sign(meshes[b*5+t],point,.025)
    if not q.result:return
    target=faces[t,q.face];src=source[s];near=False
    for i in range(3):
        for j in range(3):
            if target[i]==src[j]:near=True
    if near:return
    solid=wp.mesh_query_point_sign_winding_number(solids[b*5+t],point,.025)
    if solid.result and solid.sign<0.:
        hit=wp.mesh_eval_position(meshes[b*5+t],q.face,q.u,q.v)
        cap=wp.mesh_eval_position(solids[b*5+t],solid.face,solid.u,solid.v)
        distance=wp.length(point-hit)
        if distance<=wp.length(point-cap)+.0005:wp.atomic_max(result,b,a,t,distance*1000.)


class BatchSourceContacts:
    """Independent BVHs for each frame, no spatial shift or geometry reduction.

    All vertices and face centres are retained, as in the previous dense query.
    Only distal/distal pairs are queried: those are the sole labels consumed by
    this training mode. Normals and optimizer collision witnesses are unnecessary.
    """
    @serialized
    def __init__(self,model,topology,batch=16):
        self.device=str(model.device);self.batch=batch;self.model=model
        faces=model.faces.cpu().numpy().astype(np.int32);self.nv=len(model.vertices0)
        with torch.no_grad():rest=model(model.q0[None])[0][0].cpu().numpy()
        labels=model.weights.cpu().numpy().argmax(1)
        parts=np.array([np.bincount(labels[f],minlength=20).argmax() for f in faces])
        selected=[next(i for i,n in enumerate(model.bone_names) if n.endswith('_'+('little' if f=='pinky' else f)+'_distal')) for f in FINGERS]
        closed=closed_surface_parts(rest,faces,parts,20)
        self.vertices=wp.array(np.broadcast_to(rest,(batch,self.nv,3)).copy().reshape(-1,3),dtype=wp.vec3,device=self.device)
        self.skin=[];self.solids=[];self.regions=[];self.buffers=[];self.loops=[]
        skin_ids=[np.flatnonzero(parts==k) for k in selected]
        packed=np.zeros((5,max(map(len,skin_ids)),3),np.int32)
        for t,k in enumerate(selected):
            packed[t,:len(skin_ids[t])]=faces[skin_ids[t]]
            v,f,loops=closed[k];stride=len(v)
            buf=wp.array(np.broadcast_to(v,(batch,len(v),3)).copy().reshape(-1,3),dtype=wp.vec3,device=self.device)
            self.buffers.append(buf);self.loops.append([torch.tensor(loop,device=self.device) for loop in loops])
            for b in range(batch):
                self.skin.append(wp.Mesh(points=self.vertices,indices=wp.array((faces[skin_ids[t]]+b*self.nv).ravel(),dtype=int,device=self.device)))
                self.solids.append(wp.Mesh(points=buf,indices=wp.array((f+b*stride).ravel(),dtype=int,device=self.device),support_winding_number=True))
                region=faces[np.asarray(topology['regions'][FINGERS[t]+'_distal'])]
                self.regions.append(wp.Mesh(points=self.vertices,indices=wp.array((region+b*self.nv).ravel(),dtype=int,device=self.device)))
        # Construction is finger-major; kernels index frame-major.
        self.skin_ids=wp.array([self.skin[t*batch+b].id for b in range(batch) for t in range(5)],dtype=wp.uint64,device=self.device)
        self.solid_ids=wp.array([self.solids[t*batch+b].id for b in range(batch) for t in range(5)],dtype=wp.uint64,device=self.device)
        self.region_ids=wp.array([self.regions[t*batch+b].id for b in range(batch) for t in range(5)],dtype=wp.uint64,device=self.device)
        self.part_faces=wp.array(packed,dtype=wp.vec3i,device=self.device)
        # Source skin collision witnesses: same original classification and samples.
        all_sources=np.concatenate((np.repeat(np.arange(self.nv)[:,None],3,axis=1),faces))
        all_bc=np.concatenate((np.tile([1.,0.,0.],(self.nv,1)),np.full((len(faces),3),1/3)))
        all_parts=np.concatenate((labels,parts));ids=np.flatnonzero(np.isin(all_parts,selected))
        self.source=torch.tensor(all_sources[ids],device=self.device)
        self.bc=torch.tensor(all_bc[ids],device=self.device,dtype=torch.float32)
        self.wp_source=wp.array(all_sources[ids],dtype=wp.vec3i,device=self.device)
        lookup={k:i for i,k in enumerate(selected)}
        self.owner=wp.array([lookup[k] for k in all_parts[ids]],dtype=int,device=self.device)
        self.points=wp.empty((batch,len(ids)),dtype=wp.vec3,device=self.device)
        sources=[];bary=[];owners=[]
        for t,finger in enumerate(FINGERS):
            tri=faces[np.asarray(topology['regions'][finger+'_distal'])];verts=np.unique(tri)
            source=np.concatenate((np.repeat(verts[:,None],3,axis=1),tri))
            sources.extend(source);bary.extend(np.concatenate((np.tile([1.,0.,0.],(len(verts),1)),np.full((len(tri),3),1/3))))
            owners.extend([t]*len(source))
        self.gap_source=wp.array(sources,dtype=wp.vec3i,device=self.device)
        self.gap_bc=wp.array(bary,dtype=wp.vec3,device=self.device);self.gap_owner=wp.array(owners,dtype=int,device=self.device)
        self.gaps=wp.empty((batch,5,5),dtype=float,device=self.device);self.depth=wp.zeros((batch,5,5),dtype=float,device=self.device)
        self.query_torch_stream=torch.cuda.Stream(device=self.device);self.query_stream=wp.stream_from_torch(self.query_torch_stream)
        self.graph=None;self.graph_attempted=False;self.graph_error=None
        self.pair_a=torch.tensor([a for a in range(5) for b in range(a+1,5)],device=self.device)
        self.pair_b=torch.tensor([b for a in range(5) for b in range(a+1,5)],device=self.device)

    @torch.no_grad()
    @serialized
    def query(self,vertices):
        count=len(vertices)
        if not 1<=count<=self.batch or vertices.shape[1:]!=(self.nv,3):raise ValueError('接触标注批次无效')
        if not torch.isfinite(vertices).all():raise ValueError('源网格无效')
        if count<self.batch:vertices=torch.cat((vertices,vertices[-1:].expand(self.batch-count,-1,-1)))
        wp.to_torch(self.vertices).copy_(vertices.reshape(-1,3))
        for buffer,loops in zip(self.buffers,self.loops):
            v=wp.to_torch(buffer).reshape(self.batch,self.nv+len(loops),3);v[:,:self.nv].copy_(vertices)
            for k,loop in enumerate(loops):v[:,self.nv+k].copy_(vertices[:,loop].mean(1))
        points=(vertices[:,self.source]*self.bc[None,...,None]).sum(-2)
        wp.to_torch(self.points).copy_(points)
        wp.to_torch(self.gaps).fill_(1.e12);wp.to_torch(self.depth).zero_()
        def dispatch():
            for mesh in (*self.skin,*self.solids,*self.regions):mesh.refit()
            wp.launch(_gap,(self.batch,len(self.gap_source),5),inputs=[
                wp.from_torch(wp.to_torch(self.vertices).reshape(self.batch,self.nv,3),dtype=wp.vec3),
                self.gap_source,self.gap_bc,self.gap_owner,self.region_ids,self.gaps],device=self.device)
            wp.launch(_depth,(self.batch,len(self.source),5),inputs=[self.points,self.wp_source,self.owner,
                self.skin_ids,self.solid_ids,self.part_faces,self.depth],device=self.device)
        current=torch.cuda.current_stream(self.device);self.query_torch_stream.wait_stream(current)
        with wp.ScopedStream(self.query_stream):
            if not self.graph_attempted:
                dispatch();self.graph_attempted=True
                try:
                    with wp.ScopedCapture(stream=self.query_stream) as captured:dispatch()
                    self.graph=captured.graph
                except RuntimeError as exc:self.graph_error=str(exc)
            if self.graph is not None:wp.capture_launch(self.graph)
            else:dispatch()
        current.wait_stream(self.query_torch_stream)
        gap=wp.to_torch(self.gaps)[:count];depth=wp.to_torch(self.depth)[:count]
        a=self.pair_a;b=self.pair_b
        distances=torch.minimum(gap[:,a,b],gap[:,b,a])
        if (distances>=1.e11).any():raise ValueError('批量表面接触查询失败')
        return distances.clamp_min(0).sqrt()*1000,torch.maximum(depth[:,a,b],depth[:,b,a])


def contact_hysteresis(gaps,depths,stamps):
    """Temporal semantics stay sequential; geometry and transfers are batched."""
    active=np.zeros(10,bool)
    result=np.zeros(np.asarray(gaps).shape,bool)
    for i,(gap,depth) in enumerate(zip(gaps,depths)):
        if i==0 or stamps[i]-stamps[i-1]>250 or stamps[i]<=stamps[i-1]:active[:]=False
        active=np.where(active,(gap<=4.)|(depth>.1),(gap<=2.)|(depth>.1))
        result[i]=active
    return result

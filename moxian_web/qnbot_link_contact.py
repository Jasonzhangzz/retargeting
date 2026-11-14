"""Moving closest witnesses on ORIGINAL link-region triangles.

Topology/region membership is fixed. Closest witnesses are refreshed on the
current posed surface, then differentiated through their barycentric points
and smooth surface normals. Includes edge/triangle intersection witnesses;
neither link origins nor old fixed material anchors define contact distance.
"""
import numpy as np
import torch
import warp as wp
from .warp_runtime import serialized
from .mano_contact_loss import vertex_normals


@wp.func
def _segments(p0:wp.vec3,p1:wp.vec3,q0:wp.vec3,q1:wp.vec3):
    a=p1-p0;b=q1-q0;r=p0-q0;aa=wp.dot(a,a);bb=wp.dot(b,b);ab=wp.dot(a,b);ar=wp.dot(a,r);br=wp.dot(b,r)
    s=float(0.);t=float(0.)
    if aa>1.e-20 and bb>1.e-20:
        denominator=aa*bb-ab*ab
        if denominator>1.e-20:s=wp.clamp((ab*br-ar*bb)/denominator,0.,1.)
        t=(ab*s+br)/bb
        if t<0.:t=0.;s=wp.clamp(-ar/aa,0.,1.)
        elif t>1.:t=1.;s=wp.clamp((ab-ar)/aa,0.,1.)
    elif aa>1.e-20:s=wp.clamp(-ar/aa,0.,1.)
    elif bb>1.e-20:t=wp.clamp(br/bb,0.,1.)
    delta=r+s*a-t*b
    return wp.vec3(s,t,wp.dot(delta,delta))


@wp.kernel(enable_backward=False)
def _closest(vertices:wp.array(dtype=wp.vec3),source:wp.array2d(dtype=wp.vec3i),
             modes:wp.array2d(dtype=int),meshes:wp.array2d(dtype=wp.uint64),active:wp.array(dtype=int),
             distance:wp.array2d(dtype=float),source_bc:wp.array2d(dtype=wp.vec3),
             target:wp.array2d(dtype=wp.vec3i),target_bc:wp.array2d(dtype=wp.vec3)):
    e,s=wp.tid();distance[e,s]=1.e12;target[e,s]=wp.vec3i(0);target_bc[e,s]=wp.vec3(0.);source_bc[e,s]=wp.vec3(0.)
    mode=modes[e,s]
    if mode>=0 and active[e]!=0:
        tri=source[e,s];a=vertices[tri[0]];b=vertices[tri[1]];c=vertices[tri[2]];bc=wp.vec3(0.)
        if mode<3:bc[mode]=1.
        elif mode==3:bc=wp.vec3(1./3.)
        mesh=meshes[e,s];face=int(-1);u=float(0.);v=float(0.);d=float(1.e12)
        if mode<4:
            point=bc[0]*a+bc[1]*b+bc[2]*c;q=wp.mesh_query_point_no_sign(mesh,point,2.)
            if q.result:
                face=q.face;u=q.u;v=q.v;delta=point-wp.mesh_eval_position(mesh,face,u,v);d=wp.dot(delta,delta)
        else:
            i=mode-4;j=(i+1)%3;start=vertices[tri[i]];edge=vertices[tri[j]]-start;length=wp.length(edge)
            if length>1.e-10:
                hit=wp.mesh_query_ray(mesh,start,edge/length,length)
                if hit.result:
                    face=hit.face;u=hit.u;v=hit.v;d=0.;bc[i]=1.-hit.t/length;bc[j]=hit.t/length
                else:
                    midpoint=start+.5*edge;nearest=wp.mesh_query_point_no_sign(mesh,midpoint,2.)
                    if nearest.result:
                        face=nearest.face;u=nearest.u;v=nearest.v;bc[i]=.5;bc[j]=.5
                        delta=midpoint-wp.mesh_eval_position(mesh,face,u,v);d=wp.dot(delta,delta)
                        radius=wp.sqrt(d)+1.e-7;end=start+edge
                        query=wp.mesh_query_aabb(mesh,wp.min(start,end)-wp.vec3(radius),wp.max(start,end)+wp.vec3(radius));candidate=int(0)
                        while wp.mesh_query_aabb_next(query,candidate):
                            for k in range(3):
                                kk=(k+1)%3;q0=vertices[wp.mesh_get_index(mesh,3*candidate+k)];q1=vertices[wp.mesh_get_index(mesh,3*candidate+kk)]
                                result=_segments(start,end,q0,q1)
                                if result[2]<d:
                                    d=result[2];face=candidate;bc=wp.vec3(0.);bc[i]=1.-result[0];bc[j]=result[0]
                                    target_bary=wp.vec3(0.);target_bary[k]=1.-result[1];target_bary[kk]=result[1];u=target_bary[0];v=target_bary[1]
        if face>=0:
            distance[e,s]=d;source_bc[e,s]=bc
            target[e,s]=wp.vec3i(wp.mesh_get_index(mesh,3*face),wp.mesh_get_index(mesh,3*face+1),wp.mesh_get_index(mesh,3*face+2))
            target_bc[e,s]=wp.vec3(u,v,1.-u-v)


def disjoint_regions(faces,regions,a,b):
    aa=np.asarray(regions[a],int);bb=np.asarray(regions[b],int)
    shared=np.intersect1d(faces[aa].ravel(),faces[bb].ravel())
    if len(shared):
        aa=aa[~np.isin(faces[aa],shared).any(1)];bb=bb[~np.isin(faces[bb],shared).any(1)]
    if not len(aa) or not len(bb):raise ValueError('接触区域没有独立材料核心：'+a+'/'+b)
    return aa,bb


class LinkContactQuery:
    @serialized
    def __init__(self,vertices,faces,regions,edges):
        self.device=str(vertices.device);self.shape=tuple(vertices.shape);self.edges=edges;self.meshes=[];self.mesh_indices=[]
        self.faces=faces;nv=vertices.shape[1];f=faces.detach().cpu().numpy();self.vertex=wp.array(vertices.detach().cpu().numpy().reshape(-1,3),dtype=wp.vec3,device=self.device)
        configs=[];mesh_cache={}
        for edge in edges:
            row=edge['row'];aa,bb=disjoint_regions(f,regions,*edge['regions']);items=[]
            for source_ids,target_ids,reverse in ((aa,bb,False),(bb,aa,True)):
                key=(row,tuple(target_ids))
                if key not in mesh_cache:
                    indices=wp.array((f[target_ids]+row*nv).ravel().astype(np.int32),dtype=int,device=self.device);self.mesh_indices.append(indices)
                    mesh=wp.Mesh(points=self.vertex,indices=indices);self.meshes.append(mesh);mesh_cache[key]=mesh.id
                tris=np.repeat(f[source_ids]+row*nv,7,axis=0);modes=np.tile(np.arange(7),len(source_ids))
                items.append((tris,modes,np.full(len(tris),mesh_cache[key],np.uint64),np.full(len(tris),reverse,bool)))
            configs.append(tuple(np.concatenate([x[k] for x in items]) for k in range(4)))
        count=len(edges);self.samples=max(len(c[0]) for c in configs)
        src=np.zeros((count,self.samples,3),np.int32);modes=np.full((count,self.samples),-1,np.int32);meshes=np.zeros((count,self.samples),np.uint64);reverse=np.zeros((count,self.samples),bool)
        for i,c in enumerate(configs):
            n=len(c[0]);src[i,:n]=c[0];modes[i,:n]=c[1];meshes[i,:n]=c[2];reverse[i,:n]=c[3]
        self.source=wp.array(src,dtype=wp.vec3i,device=self.device);self.modes=wp.array(modes,dtype=int,device=self.device);self.mesh_ids=wp.array(meshes,dtype=wp.uint64,device=self.device)
        self.source_t=torch.tensor(src,device=vertices.device,dtype=torch.long);self.reverse=torch.tensor(reverse,device=vertices.device)
        self.distance=wp.zeros((count,self.samples),dtype=float,device=self.device);self.source_bc=wp.zeros((count,self.samples),dtype=wp.vec3,device=self.device)
        self.active=wp.ones(count,dtype=int,device=self.device)
        self.target=wp.zeros((count,self.samples),dtype=wp.vec3i,device=self.device);self.target_bc=wp.zeros((count,self.samples),dtype=wp.vec3,device=self.device)
        self.graph=None;self.attempted=False;self.graph_error=None
        if vertices.is_cuda:self.stream=torch.cuda.Stream(device=vertices.device);self.wp_stream=wp.stream_from_torch(self.stream)

    @torch.no_grad()
    @serialized
    def witnesses(self,vertices,active=None,normals=None):
        if tuple(vertices.shape)!=self.shape:raise ValueError('link接触批次拓扑改变')
        wp.to_torch(self.vertex).copy_(vertices.detach().reshape(-1,3))
        selected=torch.ones(len(self.edges),device=vertices.device,dtype=torch.bool) if active is None else active
        wp.to_torch(self.active).copy_(selected.to(torch.int32))
        def run():
            for mesh in self.meshes:mesh.refit()
            wp.launch(_closest,(len(self.edges),self.samples),inputs=[self.vertex,self.source,self.modes,self.mesh_ids,self.active,self.distance,self.source_bc,self.target,self.target_bc],device=self.device)
        if vertices.is_cuda:
            current=torch.cuda.current_stream(vertices.device);self.stream.wait_stream(current)
            with wp.ScopedStream(self.wp_stream):
                if not self.attempted:
                    run();self.attempted=True
                    try:
                        with wp.ScopedCapture(stream=self.wp_stream) as captured:run()
                        self.graph=captured.graph
                    except RuntimeError as exc:self.graph_error=str(exc)
                if self.graph is not None:wp.capture_launch(self.graph)
                else:run()
            current.wait_stream(self.stream)
        else:run()
        distances=wp.to_torch(self.distance);minimum=distances.min(1).values
        if normals is None:score=distances
        else:
            # Distance alone arbitrarily picks one of many intersecting points.
            # Choose a SAME-point pair jointly by distance AND facing normals,
            # with the exact weights/scales used by the differentiable loss.
            flat_normals=normals.detach().reshape(-1,3);candidate_target=wp.to_torch(self.target).long()
            ns=torch.nn.functional.normalize((flat_normals[self.source_t]*wp.to_torch(self.source_bc)[...,None]).sum(-2),dim=-1)
            nt=torch.nn.functional.normalize((flat_normals[candidate_target]*wp.to_torch(self.target_bc)[...,None]).sum(-2),dim=-1)
            score=320*distances/(3*.002**2)+40*(ns+nt).square().mean(-1)/(2*np.sin(np.deg2rad(15)/2))**2
        index=score.argmin(1);row=torch.arange(len(self.edges),device=vertices.device);values=distances[row,index]
        if bool(((values>=1.e11)&selected).any()):raise ValueError('指定link区域接触查询失败')
        source=self.source_t[row,index];target=wp.to_torch(self.target)[row,index].long();sbc=wp.to_torch(self.source_bc)[row,index].clone();tbc=wp.to_torch(self.target_bc)[row,index].clone();reverse=self.reverse[row,index]
        return dict(triangles=torch.stack((torch.where(reverse[:,None],target,source),torch.where(reverse[:,None],source,target)),1),
                    barycentric=torch.stack((torch.where(reverse[:,None],tbc,sbc),torch.where(reverse[:,None],sbc,tbc)),1),query_squared_distance=values.clone(),minimum_squared_distance=minimum.clone())

    def evaluate(self,vertices,active=None):
        normals=vertex_normals(vertices,self.faces)
        witness=self.witnesses(vertices,active,normals);tri=witness['triangles'];bc=witness['barycentric'];flat=vertices.reshape(-1,3)
        point=(flat[tri]*bc[...,None]).sum(-2)
        normals=normals.reshape(-1,3);normal=torch.nn.functional.normalize((normals[tri]*bc[...,None]).sum(-2),dim=-1)
        return point,normal,witness

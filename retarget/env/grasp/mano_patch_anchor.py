"""Fixed Shadow-derived object anchors and anatomical MANO patch witnesses.

Asset topology is prepared on CPU. FK, triangle proximity witnesses, costs and
acceptance run on CUDA. A witness is a pair of actual triangle-surface points;
convex collision proxies and a vertex-only test cannot certify patch contact.
"""
from dataclasses import dataclass, replace
import heapq
import math
import numpy as np
import torch
from retarget.solver.cem import DEFAULTS
import warp as wp

from .mano_comfree import ManoRolloutGeometry
from .contact_geometry import penetration_terms


from retarget.solver.cem import PatchSettings as PatchAnchorSettings


def patch_faces(decoder, records, radius):
    """Connected face-centroid geodesic disk within the original segment mask."""
    vertices=decoder.rest.detach().cpu().numpy()
    faces=np.asarray(decoder.base.faces)
    weights=decoder.base.weights.detach().cpu().numpy()
    centers=vertices[faces].mean(1)
    region=weights[faces].mean(1).argmax(-1)
    adjacent=[set() for _ in faces];edges={}
    for i, face in enumerate(faces):
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))))
            for other in edges.setdefault(key,[]):
                adjacent[i].add(other);adjacent[other].add(i)
            edges[key].append(i)
    chains={'thumb':(13,14,15),'index':(1,2,3),'middle':(4,5,6),
            'ring':(10,11,12),'pinky':(7,8,9)}
    result=[]
    for rec in records:
        if rec['finger']=='palm':allowed=weights[:,0]>.35
        else:
            allowed=(weights[:,rec['mano_joint_id']]>.25)&(weights[:,chains[rec['finger']]].sum(-1)>.65)
        permitted=allowed[faces].sum(-1)>=2
        seed=int(rec['mano_face_id'])
        if not permitted[seed]:raise ValueError('Mapping face outside original anatomical region')
        permitted &= region==region[seed]
        anchor=np.asarray(rec['mano_barycentric'])@vertices[faces[seed]]
        distance={seed:0.};queue=[(0.,seed)]
        while queue:
            d,i=heapq.heappop(queue)
            if d!=distance[i]:continue
            for j in adjacent[i]:
                if not permitted[j]:continue
                nd=d+float(np.linalg.norm(centers[i]-centers[j]))
                # The complete seed triangle is retained. Other triangles must
                # also have their centroid within the stated neutral disk.
                if nd<=radius and np.linalg.norm(centers[j]-anchor)<=radius and nd<distance.get(j,float('inf')):
                    distance[j]=nd;heapq.heappush(queue,(nd,j))
        result.append(sorted(distance))
    return result


def clip_patch_triangles(vertices,faces,records,regions,radius):
    """Clip skin triangles to a neutral 10mm ball using an inscribed 64-gon.

    All emitted vertices stay in the original triangle AND inside the ball.
    Interior barycentric coordinates are fixed asset data and deform with MANO.
    The curved boundary is conservatively approximated (radial deficit <0.013mm
    at radius 10mm); this cannot certify a point outside the requested patch.
    """
    all_ids=[];all_bary=[]
    circle=np.stack((np.cos(np.arange(64)*2*np.pi/64),np.sin(np.arange(64)*2*np.pi/64)),-1)
    for rec,ids in zip(records,regions):
        anchor=np.asarray(rec['mano_barycentric'])@vertices[faces[rec['mano_face_id']]]
        output_ids=[];output_bary=[]
        for face in ids:
            tri=vertices[faces[face]].astype(float)
            u=tri[1]-tri[0];u/=np.linalg.norm(u)
            normal=np.cross(u,tri[2]-tri[0]);normal/=np.linalg.norm(normal)
            v=np.cross(normal,u);height=np.dot(anchor-tri[0],normal)
            if abs(height)>=radius:continue
            center=anchor-height*normal
            xy=np.stack(((tri-center)@u,(tri-center)@v),-1)
            ring=circle*np.sqrt(radius**2-height**2)
            polygon=list(xy)
            for start,end in zip(ring,np.roll(ring,-1,axis=0)):
                if not polygon:break
                edge=end-start
                side=lambda p:edge[0]*(p[1]-start[1])-edge[1]*(p[0]-start[0])
                clipped=[]
                for a,b in zip(polygon,[*polygon[1:],polygon[0]]):
                    sa,sb=side(a),side(b);ina,inb=sa>=-1.e-16,sb>=-1.e-16
                    if ina:clipped.append(a)
                    if ina!=inb:clipped.append(a+sa/(sa-sb)*(b-a))
                polygon=clipped
            if len(polygon)<3:continue
            xy=np.asarray(polygon);points=center+xy[:,0,None]*u+xy[:,1,None]*v
            basis=np.stack((tri[1]-tri[0],tri[2]-tri[0]),-1)
            coef=np.linalg.lstsq(basis,(points-tri[0]).T,rcond=None)[0].T
            bary=np.column_stack((1-coef.sum(-1),coef))
            for j in range(1,len(bary)-1):
                selected=bary[[0,j,j+1]]
                if np.linalg.norm(np.cross(points[j]-points[0],points[j+1]-points[0]))<1.e-14:continue
                output_ids.append(face);output_bary.append(selected)
        if not output_ids:raise ValueError('Empty clipped anatomical patch')
        all_ids.append(output_ids);all_bary.append(output_bary)
    return all_ids,all_bary


@wp.func
def segment_point(p:wp.vec3,a:wp.vec3,b:wp.vec3):
    e=b-a
    t=wp.clamp(wp.dot(p-a,e)/wp.max(wp.dot(e,e),1.e-20),0.,1.)
    return a+t*e


@wp.func
def triangle_point(p:wp.vec3,a:wp.vec3,b:wp.vec3,c:wp.vec3):
    ab=b-a;ac=c-a;n=wp.cross(ab,ac);n2=wp.dot(n,n)
    q=p-wp.dot(p-a,n)/wp.max(n2,1.e-20)*n
    d00=wp.dot(ab,ab);d01=wp.dot(ab,ac);d11=wp.dot(ac,ac)
    v=(d11*wp.dot(q-a,ab)-d01*wp.dot(q-a,ac))/wp.max(n2,1.e-20)
    w=(d00*wp.dot(q-a,ac)-d01*wp.dot(q-a,ab))/wp.max(n2,1.e-20)
    if n2>1.e-20 and v>=0. and w>=0. and v+w<=1.:return q
    x=segment_point(p,a,b);y=segment_point(p,b,c);z=segment_point(p,c,a)
    if wp.length_sq(p-y)<wp.length_sq(p-x):x=y
    if wp.length_sq(p-z)<wp.length_sq(p-x):x=z
    return x


@wp.func
def segment_pair(a:wp.vec3,b:wp.vec3,c:wp.vec3,d:wp.vec3):
    u=b-a;v=d-c;r=a-c
    aa=wp.dot(u,u);ee=wp.dot(v,v);bb=wp.dot(u,v)
    cc=wp.dot(u,r);ff=wp.dot(v,r);den=aa*ee-bb*bb
    s=float(0.);t=float(0.)
    if aa<=1.e-20:
        t=wp.clamp(ff/wp.max(ee,1.e-20),0.,1.)
    elif ee<=1.e-20:s=wp.clamp(-cc/aa,0.,1.)
    else:
        if den>1.e-20:s=wp.clamp((bb*ff-cc*ee)/den,0.,1.)
        t=(bb*s+ff)/ee
        if t<0.:t=0.;s=wp.clamp(-cc/aa,0.,1.)
        elif t>1.:t=1.;s=wp.clamp((bb-cc)/aa,0.,1.)
    return a+s*u,c+t*v


@wp.func
def segment_triangle(a:wp.vec3,b:wp.vec3,c:wp.vec3,d:wp.vec3,e:wp.vec3):
    # Closed segment/triangle intersection; coplanar cases use edge witnesses.
    direction=b-a;edge1=d-c;edge2=e-c;h=wp.cross(direction,edge2)
    det=wp.dot(edge1,h);hit=bool(False);point=a
    if wp.abs(det)>1.e-14:
        inv=1./det;s=a-c;u=inv*wp.dot(s,h);q=wp.cross(s,edge1)
        v=inv*wp.dot(direction,q);t=inv*wp.dot(edge2,q)
        if u>=-1.e-6 and v>=-1.e-6 and u+v<=1.000001 and t>=0. and t<=1.:
            hit=True;point=a+t*direction
    return hit,point


@wp.func
def witness_score(h:wp.vec3,o:wp.vec3,anchor:wp.vec3,tolerance:float,near:float):
    gap=wp.length(h-o);offset=wp.length(o-anchor)
    # Every passing witness ranks before every failing one. Selection is only
    # a geometry query, never an extra optimization objective.
    value=gap/tolerance+offset/near
    if gap<=tolerance and offset<=near:return offset/near
    return 1000.+value


@wp.kernel(enable_backward=False)
def patch_witness_kernel(meshes:wp.array[wp.uint64],rows:wp.array[int],
        vertices:wp.array2d[wp.vec3],triangles:wp.array3d[int],clip:wp.array2d[wp.mat33],counts:wp.array[int],
        anchors:wp.array2d[wp.vec3],scales:wp.array[float],tolerance_world:float,near_world:float,
        hand:wp.array3d[wp.vec3],obj:wp.array3d[wp.vec3],scores:wp.array3d[float]):
    b,k,t=wp.tid()
    if t>=counts[k]:
        scores[b,k,t]=1.e20
        return
    x0=vertices[b,triangles[k,t,0]];x1=vertices[b,triangles[k,t,1]];x2=vertices[b,triangles[k,t,2]]
    w=clip[k,t]
    a=w[0,0]*x0+w[0,1]*x1+w[0,2]*x2
    c=w[1,0]*x0+w[1,1]*x1+w[1,2]*x2
    d=w[2,0]*x0+w[2,1]*x1+w[2,2]*x2
    mesh=meshes[rows[b]];anchor=anchors[b,k]
    tolerance=tolerance_world/scales[b];near=near_world/scales[b]
    best_h=triangle_point(anchor,a,c,d);best_o=anchor
    hit=wp.mesh_query_point(mesh,best_h,1.e6)
    if hit.result:best_o=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
    best=witness_score(best_h,best_o,anchor,tolerance,near)
    low=wp.min(wp.min(a,c),d)-wp.vec3(tolerance)
    high=wp.max(wp.max(a,c),d)+wp.vec3(tolerance)
    query=wp.mesh_query_aabb(mesh,low,high);face=int(0)
    while wp.mesh_query_aabb_next(query,face):
        x=wp.mesh_get_point(mesh,3*face);y=wp.mesh_get_point(mesh,3*face+1);z=wp.mesh_get_point(mesh,3*face+2)
        ht=wp.matrix_from_rows(a,c,d);ot=wp.matrix_from_rows(x,y,z)
        # Vertices, edges and interior edge/face intersections cover triangle
        # proximity; anchor projections handle parallel/interior closest sets.
        for i in range(4):
            h=a;o=x
            if i<3:
                h=ht[i];o=triangle_point(h,x,y,z)
            else:
                o=triangle_point(anchor,x,y,z);h=triangle_point(o,a,c,d)
            s=witness_score(h,o,anchor,tolerance,near)
            if s<best:best=s;best_h=h;best_o=o
            if i<3:
                o=ot[i];h=triangle_point(o,a,c,d)
                s=witness_score(h,o,anchor,tolerance,near)
                if s<best:best=s;best_h=h;best_o=o
        intersection_a=wp.vec3(0.);intersection_b=wp.vec3(0.);number=int(0)
        for i in range(3):
            for j in range(3):
                h,o=segment_pair(ht[i],ht[(i+1)%3],ot[j],ot[(j+1)%3])
                s=witness_score(h,o,anchor,tolerance,near)
                if s<best:best=s;best_h=h;best_o=o
            ok,p=segment_triangle(ht[i],ht[(i+1)%3],x,y,z)
            if ok:
                if number==0:intersection_a=p
                else:intersection_b=p
                number+=1
                s=witness_score(p,p,anchor,tolerance,near)
                if s<best:best=s;best_h=p;best_o=p
            ok,p=segment_triangle(ot[i],ot[(i+1)%3],a,c,d)
            if ok:
                if number==0:intersection_a=p
                else:intersection_b=p
                number+=1
                s=witness_score(p,p,anchor,tolerance,near)
                if s<best:best=s;best_h=p;best_o=p
        if number>=2:
            p=segment_point(anchor,intersection_a,intersection_b)
            s=witness_score(p,p,anchor,tolerance,near)
            if s<best:best=s;best_h=p;best_o=p
    hand[b,k,t]=best_h;obj[b,k,t]=best_o;scores[b,k,t]=best


class AnatomicalPatches:
    def __init__(self,decoder,records,surfaces,anchors,settings=None,backend='reference'):
        if backend not in ('reference','compact','optimized'):raise ValueError('Unknown patch backend')
        self.backend=backend
        self.pair_capacity=128
        self.settings=settings or PatchAnchorSettings()
        self.surfaces=surfaces;self.anchors=anchors.detach().clone()
        self.face_ids=patch_faces(decoder,records,self.settings.patch_radius_m)
        self.clipped_face_ids,clipped=clip_patch_triangles(decoder.rest.detach().cpu().numpy(),decoder.base.faces,
            records,self.face_ids,self.settings.patch_radius_m)
        width=max(map(len,self.clipped_face_ids));tri=np.zeros((len(records),width,3),np.int32)
        bary=np.zeros((len(records),width,3,3),np.float32)
        for k,ids in enumerate(self.clipped_face_ids):
            tri[k,:len(ids)]=decoder.base.faces[ids];bary[k,:len(ids)]=clipped[k]
        self.triangles=torch.tensor(tri,device=decoder.device,dtype=torch.int32)
        self.clip=torch.tensor(bary,device=decoder.device)
        self.counts=torch.tensor(list(map(len,self.clipped_face_ids)),device=decoder.device,dtype=torch.int32)
        self.face_table=torch.tensor([ids+[ids[0]]*(width-len(ids)) for ids in self.clipped_face_ids],device=decoder.device)
        self.packed_triangles=torch.cat([self.triangles[i,:len(ids)] for i,ids in enumerate(self.clipped_face_ids)])
        self.packed_clip=torch.cat([self.clip[i,:len(ids)] for i,ids in enumerate(self.clipped_face_ids)])
        self.packed_slots=torch.tensor([(i,j) for i,ids in enumerate(self.clipped_face_ids) for j in range(len(ids))],device=decoder.device,dtype=torch.int32)
        from .patch_query import patch_witness_compact
        self.optimized_kernel=patch_witness_compact

    @torch.no_grad()
    def query(self,vertices,rows):
        s=self.surfaces;r=s.rotation[rows];scale=s.scale[rows]
        local=(vertices-s.origin[rows,None])@r/scale[:,None,None]
        anchor=(self.anchors[rows]-s.origin[rows,None])@r/scale[:,None,None]
        b=len(rows);k,t=self.triangles.shape[:2];shape=(b,k,t)
        h=torch.empty((*shape,3),device=vertices.device);o=torch.empty_like(h)
        if self.backend!='reference':
            score=torch.full(shape,1.e20,device=vertices.device)
            kernel=self.optimized_kernel;dims=(b,len(self.packed_slots))
            patch_inputs=[wp.from_torch(self.packed_triangles),wp.from_torch(self.packed_clip,dtype=wp.mat33),wp.from_torch(self.packed_slots,dtype=wp.vec2i)]
        else:
            score=torch.empty(shape,device=vertices.device)
            kernel=patch_witness_kernel;dims=shape
            patch_inputs=[wp.from_torch(self.triangles),wp.from_torch(self.clip,dtype=wp.mat33),wp.from_torch(self.counts)]
        if self.backend=='optimized':
            from .patch_query import parallel_query
            overflow=parallel_query(self,local,anchor,rows,scale,h,o,score,capacity=self.pair_capacity)
        else:
            overflow=torch.zeros(b,device=vertices.device,dtype=torch.long)
            wp.launch(kernel,dim=dims,inputs=[s.mesh_ids,wp.from_torch(rows.to(torch.int32)),
                wp.from_torch(local.contiguous(),dtype=wp.vec3),*patch_inputs,
                wp.from_torch(anchor.contiguous(),dtype=wp.vec3),wp.from_torch(scale.contiguous()),
                self.settings.surface_tolerance_m,self.settings.anchor_tolerance_m],
                outputs=[wp.from_torch(h,dtype=wp.vec3),wp.from_torch(o,dtype=wp.vec3),wp.from_torch(score)],
                device=str(vertices.device),stream=wp.stream_from_torch(torch.cuda.current_stream(vertices.device)))
        index=score.argmin(-1)
        take=lambda x:x.gather(2,index[...,None,None].expand(-1,-1,1,3)).squeeze(2)
        h=take(h)*scale[:,None,None]@r.transpose(-1,-2)+s.origin[rows,None]
        o=take(o)*scale[:,None,None]@r.transpose(-1,-2)+s.origin[rows,None]
        gap=(h-o).norm(dim=-1);offset=(o-self.anchors[rows]).norm(dim=-1)
        fitted=(gap<=self.settings.surface_tolerance_m+1.e-8)&(offset<=self.settings.anchor_tolerance_m+1.e-8)
        fitted &= torch.isfinite(h).all(-1)&torch.isfinite(o).all(-1)
        face=self.face_table.gather(1,index.T).T
        return dict(patch_hand_points=h,patch_object_points=o,patch_gap_m=gap,
            patch_anchor_offset_m=offset,patch_fitted=fitted,patch_face_ids=face,
            patch_coverage=fitted.float().mean(-1),object_template_points=self.anchors[rows],patch_gpu_overflow_count=overflow)


def patch_penetration_terms(depth,settings,patch_settings):
    adjusted=torch.stack((depth[:,0],(depth[:,1]-patch_settings.self_allowance_m).clamp_min(0.)),-1)
    return penetration_terms(adjusted,settings)


class PatchGeometry(ManoRolloutGeometry):
    """Shared patch setup and geometric diagnostics for the current objective."""
    def __init__(self,*args,anchors,patch_settings=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.patch_settings=patch_settings or DEFAULTS.patch
        self.settings=replace(self.settings,hand_object_allowance_m=self.patch_settings.hand_object_allowance_m,
                              penetration_tolerance_m=self.patch_settings.self_allowance_m)
        self.anchors=anchors.detach().clone()
        if anchors.shape!=(len(self.surfaces.object_keys),len(self.fixed.records),3):raise ValueError('Anchor shape mismatch')
        if not torch.isfinite(anchors).all():raise ValueError('Invalid frozen object anchors')
        self.patches=AnatomicalPatches(self.decoder,self.fixed.records,self.surfaces,self.anchors,self.patch_settings,backend=self.geometry_backend)
    @torch.no_grad()
    def evaluate_candidates(self,variables,rows):
        """Store only the cheap stability/depth gate, not all population parts."""
        cost=variables.new_empty(len(variables));gate=torch.empty(len(variables),device=variables.device,dtype=torch.bool)
        device=str(self.decoder.device)
        with wp.ScopedDevice(device),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.decoder.device))):
            for start in range(0,len(variables),self.chunk):
                value=variables[start:start+self.chunk];ids=rows[start:start+self.chunk];n=len(value)
                if n<self.chunk:
                    value=torch.cat((value,value[:1].expand(self.chunk-n,-1)))
                    ids=torch.cat((ids,ids[:1].expand(self.chunk-n)))
                scores,parts=self._score(value,ids)
                good=self._candidate_gate(parts)
                cost[start:start+n]=scores[:n];gate[start:start+n]=good[:n]
        return cost,gate

    def _candidate_gate(self,parts):
        depth=parts['penetration_depth_m']
        return parts['valid']&parts['rollout_stable']&(depth[:,0]<=self.settings.hand_object_allowance_m+1.e-8)&(depth[:,1]<=self.settings.penetration_tolerance_m)

    @torch.no_grad()
    def _compute(self,value,ids,*,decoded=None):
        _,parts=super()._compute(value,ids,decoded=decoded)
        strict,soft,_=patch_penetration_terms(parts['penetration_depth_m'],self.settings,self.patch_settings)
        parts.update(penetration=strict,soft_penetration=soft,
            anchor_distance_m=(parts['mapped_hand_points']-self.anchors[ids]).norm(dim=-1),
            object_template_points=self.anchors[ids])
        return parts['rollout_cost'],parts

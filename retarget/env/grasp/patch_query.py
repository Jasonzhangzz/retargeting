"""Equivalent patch witnesses with unsigned proximity and conservative pruning.

The reference feature tests and strict tie order are retained. Object/hand
triangle lower bounds only discard pairs that cannot improve the current
witness. Patch geometry never uses the mesh inside/outside sign.
"""
import warp as wp
from .mano_patch_anchor import triangle_point,segment_point,segment_pair,segment_triangle,witness_score


@wp.func
def triangle_pair(a:wp.vec3,c:wp.vec3,d:wp.vec3,x:wp.vec3,y:wp.vec3,z:wp.vec3,
                  anchor:wp.vec3,anchor_o:wp.vec3,tolerance:float,near:float,
                  best_h:wp.vec3,best_o:wp.vec3,best:float):
    ht=wp.matrix_from_rows(a,c,d);ot=wp.matrix_from_rows(x,y,z)
    for i in range(4):
        h=a;o=x
        if i<3:
            h=ht[i];o=triangle_point(h,x,y,z)
        else:
            o=anchor_o;h=triangle_point(o,a,c,d)
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
    return best_h,best_o,best


@wp.func
def initial_witness(mesh:wp.uint64,a:wp.vec3,c:wp.vec3,d:wp.vec3,anchor:wp.vec3,tolerance:float,near:float):
    best_h=triangle_point(anchor,a,c,d);best_o=anchor
    # Same closest-point traversal as mesh_query_point, without its unused
    # inside/outside ray tests. This query never supplies contact force signs.
    hit=wp.mesh_query_point_no_sign(mesh,best_h,1.e6)
    if hit.result:best_o=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
    best=witness_score(best_h,best_o,anchor,tolerance,near)
    return best_h,best_o,best


@wp.func
def lower_bound(a:wp.vec3,c:wp.vec3,d:wp.vec3,x:wp.vec3,y:wp.vec3,z:wp.vec3,
                anchor:wp.vec3,tolerance:float,near:float):
    slack=1.e-6*(1.+wp.length(a)+wp.length(c)+wp.length(d)+wp.length(x)+wp.length(y)+wp.length(z))
    lo=wp.min(wp.min(a,c),d);hi=wp.max(wp.max(a,c),d)
    ol=wp.min(wp.min(x,y),z);oh=wp.max(wp.max(x,y),z)
    separation=wp.max(wp.max(lo-oh,ol-hi),wp.vec3(0.))
    gap_lower=wp.max(0.,wp.length(separation)-slack)
    anchor_o=triangle_point(anchor,x,y,z)
    offset_lower=wp.max(0.,wp.length(anchor_o-anchor)-slack)
    lower=1000.+gap_lower/tolerance+offset_lower/near
    if gap_lower<=tolerance and offset_lower<=near:lower=offset_lower/near
    return lower,anchor_o


@wp.func
def search_triangle(mesh:wp.uint64,a:wp.vec3,c:wp.vec3,d:wp.vec3,anchor:wp.vec3,tolerance:float,near:float):
    best_h,best_o,best=initial_witness(mesh,a,c,d,anchor,tolerance,near)
    lo=wp.min(wp.min(a,c),d);hi=wp.max(wp.max(a,c),d)
    query=wp.mesh_query_aabb(mesh,lo-wp.vec3(tolerance),hi+wp.vec3(tolerance));face=int(0)
    while wp.mesh_query_aabb_next(query,face):
        if best==0.:break
        x=wp.mesh_get_point(mesh,3*face);y=wp.mesh_get_point(mesh,3*face+1);z=wp.mesh_get_point(mesh,3*face+2)
        # Slack covers roundoff and the reference segment/triangle barycentric
        # acceptance epsilon. It only weakens pruning, never changes thresholds.
        lower,anchor_o=lower_bound(a,c,d,x,y,z,anchor,tolerance,near)
        if lower>best+1.e-5:continue
        best_h,best_o,best=triangle_pair(a,c,d,x,y,z,anchor,anchor_o,tolerance,near,best_h,best_o,best)
    return best_h,best_o,best


@wp.kernel(enable_backward=False)
def patch_witness_compact(meshes:wp.array[wp.uint64],rows:wp.array[int],vertices:wp.array2d[wp.vec3],
        triangles:wp.array2d[int],clip:wp.array[wp.mat33],slots:wp.array[wp.vec2i],
        anchors:wp.array2d[wp.vec3],scales:wp.array[float],tolerance_world:float,near_world:float,
        hand:wp.array3d[wp.vec3],obj:wp.array3d[wp.vec3],scores:wp.array3d[float]):
    b,i=wp.tid();k=slots[i][0];t=slots[i][1]
    x0=vertices[b,triangles[i,0]];x1=vertices[b,triangles[i,1]];x2=vertices[b,triangles[i,2]]
    w=clip[i]
    a=w[0,0]*x0+w[0,1]*x1+w[0,2]*x2
    c=w[1,0]*x0+w[1,1]*x1+w[1,2]*x2
    d=w[2,0]*x0+w[2,1]*x1+w[2,2]*x2
    best_h,best_o,best=search_triangle(meshes[rows[b]],a,c,d,anchors[b,k],tolerance_world/scales[b],near_world/scales[b])
    hand[b,k,t]=best_h;obj[b,k,t]=best_o;scores[b,k,t]=best


@wp.kernel(enable_backward=False)
def prepare_pairs(meshes:wp.array[wp.uint64],rows:wp.array[int],vertices:wp.array2d[wp.vec3],
        triangles:wp.array2d[int],clip:wp.array[wp.mat33],slots:wp.array[wp.vec2i],
        anchors:wp.array2d[wp.vec3],scales:wp.array[float],tolerance_world:float,near_world:float,
        coords:wp.array2d[wp.mat33],initial_h:wp.array2d[wp.vec3],initial_o:wp.array2d[wp.vec3],initial_s:wp.array2d[float],
        faces:wp.array3d[int],counts:wp.array2d[int],overflow:wp.array2d[bool],capacity:int):
    b,i=wp.tid();k=slots[i][0]
    x0=vertices[b,triangles[i,0]];x1=vertices[b,triangles[i,1]];x2=vertices[b,triangles[i,2]];w=clip[i]
    a=w[0,0]*x0+w[0,1]*x1+w[0,2]*x2;c=w[1,0]*x0+w[1,1]*x1+w[1,2]*x2;d=w[2,0]*x0+w[2,1]*x1+w[2,2]*x2
    coords[b,i]=wp.matrix_from_rows(a,c,d)
    mesh=meshes[rows[b]];anchor=anchors[b,k];tolerance=tolerance_world/scales[b];near=near_world/scales[b]
    h,o,score=initial_witness(mesh,a,c,d,anchor,tolerance,near)
    initial_h[b,i]=h;initial_o[b,i]=o;initial_s[b,i]=score
    number=int(0);full=bool(False)
    query=wp.mesh_query_aabb(mesh,wp.min(wp.min(a,c),d)-wp.vec3(tolerance),wp.max(wp.max(a,c),d)+wp.vec3(tolerance));face=int(0)
    while wp.mesh_query_aabb_next(query,face):
        if score==0.:break
        x=wp.mesh_get_point(mesh,3*face);y=wp.mesh_get_point(mesh,3*face+1);z=wp.mesh_get_point(mesh,3*face+2)
        lower,unused_anchor_o=lower_bound(a,c,d,x,y,z,anchor,tolerance,near)
        if lower>score+1.e-5:continue
        if number>=capacity:
            full=True
            break
        faces[b,i,number]=face;number+=1
    counts[b,i]=number;overflow[b,i]=full


@wp.kernel(enable_backward=False)
def evaluate_pairs(meshes:wp.array[wp.uint64],rows:wp.array[int],slots:wp.array[wp.vec2i],
        anchors:wp.array2d[wp.vec3],scales:wp.array[float],tolerance_world:float,near_world:float,
        coords:wp.array2d[wp.mat33],initial_h:wp.array2d[wp.vec3],initial_o:wp.array2d[wp.vec3],initial_s:wp.array2d[float],
        faces:wp.array3d[int],counts:wp.array2d[int],overflow:wp.array2d[bool],values:wp.array3d[float],
        pair_h:wp.array3d[wp.vec3],pair_o:wp.array3d[wp.vec3]):
    b,i,j=wp.tid()
    if j>=counts[b,i] or overflow[b,i]:return
    tri=coords[b,i];a=tri[0];c=tri[1];d=tri[2]
    mesh=meshes[rows[b]];face=faces[b,i,j];anchor=anchors[b,slots[i][0]]
    tolerance=tolerance_world/scales[b];near=near_world/scales[b]
    x=wp.mesh_get_point(mesh,3*face);y=wp.mesh_get_point(mesh,3*face+1);z=wp.mesh_get_point(mesh,3*face+2)
    anchor_o=triangle_point(anchor,x,y,z)
    h,o,score=triangle_pair(a,c,d,x,y,z,anchor,anchor_o,tolerance,near,initial_h[b,i],initial_o[b,i],initial_s[b,i])
    values[b,i,j]=score;pair_h[b,i,j]=h;pair_o[b,i,j]=o


@wp.kernel(enable_backward=False)
def finish_pairs(meshes:wp.array[wp.uint64],rows:wp.array[int],slots:wp.array[wp.vec2i],
        anchors:wp.array2d[wp.vec3],scales:wp.array[float],tolerance_world:float,near_world:float,
        coords:wp.array2d[wp.mat33],initial_h:wp.array2d[wp.vec3],initial_o:wp.array2d[wp.vec3],initial_s:wp.array2d[float],
        faces:wp.array3d[int],counts:wp.array2d[int],overflow:wp.array2d[bool],values:wp.array3d[float],
        pair_h:wp.array3d[wp.vec3],pair_o:wp.array3d[wp.vec3],
        hand:wp.array3d[wp.vec3],obj:wp.array3d[wp.vec3],scores:wp.array3d[float]):
    b,i=wp.tid();k=slots[i][0];t=slots[i][1]
    if overflow[b,i]:return
    h=initial_h[b,i];o=initial_o[b,i];score=initial_s[b,i]
    winner=int(-1)
    for j in range(counts[b,i]):
        candidate=values[b,i,j]
        if candidate<score:score=candidate;winner=j
    if winner>=0:
        h=pair_h[b,i,winner];o=pair_o[b,i,winner]
    hand[b,k,t]=h;obj[b,k,t]=o;scores[b,k,t]=score


@wp.kernel(enable_backward=False)
def fallback_pairs(meshes:wp.array[wp.uint64],rows:wp.array[int],slots:wp.array[wp.vec2i],
        anchors:wp.array2d[wp.vec3],scales:wp.array[float],tolerance_world:float,near_world:float,
        coords:wp.array2d[wp.mat33],overflow:wp.array2d[bool],
        hand:wp.array3d[wp.vec3],obj:wp.array3d[wp.vec3],scores:wp.array3d[float]):
    b,i=wp.tid()
    if not overflow[b,i]:return
    k=slots[i][0];t=slots[i][1];tri=coords[b,i]
    # Exact GPU fallback is separate so BVH traversal and its stack do not
    # inflate register/local-memory use in the normal reduction kernel.
    h,o,score=search_triangle(meshes[rows[b]],tri[0],tri[1],tri[2],anchors[b,k],tolerance_world/scales[b],near_world/scales[b])
    hand[b,k,t]=h;obj[b,k,t]=o;scores[b,k,t]=score


def parallel_query(patches,local,anchor,rows,scale,hand,obj,scores,capacity=32):
    """Parallel narrow phase with stable reference ordering and GPU overflow."""
    import torch
    b=local.shape[0];t=len(patches.packed_slots);device=local.device
    coords=torch.empty((b,t,3,3),device=device);ih=torch.empty((b,t,3),device=device);io=torch.empty_like(ih)
    initial=torch.empty((b,t),device=device);faces=torch.empty((b,t,capacity),device=device,dtype=torch.int32)
    counts=torch.empty((b,t),device=device,dtype=torch.int32);overflow=torch.empty((b,t),device=device,dtype=torch.bool)
    values=torch.empty((b,t,capacity),device=device)
    pair_h=torch.empty((b,t,capacity,3),device=device);pair_o=torch.empty_like(pair_h)
    common=[patches.surfaces.mesh_ids,wp.from_torch(rows.to(torch.int32)),wp.from_torch(patches.packed_slots,dtype=wp.vec2i),
        wp.from_torch(anchor.contiguous(),dtype=wp.vec3),wp.from_torch(scale.contiguous()),patches.settings.surface_tolerance_m,patches.settings.anchor_tolerance_m]
    scratch=[wp.from_torch(coords,dtype=wp.mat33),wp.from_torch(ih,dtype=wp.vec3),wp.from_torch(io,dtype=wp.vec3),wp.from_torch(initial),
        wp.from_torch(faces),wp.from_torch(counts),wp.from_torch(overflow)]
    stream=wp.stream_from_torch(torch.cuda.current_stream(device))
    wp.launch(prepare_pairs,dim=(b,t),inputs=[*common[:2],wp.from_torch(local.contiguous(),dtype=wp.vec3),
        wp.from_torch(patches.packed_triangles),wp.from_torch(patches.packed_clip,dtype=wp.mat33),*common[2:],*scratch,capacity],device=str(device),stream=stream)
    pair_result=[wp.from_torch(values),wp.from_torch(pair_h,dtype=wp.vec3),wp.from_torch(pair_o,dtype=wp.vec3)]
    output=[wp.from_torch(hand,dtype=wp.vec3),wp.from_torch(obj,dtype=wp.vec3),wp.from_torch(scores)]
    wp.launch(evaluate_pairs,dim=(b,t,capacity),inputs=[*common,*scratch,*pair_result],device=str(device),stream=stream)
    wp.launch(finish_pairs,dim=(b,t),inputs=[*common,*scratch,*pair_result,*output],device=str(device),stream=stream)
    wp.launch(fallback_pairs,dim=(b,t),inputs=[*common,wp.from_torch(coords,dtype=wp.mat33),wp.from_torch(overflow),*output],device=str(device),stream=stream)
    return overflow.sum(-1)

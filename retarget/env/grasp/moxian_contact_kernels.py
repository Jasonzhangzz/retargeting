"""Fused nearest-triangle witnesses and conservative clearance acceleration."""
import warp as wp
from .mano_patch_anchor import triangle_point


@wp.kernel(enable_backward=False)
def witness_cost(tri:wp.array3d[wp.vec3],normals:wp.array2d[wp.vec3],
                 anchors:wp.array2d[wp.vec3],object_normals:wp.array2d[wp.vec3],
                 longitudinal:wp.array2d[float],source_u:wp.array2d[float],
                 material:wp.array2d[wp.vec3],source_material:wp.array2d[wp.vec3],
                 samples:int,normal_length:float,cost:wp.array3d[float],eligible:wp.array3d[bool]):
    b,k=wp.tid();region=k//samples;i=k%samples
    anchor=anchors[b,region]
    p=triangle_point(anchor,tri[b,k,0],tri[b,k,1],tri[b,k,2])
    md=wp.dot(material[region,i],source_material[b,region])
    difference=longitudinal[region,i]-source_u[b,region]
    cost[b,region,i]=wp.length_sq(p-anchor)+normal_length*normal_length*wp.max(0.,1.+wp.dot(normals[b,k],object_normals[b,region]))+.0001*difference*difference+.000064*(1.-md)
    eligible[b,region,i]=md>=.5


@wp.kernel(enable_backward=False)
def clearance_query(meshes:wp.array[wp.uint64],points:wp.array2d[wp.vec3],
                    low:wp.array[wp.vec3],high:wp.array[wp.vec3],margin:wp.array[float],
                    normal:wp.array2d[wp.vec3],gap:wp.array2d[float]):
    b,k=wp.tid();p=points[b,k]
    lo=low[b]-wp.vec3(margin[b]);hi=high[b]+wp.vec3(margin[b])
    if p[0]<lo[0] or p[0]>hi[0] or p[1]<lo[1] or p[1]>hi[1] or p[2]<lo[2] or p[2]>hi[2]:
        normal[b,k]=wp.vec3(1.,0.,0.);gap[b,k]=1.e3
        return
    hit=wp.mesh_query_point_sign_normal(meshes[b],p,1.e6)
    if hit.result:
        q=wp.mesh_eval_position(meshes[b],hit.face,hit.u,hit.v)
        distance=wp.length(p-q)
        n=wp.vec3(1.,0.,0.)
        if distance>1.e-7:n=hit.sign*(p-q)/distance
        normal[b,k]=n;gap[b,k]=hit.sign*distance
    else:
        normal[b,k]=wp.vec3(1.,0.,0.);gap[b,k]=1.e6

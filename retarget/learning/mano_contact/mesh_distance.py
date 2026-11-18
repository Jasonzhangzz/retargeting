"""Exact closest triangle queries, CPU BVH, with mesh winding sign."""
import warp as wp


@wp.kernel
def triangle_distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), distance: wp.array(dtype=wp.float32)):
    i=wp.tid()
    point=points[i]
    hit=wp.mesh_query_point(mesh,point,1.0e6)
    value=float(1.0e6)
    if hit.result:
        closest=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
        value=wp.length(point-closest)*hit.sign
    distance[i]=value


@wp.kernel
def bounded_triangle_distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), radius: wp.array(dtype=wp.float32), distance: wp.array(dtype=wp.float32)):
    i=wp.tid()
    point=points[i]
    hit=wp.mesh_query_point_no_sign(mesh,point,radius[i]*1.00001)
    value=radius[i]*2.0
    if hit.result:
        closest=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
        value=wp.length(point-closest)
    distance[i]=value

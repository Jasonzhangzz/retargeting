"""Native mesh witness kernel for independent collision labels and audit."""
import warp as wp

@wp.kernel(enable_backward=False)
def _signed_witness(meshes:wp.array(dtype=wp.uint64),points:wp.array(dtype=wp.vec3),owners:wp.array(dtype=int),
    allowed:wp.array2d(dtype=int),collar:wp.array(dtype=float),
    positions:wp.array2d(dtype=wp.vec3),rotations:wp.array2d(dtype=wp.mat33),
    hits:wp.array3d(dtype=wp.vec3),normals:wp.array3d(dtype=wp.vec3),valid:wp.array3d(dtype=int)):
    b,s,t=wp.tid();o=owners[s]
    if allowed[o,t]!=0:
        world=positions[b,o]+rotations[b,o]*points[s]
        local=wp.transpose(rotations[b,t])*(world-positions[b,t])
        if not (o==0 and collar[t]>0. and wp.length(local)<=collar[t]) and not (t==0 and collar[o]>0. and wp.length(points[s])<=collar[o]):
            q=wp.mesh_query_point_sign_winding_number(meshes[t],local,1.)
            if q.result:
                hit=wp.mesh_eval_position(meshes[t],q.face,q.u,q.v)
                d=local-hit;n=wp.mesh_eval_face_normal(meshes[t],q.face)
                if wp.length(d)>1.e-8:n=q.sign*wp.normalize(d)
                hits[b,s,t]=hit;normals[b,s,t]=n;valid[b,s,t]=1

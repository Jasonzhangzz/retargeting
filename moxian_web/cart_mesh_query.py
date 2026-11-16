"""Fast signed distances to a complete closed source mesh (CPU BVH)."""
import numpy as np
import warp as wp

@wp.kernel(enable_backward=False)
def _distance(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),result:wp.array(dtype=float)):
    i=wp.tid();p=points[i]
    q=wp.mesh_query_point_sign_winding_number(mesh,p,2.)
    if q.result:
        h=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        result[i]=q.sign*wp.length(p-h)
    else:result[i]=-1.

class SignedMesh:
    def __init__(self,vertices,faces):
        # Preview stdout is a JSONL transport, so Warp diagnostics go to stderr.
        import contextlib,sys
        with contextlib.redirect_stdout(sys.stderr):
            wp.init()
            self.mesh=wp.Mesh(points=wp.array(np.asarray(vertices,np.float32),dtype=wp.vec3,device='cpu'),
                indices=wp.array(np.asarray(faces,np.int32).ravel(),dtype=int,device='cpu'),support_winding_number=True)
    def distance(self,points):
        import contextlib,sys
        p=wp.array(np.asarray(points,np.float32),dtype=wp.vec3,device='cpu');out=wp.empty(len(p),dtype=float,device='cpu')
        with contextlib.redirect_stdout(sys.stderr):wp.launch(_distance,len(p),inputs=[self.mesh.id,p,out],device='cpu')
        return out.numpy().astype(float)

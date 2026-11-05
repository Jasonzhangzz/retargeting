"""GPU depth-only signed mesh queries with a conservative object-box rejection."""
import numpy as np
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def depth_kernel(meshes:wp.array[wp.uint64],rows:wp.array[int],points:wp.array2d[wp.vec3],
                 low:wp.array[wp.vec3],high:wp.array[wp.vec3],scales:wp.array[float],
                 depth:wp.array[float],invalid:wp.array[int]):
    b,k=wp.tid();row=rows[b];p=points[b,k]
    if not wp.isfinite(p):
        wp.atomic_max(invalid,b,1)
        return
    lo=low[row];hi=high[row]
    slack=1.e-6*(1.+wp.length(lo)+wp.length(hi))
    outside=p[0]<lo[0]-slack or p[0]>hi[0]+slack or p[1]<lo[1]-slack or p[1]>hi[1]+slack or p[2]<lo[2]-slack or p[2]>hi[2]+slack
    # A point beyond the object AABB has zero penetration. Keep the original
    # failure semantics for huge coordinates beyond mesh_query_point's range.
    if outside and wp.length(p-.5*(lo+hi))+.5*wp.length(hi-lo)<999999.:
        return
    hit=wp.mesh_query_point(meshes[row],p,1.e6)
    if not hit.result:
        wp.atomic_max(invalid,b,1)
        return
    q=wp.mesh_eval_position(meshes[row],hit.face,hit.u,hit.v)
    distance=hit.sign*wp.length(p-q)*scales[row]
    if not wp.isfinite(distance) or not wp.isfinite(q):
        wp.atomic_max(invalid,b,1)
    elif distance<0.:
        wp.atomic_max(depth,b,-distance)


class TrianglePenetration:
    def __init__(self,surfaces):
        self.surfaces=surfaces
        # Static asset preparation only; no candidate pose is transferred.
        boxes={i:(np.asarray(g['vertices'],np.float32).min(0),np.asarray(g['vertices'],np.float32).max(0))
               for i,g in surfaces.geometries.items()}
        self.low=torch.tensor(np.stack([boxes[i][0] for i in surfaces.geometry_indices]),device=surfaces.device)
        self.high=torch.tensor(np.stack([boxes[i][1] for i in surfaces.geometry_indices]),device=surfaces.device)

    def query(self,world,rows):
        s=self.surfaces;n,k,_=world.shape;r=s.rotation[rows];scale=s.scale[rows]
        # Use exactly SurfaceBatch.query's coordinate transform and sign API.
        local=torch.einsum('bij,bkj->bki',r.transpose(-1,-2),world-s.origin[rows,None])/scale[:,None,None]
        depth=torch.zeros(n,device=world.device);invalid=torch.zeros(n,device=world.device,dtype=torch.int32)
        wp.launch(depth_kernel,dim=(n,k),inputs=[s.mesh_ids,wp.from_torch(rows.to(torch.int32)),
            wp.from_torch(local.contiguous(),dtype=wp.vec3),wp.from_torch(self.low,dtype=wp.vec3),wp.from_torch(self.high,dtype=wp.vec3),
            wp.from_torch(s.scale),wp.from_torch(depth),wp.from_torch(invalid)],device=str(world.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(world.device)))
        return depth,invalid==0

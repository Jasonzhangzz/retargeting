"""CUDA mesh-distance and geometric positive-span force-closure audits.

These run after the shared optimizer; they never change its candidate ranking.
Normalized LP forces do not certify realized preload or actuator feasibility.
"""
import numpy as np
import warp as wp
import torch
from forceaware import mpc_loss as fa


@wp.kernel(enable_backward=False)
def query_surface(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                  projected: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                  distance: wp.array(dtype=float)):
    i = wp.tid()
    p = points[i]
    hit = wp.mesh_query_point(mesh, p, 1.e6)
    q = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
    # Warp's mesh_get_point takes a face-vertex index, not a vertex ID.
    a = wp.mesh_get_point(mesh, 3*hit.face)
    b = wp.mesh_get_point(mesh, 3*hit.face+1)
    c = wp.mesh_get_point(mesh, 3*hit.face+2)
    normal = wp.normalize(wp.cross(b-a, c-a))
    projected[i] = q
    normals[i] = normal
    # The nearest face's plane does not classify points outside a mesh edge.
    # Use the mesh-wide inside/outside query (ray majority) at caps/vertices.
    distance[i] = hit.sign*wp.length(p-q)


@wp.kernel(enable_backward=False)
def capsule_depth(job: fa.LossJob, world: int, depth: wp.array(dtype=float)):
    i = wp.tid()
    left, right = job.model.finger_joint_pair_left[i], job.model.finger_joint_pair_right[i]
    a = fa._finger_joint_world(job, world, left, job.model.finger_joint_start[left])
    b = fa._finger_joint_world(job, world, left, job.model.finger_joint_end[left])
    c = fa._finger_joint_world(job, world, right, job.model.finger_joint_start[right])
    d = fa._finger_joint_world(job, world, right, job.model.finger_joint_end[right])
    gap = wp.sqrt(fa._segment_distance_squared(a,b,c,d)+1.e-12)
    depth[i] = wp.max(job.model.finger_joint_radius[left]+job.model.finger_joint_radius[right]-gap, 0.)


@torch.no_grad()
def force_closure_cuda(points, normals, *, friction=.6, radius=1., edges=16, tolerance=1e-7):
    """Same rank + positive normalized preload definition, with a CUDA LP.

    Infeasible-start Mehrotra primal/dual iterations solve max t subject to
    G alpha=0, sum alpha=1, alpha>=t>=0. A returned certificate is checked
    directly in float64; failed convergence cannot be reported as closure.
    """
    import torch
    if not points.is_cuda or not normals.is_cuda:
        raise ValueError('Force-closure audit requires CUDA tensors')
    p, outward = points.double(), normals.double()
    count = len(p)
    if not count:
        return dict(force_closure=False, rank=0, positive_margin=0., normalized_margin=0.,
                    equilibrium_residual=None, coefficients=[], contacts=0, backend='torch_cuda_primal_dual_LP',
                    certificate_kind='geometric_positive_span',force_units='normalized',
                    actuator_feasibility_checked=False)
    n = -torch.nn.functional.normalize(outward, dim=-1)
    axes = torch.eye(3, device=p.device, dtype=p.dtype)[n.abs().argmin(-1)]
    t1 = torch.nn.functional.normalize(torch.linalg.cross(n, axes), dim=-1)
    t2 = torch.linalg.cross(n, t1)
    angle = torch.arange(edges, device=p.device, dtype=p.dtype)*(2*torch.pi/edges)
    rays = n[...,None]+friction*(t1[...,None]*angle.cos()+t2[...,None]*angle.sin())
    rays = rays.transpose(-1,-2).reshape(-1,3)
    arms = p[:,None,:].expand(-1,edges,-1).reshape(-1,3)
    matrix = torch.cat((rays, torch.linalg.cross(arms,rays)/radius), dim=1).T
    rank = int(torch.linalg.matrix_rank(matrix, atol=1e-8, rtol=0.).item())
    k = matrix.shape[1]
    a = torch.cat((matrix, torch.ones((1,k), device=p.device, dtype=p.dtype)), dim=0)
    a = torch.cat((a, a.sum(1, keepdim=True)), dim=1)
    rhs = torch.zeros(7, device=p.device, dtype=p.dtype); rhs[-1] = 1.
    cost = torch.zeros(k+1, device=p.device, dtype=p.dtype); cost[-1] = -1.
    x = torch.full_like(cost, 1./(2*k)); slack = torch.ones_like(x); dual = torch.zeros_like(rhs)
    eye = torch.eye(7, device=p.device, dtype=p.dtype)
    def direction(rc, rp, rd):
        ratio = x/slack
        system = (a*ratio)@a.T
        residual = -rp+a@(rc/slack-ratio*rd)
        dy = torch.linalg.solve(system+1.e-13*eye, residual)
        dx = -rc/slack+ratio*(rd+a.T@dy)
        ds = -rd-a.T@dy
        return dx, dy, ds
    def step(value, delta, fraction=1.):
        limit = torch.where(delta < 0, -value/delta, torch.full_like(delta, torch.inf)).min()
        return torch.minimum(torch.ones_like(limit), fraction*limit)
    converged = False
    for iteration in range(100):
        rp, rd = a@x-rhs, a.T@dual+slack-cost
        mu = (x*slack).mean()
        if max(float(rp.abs().max()), float(rd.abs().max()), float(mu)) < 1e-9:
            converged = True
            break
        try:
            dx, dy, ds = direction(x*slack, rp, rd)
            ap, ad = step(x,dx), step(slack,ds)
            affine_mu = ((x+ap*dx)*(slack+ad*ds)).mean()
            sigma = (affine_mu/mu).clamp(0.,1.)**3
            dx, dy, ds = direction(x*slack+dx*ds-sigma*mu, rp, rd)
        except torch.linalg.LinAlgError:
            break  # An infeasible LP must never abort the batch or certify closure.
        ap, ad = step(x,dx,.995), step(slack,ds,.995)
        x, dual, slack = x+ap*dx, dual+ad*dy, slack+ad*ds
        if not bool(torch.isfinite(x).all()&torch.isfinite(slack).all()):
            break
    alpha = x[:-1]+x[-1]
    if bool(torch.isfinite(alpha).all()) and float(alpha.sum()) > 0:
        alpha = alpha/alpha.sum()
    else:
        alpha = torch.zeros_like(alpha)
    margin = float(alpha.min())
    residual = float(torch.linalg.vector_norm(matrix@alpha))
    closed = bool(converged and rank == 6 and margin > tolerance and residual < 1e-7)
    return dict(force_closure=closed, rank=rank, positive_margin=margin, normalized_margin=k*margin,
                equilibrium_residual=residual, coefficients=alpha.cpu().tolist(), contacts=count,
                backend='torch_cuda_primal_dual_LP', converged=converged, iterations=iteration+1,
                certificate_kind='geometric_positive_span',force_units='normalized',
                actuator_feasibility_checked=False)


class GraspAudit:
    def __init__(self, surface, device):
        self.device = device
        vertices, faces = np.unique(surface.triangles.reshape(-1,3), axis=0, return_inverse=True)
        self.vertices = wp.array(vertices,dtype=wp.vec3,device=device)
        self.faces = wp.array(faces,dtype=int,device=device)
        self.mesh = wp.Mesh(points=self.vertices,indices=self.faces)
        wp.synchronize_device(device)
        self.radius = float(np.linalg.norm(vertices,axis=1).max())

    def query(self, points):
        import torch
        source = wp.from_torch(points.float().contiguous(), dtype=wp.vec3)
        output = wp.empty(len(points),dtype=wp.vec3,device=self.device)
        normals = wp.empty_like(output)
        distance = wp.empty(len(points),dtype=float,device=self.device)
        wp.launch(query_surface,len(points),inputs=[self.mesh.id,source],outputs=[output,normals,distance],
                  device=self.device,stream=wp.stream_from_torch(points.device))
        return wp.to_torch(output),wp.to_torch(normals),wp.to_torch(distance)

    def evaluate_state(self,runtime,hand,args,qpos,qvel):
        """Audit a measured, physically formed state without advancing time."""
        from types import SimpleNamespace
        from comfree_warp.native_adjoint.observation_adjoint import allocate_workspace,forward
        compiled = runtime.optimizer.compiled.base
        workspace = allocate_workspace(compiled,1,requires_grad=False)
        state = forward(compiled,wp.array(np.asarray(qpos)[None],dtype=float,device=self.device),workspace)
        original = runtime.optimizer.loss.immediate[0].job
        # Warp structs own ctypes storage. copy.copy aliases that storage and
        # would replace the planner's differentiable state arrays as well.
        job,state_copy = fa.LossJob(),fa.LossState()
        for name in original.state._cls.vars:
            setattr(state_copy,name,getattr(original.state,name))
        for name in original._cls.vars:
            setattr(job,name,state_copy if name=='state' else getattr(original,name))
        job.state.body_position,job.state.body_matrix = state.body_position,state.body_matrix
        job.state.qpos = wp.array(np.asarray(qpos)[None],dtype=float,device=self.device)
        job.state.qvel = wp.array(np.asarray(qvel)[None],dtype=float,device=self.device)
        view = SimpleNamespace(index=runtime.index,output_workspace=SimpleNamespace(job=job))
        return self.evaluate(view,hand,args,0)

    @torch.no_grad()
    def evaluate(self, runtime, hand, args, selected):
        import torch
        job = runtime.output_workspace.job
        position = wp.to_torch(job.state.body_position)[selected]
        rotation = wp.to_torch(job.state.body_matrix)[selected]
        obj = runtime.index.object_body
        def tensor(value, dtype=None):
            return torch.as_tensor(value,device=str(self.device),dtype=dtype)
        def local_points(bodies, local):
            ids = tensor(bodies,torch.long)
            world = torch.bmm(rotation[ids],tensor(local,torch.float32)[...,None])[...,0]+position[ids]
            return (world-position[obj])@rotation[obj]
        points = local_points(hand.bodies,hand.local)
        projected,normals,distance = self.query(points)
        ids = tensor(hand.bodies,torch.long)
        hn = torch.bmm(rotation[ids],tensor(hand.normals,torch.float32)[...,None])[...,0]@rotation[obj]
        normal_cosine = -(hn*normals).sum(-1).clamp(-1.,1.)
        normal_residual = 2*(1-normal_cosine)
        normal_angle = torch.rad2deg(torch.acos(normal_cosine))
        active = (distance.abs() <= args.contact_tolerance_mm/1000)&(normal_cosine>.25)
        closure = force_closure_cuda(projected[active],normals[active],friction=args.friction,
                                     radius=self.radius,edges=args.cone_edges)
        dense = local_points(hand.all_bodies,hand.all_local)
        _,_,dense_distance = self.query(dense)
        penetration = float((-dense_distance.min()).clamp(min=0))*1000
        depths = wp.zeros(job.model.finger_joint_pair_count,dtype=float,device=self.device)
        wp.launch(capsule_depth,len(depths),inputs=[job,selected],outputs=[depths],device=self.device,
                  stream=wp.stream_from_torch(position.device))
        self_depth = float(wp.to_torch(depths).max())*1000
        groups = {}
        for i,body in enumerate(hand.bodies):
            name = hand.model.body(int(body)).name
            group = name.split('_link')[0] if name.startswith('right_finger') else 'palm'
            groups.setdefault(group,[]).append(i)
        required = all(bool(active[ids].any()) for ids in groups.values())
        feasible = required and max(penetration,self_depth) <= args.penetration_tolerance_mm
        report = dict(force_closure=closure, active_contacts=int(active.sum()),
                      required_groups_satisfied=required, geometry_feasible=feasible,
                      feasible=bool(feasible and closure['force_closure']),
                      gap_mean_mm=float(distance.abs().mean())*1000,
                      gap_max_mm=float(distance.abs().max())*1000,
                      normal_squared_residual=normal_residual.cpu().tolist(),
                      normal_angle_deg=normal_angle.cpu().tolist(),
                      normal_squared_residual_max=float(normal_residual.max()),
                      template_normal_alignment_in_shared_solver=False,
                      max_object_penetration_mm=penetration,max_self_penetration_mm=self_depth,
                      self_collision_measurement='same inter-finger capsules as shared GPU objective',
                      audit_backend='warp_mesh_BVH_and_torch_cuda_primal_dual_LP')
        arrays = dict(hand_points=points,realized_object_points=projected,
                      realized_hand_normals=hn,realized_normal_squared_residual=normal_residual,
                      realized_normal_angle_deg=normal_angle,
                      realized_object_normals=normals,realized_signed_distance=distance,realized_contact_mask=active)
        return report,{key:value.detach().cpu().numpy().copy() for key,value in arrays.items()}

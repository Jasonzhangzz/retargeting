"""CUDA static-load conditioning of Dexonomy inputs.

This constructs a position-servo warm start, not a new MPC objective. Geometry
alone supplies no actuator preload: tau = gravity_bias + J.T @ contact_force,
and the command is q + tau/kp, subject to the actual command/torque bounds.
The common ADMM receives the original state/reference and these corrections.
"""
from retarget.grasp_capture import capture_guard
import time

import numpy as np
import torch

from retarget.grasp_template import WujiContactKinematics


class HandLoadKinematics:
    def __init__(self, model, bodies, local, device):
        self.model = model
        self.contacts = WujiContactKinematics(model,bodies,device)
        self.tensor = self.contacts.tensor
        self.local = self.tensor(local)
        self.normals = torch.zeros_like(self.local)
        names = ['right_palm_link']+[f'right_finger{f}_link{k}'
                                   for f in range(1,6) for k in range(1,5)]
        ids = np.array([model.body(name).id for name in names])
        self.hand = WujiContactKinematics(model,ids,device)
        self.com = self.tensor(model.body_ipos[ids])
        self.mass = self.tensor(model.body_mass[ids]*(1-model.body_gravcomp[ids]))
        roots = np.array([model.body_jntadr[model.jnt_bodyid[j]] for j in range(6)])
        if not np.array_equal(roots,np.arange(6)):
            raise ValueError('Expected the six Wuji root joints before finger joints')
        root_bodies = model.jnt_bodyid[:6]
        if np.max(np.abs(model.body_ipos[root_bodies]))>1e-12:
            raise ValueError('Root COM offsets need an explicit kinematic model')
        mass = model.body_mass[root_bodies]*(1-model.body_gravcomp[root_bodies])
        self.root_mass = self.tensor([mass.sum(),mass[1:].sum(),mass[2:].sum()])
        self.gravity = self.tensor([0.,0.,-9.81])
        self.stiffness = self.tensor(model.jnt_stiffness[:26])
        self.spring = self.tensor(model.qpos_spring[:26])

    def positions(self, q):
        return self.contacts.contacts(q,self.local,self.normals)[0]

    def potential(self, q):
        p,r = self.hand.frames(q)
        center = p+(r@self.com[...,None])[...,0]
        return (-(self.mass[:,None]*center*self.gravity).sum()
                -(self.root_mass*q[:3]*self.gravity).sum()
                +.5*(self.stiffness*(q-self.spring).square()).sum())

    def evaluate(self, q):
        return (torch.func.jacfwd(self.positions)(q),
                torch.func.grad(self.potential)(q))


def load_matrices(model, bodies, local, q, points, outward, *, device,
                  mass, friction, radius, edges=8):
    """Build the force/servo constraints in the fixed object/world frame."""
    fk = HandLoadKinematics(model,bodies,local,device)
    q,points,outward = (fk.tensor(v) for v in (q,points,outward))
    jac,bias = fk.evaluate(q)
    inward = -torch.nn.functional.normalize(outward,dim=-1)
    axes = torch.eye(3,device=q.device,dtype=q.dtype)[inward.abs().argmin(-1)]
    t1 = torch.nn.functional.normalize(torch.linalg.cross(inward,axes),dim=-1)
    t2 = torch.linalg.cross(inward,t1)
    angle = torch.arange(edges,device=q.device,dtype=q.dtype)*(2*torch.pi/edges)
    rays = (inward[:,None]+friction*(t1[:,None]*angle.cos()[None,:,None]
                                +t2[:,None]*angle.sin()[None,:,None]))
    forces = rays.reshape(-1,3)
    arms = points[:,None].expand_as(rays).reshape(-1,3)
    grasp = torch.cat((forces,torch.linalg.cross(arms,forces)/radius),dim=1).T
    actuator = torch.einsum('cij,cki->jck',jac,rays).reshape(26,-1)
    kp = fk.tensor(model.actuator_gainprm[:,0])
    if (model.nu!=26 or not np.allclose(model.actuator_biasprm[:,1],-model.actuator_gainprm[:,0])
            or not np.allclose(model.actuator_gear[:,0],1.)):
        raise ValueError('Load conditioning requires the Wuji scalar position servos')
    bounds = fk.tensor(model.actuator_ctrlrange)
    low,high = kp*(bounds[:,0]-q),kp*(bounds[:,1]-q)
    limited = torch.as_tensor(model.actuator_forcelimited,device=q.device,dtype=torch.bool)
    torque = fk.tensor(model.actuator_forcerange)
    low = torch.maximum(low,torch.where(limited,torque[:,0],-torch.inf))
    high = torch.minimum(high,torch.where(limited,torque[:,1],torch.inf))
    goal = fk.tensor([0.,0.,mass*9.81,0.,0.,0.])
    return dict(q=q,jacobian=jac,bias=bias,kp=kp,rays=rays,grasp=grasp,
                actuator=actuator,low=low,high=high,goal=goal)


@torch.no_grad()
def solve_load(matrices, *, minimum_per_contact=.3, minimum_total=6., iterations=1600):
    """Small bounded pressure QP on CUDA, with explicit residual validation.

    The ADMM here is only a static input QP; it neither replaces nor changes
    the shared contact/hand MPC splitting. No CPU optimizer is used.
    """
    m = matrices
    G,B = m['grasp'],m['actuator']
    if not G.is_cuda:
        raise ValueError('Static input load solve requires CUDA')
    count,edges = m['rays'].shape[:2]
    size = G.shape[1]
    identity = torch.eye(size,device=G.device,dtype=G.dtype)
    normal = torch.kron(torch.eye(count,device=G.device,dtype=G.dtype),
                        torch.ones((1,edges),device=G.device,dtype=G.dtype))
    scale = ((m['high']-m['low'])/2).clamp(min=.01)
    A = torch.cat((G,B/scale[:,None],identity,normal,torch.ones_like(normal[:1])),dim=0)
    lower = torch.cat((m['goal'],(m['low']-m['bias'])/scale,
        G.new_zeros(size),G.new_full((count,),minimum_per_contact),G.new_tensor([minimum_total])))
    upper = torch.cat((m['goal'],(m['high']-m['bias'])/scale,
        G.new_full((size+count+1,),torch.inf)))
    rho,sigma = 10.,1e-6
    factor = torch.linalg.cholesky(.01*identity+sigma*identity+rho*A.T@A)
    inverse = torch.cholesky_inverse(factor)
    x = G.new_zeros(size)
    z,y = G.new_zeros(len(lower)),G.new_zeros(len(lower))
    # Fixed-size iterations are device-only and can be captured by callers.
    for _ in range(iterations):
        x = inverse@(sigma*x-.01+A.T@(rho*z-y))
        ax = A@x
        z = (ax+y/rho).clamp(lower,upper)
        y += rho*(ax-z)
    forces = (x[:,None]*m['rays'].reshape(-1,3)).reshape(count,edges,3).sum(1)
    torque = m['bias']+B@x
    command = m['q']+torque/m['kp']
    residual = G@x-m['goal']
    normal_force = x.reshape(count,edges).sum(1)
    violation = torch.maximum((lower-A@x).clamp(min=0.),(A@x-upper).clamp(min=0.)).max()
    finite = torch.isfinite(x).all()&torch.isfinite(command).all()
    feasible = bool(finite and violation<2e-4)
    return command.cpu().numpy(),dict(feasible=feasible,backend='torch_cuda_bounded_load_qp',
        gravity_command=(m['q']+m['bias']/m['kp']).cpu().tolist(),
        maximum_scaled_constraint_violation=float(violation),
        wrench_residual=residual.cpu().tolist(),normal_force_N=normal_force.cpu().tolist(),
        force_N=forces.cpu().tolist(),actuator_torque_Nm=torque.cpu().tolist(),
        minimum_per_contact_N=minimum_per_contact,minimum_total_N=minimum_total,
        iterations=iterations)


@torch.no_grad()
def force_cone_diagnostic(outward, friction, required_force, minimum_total):
    """A sufficient separating-direction certificate, not a feasibility solver.

    If every continuous friction cone has positive projection on this axis,
    positive contact pressure cannot produce zero force along it. This can
    prove a particular candidate lacks closure even before actuator bounds.
    Failure to find this simple witness does not imply force closure.
    """
    inward = -torch.nn.functional.normalize(outward,dim=-1)
    axis = torch.nn.functional.normalize(inward.mean(0),dim=0)
    cosine = (inward@axis).clamp(-1.,1.)
    lower = (cosine-friction*torch.sqrt((1-cosine.square()).clamp(min=0.))).min()
    target = required_force@axis
    separated = bool(lower>1e-8)
    return dict(axis=axis.cpu().tolist(),minimum_force_projection_per_normal_N=float(lower),
        required_force_projection_N=float(target),minimum_total_normal_force_N=minimum_total,
        force_closure_excluded=separated,
        static_load_excluded=bool(separated and lower*minimum_total>target+1e-7),
        scope='this candidate and continuous Coulomb cones; no global reachability claim')


def load_command(model,bodies,local,q,points,outward,*,device,mass,friction,radius,
                 minimum_per_contact=None,minimum_total=None):
    begun = time.perf_counter()
    matrices = load_matrices(model,bodies,local,q,points,outward,device=device,
        mass=mass,friction=.65*friction,radius=radius)
    minimum = max(30*mass,2*mass*9.81/max(.65*friction,.05)) if minimum_total is None else minimum_total
    if minimum_per_contact is None:
        minimum_per_contact = min(.5,3./len(points))
    command,info = solve_load(matrices,minimum_per_contact=minimum_per_contact,minimum_total=minimum)
    info['force_cone_diagnostic'] = force_cone_diagnostic(matrices['q'].new_tensor(outward),
        .65*friction,matrices['goal'][:3],minimum)
    info['seconds'] = time.perf_counter()-begun
    return command,info


def load_matrices_batch(model, bodies, local, q, points, outward, *, device,
                        mass, friction, radius, edges=8):
    """Vectorize kinematics and constraints over independent original poses."""
    fk = HandLoadKinematics(model, bodies, local, device)
    q, points, outward = (fk.tensor(value) for value in (q, points, outward))
    jac, bias = torch.vmap(fk.evaluate)(q)
    inward = -torch.nn.functional.normalize(outward, dim=-1)
    axes = torch.eye(3, device=q.device, dtype=q.dtype)[inward.abs().argmin(-1)]
    t1 = torch.nn.functional.normalize(torch.linalg.cross(inward, axes), dim=-1)
    t2 = torch.linalg.cross(inward, t1)
    angle = torch.arange(edges, device=q.device, dtype=q.dtype)*(2*torch.pi/edges)
    rays = inward[:, :, None]+friction*(t1[:, :, None]*angle.cos()[None, None, :, None]
                                      +t2[:, :, None]*angle.sin()[None, None, :, None])
    forces = rays.flatten(1, 2)
    arms = points[:, :, None].expand_as(rays).flatten(1, 2)
    grasp = torch.cat((forces, torch.linalg.cross(arms, forces)/radius), dim=-1).transpose(1, 2)
    actuator = torch.einsum('bcij,bcki->bjck', jac, rays).flatten(2)
    if (model.nu != 26 or not np.allclose(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0])
            or not np.allclose(model.actuator_gear[:, 0], 1.)):
        raise ValueError('Load conditioning requires the Wuji scalar position servos')
    kp, bounds = fk.tensor(model.actuator_gainprm[:, 0]), fk.tensor(model.actuator_ctrlrange)
    limited = torch.as_tensor(model.actuator_forcelimited, device=q.device, dtype=torch.bool)
    torque = fk.tensor(model.actuator_forcerange)
    low = torch.maximum(kp*(bounds[:, 0]-q), torch.where(limited, torque[:, 0], -torch.inf))
    high = torch.minimum(kp*(bounds[:, 1]-q), torch.where(limited, torque[:, 1], torch.inf))
    goal = fk.tensor([0., 0., mass*9.81, 0., 0., 0.]).expand(len(q), -1)
    return dict(q=q, jacobian=jac, bias=bias, kp=kp, rays=rays, grasp=grasp,
                actuator=actuator, low=low, high=high, goal=goal, friction=friction)


@torch.no_grad()
def solve_load_batch(m, *, minimum_per_contact=.3, minimum_total=6., iterations=1600):
    """One captured GPU QP schedule for every candidate; no per-pose solves."""
    G, B = m['grasp'], m['actuator']
    if not G.is_cuda:
        raise ValueError('Static input load solve requires CUDA')
    batch, contacts, edges = m['rays'].shape[:3]
    size = G.shape[-1]
    eye = torch.eye(size, device=G.device, dtype=G.dtype)
    normal = torch.kron(torch.eye(contacts, device=G.device, dtype=G.dtype),
                        torch.ones((1, edges), device=G.device, dtype=G.dtype))
    scale = ((m['high']-m['low'])/2).clamp(min=.01)
    A = torch.cat((G, B/scale[:, :, None], eye.expand(batch, -1, -1),
                   normal.expand(batch, -1, -1), torch.ones_like(normal[:1]).expand(batch, -1, -1)), dim=1)
    lower = torch.cat((m['goal'], (m['low']-m['bias'])/scale,
        G.new_zeros(batch, size), G.new_full((batch, contacts), minimum_per_contact),
        G.new_full((batch, 1), minimum_total)), dim=1)
    upper = torch.cat((m['goal'], (m['high']-m['bias'])/scale,
        G.new_full((batch, size+contacts+1), torch.inf)), dim=1)
    rho, sigma = 10., 1.e-6
    factor, status = torch.linalg.cholesky_ex(.010001*eye+rho*A.transpose(1, 2)@A, check_errors=False)
    inverse = torch.cholesky_inverse(factor)
    x, ax = G.new_zeros(batch, size), G.new_zeros(batch, A.shape[1])
    y, z = torch.zeros_like(ax), torch.zeros_like(ax)
    multiply = lambda matrix, vector: torch.bmm(matrix, vector[:, :, None])[:, :, 0]

    def step():
        x.copy_(multiply(inverse, sigma*x-.01+multiply(A.transpose(1, 2), rho*z-y)))
        ax.copy_(multiply(A, x))
        z.copy_((ax+y/rho).clamp(lower, upper))
        y.add_(rho*(ax-z))

    stream = torch.cuda.Stream(device=G.device)
    stream.wait_stream(torch.cuda.current_stream(G.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_guard(), torch.cuda.graph(graph, stream=stream):
        x.zero_(); y.zero_(); z.zero_()
        for _ in range(iterations):
            step()
    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.current_stream(G.device).wait_stream(stream)
    force = (x[:, :, None]*m['rays'].flatten(1, 2)).reshape(batch, contacts, edges, 3).sum(2)
    torque = m['bias']+multiply(B, x)
    command = m['q']+torque/m['kp']
    violation = torch.maximum((lower-ax).clamp(min=0.), (ax-upper).clamp(min=0.)).amax(1)
    finite = torch.isfinite(x).all(1)&torch.isfinite(command).all(1)
    result = dict(feasible=finite&(violation < 2.e-4)&(status == 0),
        gravity_command=m['q']+m['bias']/m['kp'], maximum_scaled_constraint_violation=violation,
        wrench_residual=multiply(G, x)-m['goal'], normal_force_N=x.reshape(batch, contacts, edges).sum(2),
        force_N=force, actuator_torque_Nm=torque)
    inward = torch.nn.functional.normalize(m['rays'].mean(2), dim=-1)
    axis = torch.nn.functional.normalize(inward.mean(1), dim=-1)
    cosine = (inward*axis[:, None]).sum(-1).clamp(-1., 1.)
    cone_lower = (cosine-m['friction']*(1-cosine.square()).clamp(min=0.).sqrt()).amin(1)
    target_projection = (m['goal'][:, :3]*axis).sum(1)
    cone = dict(axis=axis, minimum_force_projection_per_normal_N=cone_lower,
        required_force_projection_N=target_projection, force_closure_excluded=cone_lower > 1.e-8,
        static_load_excluded=(cone_lower > 1.e-8)&(cone_lower*minimum_total > target_projection+1.e-7))
    # One boundary transfer after the entire batched numerical schedule.
    commands = command.cpu().numpy()
    host = {key: value.cpu().numpy() for key, value in result.items()}
    cone_host = {key: value.cpu().numpy() for key, value in cone.items()}
    return commands, [dict({key: value[i].tolist() for key, value in host.items()},
        force_cone_diagnostic=dict({key: value[i].tolist() for key, value in cone_host.items()},
            minimum_total_normal_force_N=minimum_total,
            scope='this candidate and continuous Coulomb cones; no global reachability claim'),
        backend='cuda_graph_batched_load_qp', batch_size=batch,
        minimum_per_contact_N=minimum_per_contact, minimum_total_N=minimum_total, iterations=iterations)
        for i in range(batch)]


def load_commands_batch(model, bodies, local, q, points, outward, *, device, mass, friction, radius):
    begun = time.perf_counter()
    matrices = load_matrices_batch(model, bodies, local, q, points, outward, device=device,
        mass=mass, friction=.65*friction, radius=radius)
    minimum = max(30*mass, 2*mass*9.81/max(.65*friction, .05))
    commands, infos = solve_load_batch(matrices, minimum_per_contact=min(.5, 3./np.shape(points)[1]),
                                      minimum_total=minimum)
    elapsed = time.perf_counter()-begun
    for info in infos:
        info.update(seconds=elapsed, timing_scope='entire_candidate_batch')
    return commands, infos

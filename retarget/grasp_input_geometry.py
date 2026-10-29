"""CUDA geometry conditioning of dense power-grasp inputs.

This fits the mapped hand, not the shared ADMM objective. Final object poses
and surface anchors still come from the original Dexonomy HandObjMatcher.
"""
from retarget.grasp_capture import capture_guard
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from retarget.grasp_hand import query_path
from retarget.grasp_template import WujiContactKinematics, BoneReference, skin_reference
from retarget.grasp_regions import SurfaceClearance, contact_mask


class WujiSelfCollision:
    """Differentiable version of the shared planner's inter-finger capsules."""
    def __init__(self, model, device):
        from forceaware.mpc_loss import _fit_collision_capsule
        from retarget.grasp_hand import model_hand, body_boxes
        bodies, starts, ends, radii = [], [], [], []
        for finger in range(1, 6):
            for link in range(1, 5):
                name = f'right_finger{finger}_link{link}'
                with np.load(query_path(model, name)) as asset:
                    points = np.unique(asset['local_pos'], axis=0)
                    # The actual query assets have zero-radius sources.
                    spheres = np.c_[points, np.zeros(len(points), np.float32)]
                a, b, radius = _fit_collision_capsule(spheres)
                bodies.append(model.body(name).id)
                starts.append(a); ends.append(b); radii.append(radius)
        self.fk = WujiContactKinematics(model, np.array(bodies), device)
        self.boxes = tuple(map(self.fk.tensor, body_boxes(model, bodies))) if model_hand(model) == 'wuji2' else None
        self.starts, self.ends, self.radii = map(self.fk.tensor, (starts, ends, radii))
        pairs = torch.tensor([(i, j) for i in range(20) for j in range(i+1, 20)
                              if i//4 != j//4], device=device, dtype=torch.long)
        self.left, self.right = pairs.T

    def depths(self, q):
        position, rotation = self.fk.frames(q)
        a = position+(rotation@self.starts[..., None])[..., 0]
        b = position+(rotation@self.ends[..., None])[..., 0]
        a, b, c, d = a[self.left], b[self.left], a[self.right], b[self.right]

        def point_segment(p, a, b):
            u = b-a
            fraction = ((p-a)*u).sum(-1)/u.square().sum(-1).clamp(min=1e-12)
            return (p-a-fraction.clamp(0., 1.)[..., None]*u).square().sum(-1)

        endpoint = torch.stack((point_segment(a, c, d), point_segment(b, c, d),
                                point_segment(c, a, b), point_segment(d, a, b))).amin(0)
        u, v, offset = b-a, d-c, a-c
        aa, ab, bb = u.square().sum(-1), (u*v).sum(-1), v.square().sum(-1)
        ao, bo = (u*offset).sum(-1), (v*offset).sum(-1)
        denominator = aa*bb-ab.square()
        nonparallel = denominator > 1e-8*aa*bb
        safe = torch.where(nonparallel, denominator, torch.ones_like(denominator))
        s, t = (ab*bo-bb*ao)/safe, (aa*bo-ab*ao)/safe
        interior = (offset+s[..., None]*u-t[..., None]*v).square().sum(-1)
        inside = nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        distance = torch.sqrt(torch.where(inside, torch.minimum(endpoint, interior), endpoint)+1e-12)
        gap = distance-self.radii[self.left]-self.radii[self.right]
        if self.boxes is not None:
            center, half = self.boxes
            world = position+(rotation@center[..., None])[..., 0]
            ra, rb = rotation[self.left], rotation[self.right]
            axes_a, axes_b = ra.transpose(-1, -2), rb.transpose(-1, -2)
            cross = torch.cross(axes_a[:, :, None, :].expand(-1, 3, 3, -1),
                                axes_b[:, None, :, :].expand(-1, 3, 3, -1), dim=-1).flatten(1, 2)
            axes = torch.cat((axes_a, axes_b, cross), dim=1)
            lengths = axes.norm(dim=-1)
            axes = axes/lengths.clamp(min=1.e-12)[..., None]
            offset = world[self.left]-world[self.right]
            separation = ((offset[:, None]*axes).sum(-1).abs()
                -((axes@ra).abs()*half[self.left, None]).sum(-1)
                -((axes@rb).abs()*half[self.right, None]).sum(-1))
            certified = separation.masked_fill(lengths < 1.e-6, -torch.inf).amax(-1)
            gap = torch.maximum(gap, certified)
        return (-gap).clamp(min=0.)


def dense_power_template(hand, primitive):
    """A palm and multiple phalanges on a finger, rather than a tip pinch."""
    if not primitive or primitive['kind'] != 'cylinder':
        return False
    names = [hand.model.body(int(body)).name for body in hand.bodies]
    fingers = {}
    for name in names:
        if name.startswith('right_finger'):
            fingers.setdefault(name.split('_link')[0], set()).add(name)
    return 'right_palm_link' in names and any(len(links) >= 3 for links in fingers.values())


def fit_power_cylinder_input(hand, initial, *, device, joint_prior=None,
                             reference_qpos=None, reference_normals=None):
    """Preserve the grasp shape with optional finger and palm contact targets.

    The cylinder is aligned with the finger-base row. A one-axis opening
    stretch cannot preserve the curved multi-phalange layout of a power grasp.
    Mesh/closure/actuator checks after the original matcher remain decisive.
    The analytic cylinder is an input-fitting proxy only (the generated mesh
    has 48 sides); it is never used to generate exported object contacts.
    """
    from retarget.grasp_initialization import template_contact_layout
    begun = time.perf_counter()
    before = template_contact_layout(hand, initial)
    vertices = hand.surface.triangles.reshape(-1, 3)
    extent = vertices.max(0)-vertices.min(0)
    radius, height = extent[0]/2, extent[2]
    fk = WujiContactKinematics(hand.model, hand.bodies, device)
    canonical = np.array(initial).copy()
    canonical[:6] = 0
    hand.set(canonical)
    points = hand.points(hand.bodies, hand.local)
    normals = np.einsum('nij,nj->ni', hand.data.xmat[hand.bodies].reshape(-1, 3, 3), hand.normals)
    palm = np.flatnonzero(hand.bodies == hand.model.body('right_palm_link').id)[-1]
    radial = -normals[palm]
    axis = (hand.model.body('right_finger2_link1').pos
            -hand.model.body('right_finger5_link1').pos).copy()
    axis -= axis.dot(radial)*radial
    axis /= np.linalg.norm(axis)
    rotation = np.array([radial, np.cross(axis, radial), axis])
    canonical[:3] = np.array([radius, 0., 0.])-rotation@points[palm]
    canonical[3:6] = Rotation.from_matrix(rotation).as_euler('XYZ')
    axial = (points@rotation.T+canonical[:3])[:, 2]
    canonical[2] -= (axial.min()+axial.max())/2
    q = fk.tensor(canonical)
    local, normal = fk.tensor(hand.local), fk.tensor(hand.normals)
    cloud_fk = WujiContactKinematics(hand.model, hand.all_bodies, device)
    cloud_local = fk.tensor(hand.all_local)
    cloud_zero = torch.zeros_like(cloud_local)
    joint_ids = np.array([np.flatnonzero(hand.model.jnt_qposadr == i)[0] for i in range(26)])
    low, high = (fk.tensor(a) for a in hand.model.jnt_range[joint_ids].T)
    # Reserve command range for preload. Physical model/servo limits are not
    # changed; these tighter bounds apply only to this input fit.
    low[6:] += .08
    high[6:] -= .08
    posture = q.clone()
    q.clamp_(low, high)
    reference_qpos = np.array(initial if reference_qpos is None else reference_qpos, copy=True)
    posture[6:] = fk.tensor(reference_qpos[6:])
    center = (low[6:]+high[6:])/2
    center_weight = .03*(joint_prior or 0.)
    bones = BoneReference(hand.model, reference_qpos, device)
    region_normals = skin_reference(hand, reference_normals)
    clearance = SurfaceClearance(hand, region_normals, device)
    clearance.refresh(q)
    prior = fk.tensor([.1]*3+[.03]*3+[.1]*20)
    identity = torch.eye(26, device=fk.device, dtype=fk.dtype)
    damping = fk.tensor(1.)
    limit = fk.tensor([.01]*3+[.2]*23)
    capsules = WujiSelfCollision(hand.model, device)
    weights = fk.tensor([5. if hand.model.body(int(body)).name == 'right_palm_link'
                         else 2. if 'finger1_' in hand.model.body(int(body)).name else 1.
                         for body in hand.bodies])

    def side(p):
        r = torch.linalg.vector_norm(p[..., :2], dim=-1).clamp(min=1e-9)
        n = torch.cat((p[..., :2]/r[..., None], torch.zeros_like(r[..., None])), -1)
        return r-radius, n

    def sdf(p):
        gap, _ = side(p)
        d = torch.stack((gap, p[..., 2].abs()-height/2), -1)
        return d.amax(-1).clamp(max=0.)+torch.linalg.vector_norm(d.clamp(min=0.), dim=-1)

    def residual(value):
        p, n = fk.contacts(value, local, normal)
        gap, outward = side(p)
        cloud, _ = cloud_fk.contacts(value, cloud_local, cloud_zero)
        gap_mm = gap*1000
        reach = 1./(1.+(gap_mm/3.).square())
        return torch.cat((gap_mm/torch.sqrt(1.+(gap_mm/2.).square())*weights,
            ((n+outward)*(weights*reach)[:, None]).flatten()*10,
            (p[:, 2].amin()+p[:, 2].amax()).reshape(1)*10,
            (-sdf(cloud)-.0002).clamp(min=0.)*1000,
            (capsules.depths(value)-.0003).clamp(min=0.)*5000, prior*(value-posture),
            center_weight*(value[6:]-center), 20.*bones.style_residual(value),
            clearance.residual(value)*5000))

    def step():
        r = residual(q)
        jacobian = torch.func.jacfwd(residual)(q)
        delta = torch.linalg.solve_ex(jacobian.T@jacobian+damping*identity, -jacobian.T@r)[0]
        trial = (q+delta.clamp(-limit, limit)).clamp(low, high)
        accept = (residual(trial).square().sum() < r.square().sum()) & bones.accepts(trial)
        q.copy_(torch.where(accept, trial, q))
        damping.copy_(torch.where(accept, damping*.5, damping*4).clamp(1e-5, 1e8))

    pool = []
    for body in hand.bodies:
        with np.load(query_path(hand.model, body)) as asset:
            pool.append((asset['local_pos'], asset['local_normal']))
    size = max(len(p) for p, _ in pool)
    pp = np.zeros((len(pool), size, 3)); nn = pp.copy()
    valid = np.zeros(pp.shape[:-1], bool)
    for i, (p, n) in enumerate(pool):
        pp[i, :len(p)], nn[i, :len(n)] = p, n
        # Keep the annotated skin side; do not satisfy the geometry with a
        # dorsal patch or delete unreachable annotated contacts.
        valid[i, :len(p)] = contact_mask(hand.model, int(hand.bodies[i]), n, region_normals[i])
    if not valid.any(axis=1).all():
        raise ValueError('No query samples in an annotated contact region')
    pp, nn = fk.tensor(pp), fk.tensor(nn)
    valid = torch.tensor(valid, device=fk.device)
    rows = torch.arange(len(pool), device=fk.device)
    stream = torch.cuda.Stream(device=fk.device)
    stream.wait_stream(torch.cuda.current_stream(fk.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream(fk.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with capture_guard(), torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.current_stream(fk.device).wait_stream(stream)
    with torch.no_grad():
        for _ in range(5):
            clearance.refresh(q)
            damping.fill_(1.)
            for _ in range(40):
                graph.replay()
            p, r = fk.frames(q)
            world_p, world_n = pp@r.transpose(-1, -2)+p[:, None], nn@r.transpose(-1, -2)
            gap, outward = side(world_p)
            score = ((gap*1000).square()+((world_n+outward)*10).square().sum(-1)
                     +((world_p[..., 2].abs()-(height/2-.002)).clamp(min=0.)*1000).square())
            selected = score.masked_fill(~valid, torch.inf).argmin(-1)
            local.copy_(pp[rows, selected]); normal.copy_(nn[rows, selected])
        clearance.refresh(q)
        for _ in range(40):
            graph.replay()
        p, n = fk.contacts(q, local, normal)
        gap, outward = side(p)
        angle = torch.rad2deg(torch.acos((-(n*outward).sum(-1)).clamp(-1, 1)))
        fitted = q.cpu().numpy().copy()
        hand.local, hand.normals = local.cpu().numpy().copy(), normal.cpu().numpy().copy()
    return fitted, dict(method='CUDA_dense_power_cylinder_input_IK', device=str(fk.device),
        joint_center_prior_weight=center_weight, joint_command_reserve_rad=.08,
        bone_direction_weight=20., reference_qpos=reference_qpos.tolist(), contacts_required=False,
        bone_direction_max_deviation_deg=bones.maximum_angle(q), **clearance.diagnostics(q),
        bone_finger_weights=bones.weights.cpu().tolist(),
        reference_local_normals=region_normals.tolist(),
        contact_region_normal_dot=np.einsum('ij,ij->i', hand.normals, region_normals).tolist(),
        bone_direction_rms=float(bones.residual(q).square().mean().sqrt()),
        requested_refinement_prior_weight=joint_prior,
        seconds=time.perf_counter()-begun, surface_query_indices=selected.cpu().tolist(),
        contact_position_error_mm=np.abs(gap.cpu().numpy()*1000).tolist(),
        contact_normal_error_deg=angle.cpu().tolist(),
        maximum_self_penetration_mm=float(capsules.depths(q).max().detach())*1000,
        all_annotation_contacts_retained=True, analytic_surface_is_input_proxy=True,
        target_source='primitive envelope input; final anchors require original Dexonomy query',
        requested_opening_m=2*float(radius), before=before, after=template_contact_layout(hand, fitted),
        shared_solver_modified=False)

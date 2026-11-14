"""Material-point contact references for native MJWarp sampling rollouts."""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
from robokit.terms.simulation import SimContext
from robokit.terms.task import Task


@dataclass
class ContactReference:
    weight: np.ndarray             # [physics frame, contact], mask * confidence
    body: np.ndarray               # canonical execution-model body IDs
    hand_local: np.ndarray         # corresponding hand material points, metres
    object_local: np.ndarray       # anchors in the free object's body frame


def load_contact_reference(case, model, *, substeps=10, padding=200):
    """Map CA hand body names explicitly; resample without crossing contact switches.

    Discrete masks/identities are held at the preceding source frame. Positions
    interpolate only while the same body correspondence remains active at both
    endpoints. Thus a disabled point never interpolates towards a padded zero.
    """
    case = Path(case)
    source = mujoco.MjModel.from_xml_path(str(case / 'wuji_adapter/hand.xml'))
    with np.load(case / 'contactaware/contact_guidance.npz', allow_pickle=False) as z:
        mask = np.asarray(z['contact_mask'], np.float32)
        confidence = np.asarray(z['contact_weight'], np.float32)
        body = np.asarray(z['hand_query_body_ids'], np.int32)
        hand = np.asarray(z['hand_query_local_pos'], np.float32)
        obj = np.asarray(z['contact_pos_obj'], np.float32)
    if mask.ndim != 2 or not mask.size or confidence.shape != mask.shape or body.shape != mask.shape:
        raise ValueError('Contact masks, confidence and body IDs must have matching [T,C] shapes')
    if hand.shape != mask.shape + (3,) or obj.shape != hand.shape:
        raise ValueError('Contact points must have shape [T,C,3]')
    if not np.isfinite(mask).all() or not np.isfinite(confidence).all() or np.any(mask < 0) or np.any(confidence < 0):
        raise ValueError('Contact masks and confidence must be finite and nonnegative')
    weight = mask * confidence
    active = weight > 0
    if not active.any():
        raise ValueError('No active reference contacts')
    if not np.isfinite(hand[active]).all() or not np.isfinite(obj[active]).all():
        raise ValueError('Active reference contact points must be finite')
    mapped = np.zeros_like(body)
    for old in np.unique(body[active]):
        if old <= 0 or old >= source.nbody:
            raise ValueError(f'Invalid CA hand body ID {old}')
        name = source.body(int(old)).name
        if not name.startswith(('right_finger', 'right_palm')):
            raise ValueError(f'Contact body is not a hand body: {name}')
        mapped[(body == old) & active] = model.body(name).id
    hand = np.where(active[..., None], hand, 0)
    obj = np.where(active[..., None], obj, 0)
    frame = np.arange((len(mask) - 1) * substeps + 1) / substeps
    lo = np.floor(frame).astype(int)
    hi = np.minimum(lo + 1, len(mask) - 1)
    blend = (frame - lo)[:, None, None]
    same = active[lo] & active[hi] & (mapped[lo] == mapped[hi])
    blend = blend * same[..., None]
    def pad(x):
        return np.concatenate([x, np.repeat(x[-1:], padding, axis=0)])
    return ContactReference(
        pad(weight[lo]).astype(np.float32), pad(mapped[lo]),
        pad(hand[lo] * (1 - blend) + hand[hi] * blend).astype(np.float32),
        pad(obj[lo] * (1 - blend) + obj[hi] * blend).astype(np.float32),
    )


@wp.kernel
def contact_anchor_cost(
    xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
    object_body: int, clock: wp.array(dtype=int),
    weights: wp.array2d(dtype=float), bodies: wp.array2d(dtype=int),
    hand_local: wp.array2d(dtype=wp.vec3), object_local: wp.array2d(dtype=wp.vec3),
    scale: float, weight: float, out: wp.array(dtype=float),
):
    world = wp.tid()
    frame = wp.min(clock[0] + clock[1] + 1, weights.shape[0] - 1)
    total = float(0.0)
    error = float(0.0)
    for c in range(weights.shape[1]):
        confidence = weights[frame, c]
        if confidence > 0.0:
            body = bodies[frame, c]
            point = xpos[world, body] + xmat[world, body] * hand_local[frame, c]
            target = xpos[world, object_body] + xmat[world, object_body] * object_local[frame, c]
            delta = (point - target) / scale
            error += confidence * wp.dot(delta, delta)
            total += confidence
    if total > 0.0:
        out[world] += weight * error / total


class ContactAnchorTask(Task):
    """Mean squared material-point distance, normalized by a 10 mm length scale.

    The target follows the *predicted actual object*, rather than its desired
    trajectory, so the term penalizes slip/separation in the object's frame.
    It does not replace the existing object-tracking or physical-force costs.
    """
    def __init__(self, reference, *, weight=1.0, scale=.01):
        if not np.isfinite(weight) or weight <= 0 or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Positive finite contact weight and length scale required')
        self.reference, self.weight, self.scale = reference, weight, scale

    def register(self, sim):
        self.sim = sim

    def resolve(self, values, context):
        self.clock = context[SimContext.CLOCK]
        self.object_body = self.sim.mjm.body('tracked_object').id
        ref, device = self.reference, self.sim.device
        self.weights = wp.array(ref.weight, dtype=float, device=device)
        self.bodies = wp.array(ref.body, dtype=int, device=device)
        self.hand = wp.array(ref.hand_local, dtype=wp.vec3, device=device)
        self.obj = wp.array(ref.object_local, dtype=wp.vec3, device=device)

    def reset(self, sim):
        pass

    def compute_weighted_cost(self, values, *, out_cost):
        # mjw.step leaves body transforms at the pre-integration pose. Refresh
        # only GPU kinematics, preserving the just-computed constraint forces.
        self.sim.kinematics()
        d = self.sim._d
        wp.launch(contact_anchor_cost, dim=self.sim.nworld, inputs=[
            d.xpos, d.xmat, self.object_body, self.clock, self.weights,
            self.bodies, self.hand, self.obj, self.scale, self.weight, out_cost,
        ], device=self.sim.device)


def measure_contact_tracking(model, qpos, reference, *, start_frame=1):
    """Offline errors of the exact same material points, using MJWarp GPU FK."""
    from robokit.sim import MjWarpSim
    qpos = np.asarray(qpos, np.float32)
    n = len(qpos)
    end = start_frame + n
    if end > len(reference.weight):
        raise ValueError('Contact reference shorter than recorded states')
    with wp.ScopedDevice('cuda:0'):
        sim = MjWarpSim(model, n, mujoco.MjData(model), nconmax=1, njmax=1, device='cuda:0')
        with wp.ScopedStream(sim.stream):
            sim.data.qpos.assign(qpos)
            sim.kinematics()
            wp.synchronize_stream(sim.stream)
            pos, mat = sim._d.xpos.numpy(), sim._d.xmat.numpy()
    body = reference.body[start_frame:end]
    rows = np.arange(n)[:, None]
    hand = pos[rows, body] + np.einsum('ncij,ncj->nci', mat[rows, body], reference.hand_local[start_frame:end])
    obj = model.body('tracked_object').id
    anchors = pos[:, obj, None] + np.einsum('nij,ncj->nci', mat[:, obj], reference.object_local[start_frame:end])
    distance = np.linalg.norm(hand - anchors, axis=-1)
    weight = reference.weight[start_frame:end]
    active = weight > 0
    result = dict(
        weighted_rmse_mm=float(1000 * np.sqrt(np.sum(weight * distance**2) / np.sum(weight))),
        median_mm=float(1000 * np.median(distance[active])),
        p95_mm=float(1000 * np.percentile(distance[active], 95)),
        active_samples=int(active.sum()),
        measurement='Material-point separation from corresponding anchor on actual object; MJWarp GPU FK',
    )
    return result

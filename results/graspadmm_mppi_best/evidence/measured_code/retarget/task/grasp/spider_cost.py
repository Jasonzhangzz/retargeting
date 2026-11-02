"""SPIDER's one-hand tracking reward as positive batched rollout costs.

Derived from facebookresearch/spider commit
44717007de41cbef7565dff7ff9f4453557a2d3d (CC BY-NC 4.0;
Copyright Meta Platforms, Inc. and affiliates; see licenses/SPIDER.txt).
The retained source bodies are in retarget/task/humanoid/cost.py and math.py.

This is the ``right``/``left``, one free-object configuration convention:
scalar hand coordinates, object xyz, then object quaternion WXYZ. The state
cost is ONE weighted configuration L2 norm, plus velocity L2 and masked
contact-point L2 distances. Hand/object norms are non-additive diagnostics.
It does not reuse ForceAware's squared millimetre costs or virtual forces.

The optional ``contact_keep`` component is a LOCAL extension, not part of
original SPIDER. It scores actual measured per-finger normal force shortages
and does not invoke a ForceAware optimizer or infer forces from geometry.
"""

from dataclasses import dataclass, fields
import math

import torch


SPIDER_COMMIT = "44717007de41cbef7565dff7ff9f4453557a2d3d"
SOURCE = "retarget/task/humanoid/cost.py:get_reward; right/left embodiment"


@dataclass(frozen=True)
class SpiderCostSettings:
    """Original SPIDER scales, plus an explicitly opt-in local force term.

    The original reward only checks ``contact_rew_scale > 0``. It does NOT
    multiply contact distances by that scale. Enable it explicitly when
    supplying ContactAware correspondences, rather than changing this rule.
    ``contact_keep_rew_scale`` instead multiplies the local contact-keeping
    term; its zero default preserves the original SPIDER objective.
    """

    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0
    contact_rew_scale: float = 0.0
    contact_keep_rew_scale: float = 0.0
    contact_min_force_n: float = 0.05
    palm_keep_rew_scale: float = 0.0
    wrench_rew_scale: float = 0.0
    contact_worst_rew_scale: float = 0.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        if self.contact_min_force_n <= 0:
            raise ValueError("contact_min_force_n must be strictly positive")


def make_hand_scale(settings=SpiderCostSettings(), *, hand_dimension=26, device=None, dtype=torch.float32):
    """Allocate constant weights once before capturing a rollout CUDA graph."""
    if not isinstance(settings, SpiderCostSettings):
        raise TypeError("settings must be SpiderCostSettings")
    if isinstance(hand_dimension, bool) or not isinstance(hand_dimension, int) or hand_dimension < 6:
        raise ValueError("hand_dimension must include six scalar wrist coordinates")
    return torch.tensor([settings.base_pos_rew_scale]*3 + [settings.base_rot_rew_scale]*3
        + [settings.joint_rew_scale]*(hand_dimension-6), device=device, dtype=dtype)


def _reference(value, shape, like, name, *, mask=False):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if value.device != like.device or (value.dtype != like.dtype and not (mask and value.dtype == torch.bool)):
        raise ValueError(f"{name} must share the rollout dtype and device")
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as error:
        raise ValueError(f"{name} does not broadcast to {tuple(shape)}") from error


def quaternion_difference(actual, reference):
    """Branch-free form of the retained SPIDER WXYZ ``quat_sub`` helper.

    MuJoCo supplies unit quaternions. As in the source helper, this computes
    conjugate(reference) * actual and wraps its rotation vector to [-pi, pi].
    Quaternion sign changes therefore represent the same configuration.
    """
    aw, av = actual[..., :1], actual[..., 1:]
    bw, bv = reference[..., :1], reference[..., 1:]
    vector = bw * av - aw * bv - torch.cross(bv, av, dim=-1)
    scalar = bw * aw + (bv * av).sum(dim=-1, keepdim=True)
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, scalar)
    angle = torch.where(angle > torch.pi, angle - 2.0 * torch.pi, angle)
    tiny = torch.finfo(actual.dtype).tiny
    return torch.where(sine > 0.0, vector * angle / sine.clamp_min(tiny), torch.zeros_like(vector))


def step_cost(
    qpos,
    qvel,
    qpos_reference,
    qvel_reference,
    *,
    contact_positions=None,
    contact_target_world=None,
    contact_mask=None,
    finger_normal_forces=None,
    required_finger_mask=None,
    palm_normal_forces=None,
    wrench_support_cost=None,
    settings=SpiderCostSettings(),
    hand_dimension=26,
    hand_scale=None,
):
    """Return positive SPIDER costs from caller-supplied simulation observations.

    ``qpos`` is [B,...,hand_dimension+7], ``qvel`` is
    [B,...,hand_dimension+6]. References broadcast over the batch dimensions.
    Contact positions are [B,...,C,3]; targets/masks broadcast to that shape.
    CA object-local anchors must be transformed by the REFERENCE object pose
    before passing ``contact_target_world``, matching SPIDER's world targets.
    Contact geometry is scored exactly as supplied. The original post-step
    reward reads simulator site geometry, which may precede integration;
    a runner must record that observation stage rather than silently mixing
    in a new FK convention. This helper does not recompute body geometry.

    Pass a preallocated ``hand_scale=make_hand_scale(...)`` when capturing a
    CUDA graph, avoiding constant host-to-device tensor creation in capture.

    The helper never infers contact forces or success from geometric error.
    When ``contact_keep_rew_scale > 0``, actual ``finger_normal_forces`` and
    a deduplicated ``required_finger_mask`` are mandatory and broadcast to
    [B,...,5]. Forces share the rollout float dtype/device; masks are bool
    or the same float dtype/device with entries in [0,1]. Physics supplies one entry per finger,
    independently of the number of active CA contact slots. The LOCAL term
    is ``w * sum(mask * relu(1 - max(Fn,0)/f_min)**2)``. A required finger
    contributes w at zero/negative load and zero at/above f_min. Disabled
    contact keeping needs no force inputs and leaves original SPIDER exact.
    Invalid/nonfinite sampled states receive infinite total cost so that an
    MPPI finite-candidate check cannot select an invalid quaternion or state.
    Enabled force/mask inputs must also be finite, including inactive slots.
    """
    if not isinstance(settings, SpiderCostSettings):
        raise TypeError("settings must be SpiderCostSettings")
    if isinstance(hand_dimension, bool) or not isinstance(hand_dimension, int) or hand_dimension < 6:
        raise ValueError("hand_dimension must include six scalar wrist coordinates")
    if (not isinstance(qpos, torch.Tensor) or not qpos.is_floating_point()
            or qpos.ndim < 2 or qpos.shape[-1] != hand_dimension + 7):
        raise ValueError("qpos must contain batched scalar hand coordinates and one free-object pose")
    if (not isinstance(qvel, torch.Tensor) or qvel.shape != qpos.shape[:-1] + (hand_dimension + 6,)
            or qvel.device != qpos.device or qvel.dtype != qpos.dtype):
        raise ValueError("qvel must match qpos batch dimensions, dtype and device")
    qref = _reference(qpos_reference, qpos.shape, qpos, "qpos_reference")
    vref = _reference(qvel_reference, qvel.shape, qpos, "qvel_reference")
    hand = qpos[..., :hand_dimension] - qref[..., :hand_dimension]
    position = qpos[..., hand_dimension:hand_dimension+3] - qref[..., hand_dimension:hand_dimension+3]
    rotation = quaternion_difference(qpos[..., -4:], qref[..., -4:])
    if hand_scale is None:
        hand_scale = make_hand_scale(settings, hand_dimension=hand_dimension, device=qpos.device, dtype=qpos.dtype)
    else:
        hand_scale = _reference(hand_scale, (hand_dimension,), qpos, "hand_scale")
    hand_error = hand * hand_scale
    object_error = torch.cat((settings.pos_rew_scale * position, settings.rot_rew_scale * rotation), dim=-1)
    qpos_cost = torch.linalg.vector_norm(torch.cat((hand_error, object_error), dim=-1), dim=-1)
    qvel_cost = settings.vel_rew_scale * torch.linalg.vector_norm(qvel-vref, dim=-1)
    contact_cost = torch.zeros_like(qpos_cost)
    if settings.contact_rew_scale > 0.0:
        if (not isinstance(contact_positions, torch.Tensor) or contact_positions.ndim != qpos.ndim+1
                or contact_positions.shape[:-2] != qpos.shape[:-1] or contact_positions.shape[-1] != 3
                or contact_positions.device != qpos.device or contact_positions.dtype != qpos.dtype):
            raise ValueError("Enabled contact tracking requires contact_positions with rollout batch dimensions and [C,3]")
        targets = _reference(contact_target_world, contact_positions.shape, qpos, "contact_target_world")
        mask = _reference(contact_mask, contact_positions.shape[:-1], qpos, "contact_mask", mask=True)
        contact_cost = (torch.linalg.vector_norm(contact_positions-targets, dim=-1)*mask).sum(dim=-1)
    total = qpos_cost + qvel_cost + contact_cost
    palm_cost=torch.zeros_like(total)
    wrench_cost=torch.zeros_like(total)
    if settings.palm_keep_rew_scale>0:
        palm=_reference(palm_normal_forces,total.shape,qpos,'palm_normal_forces')
        palm_cost=settings.palm_keep_rew_scale*torch.relu(1.-palm.clamp_min(0)/settings.contact_min_force_n).square()
        total=total+palm_cost
    if settings.wrench_rew_scale>0:
        support=_reference(wrench_support_cost,total.shape,qpos,'wrench_support_cost')
        wrench_cost=settings.wrench_rew_scale*support
        total=total+wrench_cost
    contact_keep_cost = torch.zeros_like(qpos_cost)
    if settings.contact_keep_rew_scale > 0.0:
        if not isinstance(finger_normal_forces, torch.Tensor) or not isinstance(required_finger_mask, torch.Tensor):
            raise ValueError("Enabled contact keeping requires finger_normal_forces and required_finger_mask tensors")
        force_shape = qpos.shape[:-1] + (5,)
        finger_force = _reference(finger_normal_forces, force_shape, qpos, "finger_normal_forces")
        finger_mask = _reference(required_finger_mask, force_shape, qpos, "required_finger_mask", mask=True)
        shortage = torch.relu(1.0 - finger_force.clamp_min(0.0)/settings.contact_min_force_n)
        contact_keep_cost = settings.contact_keep_rew_scale * (finger_mask * shortage.square()).sum(dim=-1)
        total = total + contact_keep_cost
    valid = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
    valid = valid & torch.isfinite(qref).all(dim=-1) & torch.isfinite(vref).all(dim=-1)
    valid = valid & (torch.linalg.vector_norm(qpos[..., -4:], dim=-1) > 0)
    valid = valid & (torch.linalg.vector_norm(qref[..., -4:], dim=-1) > 0)
    if settings.contact_keep_rew_scale > 0.0:
        valid = valid & torch.isfinite(finger_force).all(dim=-1) & torch.isfinite(finger_mask).all(dim=-1)
        valid = valid & (finger_mask >= 0).all(dim=-1) & (finger_mask <= 1).all(dim=-1)
    total = torch.where(valid & torch.isfinite(total), total, torch.inf)
    return dict(total=total, qpos=qpos_cost, qvel=qvel_cost, contact=contact_cost, contact_keep=contact_keep_cost,
        palm_keep=palm_cost,wrench=wrench_cost,
        hand_diagnostic=torch.linalg.vector_norm(hand_error, dim=-1),
        object_diagnostic=torch.linalg.vector_norm(object_error, dim=-1))


def trajectory_cost(
    qpos,
    qvel,
    qpos_reference,
    qvel_reference,
    *,
    contact_positions=None,
    contact_target_world=None,
    contact_mask=None,
    finger_normal_forces=None,
    required_finger_mask=None,
    palm_normal_forces=None,
    wrench_support_cost=None,
    settings=SpiderCostSettings(),
    hand_dimension=26,
    hand_scale=None,
    temporal_weights=None,
):
    """Mean post-step cost [B,H,...], with original terminal reward scaling.

    The last state's TOTAL reward/cost is multiplied by terminal_rew_scale,
    then divided by H, as in the original SPIDER rollout. This is not a sum
    over q0, not a squared cost, and not normalized by the sum of weights.
    Enabled local ``contact_keep`` receives the same final-state scaling/H.
    """
    if not isinstance(qpos, torch.Tensor) or qpos.ndim != 3 or qpos.shape[1] < 1:
        raise ValueError("Trajectory qpos must have shape [batch, nonempty_horizon, coordinates]")
    steps = step_cost(qpos, qvel, qpos_reference, qvel_reference,
        contact_positions=contact_positions, contact_target_world=contact_target_world,
        contact_mask=contact_mask, finger_normal_forces=finger_normal_forces,
        required_finger_mask=required_finger_mask,palm_normal_forces=palm_normal_forces,
        wrench_support_cost=wrench_support_cost,settings=settings, hand_dimension=hand_dimension, hand_scale=hand_scale)
    if temporal_weights is None:
        temporal = qpos.new_ones(qpos.shape[1])
        temporal[-1] = settings.terminal_rew_scale
    else:
        if (temporal_weights.shape != (qpos.shape[1],) or temporal_weights.device != qpos.device
                or temporal_weights.dtype != qpos.dtype):
            raise ValueError("Cached temporal weights must match the horizon, dtype and device")
        temporal = temporal_weights
    result = {name: (value*temporal).mean(dim=1) for name, value in steps.items()
              if name not in ("hand_diagnostic", "object_diagnostic")}
    if settings.contact_worst_rew_scale>0:
        risk=settings.contact_worst_rew_scale*(steps['contact_keep']+steps['palm_keep']).amax(dim=1)
        result['contact_worst']=risk
        result['total']=result['total']+risk
    # Keep these non-additive diagnostics independent of terminal scaling,
    # matching the original rollout's unscaled mean-info convention.
    result.update({name: steps[name].mean(dim=1) for name in ("hand_diagnostic", "object_diagnostic")})
    # 0 * inf at a zero terminal weight must never make an invalid rollout
    # attractive to the sampler; invalid sampled states remain rejected.
    result["total"] = torch.where(torch.isfinite(steps["total"]).all(dim=1), result["total"], torch.inf)
    return result

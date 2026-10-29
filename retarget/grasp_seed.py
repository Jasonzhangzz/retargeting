"""GPU bone-direction seed for one independent static grasp input."""
from retarget.grasp_capture import capture_guard
import time
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from retarget.mano_noise import FINGERS, palm_frame
from retarget.grasp_template import WujiContactKinematics


def bone_seed_cuda(model, joints, *, device):
    """Fit all five four-DOF fingers together, without CPU least-squares.

    The neutral model supplies immutable palm geometry. Target coordinate-frame
    packing is preparation; all residuals, Jacobians, bounded LM updates and
    acceptance decisions run in one fixed-budget CUDA graph.
    """
    begun = time.perf_counter()
    human = np.asarray(joints)
    if human.shape[0] != 1:
        raise ValueError('Static grasp bone seeding expects one annotated frame')
    human = human[0]
    joint_ids = np.array([np.flatnonzero(model.jnt_qposadr == i)[0] for i in range(26)])
    low, high = model.jnt_range[joint_ids].T
    data = mujoco.MjData(model)
    neutral = np.clip(model.qpos0[:26], low+1.e-8, high-1.e-8)
    data.qpos[:26] = neutral
    mujoco.mj_kinematics(model, data)
    robot_roots = [data.xpos[model.body(f'right_finger{robot}_link2').id].copy()
                   for _, robot, _ in FINGERS.values()]
    robot_origin, robot_basis = palm_frame(np.zeros(3), np.asarray(robot_roots))
    roots = human[[value[2] for value in FINGERS.values()]]
    origin, basis = palm_frame(human[0], roots)
    rotation = basis@robot_basis.T
    base = np.clip(neutral, low+.1, high-.1)
    base[:3] = origin-rotation@robot_origin
    base[3:6] = Rotation.from_matrix(rotation).as_euler('XYZ')
    bodies, local, desired = [], [], []
    by_robot = {robot: (name, mano) for name, (_, robot, mano) in FINGERS.items()}
    for robot in range(1, 6):
        name, mano = by_robot[robot]
        ids = [model.body(f'right_finger{robot}_link{k}').id for k in (2, 3, 4)]
        tip = model.site(f'right_{name}_tip').id
        if model.site_bodyid[tip] != ids[-1]:
            raise ValueError('Expected Wuji fingertip on the distal finger body')
        bodies.extend([*ids, ids[-1]])
        local.extend([np.zeros(3), np.zeros(3), np.zeros(3), model.site_pos[tip]])
        vectors = np.diff(human[mano:mano+4], axis=0)
        desired.append(vectors/np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1.e-12))
    fk = WujiContactKinematics(model, np.array(bodies), device)
    root, positions, target = fk.tensor(base[:6]), fk.tensor(local), fk.tensor(desired)
    normals = torch.zeros_like(positions)
    # Fixed sampling is independent of the grasp optimizer and task RNG.
    rng = np.random.default_rng(0)
    fractions = np.r_[np.full((1, 20), .5), np.full((1, 20), .25),
                      np.full((1, 20), .75), rng.uniform(size=(4, 20))]
    starts = np.r_[base[None, 6:], low[None, 6:]+fractions*(high-low)[None, 6:]]
    q = fk.tensor(starts).reshape(8, 5, 4)
    original = q.clone()
    lo, hi = fk.tensor(low[6:]).reshape(5, 4), fk.tensor(high[6:]).reshape(5, 4)
    identity = torch.eye(4, device=fk.device, dtype=fk.dtype)
    damping = torch.full((8, 5), .01, device=fk.device, dtype=fk.dtype)

    def residual(value):
        p, _ = fk.contacts(torch.cat((root, value.flatten())), positions, normals)
        vector = p.reshape(5, 4, 3).diff(dim=1)
        unit = vector/vector.norm(dim=-1, keepdim=True).clamp(min=1.e-12)
        return (unit-target).reshape(5, 9)

    batched_residual = torch.vmap(residual)
    batched_jacobian = torch.vmap(torch.func.jacfwd(residual))

    def step():
        r = batched_residual(q)
        full = batched_jacobian(q)
        jacobian = torch.stack([full[:, i, :, i, :] for i in range(5)], dim=1)
        jt = jacobian.transpose(-1, -2)
        delta = torch.linalg.solve_ex(jt@jacobian+damping[..., None, None]*identity,
                                      -(jt@r[..., None]), check_errors=False)[0][..., 0]
        trial = (q+delta.clamp(-.25, .25)).clamp(lo, hi)
        cost = batched_residual(trial).square().sum(-1)
        accepted = torch.isfinite(trial).all(-1) & torch.isfinite(cost) & (cost < r.square().sum(-1))
        q.copy_(torch.where(accepted[..., None], trial, q))
        damping.copy_(torch.where(accepted, damping*.5, damping*4).clamp(1.e-8, 1.e8))

    stream = torch.cuda.Stream(device=fk.device)
    stream.wait_stream(torch.cuda.current_stream(fk.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_guard(), torch.cuda.graph(graph, stream=stream):
        q.copy_(original)
        damping.fill_(.01)
        for _ in range(96):
            step()
    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.current_stream(fk.device).wait_stream(stream)
    costs = batched_residual(q).square().mean(-1)
    costs = costs.masked_fill(~torch.isfinite(costs) | ~torch.isfinite(q).all(-1), torch.inf)
    selected = costs.argmin(0)
    fingers = torch.arange(5, device=fk.device)
    error = costs[selected, fingers].sqrt().cpu().numpy()
    if not np.isfinite(error).all():
        raise ValueError('No finite bone IK candidate')
    result = np.r_[base[:6], q[selected, fingers].flatten().cpu().numpy()]
    return result[None], dict(method='CUDA_batched_finger_bone_direction_LM',
        device=str(fk.device), iterations=96, fingers=5, root_fixed=True, multistarts=8,
        sampling_seed=0, selected_candidates=selected.cpu().tolist(),
        finger_direction_rms=error.tolist(), candidate_direction_rms=costs.sqrt().cpu().tolist(),
        chain_direction_rms=float(error.mean()), chain_direction_rms_max=float(error.max()),
        original_wuji_reference_used=False, object_or_contact_objective_used=False,
        seconds=time.perf_counter()-begun)

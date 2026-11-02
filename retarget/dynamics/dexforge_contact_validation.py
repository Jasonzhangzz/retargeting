"""An actual hand-object action-gradient checkpoint, independent of floor contact.

The checkpoint copies a physically touching frame from the unchanged original
reference. It is validation only and never replaces the controller's initial
state or edits the reference. Both object-pose and object-velocity outputs are
tested against the 29 real actuator controls, with no perturbed object inputs.
"""

from __future__ import annotations

import mujoco
import numpy as np


def _hand_object_slots(model, geom_pairs, hand_sites, object_body):
    pairs = np.asarray(geom_pairs, np.int32).reshape(-1, 2)
    bodies = model.geom_bodyid[pairs]
    hand_weld = model.body_weldid[model.site_bodyid[np.asarray(hand_sites, int)]]
    objects = np.zeros(model.nbody, bool)
    objects[object_body] = True
    for body in range(object_body+1, model.nbody):
        objects[body] = objects[int(model.body_parentid[body])]
    is_hand = np.isin(model.body_weldid[bodies], hand_weld)
    is_object = objects[bodies]
    return (is_hand[:, 0] & is_object[:, 1]) | (is_hand[:, 1] & is_object[:, 0])


def select_contact_frame(model, reference_qpos, *, hand_names=("left_palm", "right_palm"), object_body=None):
    """Choose a reference frame with robust physical hand-object contact.

    Prefer bilateral contact, then the largest minimum hand penetration depth.
    This only runs collision on original poses; there is no IK or pose repair.
    """
    sites = tuple(model.site(name).id for name in hand_names)
    if object_body is None:
        free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
        object_body = int(model.jnt_bodyid[free[-1]])
    hand_weld = model.body_weldid[model.site_bodyid[list(sites)]]
    data = mujoco.MjData(model)
    best, score = None, None
    for frame, qpos in enumerate(reference_qpos):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        pairs = np.asarray([contact.geom.copy() for contact in data.contact], np.int32).reshape(-1, 2)
        relevant = _hand_object_slots(model, pairs, sites, object_body)
        depths = np.zeros(2)
        contacts = []
        for index in np.flatnonzero(relevant):
            contact = data.contact[int(index)]
            if contact.dist > 0:
                continue
            weld = model.body_weldid[model.geom_bodyid[pairs[index]]]
            for hand in range(2):
                if hand_weld[hand] in weld:
                    depths[hand] = max(depths[hand], -float(contact.dist))
            contacts.append(dict(geoms=[model.geom(int(geom)).name for geom in contact.geom], distance=float(contact.dist)))
        touching = depths > 0
        if not touching.any():
            continue
        candidate = (int(touching.sum()), float(depths[touching].min()))
        if score is None or candidate > score:
            score = candidate
            best = dict(frame=int(frame), hand_count=int(touching.sum()), hand_depth_m=depths.tolist(),
                        original_hand_object_contact_count=len(contacts), original_contacts=contacts)
    return best


def _signature(workspace):
    collision = workspace.forward.base.collision
    result = [collision.contacts.active.numpy().copy()]
    for batch in collision.batches:
        selected = batch.selection
        for name in ("source_id", "target_id", "candidate_source", "candidate_target"):
            result.append(getattr(selected, name).numpy().copy())
        # The fourth coordinate stores the signed mesh feature kind.
        result.append(selected.candidate_axis.numpy()[..., 3].copy())
    return result


def _same_signature(first, second):
    return len(first) == len(second) and all(np.array_equal(a, b) for a, b in zip(first, second))


def validate_hand_object_control(prepared, *, epsilons=(.01, .003, .001, .0003),
                                 rtol=.01, atol=.002, directions=2, seed=0):
    """Check a real one-step control-to-object VJP and central differences."""
    if isinstance(directions, bool) or not isinstance(directions, int) or directions < 2:
        raise ValueError("hand-object validation requires at least two control directions")
    epsilons = tuple(epsilons)
    if len(epsilons) < 3 or len(set(epsilons)) != len(epsilons) or any(not np.isfinite(e) or e <= 0 for e in epsilons):
        raise ValueError("hand-object validation requires three distinct positive finite epsilons")
    if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) <= 0:
        raise ValueError("hand-object validation tolerances must be finite and positive")
    import warp as wp
    from comfree_warp.native_adjoint.dynamics import DynamicsInput
    from comfree_warp.native_adjoint.fast_step import compile_fast_step, clear_contact_status, check_contact_status
    from comfree_warp.native_adjoint.fast_step_adjoint import (
        BackwardCall, StepCotangent, allocate_workspace, record, forward, backward,
    )
    from comfree_warp.native_adjoint.runtime import StepInput
    from retarget.env.humanoid.dexforge_runner import _fd_assessment

    reference = prepared.reference
    original = prepared.artifacts.source_model
    checkpoint = select_contact_frame(original, reference.qpos)
    report = dict(kind="hand_object_control", horizon=1, passed=False,
                  checkpoint_only=True, controller_initial_state_changed=False,
                  objective="separate linear object pose and velocity outputs; only 29 actuator controls perturbed",
                  checkpoint=checkpoint, directional_tests=[])
    if checkpoint is None:
        report["reason"] = "original reference has no physical hand-object contact frame"
        return report
    cpu, device_model = prepared.models.planner_cpu, prepared.models.planner_device
    compiled = getattr(getattr(prepared, "optimizer", None), "compiled", None)
    if compiled is None:
        compiled = compile_fast_step(cpu, device_model)
    device, base = compiled.base.device, compiled.base
    frame = checkpoint["frame"]
    qpos, qvel = reference.initial_state(start_frame=frame)
    qpos, qvel = qpos.astype(np.float32), qvel.astype(np.float32)
    future = frame + float(cpu.opt.timestep)/reference.ref_dt
    controls = np.zeros(cpu.nu, np.float32)
    controls[reference.index.hand_ctrl] = reference.qpos_at([future])[0, reference.index.hand_qpos]
    matrix = lambda values: wp.array(np.asarray(values, np.float32), dtype=float, device=device,
                                    requires_grad=True, retain_grad=True)
    dynamics = DynamicsInput()
    dynamics.qpos, dynamics.qvel = matrix(qpos[None]), matrix(qvel[None])
    dynamics.ctrl = matrix(controls[None])
    dynamics.qfrc_applied = matrix(np.zeros((1, cpu.nv), np.float32))
    inputs = StepInput(dynamics, matrix(np.asarray([frame*reference.ref_dt], np.float32)), matrix(np.zeros(1)))
    workspace = allocate_workspace(compiled, 1)
    recorded = record(compiled, inputs, workspace, freeze_frame_vjp=False)
    zeros = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device=device)
    cotangent = StepCotangent(
        zeros((1,cpu.nq)), zeros((1,cpu.nv)), zeros((1,cpu.nv)), zeros(1), zeros((1,cpu.nv)),
        zeros((1,cpu.nbody),wp.vec3), zeros((1,cpu.nbody),wp.mat33),
        zeros((1,base.collision.contact_count)), zeros((1,base.collision.contact_count),wp.vec3),
        zeros((1,base.collision.contact_count),wp.mat33),
    )
    obj_q = int(reference.index.object_qpos)
    obj_joint = int(np.flatnonzero(cpu.jnt_qposadr == obj_q)[0])
    obj_v = int(cpu.jnt_dofadr[obj_joint])
    rng = np.random.default_rng(seed)
    # Position/linear velocity use millimetres to resolve float32 forward
    # quantization. Rotation components retain a documented fixed scale.
    pose_seed = rng.choice([-1.,1.], 7).astype(np.float32) * np.asarray([1000.]*3+[10.]*4,np.float32)
    velocity_seed = rng.choice([-1.,1.], 6).astype(np.float32) * np.asarray([1000.]*3+[10.]*3,np.float32)
    hand_sites = [cpu.site(name).id for name in ("left_palm", "right_palm")]
    geom_pairs = base.collision.contact_layout.contact_geom.numpy()
    slots = _hand_object_slots(cpu, geom_pairs, hand_sites, reference.index.object_body)

    def evaluate(control):
        dynamics.ctrl.assign(np.asarray(control, np.float32)[None])
        clear_contact_status(compiled, workspace.forward)
        forward(compiled, inputs, workspace)
        check_contact_status(compiled, workspace.forward)
        state_q = recorded.result.qpos.numpy()[0, obj_q:obj_q+7].astype(np.float64)
        state_v = recorded.result.qvel.numpy()[0, obj_v:obj_v+6].astype(np.float64)
        return dict(object_pose=float(state_q@pose_seed.astype(np.float64)),
                    object_velocity=float(state_v@velocity_seed.astype(np.float64))), _signature(workspace)

    scores, signature = evaluate(controls)
    active = workspace.forward.base.collision.contacts.active.numpy()[0].astype(bool)
    distance = recorded.result.contact_distance.numpy()[0]
    active_hands = slots & active
    penetrating = active_hands & (distance <= 0.)
    report.update(source_frame=frame, source_time_seconds=frame*reference.ref_dt,
                  native_timestep=float(cpu.opt.timestep), hand_object_contact_count=int(active_hands.sum()),
                  penetrating_hand_object_contact_count=int(penetrating.sum()),
                  hand_object_geom_pairs=[list(map(int, geom)) for geom in geom_pairs[active_hands]],
                  hand_object_geom_names=[[cpu.geom(int(g)).name for g in geom] for geom in geom_pairs[active_hands]],
                  hand_object_distance_m=distance[active_hands].tolist(),
                  object_pose_seed=pose_seed.tolist(), object_velocity_seed=velocity_seed.tolist())
    if not penetrating.any():
        report["reason"] = "selected original contact frame has no penetrating native hand-object collision"
        return report
    gradients = {}
    for name in ("object_pose", "object_velocity"):
        cotangent.qpos.zero_()
        cotangent.qvel.zero_()
        if name == "object_pose":
            value = np.zeros((1,cpu.nq),np.float32)
            value[0,obj_q:obj_q+7] = pose_seed
            cotangent.qpos.assign(value)
        else:
            value = np.zeros((1,cpu.nv),np.float32)
            value[0,obj_v:obj_v+6] = velocity_seed
            cotangent.qvel.assign(value)
        gradient = backward(BackwardCall(compiled, inputs, workspace, recorded, cotangent))
        gradients[name] = gradient.ctrl.numpy()[0].copy()
    norms = {name: float(np.linalg.norm(value)) for name,value in gradients.items()}
    report["control_gradient_norms"] = norms
    nonzero = all(np.isfinite(value) and value > 1e-7 for value in norms.values())
    tests = []
    for number in range(directions):
        direction = gradients["object_pose"].astype(float) if number == 0 else rng.standard_normal(cpu.nu)
        direction /= max(np.linalg.norm(direction),1e-30)
        trials = {name: [] for name in gradients}
        for epsilon in sorted(epsilons, reverse=True):
            values, signatures = [], []
            for sign in (-1.,1.):
                value, current = evaluate(controls+sign*epsilon*direction)
                values.append(value)
                signatures.append(current)
            stable = _same_signature(signature,signatures[0]) and _same_signature(signature,signatures[1])
            for name in gradients:
                trials[name].append(dict(epsilon=float(epsilon), derivative=(values[1][name]-values[0][name])/(2*epsilon),
                                         same_contact_active=stable))
        for name in gradients:
            derivative = float(gradients[name]@direction)
            assessment = _fd_assessment(derivative,trials[name],rtol=rtol,atol=atol)
            assessment.update(direction=number, output=name, same_geometry_features_required=True)
            tests.append(assessment)
    report.update(directional_tests=tests, nonzero_control_to_object=nonzero,
                  passed=bool(nonzero and all(test["passed"] for test in tests)),
                  reason="passed" if nonzero and all(test["passed"] for test in tests) else "nonzero hand-object control gradient/finite difference certification failed")
    return report

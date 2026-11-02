"""Independent native ForceAware grasp diagnostics, without running an optimizer.

Run with .venv-forceaware/bin/python.  Native dynamics and adjoints are the
byte-verified DexForge implementation; the linear probe is a derivative test,
not a replacement grasp objective or a certification of the FA objective.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import sys

import numpy as np

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _native_step_profile():
    """Describe the transition actually active in this diagnostic process."""
    extension = sys.modules.get("retarget.dynamics.dexforge_servo_integration")
    if extension is not None:
        metadata = extension.servo_integration_metadata()
        if metadata["active"]:
            return metadata["profile"], True
    return "original_native", False


def object_motion(qpos, object_qpos):
    """WXYZ free-object rotation; the hand's XYZ Euler root is untouched."""
    pose = np.asarray(qpos, np.float64)[:, object_qpos:object_qpos + 7]
    if pose.shape[1] != 7 or not np.isfinite(pose).all():
        raise ValueError("Expected finite full qpos containing the free object")
    quaternion = pose[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero free-object quaternion")
    quaternion = quaternion / norms[:, None]
    dot = np.clip(np.abs(quaternion @ quaternion[0]), 0., 1.)
    return dict(object_position_m=pose[:, :3].copy(), object_quaternion_wxyz=quaternion,
                object_displacement_mm=np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1) * 1000.,
                object_rotation_rad=2 * np.arccos(dot))


def decode_contact_forces(row_contact, row_edge, friction, frames, row_force, active):
    """Decode original pyramidal rows as force on geom2, and normal pressure."""
    frames, row_force = np.asarray(frames), np.asarray(row_force)
    active = np.asarray(active, bool)
    if frames.shape != (*active.shape, 3, 3):
        raise ValueError("Contact frame shape does not match active slots")
    force = np.zeros((*active.shape, 3), np.float64)
    normal = np.zeros(active.shape, np.float64)
    for row, (contact, edge) in enumerate(zip(row_contact, row_edge)):
        if contact < 0 or edge < 0:  # Joint limits are not contact rays.
            continue
        ray = frames[:, contact, 0].astype(np.float64).copy()
        axis = int(edge) // 2
        if axis < 2:
            ray += (1. if edge % 2 == 0 else -1.) * friction[contact, axis] * frames[:, contact, axis + 1]
        force[:, contact] += row_force[:, row, None] * ray
        normal[:, contact] += row_force[:, row]
    return np.where(active[..., None], force, 0.), np.where(active, normal, 0.)


def finger_body_labels(cpu):
    """Include fixed sensor/pad descendants in their physical finger chain."""
    labels = np.full(cpu.nbody, -1, np.int32)
    for body in range(1, cpu.nbody):
        owner = body
        while owner:
            match = re.search(r"right_finger([1-5])_link", cpu.body(owner).name)
            if match:
                labels[body] = int(match.group(1)) - 1
                break
            owner = int(cpu.body_parentid[owner])
    return labels


def finger_contact_metrics(contact_geom, geom_body, body_finger, object_body,
                           collision_active, distances, force_on_geom2, normal_force):
    """Count actual object contacts separately from their normal load."""
    pairs = np.asarray(geom_body, int)[np.asarray(contact_geom, int)]
    counts = np.zeros((len(distances), 5), np.int32)
    gap = np.full((len(distances), 5), np.nan)
    pressure = np.zeros((len(distances), 5), np.float64)
    object_force = np.zeros((len(distances), 3), np.float64)
    for contact, (first, second) in enumerate(pairs):
        if first == object_body:
            hand, sign = second, -1.
        elif second == object_body:
            hand, sign = first, 1.
        else:
            continue
        object_force += sign * force_on_geom2[:, contact]
        finger = int(body_finger[hand])
        if finger < 0:
            continue
        enabled = np.asarray(collision_active[:, contact], bool)
        counts[:, finger] += enabled & (normal_force[:, contact] > 1e-6)
        pressure[:, finger] += np.maximum(normal_force[:, contact], 0.)
        for step in np.flatnonzero(enabled):
            value = distances[step, contact] * 1000.
            gap[step, finger] = value if np.isnan(gap[step, finger]) else min(value, gap[step, finger])
    return dict(finger_normal_force_n=pressure, finger_loaded_contact_count=counts,
                finger_min_native_gap_mm=gap, object_contact_force_world_n=object_force)


def constant_native_control(index, qpos, nu, control=None):
    """Map a finite constant FA hand command to the native actuator layout.

    When hand/action and full-actuator dimensions coincide, the supplied
    command uses FA hand-action order, as do official saved control arrays.
    This helper never changes qpos or the tracked hand reference.
    """
    hand_ctrl, hand_qpos = np.asarray(index.hand_ctrl), np.asarray(index.hand_qpos)
    if control is None:
        values = np.asarray(qpos, np.float32)[hand_qpos]
    else:
        values = np.asarray(control, np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Constant native control must contain finite values")
    if values.shape == (len(hand_ctrl),):
        native = np.zeros(nu, np.float32)
        native[hand_ctrl] = values
        return native
    if values.shape == (nu,):
        return values.copy()
    raise ValueError(f"Constant native control must have shape ({len(hand_ctrl)},) in hand order or ({nu},) in actuator order")


def native_baseline(cpu, device_model, index, qpos, qvel, *, steps, control=None,
                    controls=None, guidance=None):
    """Recompute native collisions and integrate q/v on every physics step."""
    import warp as wp
    from comfree_warp.native_adjoint.dynamics import DynamicsInput
    from comfree_warp.native_adjoint.fast_step import (
        allocate_workspace, compile_fast_step, clear_contact_status, check_contact_status, step,
    )
    from comfree_warp.native_adjoint.runtime import StepInput

    q0, v0 = np.asarray(qpos, np.float32), np.asarray(qvel, np.float32)
    if q0.shape != (cpu.nq,) or v0.shape != (cpu.nv,) or not np.isfinite(q0).all() or not np.isfinite(v0).all():
        raise ValueError("Invalid native initial qpos/qvel")
    if steps < 1:
        raise ValueError("At least one native step is required")
    compiled = compile_fast_step(cpu, device_model)
    device = compiled.base.device
    matrix = lambda value: wp.array(np.asarray(value, np.float32), dtype=float, device=device)
    inputs_dynamics = DynamicsInput()
    inputs_dynamics.qpos, inputs_dynamics.qvel = matrix(q0[None]), matrix(v0[None])
    u = constant_native_control(index, q0, cpu.nu, control)
    if controls is None:
        control_sequence = np.repeat(u[None], steps, axis=0)
    else:
        control_sequence = np.asarray(controls, np.float32)
        if control is not None or control_sequence.shape != (steps, cpu.nu) or not np.isfinite(control_sequence).all():
            raise ValueError("Native replay requires finite per-physics-step controls[steps,nu]")
    inputs_dynamics.ctrl = matrix(u[None])
    inputs_dynamics.qfrc_applied = matrix(np.zeros((1, cpu.nv)))
    inputs = StepInput(inputs_dynamics, matrix(np.zeros(1)), matrix(np.zeros(1)))
    workspace = allocate_workspace(compiled, 1)
    arrays = {name: [] for name in ("qpos", "qvel", "time", "contact_distance", "contact_position",
        "contact_frame", "contact_collision_active", "contact_row_force", "generalized_constraint_force")}
    if guidance is not None:
        arrays.update(guidance_query_world_m=[], guidance_target_world_m=[])
    for step_index in range(steps):
        inputs_dynamics.ctrl.assign(control_sequence[step_index:step_index + 1])
        clear_contact_status(compiled, workspace)
        result = step(compiled, inputs, workspace)
        check_contact_status(compiled, workspace)
        for name in ("qpos", "qvel", "time", "contact_distance", "contact_position", "contact_frame"):
            arrays[name].append(getattr(result, name).numpy()[0].copy())
        arrays["contact_collision_active"].append(workspace.base.collision.contacts.active.numpy()[0].copy())
        arrays["contact_row_force"].append(workspace.contact.output.force.numpy()[0].copy())
        arrays["generalized_constraint_force"].append(result.constraint_force.numpy()[0].copy())
        if guidance is not None:
            position, rotation = result.body_position.numpy()[0], result.body_matrix.numpy()[0]
            bodies, local = guidance.query_body[0], guidance.query_local[0]
            arrays["guidance_query_world_m"].append(position[bodies] + np.einsum("kij,kj->ki", rotation[bodies], local))
            arrays["guidance_target_world_m"].append(position[index.object_body]
                + guidance.position_object[0] @ rotation[index.object_body].T)
        if not all(np.isfinite(arrays[name][-1]).all() for name in ("qpos", "qvel", "time")):
            break  # Preserve the first failing state rather than feeding NaNs into collision.
        inputs_dynamics.qpos.assign(arrays["qpos"][-1][None])
        inputs_dynamics.qvel.assign(arrays["qvel"][-1][None])
        inputs.time.assign(np.asarray([arrays["time"][-1]], np.float32))
    arrays = {name: np.stack(value) for name, value in arrays.items()}
    parameters = compiled.base.contact_rows.parameters
    arrays["contact_geom_ids"] = parameters.contact_geom.numpy().copy()
    margin = parameters.contact_includemargin.numpy()
    arrays["contact_active"] = arrays["contact_collision_active"].astype(bool) & (arrays["contact_distance"] < margin)
    force, pressure = decode_contact_forces(parameters.row_contact.numpy(), parameters.row_edge.numpy(),
        parameters.contact_friction.numpy(), arrays["contact_frame"], arrays["contact_row_force"], arrays["contact_active"])
    arrays["contact_force_on_geom2_n"], arrays["contact_normal_force_n"] = force, pressure
    arrays.update(finger_contact_metrics(arrays["contact_geom_ids"], cpu.geom_bodyid, finger_body_labels(cpu),
        index.object_body, arrays["contact_collision_active"], arrays["contact_distance"], force, pressure))
    arrays["qpos_with_initial"] = np.vstack((q0, arrays["qpos"]))
    arrays["qvel_with_initial"] = np.vstack((v0, arrays["qvel"]))
    arrays["control"] = control_sequence[:len(arrays["qpos"])].copy()
    if guidance is not None:
        arrays["guidance_active_mask"] = guidance.mask[0].copy()
        arrays["guidance_body_ids"] = guidance.query_body[0].copy()
        arrays["guidance_native_output_index"] = guidance.output_index[0].copy()
        arrays["guidance_query_local_m"] = guidance.query_local[0].copy()
        arrays["guidance_target_distance_mm"] = np.linalg.norm(
            arrays["guidance_query_world_m"] - arrays["guidance_target_world_m"], axis=-1) * 1000.
    finite = all(np.isfinite(arrays[name]).all() for name in ("qpos", "qvel", "contact_row_force", "contact_force_on_geom2_n"))
    if finite:
        arrays.update(object_motion(arrays["qpos_with_initial"], index.object_qpos))
    else:
        pose = arrays["qpos_with_initial"][:, index.object_qpos:index.object_qpos + 7]
        arrays.update(object_position_m=pose[:, :3], object_quaternion_wxyz=pose[:, 3:7],
            object_displacement_mm=np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1) * 1000.,
            object_rotation_rad=np.full(len(pose), np.nan))
    kind = "supplied_control_replay" if controls is not None else "supplied_constant_preload" if control is not None else "constant_CA_reference_control"
    control_source = "supplied per-physics-step native actuator controls" if controls is not None else "caller-supplied constant preload command" if control is not None else "accepted CA hand reference positions"
    summary = dict(kind=kind, control_source=control_source,
        optimizer_run=False, finite=bool(finite),
        steps=len(arrays["qpos"]), requested_steps=steps, dt=float(cpu.opt.timestep),
        duration_seconds=len(arrays["qpos"]) * float(cpu.opt.timestep),
        initial_state="unmodified CA qpos, supplied qvel" if controls is None else "caller-supplied qpos/qvel",
        collision_stage="pre_integration; qpos/qvel are post_integration",
        finger_names=FINGER_NAMES, object_displacement_max_mm=float(arrays["object_displacement_mm"].max()),
        object_rotation_max_rad=float(arrays["object_rotation_rad"].max()),
        qvel_abs_max=float(np.abs(arrays["qvel"]).max()),
        physical_penetration_max_mm=float(np.maximum(-arrays["contact_distance"][arrays["contact_active"]], 0.).max(initial=0.) * 1000.),
        finger_normal_force_min_n=arrays["finger_normal_force_n"].min(0).tolist(),
        finger_normal_force_max_n=arrays["finger_normal_force_n"].max(0).tolist(),
        finger_has_positive_force_all_steps=(arrays["finger_normal_force_n"] > 1e-6).all(0).tolist(),
        native_qvel_state=True, native_collision_every_step=True,
        body_mass_total_kg=float(cpu.body_mass.sum()),
        dof_damping_nonzero=int(np.count_nonzero(cpu.dof_damping)),
        actuator_position_gain_nonzero=int(np.count_nonzero(cpu.actuator_gainprm[:, 0])),
        actuator_velocity_bias_nonzero=int(np.count_nonzero(cpu.actuator_biasprm[:, 2])))
    if not finite:
        bad = np.flatnonzero(~np.isfinite(arrays["qpos"]).all(1) | ~np.isfinite(arrays["qvel"]).all(1))
        summary["first_nonfinite_post_step"] = None if not len(bad) else int(bad[0] + 1)
    if guidance is not None:
        active = guidance.mask[0] > 0
        summary.update(guidance_active_count=int(active.sum()),
            guidance_active_target_distance_max_mm=float(arrays["guidance_target_distance_mm"][:, active].max()),
            geometry_force_mapping="guidance body to unique whole-hand GS output; normal forces from physical object contact rows")
    return summary, arrays


def expand_controls(controls, hand_ctrl, nu, *, control_dt, exec_dt):
    """Replay official planner controls for the exact configured executor substeps."""
    values = np.asarray(controls, np.float32)
    ratio = float(control_dt) / float(exec_dt)
    repeats = round(ratio)
    if not np.isfinite(ratio) or repeats < 1 or not np.isclose(ratio, repeats, rtol=0., atol=1e-7):
        raise ValueError("Control interval must be a positive integer multiple of executor dt")
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite controls[T,action_dim or nu]")
    if values.shape[1] == len(hand_ctrl):
        full = np.zeros((len(values), nu), np.float32)
        full[:, np.asarray(hand_ctrl)] = values
    elif values.shape[1] == nu:
        full = values
    else:
        raise ValueError("Control dimension disagrees with native actuator layout")
    return np.repeat(full, repeats, axis=0)


def audit_native_controls(artifacts, cfg, output_dir, *, qpos, qvel, controls, control_dt=None):
    """Read-only true-dynamics replay of an official FA result, including init correction."""
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment
    require_environment(); activate_upstream()
    import comfree_warp
    from forceaware.workflow import _load_runtime_models
    from retarget.env.grasp.forceaware_reference import load_reference
    from retarget.common.results import write_json

    models = _load_runtime_models(cfg, comfree_warp)
    reference = load_reference(artifacts, models.planner_cpu, models.planner_device, cfg)
    grid = cfg.simulator.time_grid
    expanded = expand_controls(controls, reference.index.hand_ctrl, models.executor.cpu_model.nu,
        control_dt=grid.mpc_dt if control_dt is None else control_dt, exec_dt=grid.exec_dt)
    summary, arrays = native_baseline(models.executor.cpu_model, models.executor.device_model,
        reference.index, qpos, qvel, steps=len(expanded), controls=expanded, guidance=reference.sequence.contact)
    summary.update(control_dt=grid.mpc_dt if control_dt is None else float(control_dt),
                   control_source="supplied official FA controls; initial state supplied by caller",
                   force_source="original native physical contact row pressure, grouped by finger body")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "native_replay.npz", **arrays)
    write_json(output_dir / "replay.json", summary)
    return summary


def _native_program_branch_signatures(compiled, workspace, inputs, *, world=0):
    """Read the actual hard-solver and actuator clipping branches.

    Collision features alone do not identify the differentiated program:
    controls change every hard Newton active set even at a fixed contact pose.
    """
    contact = workspace.forward.contact
    digest = hashlib.sha256()
    digest.update(workspace.forward.base.contact_rows.rows.active.numpy()[world].tobytes())
    for state in contact.states:
        digest.update((state.rows.slope.numpy()[world] > 0.).tobytes())
        if hasattr(state, "pcg"):
            digest.update(state.pcg.iterations.numpy()[world:world+1].tobytes())
            digest.update(state.pcg.status.numpy()[world:world+1].tobytes())
    # The original hard final projection is max(raw,0); reading its actual
    # output avoids re-evaluating the threshold with a different host FMA.
    digest.update((contact.output.force.numpy()[world] > 0.).tobytes())
    actuator_digest = hashlib.sha256()
    model = compiled.base.dynamics.model
    dofs = model.actuator_dofid.numpy()
    gear = model.actuator_gear.numpy()[world % model.actuator_gear.shape[0], :, 0]
    if len(np.unique(dofs)) != len(dofs) or np.any(gear == 0.):
        raise ValueError("Grasp native gradient checks require one scalar actuator per controlled DOF")
    # Each WUJI scalar DOF has one actuator. Its native generalized force is
    # the exact clipped actuator force times gear, not a host reconstruction.
    force = workspace.forward.base.dynamics.actuator_force.numpy()[world, dofs] / gear
    force_limits = model.actuator_forcerange.numpy()[world % model.actuator_forcerange.shape[0]]
    force_limited = model.actuator_forcelimited.numpy().astype(bool)
    actuator_digest.update((force_limited & (force <= force_limits[:, 0])).tobytes())
    actuator_digest.update((force_limited & (force >= force_limits[:, 1])).tobytes())
    ctrl = inputs.dynamics.ctrl.numpy()[world]
    ctrl_limits = model.actuator_ctrlrange.numpy()[world % model.actuator_ctrlrange.shape[0]]
    ctrl_limited = model.actuator_ctrllimited.numpy().astype(bool) & bool(model.clamp_control)
    actuator_digest.update((ctrl_limited & (ctrl <= ctrl_limits[:, 0])).tobytes())
    actuator_digest.update((ctrl_limited & (ctrl >= ctrl_limits[:, 1])).tobytes())
    return digest.hexdigest(), actuator_digest.hexdigest()


def _native_fd_assessment(adjoint, trials, *, rtol, atol, max_control_direction_error=.02):
    """Select FD plateaus independently of AD, within the same native branch."""
    from retarget.env.humanoid.dexforge_runner import _fd_assessment

    gated = []
    for trial in trials:
        eligible = (trial.get("same_contact_solver_branch", False)
            and trial.get("same_actuator_branch", False)
            and trial.get("control_direction_error_norm", float("inf")) <= max_control_direction_error)
        gated.append(dict(trial, same_contact_features=trial["same_contact_features"] and eligible))
    result = _fd_assessment(adjoint, gated, rtol=rtol, atol=atol)
    result["trials"] = trials
    result["max_control_direction_error_norm"] = max_control_direction_error
    result["outcome"] = "passed" if result["passed"] else "mismatch_on_same_branch" if result["selected_epsilon"] is not None else "inconclusive"
    if result["selected_epsilon"] is None and np.isfinite(adjoint):
        result["reason"] = "no stable finite-difference plateau preserving collision features, hard solver/actuator branches and resolved float32 controls"
    return result


# Read-only public helpers shared by the independent official-objective audit.
native_program_branch_signatures = _native_program_branch_signatures
native_fd_assessment = _native_fd_assessment


def native_contact_signature(compiled, workspace, *, world=0):
    """Hash active native collision IDs and their used static mesh features."""
    from retarget.env.humanoid.dexforge_runner import _contact_signature
    return _contact_signature(compiled, workspace, world=world)


def native_control_gradient_probe(cpu, device_model, index, qpos, qvel, *, horizon,
                                  directions=2, epsilons=(.01, .003, .001, .0003, .0001, .00003, .00001, .000003),
                                  rtol=.05, atol=.002, seed=177, control=None):
    """Actual full native-step adjoint vs controls-only central differences."""
    import warp as wp
    from comfree_warp.native_adjoint.dynamics import DynamicsInput
    from comfree_warp.native_adjoint.fast_step import compile_fast_step, clear_contact_status, check_contact_status
    from comfree_warp.native_adjoint.fast_step_adjoint import BackwardCall, StepCotangent, allocate_workspace, record, forward, backward
    from comfree_warp.native_adjoint.runtime import StepInput

    profile, integration_extended = _native_step_profile()
    compiled = compile_fast_step(cpu, device_model)
    device = compiled.base.device
    matrix = lambda value: wp.array(np.asarray(value, np.float32), dtype=float, device=device, requires_grad=True, retain_grad=True)
    qbase, vbase = np.asarray(qpos, np.float32).copy(), np.asarray(qvel, np.float32).copy()
    # Deliberately no hand-root quaternion normalization: WUJI uses 6 scalars.
    initial_q, initial_v = matrix(qbase[None]), matrix(vbase[None])
    current_q, current_v, current_time = initial_q, initial_v, matrix(np.zeros(1))
    command = constant_native_control(index, qbase, cpu.nu, control)
    ubase = np.repeat(command[None], horizon, axis=0)
    control_source = "accepted CA hand reference positions" if control is None else "caller-supplied constant preload command"
    stages = []
    for step_index in range(horizon):
        dynamics = DynamicsInput()
        dynamics.qpos, dynamics.qvel = current_q, current_v
        dynamics.ctrl = matrix(ubase[step_index:step_index + 1])
        dynamics.qfrc_applied = matrix(np.zeros((1, cpu.nv)))
        inputs = StepInput(dynamics, current_time, matrix(np.zeros(1)))
        workspace = allocate_workspace(compiled, 1)
        recorded = record(compiled, inputs, workspace, freeze_frame_vjp=False)
        zeros = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device=device)
        count = compiled.base.collision.contact_count
        cotangent = StepCotangent(zeros((1, cpu.nq)), zeros((1, cpu.nv)), zeros((1, cpu.nv)), zeros(1),
            zeros((1, cpu.nv)), zeros((1, cpu.nbody), wp.vec3), zeros((1, cpu.nbody), wp.mat33),
            zeros((1, count)), zeros((1, count), wp.vec3), zeros((1, count), wp.mat33))
        stages.append((inputs, workspace, recorded, cotangent))
        current_q, current_v, current_time = recorded.result.qpos, recorded.result.qvel, recorded.result.time
    rng = np.random.default_rng(seed)
    qseed = rng.choice([-1., 1.], cpu.nq).astype(np.float32)
    vseed = (.125 * rng.choice([-1., 1.], cpu.nv)).astype(np.float32)

    def evaluate(controls):
        initial_q.assign(qbase[None]); initial_v.assign(vbase[None])
        signatures, branches, actuator_branches, counts = [], [], [], []
        for step_index, (inputs, workspace, recorded, _) in enumerate(stages):
            inputs.dynamics.ctrl.assign(controls[step_index:step_index + 1].astype(np.float32))
            clear_contact_status(compiled, workspace.forward)
            forward(compiled, inputs, workspace)
            check_contact_status(compiled, workspace.forward)
            signature, count = native_contact_signature(compiled, workspace)
            signatures.append(signature); counts.append(count)
            branch, actuator_branch = _native_program_branch_signatures(compiled, workspace, inputs)
            branches.append(branch); actuator_branches.append(actuator_branch)
        final = stages[-1][2].result
        host_q, host_v = final.qpos.numpy()[0].astype(np.float64), final.qvel.numpy()[0].astype(np.float64)
        score = float(np.dot(host_q, qseed) + np.dot(host_v, vseed))
        return score, signatures, counts, branches, actuator_branches, host_q, host_v

    score, signature, contacts, branch, actuator_branch, _, _ = evaluate(ubase)
    if not np.isfinite(score):
        return dict(kind="native_control_chain", horizon=horizon, score=score,
            active_contacts_per_step=contacts, has_actual_contact=bool(any(contacts)), passed=False,
            reason="unperturbed native chain is nonfinite; no finite derivative exists for this probe",
            directional_tests=[], initial_state_optimized=False, objective_validation=False,
            native_step_profile=profile, native_integration_extended=integration_extended,
            control_source=control_source, constant_control=command.tolist())
    stages[-1][3].qpos.assign(qseed[None]); stages[-1][3].qvel.assign(vseed[None])
    gradients = [None] * horizon
    for step_index in reversed(range(horizon)):
        inputs, workspace, recorded, cotangent = stages[step_index]
        gradient = backward(BackwardCall(compiled, inputs, workspace, recorded, cotangent))
        gradients[step_index] = gradient.ctrl.numpy()[0].copy()
        if step_index:
            wp.copy(stages[step_index - 1][3].qpos, gradient.qpos)
            wp.copy(stages[step_index - 1][3].qvel, gradient.qvel)
    tests = []
    for direction_index in range(directions):
        du = rng.standard_normal(ubase.shape)
        du /= np.linalg.norm(du)
        derivative = float(np.sum(np.asarray(gradients) * du))
        trials = []
        for epsilon in epsilons:
            low_control, high_control = (ubase - epsilon*du).astype(np.float32), (ubase + epsilon*du).astype(np.float32)
            low = evaluate(low_control)
            high = evaluate(high_control)
            unchanged = low[1] == signature and high[1] == signature
            # Subtract each actual float32 output coordinate in host float64
            # before reduction; never subtract two large rounded scalar losses.
            fd = float((np.dot(high[5]-low[5], qseed) + np.dot(high[6]-low[6], vseed)) / (2*epsilon))
            effective_du = (high_control.astype(np.float64)-low_control.astype(np.float64))/(2*epsilon)
            trials.append(dict(epsilon=float(epsilon), derivative=fd,
                same_contact_active=unchanged, same_contact_features=unchanged,
                same_contact_solver_branch=low[3] == branch and high[3] == branch,
                same_actuator_branch=low[4] == actuator_branch and high[4] == actuator_branch,
                control_direction_error_norm=float(np.linalg.norm(effective_du-du)),
                quantized_direction_adjoint=float(np.sum(np.asarray(gradients)*effective_du))))
        assessment = _native_fd_assessment(derivative, trials, rtol=rtol, atol=atol)
        assessment["direction"] = direction_index
        tests.append(assessment)
    return dict(kind="native_control_chain", horizon=horizon, score=score,
        active_contacts_per_step=contacts, has_actual_contact=bool(any(contacts)),
        passed=bool(any(contacts) and all(test["passed"] for test in tests)), directional_tests=tests,
        initial_state_optimized=False, initial_qpos_unchanged=bool(np.array_equal(qbase, np.asarray(qpos, np.float32))),
        objective=f"linear final qpos/qvel cotangent of {profile} transition; controls only", objective_validation=False,
        control_source=control_source, constant_control=command.tolist(),
        native_step_profile=profile, native_integration_extended=integration_extended,
        finite_difference_reduction="host float64 difference of actual float32 output coordinates",
        branch_policy="collision features, every hard Newton slope, final projection, row activation and actuator clipping",
        diagnostic_revision="full_native_branch_and_float32_resolution")


def diagnose_native(artifacts, cfg, output_dir, *, duration_seconds=.1, horizon=10,
                    directions=1, skip_gradients=False, control=None):
    """Diagnostic only: preserve original config, optimizer and initial-state policy.

    A gradient failure is recorded, rather than disguised as certification or
    used to replace the official optimizer. A nonfinite baseline is explicit.
    """
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment, verify_upstream
    versions = require_environment(); activate_upstream()
    import warp as wp
    import comfree_warp
    from forceaware.workflow import _load_runtime_models
    from retarget.env.grasp.forceaware_reference import load_reference
    from retarget.common.results import clean_json, write_json

    wp.init()
    models = _load_runtime_models(cfg, comfree_warp)
    reference = load_reference(artifacts, models.planner_cpu, models.planner_device, cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(upstream_commit=verify_upstream()["commit"], versions=versions,
                  native_scene=str(Path(artifacts.xml).resolve()),
                  reference_sequence=str(Path(artifacts.sequence_dir).resolve()),
                  certified=False, optimizer_replaced=False)
    report["native_step_profile"], report["native_integration_extended"] = _native_step_profile()
    steps = max(1, int(round(duration_seconds / cfg.simulator.time_grid.exec_dt)))
    print(f"[grasp FA validation] constant-control native {steps} steps", flush=True)
    try:
        report["baseline"], arrays = native_baseline(models.executor.cpu_model, models.executor.device_model,
            reference.index, reference.qpos, reference.qvel, steps=steps, guidance=reference.sequence.contact,
            control=control)
    except Exception as error:
        report.update(finite=False, native_control_gradient_passed=False,
                      baseline=dict(finite=False, error_type=type(error).__name__, error=str(error)))
        write_json(output_dir / "validation.json", report)
        return report
    np.savez_compressed(output_dir / "native_baseline.npz", **arrays)
    write_json(output_dir / "validation.json", report)
    report["gradient_probes"] = []
    if not skip_gradients:
        for length in sorted(set((1, horizon))):
            print(f"[grasp FA validation] native control VJP horizon={length}", flush=True)
            try:
                probe = native_control_gradient_probe(models.planner_cpu, models.planner_device,
                    reference.index, reference.qpos, reference.qvel, horizon=length, directions=directions,
                    control=control)
            except Exception as error:
                probe = dict(kind="native_control_chain", horizon=length, passed=False,
                             error_type=type(error).__name__, error=str(error))
            report["gradient_probes"].append(probe)
            write_json(output_dir / "validation.json", report)
        report["native_control_gradient_passed"] = all(probe["passed"] for probe in report["gradient_probes"])
    else:
        report["native_control_gradient_passed"] = False
        report["gradient_skip_reason"] = "explicit diagnostic-only request"
    report["finite"] = report["baseline"]["finite"]
    write_json(output_dir / "validation.json", report)
    print(json.dumps(clean_json({"baseline": report["baseline"], "native_control_gradient_passed": report["native_control_gradient_passed"]})), flush=True)
    return report

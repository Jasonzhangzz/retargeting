"""Read-only diagnostics from the selected native ForceAware trajectory."""

import numpy as np


def initial_correction(qpos_before, qpos_after, index, *, bound, allow_init=True, atol=1e-6):
    before, after = np.asarray(qpos_before), np.asarray(qpos_after)
    if before.shape != after.shape or before.ndim != 1 or not np.isfinite(after).all():
        raise ValueError("Invalid ForceAware initial correction states")
    joints = np.asarray(index.hand_qpos, dtype=int)
    delta = after - before
    fixed = np.ones(len(before), dtype=bool)
    fixed[joints] = False
    if np.any(np.abs(delta[fixed]) > atol):
        raise RuntimeError("ForceAware initial correction changed the fixed root or object state")
    maximum = float(np.max(np.abs(delta[joints]), initial=0.0))
    limit = float(bound) if allow_init else 0.0
    if maximum > limit + atol:
        raise RuntimeError("ForceAware initial joint correction exceeds its allowed bound")
    return dict(initial_joint_delta=delta[joints].tolist(), initial_joint_delta_max_rad=maximum,
                initial_joint_delta_norm_rad=float(np.linalg.norm(delta[joints])),
                initial_root_unchanged=True, initial_object_unchanged=True)


def _angles(quaternion, reference):
    q = np.asarray(quaternion, np.float64)
    r = np.asarray(reference, np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    r = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-8)
    r = r * np.where(np.sum(q*r, axis=-1, keepdims=True) < 0, -1.0, 1.0)
    # This is the pinned objective's WXYZ relative-quaternion expression.
    xyz = np.stack((r[...,0]*q[...,1]-r[...,1]*q[...,0]-r[...,2]*q[...,3]+r[...,3]*q[...,2],
                    r[...,0]*q[...,2]+r[...,1]*q[...,3]-r[...,2]*q[...,0]-r[...,3]*q[...,1],
                    r[...,0]*q[...,3]-r[...,1]*q[...,2]+r[...,2]*q[...,1]-r[...,3]*q[...,0]), -1)
    sine = np.sqrt(np.maximum(np.sum(xyz*xyz, axis=-1), 1e-12))
    return 2 * np.arctan2(sine, np.sum(q*r, axis=-1))


def tracking_components(qpos, targets, index, cfg, *, root_pos_weight, root_rot_weight):
    qpos = np.asarray(qpos, np.float64)
    root = np.asarray(targets.root_qpos, np.float64)
    obj = np.asarray(targets.object_qpos, np.float64)
    if qpos.ndim != 2 or root.shape != (len(qpos), 7) or obj.shape != root.shape:
        raise ValueError("ForceAware tracking diagnostics require full dense qpos and root/object references")
    weights = np.full(index.action_dim, cfg.loss.action_ref_finger)
    weights[:3], weights[3:6] = cfg.loss.action_ref_base, cfg.loss.action_ref_base_rot
    joint_error = qpos[:, index.hand_qpos] - np.asarray(targets.dense_actions)
    root_error = qpos[:, :3] - root[:, :3]
    root_angle = _angles(qpos[:,3:7], root[:,3:7])
    object_pose = qpos[:, index.object_qpos:index.object_qpos+7]
    object_error = object_pose[:,:3] - obj[:,:3]
    object_angle = _angles(object_pose[:,3:], obj[:,3:])
    joint_terms = np.sum((joint_error * weights)**2, axis=1)
    root_terms = root_pos_weight**2*np.sum(root_error**2, axis=1) + (root_rot_weight*root_angle)**2
    object_terms = cfg.loss.reference_pose_scale * (
        cfg.loss.obj_pos**2*np.sum(object_error**2, axis=1) + (cfg.loss.obj_rot*object_angle)**2)
    stage_scale = np.ones(len(qpos)); stage_scale[-1] = cfg.loss.terminal
    return dict(
        joint_tracking_loss=float(1e6*np.dot(stage_scale, joint_terms)),
        root_tracking_loss=float(1e6*np.dot(stage_scale, root_terms)),
        object_tracking_loss=float(1e6*np.dot(stage_scale, object_terms)),
        joint_error_rms_rad=float(np.sqrt(np.mean(joint_error**2))),
        root_position_rms_m=float(np.sqrt(np.mean(np.sum(root_error**2, axis=1)))),
        root_rotation_rms_rad=float(np.sqrt(np.mean(root_angle**2))),
        object_position_rms_m=float(np.sqrt(np.mean(np.sum(object_error**2, axis=1)))),
        object_rotation_rms_rad=float(np.sqrt(np.mean(object_angle**2))),
    )


def _selected(array, candidate):
    return np.asarray(array.numpy()[candidate]).copy()


def _contact_forces(optimizer, frames, row_forces, active):
    parameters = optimizer.compiled.base.contact_rows.parameters
    row_contact = np.asarray(parameters.row_contact.numpy(), dtype=int)
    row_edge = np.asarray(parameters.row_edge.numpy(), dtype=int)
    friction = np.asarray(parameters.contact_friction.numpy(), dtype=np.float64)
    force = np.zeros((*active.shape, 3), dtype=np.float64)
    for row, (contact, edge) in enumerate(zip(row_contact, row_edge)):
        # Joint-limit rows share the response vector but are not contact rays.
        if contact < 0 or edge < 0:
            continue
        ray = frames[:, contact, 0].astype(np.float64).copy()
        axis = edge // 2
        if axis < 2:
            ray += (1.0 if edge % 2 == 0 else -1.0) * friction[contact, axis] * frames[:, contact, axis+1]
        force[:, contact] += row_forces[:,row,None] * ray
    return np.where(active[...,None], force, 0.0).astype(np.float32)


def collect_window_diagnostics(optimizer, plan, targets, *, qpos_before, settings,
                               observation_site_ids=(), allow_init=True):
    """Return JSON metrics and NPZ arrays without stepping or modifying a graph.

    Contact arrays describe pre-integration geometry. Dense qpos/qvel arrays are
    the post-integration states optimized against the reference. Reactions are
    explicitly the forces on the second geom in each native collision pair.
    """
    selected = int(plan.diag["selected"])
    qpos = np.stack([_selected(step.recorded.result.qpos, selected) for step in optimizer.steps])
    qvel = np.stack([_selected(step.recorded.result.qvel, selected) for step in optimizer.steps])
    distance = np.stack([_selected(step.recorded.result.contact_distance, selected) for step in optimizer.steps])
    position = np.stack([_selected(step.recorded.result.contact_position, selected) for step in optimizer.steps])
    frame = np.stack([_selected(step.recorded.result.contact_frame, selected) for step in optimizer.steps])
    collision_active = np.stack([_selected(step.workspace.forward.base.collision.contacts.active, selected)
                                 for step in optimizer.steps]).astype(bool)
    parameters = optimizer.compiled.base.contact_rows.parameters
    margins = np.asarray(parameters.contact_includemargin.numpy())
    active = collision_active & (distance < margins[None])
    row_force = np.stack([_selected(step.workspace.forward.contact.output.force, selected) for step in optimizer.steps])
    force = _contact_forces(optimizer, frame, row_force, active)
    geom = np.asarray(parameters.contact_geom.numpy(), np.int32)
    summary = initial_correction(qpos_before, plan.initial_qpos, optimizer.index,
                                 bound=settings.init_joint_bound, allow_init=allow_init)
    summary.update(tracking_components(qpos, targets, optimizer.index, optimizer.cfg,
                                      root_pos_weight=optimizer.objective.root_pos_weight,
                                      root_rot_weight=optimizer.objective.root_rot_weight))
    tracking = sum(summary[key] for key in ("joint_tracking_loss", "root_tracking_loss", "object_tracking_loss"))
    summary.update(gradient_norm=float(plan.grad_norm), selected_candidate=selected,
                   valid_candidates=int(np.asarray(plan.diag["valid"]).sum()),
                   active_contact_count_max=int(active.sum(1).max(initial=0)),
                   contact_stage="pre_integration", tracking_loss=tracking,
                   regularization_and_penetration_loss=float(plan.loss)-tracking,
                   physical_penetration_max_m=float(np.maximum(-distance[active], 0).max(initial=0)),
                   contact_force_max_n=float(np.linalg.norm(force,axis=-1).max(initial=0)))
    arrays = dict(qpos_predicted=qpos, qvel_predicted=qvel,
                  contact_collision_active=collision_active, contact_active=active,
                  contact_distance=distance, contact_position=position,
                  contact_frame=frame, contact_normal=frame[:,:,0], contact_force_on_geom2=force,
                  contact_geom_ids=geom, contact_row_force=row_force,
                  generalized_constraint_force=np.stack([_selected(step.recorded.result.constraint_force,selected)
                                                          for step in optimizer.steps]),
                  initial_qpos_before=np.asarray(qpos_before), initial_qpos_after=np.asarray(plan.initial_qpos))
    if len(observation_site_ids):
        cpu = optimizer.cpu_model
        ids = np.asarray(observation_site_ids, dtype=int)
        bodies, local = cpu.site_bodyid[ids], cpu.site_pos[ids]
        observed = []
        for step in optimizer.steps:
            xyz = _selected(step.recorded.result.body_position, selected)
            matrix = _selected(step.recorded.result.body_matrix, selected)
            observed.append(xyz[bodies] + np.einsum("kij,kj->ki",matrix[bodies],local))
        arrays["observation_site_ids"] = ids
        arrays["observation_site_position"] = np.stack(observed)
    return summary, arrays

"""Explicit static-load control initialization for an accepted CA hand state.

This reuses the existing bounded load QP and returns data only.  It does not
change the CA hand reference, initial state, FA objective, or rolling workflow.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from numbers import Real

import mujoco
import numpy as np

from .forceaware_reference import _read_ca


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _minimum_normal_forces(minimum, body_names):
    """Resolve scalar, five-finger mapping, or active-contact-order vector."""
    def valid(value):
        return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value) and value > 0.)
    if isinstance(minimum, Mapping):
        if set(minimum) != set(FINGERS) or not all(valid(value) for value in minimum.values()):
            raise ValueError("Preload finger minima must provide positive finite values for exactly all five fingers")
        values = []
        for name in body_names:
            if not name.startswith("right_finger"):
                raise ValueError("A finger-minimum mapping cannot assign an active palm slot; use a contact vector")
            values.append(minimum[FINGERS[int(name[len("right_finger")])-1]])
        return np.asarray(values, np.float64), "finger_mapping"
    if isinstance(minimum, Real):
        if not valid(minimum):
            raise ValueError("Preload normal-force minimum must be positive and finite")
        return np.full(len(body_names), minimum, np.float64), "scalar"
    values = np.asarray(minimum)
    if values.shape != (len(body_names),) or not all(valid(value) for value in values):
        raise ValueError("Preload contact minima must be a positive finite vector in active-contact order")
    return values.astype(np.float64), "contact_vector"


def _cpu_load_matrices(values, *, edges=8):
    """The existing load QP's matrices, with exact source MuJoCo derivatives."""
    model, q = values["model"], values["qref"]
    kp = np.asarray(model.actuator_gainprm[:, 0], np.float64)
    if (model.nu != 26 or np.any(kp <= 0.)
            or not np.allclose(model.actuator_biasprm[:, 1], -kp)
            or not np.allclose(model.actuator_gear[:, 0], 1.)):
        raise ValueError("Load conditioning requires the Wuji scalar position servos")
    inward = -np.asarray(values["outward"], np.float64)
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    axes = np.eye(3)[np.abs(inward).argmin(1)]
    tangent = np.cross(inward, axes); tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    second = np.cross(inward, tangent)
    angles = np.arange(edges)*2*np.pi/edges
    rays = inward[:, None] + values["friction"]*(
        tangent[:, None]*np.cos(angles)[None, :, None]
        + second[:, None]*np.sin(angles)[None, :, None])
    force = rays.reshape(-1, 3)
    arms = np.repeat(values["arms"], edges, axis=0)
    grasp = np.r_[force.T, (np.cross(arms, force)/values["radius"]).T]
    actuator = np.einsum("cij,cki->jck", values["jacobian"], rays).reshape(26, -1)
    low = np.maximum(kp*(model.actuator_ctrlrange[:, 0]-q),
        np.where(model.actuator_forcelimited, model.actuator_forcerange[:, 0], -np.inf))
    high = np.minimum(kp*(model.actuator_ctrlrange[:, 1]-q),
        np.where(model.actuator_forcelimited, model.actuator_forcerange[:, 1], np.inf))
    return dict(q=q, kp=kp, bias=values["bias"], rays=rays, grasp=grasp,
                actuator=actuator, low=low, high=high,
                goal=np.asarray([0., 0., values["mass"]*9.81, 0., 0., 0.]))


def _solve_clarabel_load(matrices, minima, *, minimum_total, iterations):
    """Solve the same .005||x||²+.01sum(x) bounded static input QP on CPU."""
    import clarabel
    from scipy import sparse

    m = matrices
    count, edges = m["rays"].shape[:2]
    size = count*edges
    normal = np.kron(np.eye(count), np.ones((1, edges)))
    scale = np.maximum((m["high"]-m["low"])/2, .01)
    if not np.isfinite(scale).all():
        raise ValueError("Static Clarabel preload requires finite source control/force bounds")
    rows = np.vstack([m["grasp"], m["actuator"]/scale[:, None], -m["actuator"]/scale[:, None],
                      -np.eye(size), -normal, -np.ones((1, size))])
    bounds = np.r_[m["goal"], (m["high"]-m["bias"])/scale, -(m["low"]-m["bias"])/scale,
                   np.zeros(size), -np.asarray(minima), -minimum_total]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = int(iterations)
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    solution = clarabel.DefaultSolver(sparse.eye(size, format="csc")*.01,
        np.full(size, .01), sparse.csc_matrix(rows), bounds,
        [clarabel.ZeroConeT(6), clarabel.NonnegativeConeT(len(bounds)-6)], settings).solve()
    x = np.asarray(solution.x, np.float64)
    if x.shape != (size,):
        x = np.full(size, np.nan)
    torque = m["bias"]+m["actuator"]@x
    command = m["q"]+torque/m["kp"]
    pressure = normal@x
    wrench = m["grasp"]@x-m["goal"]
    violations = np.r_[abs(wrench), (m["low"]-torque)/scale,
                        (torque-m["high"])/scale, -x, np.asarray(minima)-pressure,
                        minimum_total-x.sum()]
    violation = float(max(violations.max(), 0.))
    finite = bool(np.isfinite(x).all() and np.isfinite(command).all())
    feasible = bool(str(solution.status) == "Solved" and finite and violation < 2e-4)
    return command, dict(feasible=feasible, backend="clarabel_cpu_bounded_load_qp",
        clarabel_version=clarabel.__version__, solver_status=str(solution.status),
        solver_primal_residual=float(solution.r_prim), solver_dual_residual=float(solution.r_dual),
        gravity_command=(m["q"]+m["bias"]/m["kp"]).tolist(),
        maximum_scaled_constraint_violation=violation, wrench_residual=wrench.tolist(),
        normal_force_N=pressure.tolist(), force_N=np.einsum("ce,cei->ci", x.reshape(count, edges), m["rays"]).tolist(),
        actuator_torque_Nm=torque.tolist(), minimum_per_contact_N=np.asarray(minima).tolist(),
        minimum_total_N=minimum_total, iterations=solution.iterations,
        maximum_solver_iterations=int(iterations),
        ray_coefficients=x.reshape(count, edges).tolist(),
        objective_value=float(.005*x@x+.01*x.sum()),
        objective="0.005 ||ray_coefficients||^2 + 0.01 sum(ray_coefficients)")


def _object_surface(model, data, object_body):
    import trimesh

    physical = (model.geom_contype | model.geom_conaffinity) != 0
    objects = np.flatnonzero(physical & (model.geom_bodyid == object_body))
    if len(objects) != 1 or model.geom_type[objects[0]] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("Static preload requires the source primitive's single closed mesh")
    geom = int(objects[0]); mesh = int(model.geom_dataid[geom])
    vertex_start, vertex_count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    vertices = model.mesh_vert[vertex_start:vertex_start+vertex_count]
    vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    surface = trimesh.Trimesh(vertices, model.mesh_face[face_start:face_start+face_count], process=False)
    # A common source coefficient is unambiguous for the current WUJI scene.
    friction = model.geom_friction[physical, 0]
    if model.npair or not np.allclose(friction, friction[0], rtol=0, atol=1e-12):
        raise ValueError("Static preload needs explicit pair-material handling for nonuniform friction")
    return surface, float(friction[0])


def convex_surface_normals(surface, points, closest, triangles):
    """Match the native signed-distance feature normal at faces and rims.

    A triangle normal at an edge is ambiguous. For the closed convex primitive
    use the signed point-to-projection direction, with the face normal only
    inside the same 0.1um surface band as the CA query adapter.
    """
    points, closest = np.asarray(points, np.float64), np.asarray(closest, np.float64)
    face_normals = np.asarray(surface.face_normals, np.float64)
    planes = np.einsum("fi,fi->f", face_normals, surface.triangles[:, 0])
    vertices = np.asarray(surface.vertices, np.float64)
    if (not surface.is_watertight or not surface.is_winding_consistent or surface.volume <= 0
            or np.max(vertices @ face_normals.T - planes, initial=0.) > 1e-8):
        raise ValueError("Preload feature normals require an outward-wound closed convex primitive")
    outside = np.max(points @ face_normals.T - planes, axis=1) > 0.
    delta = points - closest
    distance = np.linalg.norm(delta, axis=1)
    normals = np.where(outside[:, None], 1., -1.) * delta / np.maximum(distance[:, None], 1e-12)
    near = distance < 1e-7
    normals[near] = face_normals[np.asarray(triangles)[near]]
    return normals


def preload_inputs(artifacts, *, require_all_fingers=True):
    """Read actual CA material points and native-compatible object normals."""
    _, card, report, guidance, poses = _read_ca(artifacts.root)
    model = mujoco.MjModel.from_xml_path(str(Path(artifacts.viewer_xml).resolve()))
    if (model.nq, model.nv, model.nu) != (33, 32, 26):
        raise ValueError("Static WUJI preload requires nq=33, nv=32, nu=26")
    if not np.allclose(model.opt.gravity, [0., 0., -9.81], rtol=0, atol=1e-12):
        raise ValueError("Existing load QP assumes world gravity [0,0,-9.81]")
    qref = np.asarray(card["hand_reference_qpos"], np.float64).copy()
    mask = np.asarray(card["reference_contact_mask"], bool)
    bodies = guidance["hand_query_body_ids"][0, mask].astype(int)
    local = guidance["hand_query_local_pos"][0, mask].astype(np.float64)
    names = [model.body(int(body)).name for body in bodies]
    represented = {FINGERS[int(name[len("right_finger")])-1]
                   for name in names if name.startswith("right_finger")}
    if require_all_fingers and represented != set(FINGERS):
        raise ValueError("Static preload requires active CA points on all five fingers")
    data = mujoco.MjData(model)
    data.qpos[:26] = qref
    object_joint = model.joint("obj_joint").id
    object_address = int(model.jnt_qposadr[object_joint])
    # Original CA sequence convention: quaternion XYZW, then position XYZ.
    pose = poses[0]
    data.qpos[object_address:object_address+7] = np.r_[pose[4:7], pose[3], pose[:3]]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    object_body = model.body("grasp_object").id
    points = data.xpos[bodies] + np.einsum("nij,nj->ni", data.xmat[bodies].reshape(-1, 3, 3), local)
    surface, friction = _object_surface(model, data, object_body)
    import trimesh
    closest, distance, triangles = trimesh.proximity.closest_point_naive(surface, points)
    outward = convex_surface_normals(surface, points, closest, triangles)
    radius = float(np.max(np.linalg.norm(surface.vertices-data.xipos[object_body], axis=1)))
    jacobian = []
    for body, point in zip(bodies, points):
        position, rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, position, rotation, point, int(body))
        jacobian.append(position[:, :26])
    return dict(model=model, qref=qref, bodies=bodies, local=local, points=points,
        arms=points-data.xipos[object_body], outward=outward, closest=closest,
        distances=distance, jacobian=np.asarray(jacobian), radius=radius,
        mass=float(model.body_mass[object_body]), friction=friction,
        bias=(data.qfrc_bias-data.qfrc_passive)[:26].copy(), body_names=names,
        source_ca_commit=report["source_commit"], active_mask=mask)


def _command_report(model, qref, command, info):
    command = np.asarray(command, np.float64)
    torque = np.asarray(info["actuator_torque_Nm"], np.float64)
    command_violation = np.maximum(model.actuator_ctrlrange[:, 0]-command,
                                   command-model.actuator_ctrlrange[:, 1]).clip(min=0.)
    force_violation = np.maximum(model.actuator_forcerange[:, 0]-torque,
                                 torque-model.actuator_forcerange[:, 1]).clip(min=0.)
    force_violation = np.where(model.actuator_forcelimited, force_violation, 0.)
    finite = bool(np.isfinite(command).all() and np.isfinite(torque).all()
                  and np.isfinite(np.asarray(info["force_N"])).all())
    command_ok = bool(finite and command_violation.max(initial=0.) <= 1e-8)
    force_ok = bool(finite and force_violation.max(initial=0.) <= 1e-8)
    offset = command-qref
    fingers = np.rad2deg(offset[6:]).reshape(5, 4)
    return dict(feasible=bool(info["feasible"] and finite and command_ok and force_ok), finite=finite,
        command_within_bounds=command_ok, actuator_force_within_bounds=force_ok,
        command_max_violation_rad=float(command_violation.max(initial=0.)),
        actuator_force_max_violation=float(force_violation.max(initial=0.)),
        command_offsets=offset.tolist(), wrist_translation_offset_mm=(offset[:3]*1000.).tolist(),
        wrist_rotation_offset_deg=np.rad2deg(offset[3:6]).tolist(),
        finger_offsets_deg={name: dict(joints=fingers[i].tolist(), maximum_abs=float(abs(fingers[i]).max()))
                            for i, name in enumerate(FINGERS)})


def prepare_preload(artifacts, cfg, *, device="cuda:0", minimum_per_contact=.2,
                    minimum_total=6., iterations=10000, require_all_fingers=True):
    """Return `(command, report)`; return `None` command if strict QP checks fail.

    `command` is an actuator target, never a hand-state reference.  A caller may
    explicitly initialize previous_action=command and knot corrections to
    command-qref, keeping CA qpos/qvel and the original FA targets unchanged.
    """
    values = preload_inputs(artifacts) if require_all_fingers else preload_inputs(artifacts, require_all_fingers=False)
    model, qref = values["model"], values["qref"]
    if cfg.sequence.hand != "wuji_right":
        raise ValueError("Static preload is only adapted for the WUJI2 right hand")
    minima, minimum_kind = _minimum_normal_forces(minimum_per_contact, values["body_names"])
    if (not isinstance(minimum_total, Real) or isinstance(minimum_total, (bool, np.bool_))
            or not np.isfinite(minimum_total) or minimum_total <= 0):
        raise ValueError("Preload total pressure minimum must be positive and finite")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("Preload iterations must be a positive integer")
    if minimum_kind == "scalar":
        from .load import load_matrices, solve_load
        matrices = dict(load_matrices(model, values["bodies"], values["local"], qref,
            values["arms"], values["outward"], device=device, mass=values["mass"],
            friction=values["friction"], radius=values["radius"]))
        jacobian_error = float(np.max(abs(matrices["jacobian"].cpu().numpy()-values["jacobian"])))
        if jacobian_error > 1e-7:
            raise ValueError("Load kinematics disagree with actual source material-point Jacobians")
        approximate_bias = matrices["bias"].cpu().numpy()
        # Include all fixed pad/mount masses in this private matrices copy.
        matrices["bias"] = matrices["q"].new_tensor(values["bias"])
        command, info = solve_load(matrices, minimum_per_contact=float(minimum_per_contact),
                                  minimum_total=minimum_total, iterations=iterations)
        bias_difference = float(abs(approximate_bias-values["bias"]).max())
    else:
        matrices = _cpu_load_matrices(values)
        command, info = _solve_clarabel_load(matrices, minima,
            minimum_total=float(minimum_total), iterations=iterations)
        jacobian_error, bias_difference = 0., None
    status = _command_report(model, qref, command, info)
    scaled_wrench = np.asarray(info["wrench_residual"], np.float64)
    report = dict(info, **{key: value for key, value in status.items() if key != "feasible"})
    report.update(feasible=status["feasible"], method="existing grasp.load_matrices + solve_load bounded static QP"
        if minimum_kind == "scalar" else "Clarabel CPU solve of the existing bounded static input QP with per-contact minima",
        minimum_specification=minimum_kind, resolved_minimum_per_contact_N=minima.tolist(),
        minimum_per_finger_N=dict(minimum_per_contact) if minimum_kind == "finger_mapping" else None,
        static_qp_only=True, forceaware_objective_modified=False,
        input_scope="actual active CA hand material points; signed-distance feature normals on the convex object mesh; original source servos/material",
        normal_convention="outward signed point-to-closest-feature direction; nearest triangle normal only within 0.1um of the surface",
        physical_hold_certified=False, force_source="static QP proposal; not measured native contact row forces",
        command_role="explicit actuator-control warm initialization only; never hand_reference_qpos",
        hand_reference_unchanged=True, initial_state_unchanged=True,
        source_ca_commit=values["source_ca_commit"], mass_kg=values["mass"], friction=values["friction"],
        object_weight_N=values["mass"]*9.81, active_bodies=values["body_names"],
        actual_material_points_world=values["points"].tolist(),
        actual_object_outward_normals=values["outward"].tolist(), actual_mesh_gap_mm=(values["distances"]*1000.).tolist(),
        static_bias_source="full MuJoCo qfrc_bias - qfrc_passive at CA qpos, zero qvel",
        static_bias=values["bias"].tolist(), approximate_load_bias_max_difference=bias_difference,
        actual_point_jacobian_max_difference=jacobian_error,
        force_balance_residual_N=scaled_wrench[:3].tolist(),
        torque_balance_residual_Nm=(scaled_wrench[3:]*values["radius"]).tolist(),
        command=np.asarray(command).tolist(), hand_reference_qpos=qref.tolist(),
        recommended_previous_action=np.asarray(command).tolist() if status["feasible"] else None,
        recommended_knot_correction=(np.asarray(command)-qref).tolist() if status["feasible"] else None,
        horizon_knots=cfg.optimizer.horizon, strict_scaled_qp_tolerance=2e-4,
        candidate_command_clipped=False)
    if not status["feasible"]:
        report["reason"] = "static QP or actual command/force bound check failed; no usable preload returned"
        return None, report
    return np.asarray(command).copy(), report

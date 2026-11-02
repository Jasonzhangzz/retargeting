"""Opt-in, process-local implicit integration of fixed scalar PD servos.

The original native force law, including its control and force clamps, stays
unchanged. Only the integration matrix becomes ``M + h*(Dpassive + Dact)``,
where a force-saturated actuator contributes zero to Dact. This is a local
numerical capability extension, not the complete MuJoCo implicitfast method.
Activate it before constructing any native planner or executor.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import weakref

import mujoco
import numpy as np
import warp as wp


PROFILE = "local_fixed_pd_implicit_v1"
_ACTIVE = False
_ORIGINALS = {}
_COMPILED = {}


@dataclass(frozen=True)
class _ServoData:
    dof_to_actuator: object
    velocity_damping: object
    contract: dict


def validate_servo_model(model):
    """Fail closed outside one unit-gear, fixed PD actuator per scalar DOF."""
    if model.nu < 1:
        raise ValueError("PD integration requires at least one scalar PD actuator")
    kinds = (("actuator_dyntype", mujoco.mjtDyn.mjDYN_NONE),
             ("actuator_gaintype", mujoco.mjtGain.mjGAIN_FIXED),
             ("actuator_biastype", mujoco.mjtBias.mjBIAS_AFFINE),
             ("actuator_trntype", mujoco.mjtTrn.mjTRN_JOINT))
    for field, expected in kinds:
        if np.any(np.asarray(getattr(model, field)) != int(expected)):
            raise ValueError(f"PD integration requires supported {field}")
    gear = np.zeros_like(model.actuator_gear)
    gear[:, 0] = 1.
    if not np.allclose(model.actuator_gear, gear, atol=1e-12, rtol=0):
        raise ValueError("PD integration requires unit scalar actuator gear")
    joints = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    if np.any(~np.isin(model.jnt_type[joints], [int(mujoco.mjtJoint.mjJNT_HINGE),
                                                     int(mujoco.mjtJoint.mjJNT_SLIDE)])):
        raise ValueError("PD integration only supports actuated hinge/slide joints")
    dofs = np.asarray(model.jnt_dofadr[joints], dtype=np.int32)
    if len(np.unique(dofs)) != model.nu:
        raise ValueError("PD integration requires one actuator per actuated DOF")
    kp = np.asarray(model.actuator_gainprm[:, 0])
    bias = np.asarray(model.actuator_biasprm)
    if (not np.isfinite(kp).all() or np.any(kp <= 0) or not np.isfinite(bias).all()
            or np.any(bias[:, 2] > 0)
            or not np.allclose(bias[:, 0], 0., atol=1e-12, rtol=0)
            or not np.allclose(bias[:, 1], -kp, atol=1e-12, rtol=0)):
        raise ValueError("PD integration requires fixed positive kp and nonnegative kv")
    disabled = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP | mujoco.mjtDisableBit.mjDSBL_DAMPER)
    if int(model.opt.disableflags) & disabled:
        raise ValueError("PD integration requires the native implicit passive-damping path enabled")
    if int(model.opt.disableactuator):
        raise ValueError("PD integration does not support actuator-group disabling")
    if not np.isfinite(model.dof_damping).all() or np.any(model.dof_damping < 0):
        raise ValueError("PD integration requires finite nonnegative passive damping")
    # The upstream native force routine implements actuator clamps, not a second
    # generalized joint-force clamp. WUJI's one-to-one limits coincide exactly.
    for actuator, joint in enumerate(joints):
        if model.jnt_actfrclimited[joint]:
            joint_lo, joint_hi = model.jnt_actfrcrange[joint]
            lo, hi = model.actuator_forcerange[actuator]
            if (not model.actuator_forcelimited[actuator]
                    or joint_lo > lo or joint_hi < hi):
                raise ValueError("A separate tighter joint actuator-force clamp is unsupported")
    digest = hashlib.sha256()
    for field in ("actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
                  "actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcelimited",
                  "actuator_forcerange", "dof_damping"):
        digest.update(field.encode())
        digest.update(np.asarray(getattr(model, field)).tobytes())
    return dict(profile=PROFILE, nq=int(model.nq), nv=int(model.nv), nu=int(model.nu),
                actuator_dof_ids=dofs.tolist(), kp=kp.tolist(), kv=(-bias[:, 2]).tolist(),
                actuator_forcelimited=np.asarray(model.actuator_forcelimited, dtype=bool).tolist(),
                fixed_servo_contract_sha256=digest.hexdigest())


@wp.kernel(enable_backward=False)
def _servo_mass(timestep: wp.array(dtype=float), damping: wp.array2d(dtype=float),
                actuator_force: wp.array2d(dtype=float), dof_to_actuator: wp.array(dtype=int),
                velocity_damping: wp.array(dtype=float), limited: wp.array(dtype=bool),
                limits: wp.array2d(dtype=wp.vec2), actuation_enabled: int,
                mass: wp.array3d(dtype=float), output: wp.array3d(dtype=float)):
    world, row, column = wp.tid()
    value = mass[world, row, column]
    if row == column:
        coefficient = damping[world % damping.shape[0], row]
        actuator = dof_to_actuator[row]
        if actuator >= 0 and actuation_enabled != 0:
            force = actuator_force[world, row]
            bounds = limits[world % limits.shape[0], actuator]
            interior = not limited[actuator] or (force > bounds[0] and force < bounds[1])
            if interior:
                coefficient += velocity_damping[actuator]
        value += timestep[world % timestep.shape[0]] * coefficient
    output[world, row, column] = value


def _checked_compile(cpu_model, device_model):
    contract = validate_servo_model(cpu_model)
    if device_model.actuator_biasprm.shape[0] != 1:
        raise ValueError("PD integration requires unbatched, fixed actuator parameters")
    expected = cpu_model.actuator_biasprm[:, 2]
    actual = device_model.actuator_biasprm.numpy()[0, :, 2]
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-8):
        raise ValueError("Device and CPU servo velocity biases differ")
    compiled = _ORIGINALS["compile_fast_step"](cpu_model, device_model)
    if not compiled.base.integration.implicit_damping:
        raise ValueError("The native integration workspace must include an implicit mass solve")
    mapping = np.full(cpu_model.nv, -1, np.int32)
    mapping[np.asarray(contract["actuator_dof_ids"])] = np.arange(cpu_model.nu)
    data = _ServoData(wp.array(mapping, dtype=int, device=compiled.base.device),
                      wp.array(contract["kv"], dtype=float, device=compiled.base.device), contract)
    key = id(compiled)
    reference = weakref.ref(compiled, lambda _: _COMPILED.pop(key, None))
    _COMPILED[key] = (reference, data)
    return compiled


def _compiled_data(compiled):
    saved = _COMPILED.get(id(compiled))
    if saved is None or saved[0]() is not compiled:
        raise RuntimeError("Activate PD integration before compiling every planner/executor")
    return saved[1]


def _integration_acceleration(compiled, workspace, *, mass, rhs, qacc):
    from comfree_warp.native_adjoint.fast_mass import solve

    data = _compiled_data(compiled)
    integration = compiled.base.integration
    dynamics = compiled.base.dynamics.model
    wp.launch(_servo_mass, dim=mass.shape,
              inputs=[integration.model.timestep, integration.model.dof_damping,
                      workspace.base.dynamics.actuator_force, data.dof_to_actuator,
                      data.velocity_damping, dynamics.actuator_forcelimited,
                      dynamics.actuator_forcerange, dynamics.actuation_enabled, mass],
              outputs=[workspace.integration_matrix], device=compiled.base.device)
    # On each fixed saturation branch Dact is constant. The original integration
    # mass adjoint differentiates A=M+hD and rhs exactly on that branch. The
    # separate physical qacc=M^-1 rhs output and its adjoint remain unchanged.
    return solve(workspace.integration_matrix, rhs, workspace.integration_mass)


def servo_integration_metadata():
    from .dexforge_upstream import COMMIT, verify_upstream

    upstream = verify_upstream()
    files = ("forceaware/third_party/comfree_warp/comfree_warp/native_adjoint/fast_step.py",
             "forceaware/third_party/comfree_warp/comfree_warp/native_adjoint/fast_step_adjoint.py",
             "forceaware/third_party/comfree_warp/comfree_warp/native_adjoint/integration.py")
    return dict(profile=PROFILE, active=bool(_ACTIVE), upstream_commit=COMMIT,
                upstream_source_modified=False, loss_modified=False, optimizer_modified=False,
                physical_parameters_modified=False, native_integration_extended=True,
                implementation=str(Path(__file__).relative_to(Path(__file__).resolve().parents[2])),
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                original_source_sha256={name: upstream["sha256"][name] for name in files},
                equation="a_integrate=(M+h*(Dpassive+Dact))^-1*(original_clamped_smooth_force+contact_force)",
                actuator_mask="Dact=-bias_velocity only strictly inside the original force limits; zero when saturated",
                derivative="Exact on each fixed force-clamp branch for the supported fixed unit-gear PD model; mask boundaries are nonsmooth",
                unchanged_outputs="Physical actuator force clamps and physical qacc=M^-1 total_force stay unchanged",
                limitation="Local fixed-PD implicit integration extension, not complete MuJoCo implicitfast; no additional Coriolis velocity derivative or other force laws",
                activation_scope="Process-local; activate separately in each solve/validation/audit worker before model compilation")


def activate_servo_integration():
    """Enable the same local transition in official planner, executor and probes."""
    global _ACTIVE
    from .dexforge_upstream import activate_upstream, require_environment

    require_environment()
    activate_upstream()
    from comfree_warp.native_adjoint import fast_step, fast_step_adjoint

    if _ACTIVE:
        return servo_integration_metadata()
    _ORIGINALS["compile_fast_step"] = fast_step.compile_fast_step
    _ORIGINALS["integration_acceleration"] = fast_step.integration_acceleration
    _ORIGINALS["adjoint_integration_acceleration"] = fast_step_adjoint.integration_acceleration
    original_compile = fast_step.compile_fast_step
    fast_step.compile_fast_step = _checked_compile
    fast_step.integration_acceleration = _integration_acceleration
    fast_step_adjoint.integration_acceleration = _integration_acceleration
    # Original FA consumers import compile_fast_step by value. Cover already
    # imported consumers as well as future imports, without changing algorithms.
    for module in tuple(sys.modules.values()):
        if module is not None and vars(module).get("compile_fast_step") is original_compile:
            setattr(module, "compile_fast_step", _checked_compile)
    _ACTIVE = True
    return servo_integration_metadata()


def deactivate_servo_integration():
    """Restore the original process functions, primarily for isolated tests."""
    global _ACTIVE
    if not _ACTIVE:
        return
    from comfree_warp.native_adjoint import fast_step, fast_step_adjoint

    for module in tuple(sys.modules.values()):
        if module is not None and vars(module).get("compile_fast_step") is _checked_compile:
            setattr(module, "compile_fast_step", _ORIGINALS["compile_fast_step"])
    fast_step.integration_acceleration = _ORIGINALS["integration_acceleration"]
    fast_step_adjoint.integration_acceleration = _ORIGINALS["adjoint_integration_acceleration"]
    _ACTIVE = False


def servo_integration_diagnostics(compiled, workspace, *, inputs=None, reference=None,
                                  boundary_atol=1e-6):
    """Report local masks and, when inputs are supplied, raw clamp-kink margins.

    This explicit host-side diagnostic synchronizes device arrays; never call it
    inside a captured training graph. Saturation itself is not a kink: raw force
    is required to distinguish deep saturation from a nearby clamp boundary.
    """
    data = _compiled_data(compiled)
    forward = workspace.forward if hasattr(workspace, "forward") else workspace
    model = compiled.base.dynamics.model
    dofs = np.asarray(data.contract["actuator_dof_ids"])
    force = forward.base.dynamics.actuator_force.numpy()[:, dofs]
    bounds = model.actuator_forcerange.numpy()
    limited = model.actuator_forcelimited.numpy().astype(bool)
    bounds = bounds[np.arange(len(force)) % len(bounds)]
    active = (~limited[None]) | ((force > bounds[:, :, 0]) & (force < bounds[:, :, 1]))
    active &= bool(model.actuation_enabled)
    result = dict(profile=PROFILE, model_contract=data.contract,
                  actuator_implicit_damping_active=active.tolist(),
                  actuator_force_n_or_nm=force.tolist(),
                  derivative_scope="Piecewise smooth only; finite differences must retain force masks and every contact solver branch",
                  raw_boundary_diagnostics_available=inputs is not None)
    if reference is not None:
        before = np.asarray(reference["actuator_implicit_damping_active"], dtype=bool)
        if before.shape != active.shape:
            raise ValueError("Reference actuator saturation mask has a different shape")
        result["actuator_mask_changes"] = int(np.count_nonzero(before != active))
    if inputs is not None:
        dynamics = inputs.dynamics if hasattr(inputs, "dynamics") else inputs
        control = dynamics.ctrl.numpy()
        ctrl_bounds = model.actuator_ctrlrange.numpy()
        ctrl_bounds = ctrl_bounds[np.arange(len(force)) % len(ctrl_bounds)]
        ctrl_limited = model.actuator_ctrllimited.numpy().astype(bool) & bool(model.clamp_control)
        effective = np.where(ctrl_limited[None],
                             np.clip(control, ctrl_bounds[:, :, 0], ctrl_bounds[:, :, 1]), control)
        q = dynamics.qpos.numpy()[:, model.actuator_qposadr.numpy()]
        v = dynamics.qvel.numpy()[:, dofs]
        gain, bias = model.actuator_gainprm.numpy()[0], model.actuator_biasprm.numpy()[0]
        raw = gain[:, 0] * effective + bias[:, 0] + bias[:, 1] * q + bias[:, 2] * v
        force_margin = np.minimum(np.abs(raw - bounds[:, :, 0]), np.abs(raw - bounds[:, :, 1]))
        ctrl_margin = np.minimum(np.abs(control - ctrl_bounds[:, :, 0]), np.abs(control - ctrl_bounds[:, :, 1]))
        force_near = limited[None] & (force_margin <= boundary_atol)
        ctrl_near = ctrl_limited[None] & (ctrl_margin <= boundary_atol)
        result.update(raw_actuator_force_n_or_nm=raw.tolist(),
                      near_force_clamp_boundary=force_near.tolist(),
                      near_control_clamp_boundary=ctrl_near.tolist(),
                      boundary_atol=float(boundary_atol),
                      local_actuator_nonsmooth=bool(np.any(force_near) or np.any(ctrl_near)))
    return result

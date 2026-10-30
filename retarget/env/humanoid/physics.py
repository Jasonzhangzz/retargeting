"""SPIDER MJWarp physics and state restoration, with a dtype-only copy adapter.
Copyright Meta Platforms, Inc.; CC BY-NC 4.0, see licenses/SPIDER.txt.
"""
from __future__ import annotations
from .config import PhysicsConfig as Config
from dataclasses import dataclass
import numpy as np
import torch
import mujoco
import mujoco_warp as mjwarp
import warp as _warp
import loguru

class _ArrayAPI:
    def __getattr__(self,name):return getattr(_warp,name)
    def copy(self,destination,source,*args,**kwargs):
        if destination.dtype!=source.dtype:
            source=_warp.from_torch(_warp.to_torch(source),dtype=destination.dtype)
        return _warp.copy(destination,source,*args,**kwargs)
wp=_ArrayAPI()

@dataclass
class MJWPEnv:
    model_cpu: mujoco.MjModel
    data_cpu: mujoco.MjData
    # Unified data sink always reflecting last step's state
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data
    data_wp_prev: mjwarp.Data
    graph: wp.ScopedCapture.Graph
    # Device alias used for Warp allocations/launches (e.g., "cuda:1" or "cpu")
    device: str
    num_worlds: int


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data
) -> wp.ScopedCapture.Graph:
    """Warm up and capture a CUDA graph that runs a single mjwarp.step."""

    def _step_once():
        mjwarp.step(model_wp, data_wp)

    # Warmup/compile
    # _step_once()
    # _step_once()
    # wp.synchronize()
    # Capture
    with wp.ScopedCapture() as capture:
        _step_once()
    wp.synchronize()
    return capture.graph


def setup_mj_model(config: Config) -> mujoco.MjModel:
    model_cpu = mujoco.MjModel.from_xml_path(config.model_path)
    model_cpu.opt.timestep = float(config.sim_dt)
    if config.embodiment_type in ["left", "right", "bimanual"]:
        # setup for hand
        model_cpu.opt.iterations = 20
        model_cpu.opt.ls_iterations = 50
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.0,
            0.95,
            0.03,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    elif config.embodiment_type in ["humanoid", "humanoid_object"]:
        # setup for humanoid
        model_cpu.opt.iterations = 5
        model_cpu.opt.ls_iterations = 10
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.9,
            0.95,
            0.001,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model_cpu


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> MJWPEnv:
    """Setup and reset the environment backed by MJWP.
    Returns an MJWPEnv with captured graph.
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_data
    qpos_init = qpos_ref[0]

    # CPU model/data
    model_cpu = setup_mj_model(config)
    data_cpu = mujoco.MjData(model_cpu)
    # Seed initial state
    arrs = (qpos_init, qvel_ref[0], ctrl_ref[0])
    data_cpu.qpos[:] = arrs[0].detach().cpu().numpy()
    data_cpu.qvel[:] = arrs[1].detach().cpu().numpy()
    data_cpu.ctrl[:] = arrs[2].detach().cpu().numpy()
    mujoco.mj_step(model_cpu, data_cpu)

    # Move to Warp (batched worlds)
    # Set Warp default device to match config to ensure kernels/modules load on it
    wp.set_device(str(config.device))
    # Build default model/data/graph on the configured device
    dev = str(config.device)
    with wp.ScopedDevice(dev):
        default_model_wp = mjwarp.put_model(model_cpu)
        # pair_margin_override_np = (
        #     np.zeros((int(config.num_samples), model_cpu.npair)).astype(np.float32)
        #     + 0.01
        # )
        # pair_margin_override_wp = wp.from_numpy(
        #     pair_margin_override_np, dtype=wp.float32, device=dev
        # )
        # default_model_wp.pair_margin = pair_margin_override_wp

        default_data_wp = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        data_wp_prev = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        default_graph = _compile_step(default_model_wp, default_data_wp)

    # Initialize env; default active is main
    return MJWPEnv(
        model_cpu=model_cpu,
        data_cpu=data_cpu,
        model_wp=default_model_wp,
        data_wp=default_data_wp,
        data_wp_prev=data_wp_prev,
        graph=default_graph,
        device=dev,
        num_worlds=int(config.num_samples),
    )


def _object_pose_slice(
    qpos: np.ndarray, embodiment_type: str, nq_obj: int
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract object position(s) and orientation quaternion(s) from a qpos vector.

    Mirrors the slicing in `_diff_qpos`. Returns
    `(right_pos, right_quat, left_pos, left_quat)`. Quaternions are `None`
    when the embodiment / nq_obj combination has no rotation DOF (e.g.,
    contact-guidance which only actuates positions). Sides that don't apply
    are returned as `None`.

    Args:
        qpos: 1D numpy array of qpos for a single environment.
        embodiment_type: One of "bimanual", "right", "left", "humanoid_object".
        nq_obj: object DOF count (matches `Config.nq_obj`).

    Returns:
        Tuple `(right_pos, right_quat, left_pos, left_quat)`.
    """
    qpos = np.asarray(qpos)
    if embodiment_type == "bimanual":
        if nq_obj == 12:
            # contact_guidance: positions only, [right(3), left(3)]
            right_pos = qpos[-12:-9].copy()
            left_pos = qpos[-6:-3].copy()
            return right_pos, None, left_pos, None
        # nq_obj == 14: [right_pos(3), right_quat(4), left_pos(3), left_quat(4)]
        right_pos = qpos[-14:-11].copy()
        right_quat = qpos[-11:-7].copy()
        left_pos = qpos[-7:-4].copy()
        left_quat = qpos[-4:].copy()
        return right_pos, right_quat, left_pos, left_quat
    if embodiment_type in ["right", "left"]:
        if nq_obj == 6:
            pos = qpos[-6:-3].copy()
            if embodiment_type == "right":
                return pos, None, None, None
            return None, None, pos, None
        # nq_obj == 7: [pos(3), quat(4)]
        pos = qpos[-7:-4].copy()
        quat = qpos[-4:].copy()
        if embodiment_type == "right":
            return pos, quat, None, None
        return None, None, pos, quat
    if embodiment_type == "humanoid_object":
        pos = qpos[-7:-4].copy()
        quat = qpos[-4:].copy()
        return pos, quat, None, None
    # No object on this embodiment (humanoid only)
    return None, None, None, None


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Return the rotation angle in radians between two unit quaternions.

    Quaternions are in MuJoCo `[w, x, y, z]` order. Computes
    `2 * acos(|<q0, q1>|)` for numerical stability and sign invariance.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 < 1e-8 or n1 < 1e-8:
        return 0.0
    dot = float(np.clip(np.abs(np.dot(q0 / n0, q1 / n1)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _initial_state_sanity_check(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_ref: torch.Tensor,
    qvel_ref: torch.Tensor,
    ctrl_ref: torch.Tensor,
    config: Config,
    save_video_dir: str | None = None,
) -> None:
    """Verify the initial scene is stable when controls are held at ctrl_ref[0].

    Resets `mj_data` to the first reference frame, holds `ctrl_ref[0]` fixed,
    and steps CPU MuJoCo for `config.sanity_check_seconds`. If the object
    drifts past `config.sanity_check_pos_thresh` (meters) or rotates past
    `config.sanity_check_rot_thresh` (radians), raises `RuntimeError` with
    actionable hints. CPU-only — does not touch the warp/GPU path.

    If `save_video_dir` is provided and `config.save_video` is enabled, an
    MP4 of the rollout is saved to `<save_video_dir>/sanity_check_drift.mp4`
    and its path is included in the error message.

    Args:
        mj_model: CPU MuJoCo model (already configured).
        mj_data: CPU MuJoCo data buffer; will be mutated and left at the
            final stepped state.
        qpos_ref: Reference qpos tensor of shape `(T, nq)`.
        qvel_ref: Reference qvel tensor of shape `(T, nv)`.
        ctrl_ref: Reference ctrl tensor of shape `(T, nu)`.
        config: SPIDER `Config` (uses `embodiment_type`, `nq_obj`,
            `sanity_check_seconds`, `sanity_check_pos_thresh`,
            `sanity_check_rot_thresh`, `save_video`).
        save_video_dir: Optional output directory for the drift MP4.
    """
    seconds = float(config.sanity_check_seconds)
    if seconds <= 0.0:
        return

    # Force a fine CPU MuJoCo timestep for the stability rollout. The
    # production sim_dt (e.g., 0.01) is too coarse for the implicit-fast
    # integrator to keep contact stable on rigid props, producing spurious
    # drift even when the scene is well-supported. 0.002s matches the
    # default MuJoCo timestep and cleanly resolves contact dynamics.
    _SANITY_TIMESTEP = 0.002
    original_timestep = float(mj_model.opt.timestep)
    if original_timestep <= 0.0:
        loguru.logger.warning(
            "sanity check skipped: invalid mj_model.opt.timestep={}",
            original_timestep,
        )
        return
    mj_model.opt.timestep = _SANITY_TIMESTEP
    timestep = _SANITY_TIMESTEP
    n_steps = int(seconds / timestep)
    if n_steps <= 0:
        loguru.logger.warning(
            "sanity check skipped: sanity_check_seconds ({}) < timestep ({})",
            seconds,
            timestep,
        )
        mj_model.opt.timestep = original_timestep
        return

    try:
        _run_sanity_check_body(
            mj_model,
            mj_data,
            qpos_ref,
            ctrl_ref,
            config,
            save_video_dir,
            seconds,
            timestep,
            n_steps,
        )
    finally:
        mj_model.opt.timestep = original_timestep


def _run_sanity_check_body(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_ref: torch.Tensor,
    ctrl_ref: torch.Tensor,
    config: Config,
    save_video_dir: str | None,
    seconds: float,
    timestep: float,
    n_steps: int,
) -> None:
    qpos0_np = qpos_ref[0].detach().cpu().numpy().astype(np.float64)
    ctrl0_np = ctrl_ref[0].detach().cpu().numpy().astype(np.float64)

    mj_data.qpos[:] = qpos0_np.copy()
    # Zero qvel: the sanity check tests whether the scene is in static
    # equilibrium under gravity + contact with controls held. qvel_ref[0]
    # is a finite-difference of the trajectory and propagates real motion
    # (e.g., a teapot reference may carry several rad/s of angular velocity
    # from the trajectory). That motion would integrate to drift even on
    # a perfectly supported object, masking real failures.
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = ctrl0_np.copy()
    mj_data.time = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Capture the post-forward initial state
    right_pos_0, right_quat_0, left_pos_0, left_quat_0 = _object_pose_slice(
        np.array(mj_data.qpos), config.embodiment_type, int(config.nq_obj)
    )

    # Optionally render
    record_video = bool(save_video_dir) and bool(getattr(config, "save_video", False))
    renderer = None
    frames: list[np.ndarray] = []
    render_every = 1
    if record_video:
        try:
            mj_model.vis.global_.offwidth = 256
            mj_model.vis.global_.offheight = 256
            renderer = mujoco.Renderer(mj_model, height=256, width=256)
            # ~30fps
            target_fps = 30.0
            render_every = max(int(round(1.0 / (target_fps * timestep))), 1)
        except Exception as exc:
            loguru.logger.warning("sanity check renderer setup failed: {}", exc)
            renderer = None
            record_video = False

    loguru.logger.info(
        "Running initial-state sanity check: holding ctrl_ref[0] for {:.2f}s ({} steps).",
        seconds,
        n_steps,
    )

    # Track MAX drift across the rollout, not just end-vs-start. An object
    # that bounces and then settles back near its origin would otherwise
    # silently pass the check.
    max_drift_pos_right = 0.0
    max_drift_pos_left = 0.0
    max_drift_rot_right = 0.0
    max_drift_rot_left = 0.0
    try:
        for step_i in range(n_steps):
            mj_data.ctrl[:] = ctrl0_np
            mujoco.mj_step(mj_model, mj_data)
            r_p, r_q, l_p, l_q = _object_pose_slice(
                np.array(mj_data.qpos), config.embodiment_type, int(config.nq_obj)
            )
            if right_pos_0 is not None and r_p is not None:
                max_drift_pos_right = max(
                    max_drift_pos_right, float(np.linalg.norm(r_p - right_pos_0))
                )
            if right_quat_0 is not None and r_q is not None:
                max_drift_rot_right = max(
                    max_drift_rot_right, _quat_angle(right_quat_0, r_q)
                )
            if left_pos_0 is not None and l_p is not None:
                max_drift_pos_left = max(
                    max_drift_pos_left, float(np.linalg.norm(l_p - left_pos_0))
                )
            if left_quat_0 is not None and l_q is not None:
                max_drift_rot_left = max(
                    max_drift_rot_left, _quat_angle(left_quat_0, l_q)
                )
            if renderer is not None and step_i % render_every == 0:
                try:
                    try:
                        renderer.update_scene(mj_data, "front")
                    except Exception:
                        renderer.update_scene(mj_data, 0)
                    frames.append(renderer.render())
                except Exception as exc:
                    loguru.logger.warning("sanity check render frame failed: {}", exc)
                    renderer = None
                    record_video = False
    finally:
        if renderer is not None:
            import contextlib

            with contextlib.suppress(Exception):
                renderer.close()

    right_pos_t, right_quat_t, left_pos_t, left_quat_t = _object_pose_slice(
        np.array(mj_data.qpos), config.embodiment_type, int(config.nq_obj)
    )

    drifts: list[tuple[str, float, float]] = []  # (label, value, threshold)
    pos_thresh = float(config.sanity_check_pos_thresh)
    rot_thresh = float(config.sanity_check_rot_thresh)

    # Use MAX-over-rollout drift so transient bounces (e.g., teapot
    # oscillating against a thin support cube) are caught even when the
    # end-frame pose returns close to the start.
    if right_pos_0 is not None and right_pos_t is not None:
        drifts.append(("right object max pos drift", max_drift_pos_right, pos_thresh))
    if right_quat_0 is not None and right_quat_t is not None:
        drifts.append(("right object max rot drift", max_drift_rot_right, rot_thresh))
    if left_pos_0 is not None and left_pos_t is not None:
        drifts.append(("left object max pos drift", max_drift_pos_left, pos_thresh))
    if left_quat_0 is not None and left_quat_t is not None:
        drifts.append(("left object max rot drift", max_drift_rot_left, rot_thresh))

    failures = [(label, val, thr) for (label, val, thr) in drifts if val > thr]

    video_path: str | None = None
    if record_video and frames:
        import os

        os.makedirs(save_video_dir, exist_ok=True)
        video_path = os.path.join(save_video_dir, "sanity_check_drift.mp4")
        try:
            import imageio

            imageio.mimsave(video_path, frames, fps=30)
            loguru.logger.info("Saved sanity-check video to {}", video_path)
        except Exception as exc:
            loguru.logger.warning("Failed to save sanity-check video: {}", exc)
            video_path = None

    if not failures:
        summary_parts = []
        for label, val, _ in drifts:
            unit = "cm" if "pos" in label else "rad"
            shown = val * 100.0 if unit == "cm" else val
            summary_parts.append(f"{label}={shown:.4f}{unit}")
        loguru.logger.info(
            "Sanity check passed ({} steps held at ctrl_ref[0]). {}",
            n_steps,
            "; ".join(summary_parts) if summary_parts else "no object DOFs to check",
        )
        return

    # Build error message
    detail_lines = []
    for label, val, thr in drifts:
        if "pos" in label:
            detail_lines.append(
                f"  - {label}: pos={val * 100.0:.3f}cm (threshold "
                f"{thr * 100.0:.3f}cm) over {seconds:.2f}s with controls held at "
                "ctrl_ref[0]"
            )
        else:
            detail_lines.append(
                f"  - {label}: rot={val:.4f}rad (threshold {thr:.4f}rad) over "
                f"{seconds:.2f}s with controls held at ctrl_ref[0]"
            )
    hints = [
        "  - If the object should rest on the floor: check the initial qpos "
        "and scene XML — the object may be floating or intersecting another "
        "body, causing it to fall or be pushed when physics starts.",
        "  - If the object should be in the hand: the IK / preprocessing "
        "produced an unstable grasp. Re-run preprocess/ik.py with a tighter "
        "wrist target, verify contact_pos targets, or adjust scene margins.",
    ]
    msg_parts = [
        "Initial-state sanity check FAILED: object drifted past thresholds "
        f"after holding ctrl_ref[0] for {seconds:.2f}s ({n_steps} CPU "
        "MuJoCo steps).",
        "Drift detected:",
        *detail_lines,
        "Hints:",
        *hints,
    ]
    if video_path:
        msg_parts.append(f"Drift video saved to: {video_path}")
    msg_parts.append(
        "Disable this check by setting `sanity_check_seconds=0.0` in your config."
    )
    raise RuntimeError("\n".join(msg_parts))


def get_trace(config: Config, env: MJWPEnv) -> torch.Tensor:
    """Return per-world trace points used for visualization. Minimal default returns
    an empty trace set of shape (N, 0, 3) when not configured.
    """
    site_xpos = wp.to_torch(env.data_wp.site_xpos)  # (N, nsite, 3)
    return site_xpos[:, config.trace_site_ids, :]


def save_state(env: MJWPEnv):
    """Clone the essential set of Warp arrays to restore later.
    Includes core state variables and key derived quantities.
    """
    _copy_state(env.data_wp, env.data_wp_prev)
    return env


def load_state(env: MJWPEnv, state):
    _copy_state(env.data_wp_prev, env.data_wp)
    return env


def apply_perturbation(config: Config, env: MJWPEnv):
    # get object id
    right_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "right_object"
    )
    left_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "left_object"
    )
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    if right_obj_id != -1:
        xfrc_applied[:, right_obj_id, :3] = config.perturb_force
        xfrc_applied[:, right_obj_id, 3:] = config.perturb_torque
    if left_obj_id != -1:
        xfrc_applied[:, left_obj_id, :3] = config.perturb_force
        xfrc_applied[:, left_obj_id, 3:] = config.perturb_torque
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    return env


def step_env(config: Config, env: MJWPEnv, ctrl_mujoco: torch.Tensor):
    """Step all worlds with provided MuJoCo-format controls of shape (N, nu)."""
    if ctrl_mujoco.dim() == 1:
        ctrl_mujoco = ctrl_mujoco.unsqueeze(0).repeat(env.num_worlds, 1)
    # Ensure we operate on the correct CUDA context/device
    with wp.ScopedDevice(env.device):
        # apply perturbation
        env = apply_perturbation(config, env)
        # step control
        wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl_mujoco.to(torch.float32)))
        wp.capture_launch(env.graph)


def save_env_params(config: Config, env: MJWPEnv):
    """Save the current simulation parameters."""
    # Only record which group is active; parameters are embedded in separate models
    # TODO: explicitly read pair_margin and xy_offset from env.data_wp
    # currently we choose this solution since pair_margin has a huge virtual dimension,
    # convert it to torch would lead to OOM
    pair_margin = 0.0
    xy_offset = 0.0
    return {"pair_margin": pair_margin, "xy_offset": xy_offset}


def load_env_params(config: Config, env: MJWPEnv, env_param: dict):
    """Load the simulation parameters.

    Parameters to be updated:
    - pair_margin
    - xy_offset of the object
    """
    # update model parameters (pair_margin)
    if "pair_margin" in env_param:
        pair_margin_single_np = np.full(
            shape=(config.npair,), fill_value=env_param["pair_margin"], dtype=np.float32
        )

        # 2. Copy this small array to the GPU
        pair_margin_override_wp = wp.from_numpy(
            pair_margin_single_np, dtype=wp.float32, device=config.device
        )

        # 3. Apply the stride trick to broadcast it
        # This makes Warp treat the single instance as if it were num_samples copies
        # without allocating any new memory.
        pair_margin_override_wp.strides = (0,) + pair_margin_override_wp.strides
        pair_margin_override_wp.shape = (
            config.num_samples,
        ) + pair_margin_override_wp.shape
        pair_margin_override_wp.ndim += 1
        wp.copy(env.model_wp.pair_margin, pair_margin_override_wp)

    # update object position (NOTE: currently, xy_offset is only one scalar, which means we only update in the diagonal direction)
    if "xy_offset" in env_param:
        qpos_override_th = wp.to_torch(env.data_wp.qpos)
        # TODO: make object pos detection automatic
        if config.embodiment_type == "bimanual":
            qpos_override_th[:, -14:-12] = (
                qpos_override_th[:, -14:-12] + env_param["xy_offset"]
            )
            qpos_override_th[:, -12:-10] = (
                qpos_override_th[:, -12:-10] + env_param["xy_offset"]
            )
        elif config.embodiment_type in ["right", "left"]:
            qpos_override_th[:, -7:-5] = (
                qpos_override_th[:, -7:-5] + env_param["xy_offset"]
            )

        wp.copy(env.data_wp.qpos, wp.from_torch(qpos_override_th))

    # update object actuator gains
    if "kp" in env_param or "kd" in env_param:
        actuator_ids = config.object_actuator_ids
        if not actuator_ids:
            loguru.logger.warning(
                "Object actuator ids are empty; skipping kp/kd updates."
            )
        else:
            kp = env_param.get("kp")
            kd = env_param.get("kd")
            if kp is None or kd is None:
                loguru.logger.warning(
                    "Both kp and kd are required to update actuator gains; skipping."
                )
            else:
                kp_np = np.asarray(kp, dtype=np.float32)
                kd_np = np.asarray(kd, dtype=np.float32)
                if kp_np.ndim == 0:
                    kp_np = np.full((len(actuator_ids),), kp_np, dtype=np.float32)
                if kd_np.ndim == 0:
                    kd_np = np.full((len(actuator_ids),), kd_np, dtype=np.float32)
                if kp_np.shape[0] != len(actuator_ids) or kd_np.shape[0] != len(
                    actuator_ids
                ):
                    raise ValueError(
                        "kp/kd size mismatch for object actuators: "
                        f"kp={kp_np.shape}, kd={kd_np.shape}, "
                        f"expected={len(actuator_ids)}"
                    )

                # Update CPU model (used for viewer and as source of truth)
                env.model_cpu.actuator_gainprm[actuator_ids, 0] = kp_np
                env.model_cpu.actuator_biasprm[actuator_ids, 1] = -kd_np

                # Propagate to MJWarp model if available
                if hasattr(env.model_wp, "actuator_gainprm") and hasattr(
                    env.model_wp, "actuator_biasprm"
                ):
                    gain_full = np.array(
                        env.model_cpu.actuator_gainprm, dtype=np.float32
                    )
                    bias_full = np.array(
                        env.model_cpu.actuator_biasprm, dtype=np.float32
                    )
                    wp.copy(
                        env.model_wp.actuator_gainprm,
                        wp.from_numpy(
                            gain_full, dtype=wp.float32, device=config.device
                        ),
                    )
                    wp.copy(
                        env.model_wp.actuator_biasprm,
                        wp.from_numpy(
                            bias_full, dtype=wp.float32, device=config.device
                        ),
                    )
                else:
                    loguru.logger.warning(
                        "MJWarp model has no actuator_gainprm/biasprm; updated CPU model only."
                    )

    return env


def _broadcast_state(data_wp, num_worlds: int):
    """Broadcast state from first world/env to all worlds/envs.

    This is a generic function that can be used by both MJWP and HDMI simulators.

    Args:
        data_wp: MuJoCo Warp data object (mjwarp.Data or wrapped version)
        num_worlds: Number of parallel worlds/environments
    """
    # Core state variables - always try these first
    qpos0 = wp.to_torch(data_wp.qpos)[:1]
    qvel0 = wp.to_torch(data_wp.qvel)[:1]
    time0 = wp.to_torch(data_wp.time)[:1]
    ctrl0 = wp.to_torch(data_wp.ctrl)[:1]

    # Handle time specially as it might be 1D
    if time0.dim() == 1:
        time_repeated = time0.repeat(num_worlds)
    else:
        time_repeated = time0.repeat(num_worlds, 1)

    wp.copy(data_wp.qpos, wp.from_torch(qpos0.repeat(num_worlds, 1)))
    wp.copy(data_wp.qvel, wp.from_torch(qvel0.repeat(num_worlds, 1)))
    wp.copy(data_wp.time, wp.from_torch(time_repeated))
    wp.copy(data_wp.ctrl, wp.from_torch(ctrl0.repeat(num_worlds, 1)))

    # Additional core state variables
    qacc0 = wp.to_torch(data_wp.qacc)[:1]
    wp.copy(data_wp.qacc, wp.from_torch(qacc0.repeat(num_worlds, 1)))

    act0 = wp.to_torch(data_wp.act)[:1]
    wp.copy(data_wp.act, wp.from_torch(act0.repeat(num_worlds, 1)))

    act_dot0 = wp.to_torch(data_wp.act_dot)[:1]
    wp.copy(data_wp.act_dot, wp.from_torch(act_dot0.repeat(num_worlds, 1)))

    # Forces and applied forces
    qfrc_applied0 = wp.to_torch(data_wp.qfrc_applied)[:1]
    wp.copy(data_wp.qfrc_applied, wp.from_torch(qfrc_applied0.repeat(num_worlds, 1)))

    xfrc_applied0 = wp.to_torch(data_wp.xfrc_applied)[:1]
    wp.copy(data_wp.xfrc_applied, wp.from_torch(xfrc_applied0.repeat(num_worlds, 1, 1)))

    # Mocap data
    mocap_pos0 = wp.to_torch(data_wp.mocap_pos)[:1]
    wp.copy(data_wp.mocap_pos, wp.from_torch(mocap_pos0.repeat(num_worlds, 1, 1)))

    mocap_quat0 = wp.to_torch(data_wp.mocap_quat)[:1]
    wp.copy(data_wp.mocap_quat, wp.from_torch(mocap_quat0.repeat(num_worlds, 1, 1)))

    # Spatial transformations
    xpos0 = wp.to_torch(data_wp.xpos)[:1]
    wp.copy(data_wp.xpos, wp.from_torch(xpos0.repeat(num_worlds, 1, 1)))

    xquat0 = wp.to_torch(data_wp.xquat)[:1]
    wp.copy(data_wp.xquat, wp.from_torch(xquat0.repeat(num_worlds, 1, 1)))

    xmat0 = wp.to_torch(data_wp.xmat)[:1]
    wp.copy(data_wp.xmat, wp.from_torch(xmat0.repeat(num_worlds, 1, 1, 1)))

    # Geometry positions
    geom_xpos0 = wp.to_torch(data_wp.geom_xpos)[:1]
    wp.copy(data_wp.geom_xpos, wp.from_torch(geom_xpos0.repeat(num_worlds, 1, 1)))

    geom_xmat0 = wp.to_torch(data_wp.geom_xmat)[:1]
    wp.copy(data_wp.geom_xmat, wp.from_torch(geom_xmat0.repeat(num_worlds, 1, 1, 1)))

    # Site positions
    site_xpos0 = wp.to_torch(data_wp.site_xpos)[:1]
    wp.copy(data_wp.site_xpos, wp.from_torch(site_xpos0.repeat(num_worlds, 1, 1)))


def sync_env(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Broadcast the state from first env to all envs

    This function synchronizes states from the first environment to all environments.
    Uses safe copying with buffer size validation to avoid mismatches.
    """
    _broadcast_state(env.data_wp, env.num_worlds)


def _copy_state(src: mjwarp.Data, dst: mjwarp.Data):
    """Copy the state from src to dst

    TODO: this function is a temporary solution for domain randomization. A better way should be defining a new warp kernel to update simulation parameter accordingly.

    Args:
        src: mjwarp.Data
            the source data to be copied from
        dst: mjwarp.Data
            the destination data to be copied to
    """
    # Core state variables
    wp.copy(dst.qpos, src.qpos)
    wp.copy(dst.qvel, src.qvel)
    wp.copy(dst.qacc, src.qacc)
    wp.copy(dst.time, src.time)
    wp.copy(dst.ctrl, src.ctrl)
    wp.copy(dst.act, src.act)
    wp.copy(dst.act_dot, src.act_dot)
    wp.copy(dst.qacc_warmstart, src.qacc_warmstart)

    # Forces and applied forces
    wp.copy(dst.qfrc_applied, src.qfrc_applied)
    wp.copy(dst.xfrc_applied, src.xfrc_applied)

    # Energy tracking
    wp.copy(dst.energy, src.energy)

    # Mocap data
    wp.copy(dst.mocap_pos, src.mocap_pos)
    wp.copy(dst.mocap_quat, src.mocap_quat)

    # Spatial transformations
    wp.copy(dst.xpos, src.xpos)
    wp.copy(dst.xquat, src.xquat)
    wp.copy(dst.xmat, src.xmat)
    wp.copy(dst.xipos, src.xipos)
    wp.copy(dst.ximat, src.ximat)

    # Geometry positions
    wp.copy(dst.geom_xpos, src.geom_xpos)
    wp.copy(dst.geom_xmat, src.geom_xmat)
    wp.copy(dst.site_xpos, src.site_xpos)
    wp.copy(dst.site_xmat, src.site_xmat)

    # Camera and lighting (if present)
    if hasattr(src, "cam_xpos") and hasattr(dst, "cam_xpos"):
        wp.copy(dst.cam_xpos, src.cam_xpos)
        wp.copy(dst.cam_xmat, src.cam_xmat)
    if hasattr(src, "light_xpos") and hasattr(dst, "light_xpos"):
        wp.copy(dst.light_xpos, src.light_xpos)
        wp.copy(dst.light_xdir, src.light_xdir)

    # Body dynamics
    wp.copy(dst.cacc, src.cacc)
    wp.copy(dst.cfrc_int, src.cfrc_int)
    wp.copy(dst.cfrc_ext, src.cfrc_ext)

    # Sensor data
    wp.copy(dst.sensordata, src.sensordata)

    # Actuator data
    wp.copy(dst.actuator_length, src.actuator_length)
    wp.copy(dst.actuator_velocity, src.actuator_velocity)
    wp.copy(dst.actuator_force, src.actuator_force)

    # Tendon data
    wp.copy(dst.ten_length, src.ten_length)
    wp.copy(dst.ten_velocity, src.ten_velocity)

    # Contact struct - copy all fields
    wp.copy(dst.contact.dist, src.contact.dist)
    wp.copy(dst.contact.pos, src.contact.pos)
    wp.copy(dst.contact.frame, src.contact.frame)
    wp.copy(dst.contact.includemargin, src.contact.includemargin)
    wp.copy(dst.contact.friction, src.contact.friction)
    wp.copy(dst.contact.solref, src.contact.solref)
    wp.copy(dst.contact.solreffriction, src.contact.solreffriction)
    wp.copy(dst.contact.solimp, src.contact.solimp)
    wp.copy(dst.contact.dim, src.contact.dim)
    wp.copy(dst.contact.geom, src.contact.geom)
    wp.copy(dst.contact.efc_address, src.contact.efc_address)
    wp.copy(dst.contact.worldid, src.contact.worldid)

    # Constraint (efc) struct - copy all fields
    wp.copy(dst.efc.type, src.efc.type)
    wp.copy(dst.efc.id, src.efc.id)
    wp.copy(dst.efc.J, src.efc.J)
    wp.copy(dst.efc.pos, src.efc.pos)
    wp.copy(dst.efc.margin, src.efc.margin)
    wp.copy(dst.efc.D, src.efc.D)
    wp.copy(dst.efc.vel, src.efc.vel)
    wp.copy(dst.efc.aref, src.efc.aref)
    wp.copy(dst.efc.frictionloss, src.efc.frictionloss)
    wp.copy(dst.efc.force, src.efc.force)
    # Note: The workspace fields (Jaref, Ma, grad, etc.) are typically not needed for state transfer
    # as they are recomputed during solving, but include if needed:
    # wp.copy(dst.efc.Jaref, src.efc.Jaref)
    # wp.copy(dst.efc.Ma, src.efc.Ma)
    # wp.copy(dst.efc.grad, src.efc.grad)
    # ... (other workspace fields)
    #
    return dst

"""Humanoid-only losses for DexForge's unchanged production FA optimizer.

Four palm/sole observation sites do not create contacts or CA anchors. Joint,
floating-root and object tracking use native rollout qpos; penetration uses all
physical collision rows, including the original foot patches and object floor.
"""

import math

import numpy as np
import warp as wp

from forceaware import mpc_loss
from forceaware.numerics import LOSS_GRADIENT_SCALE
from forceaware.servo_prior import ServoPriorOptimizer


@wp.struct
class RootTracking:
    qpos: wp.array(dtype=float)
    position_weight: float
    rotation_weight: float


@wp.func
def _root_reference_squared(job: mpc_loss.LossJob, root: RootTracking, world: int) -> float:
    squared = float(0.0)
    for axis in range(3):
        difference = root.position_weight * (job.state.qpos[world, axis] - root.qpos[axis])
        squared += difference * difference
    quaternion = wp.vec4(job.state.qpos[world, 3], job.state.qpos[world, 4],
                         job.state.qpos[world, 5], job.state.qpos[world, 6])
    reference = wp.vec4(root.qpos[3], root.qpos[4], root.qpos[5], root.qpos[6])
    angle = root.rotation_weight * mpc_loss._normalized_quaternion_angle(quaternion, reference)
    return squared + angle * angle


@wp.kernel
def _humanoid_loss(job: mpc_loss.LossJob, root: RootTracking, output: wp.array(dtype=float)):
    world = wp.tid()
    tracking = (mpc_loss._reference_pose_loss(job, world)
                + mpc_loss.MILLIMETERS * mpc_loss.MILLIMETERS
                * _root_reference_squared(job, root, world))
    # No marked-query or grasp-depth term exists without a CA trajectory.
    output[world] = job.reference_scale * tracking + mpc_loss._penetration_loss(job, world)


def _loss_model(index, contact_rows, *, guidance_count, device):
    model = mpc_loss.LossModel()
    model.hand_qpos = wp.array(index.hand_qpos, dtype=int, device=device)
    model.finger_joint_body = wp.empty(0, dtype=int, device=device)
    model.finger_joint_start = wp.empty(0, dtype=wp.vec3, device=device)
    model.finger_joint_end = wp.empty(0, dtype=wp.vec3, device=device)
    model.finger_joint_radius = wp.empty(0, dtype=float, device=device)
    model.finger_joint_pair_left = wp.empty(0, dtype=int, device=device)
    model.finger_joint_pair_right = wp.empty(0, dtype=int, device=device)
    model.output_contact_rows = wp.array(contact_rows, dtype=int, device=device)
    count = len(contact_rows)
    model.output_includemargin = wp.zeros(count, dtype=float, device=device)
    model.output_force_sign = wp.zeros(count, dtype=float, device=device)
    model.output_friction = wp.zeros(count, dtype=wp.vec2, device=device)
    model.wrench_force_direction = wp.empty(0, dtype=wp.vec3, device=device)
    model.wrench_torque_direction = wp.empty(0, dtype=wp.vec3, device=device)
    model.object_com_local = wp.vec3(0.0)
    model.object_radius = 1.0
    model.object_qpos = int(index.object_qpos)
    model.object_body = int(index.object_body)
    model.action_dim = int(index.action_dim)
    model.guidance_count = guidance_count
    model.output_count = count
    model.finger_joint_pair_count = 0
    model.wrench_direction_count = 0
    # Inactive rows return zero depth and contribute no penetration loss.
    model.contact_thresholds = wp.zeros(count, dtype=float, device=device)
    return model


class HumanoidObjective(mpc_loss.MpcObjective):
    """Objective extension; native optimizer and its adjoint are unmodified."""

    def __init__(self, *args, settings, root_pos_weight, root_rot_weight,
                 penetration_output_indices=None, **kwargs):
        self.settings = settings
        self.root_pos_weight = float(root_pos_weight)
        self.root_rot_weight = float(root_rot_weight)
        if any(not math.isfinite(value) or value < 0 for value in (self.root_pos_weight, self.root_rot_weight)):
            raise ValueError("DexForge root tracking weights must be finite and nonnegative")
        self.penetration_output_indices = penetration_output_indices
        self._root_jobs = {}
        super().__init__(*args, **kwargs)

    def _make_regularizer(self, cpu_model, index, cfg, *, worlds, device):
        regularizer = mpc_loss.MpcRegularizer(cpu_model, index, cfg, worlds=worlds, device=device)
        model, options = regularizer.model, self.settings
        # Every controlled G1 coordinate is a scalar joint angle. The upstream
        # first-six virtual-wrist unit rules do not apply to this ModelIndex.
        model.init_bound.fill_(options.init_joint_bound)
        model.control_velocity_scale.fill_(options.control_velocity_scale)
        model.control_velocity_weight.fill_(cfg.smoothing.control_velocity_weight * options.joint_component_ratio)
        model.control_acceleration_scale.fill_(options.control_acceleration_scale)
        model.state_velocity_scale.fill_(options.state_velocity_scale)
        model.state_acceleration_scale.fill_(options.state_acceleration_scale)
        model.state_component_ratio.fill_(options.joint_component_ratio)
        return regularizer

    def _contact_rows(self):
        count = int(self.steps[0].recorded.result.contact_distance.shape[1])
        if self.penetration_output_indices is None:
            # Contact rows include native floor and every original physical
            # collision pair, not merely the four logical observation outputs.
            return np.arange(count, dtype=np.int32)
        collision = self.compiled.base.collision
        outputs = np.asarray(collision.output_contact_rows.numpy(), np.int32)
        selected = np.asarray(self.penetration_output_indices, dtype=np.int32)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(outputs)):
            raise ValueError("Invalid DexForge penetration output indices")
        rows = outputs[selected]
        return np.unique(rows[(rows >= 0) & (rows < count)])

    def _workspace(self, runtime, *, qpos=None, reference_scale=1.0, contact_active=None):
        result = runtime.recorded.result
        state = mpc_loss.LossState()
        state.qpos = result.qpos if qpos is None else qpos
        for name in ("body_position", "body_matrix", "contact_distance", "contact_position", "contact_frame"):
            setattr(state, name, getattr(result, name))
        if contact_active is None:
            contact_active = runtime.workspace.forward.base.collision.contacts.active
        state.contact_active = contact_active
        job = mpc_loss.LossJob()
        job.model = _loss_model(self.index, self._contact_rows(), guidance_count=self.guidance_count, device=self.device)
        job.weights = mpc_loss.make_weights(self.cfg)
        job.state = state
        job.target = mpc_loss.allocate_target(self.index.action_dim, self.guidance_count, self.device)
        auxiliary = mpc_loss.LossAuxiliary()
        auxiliary.marked_query_weight = wp.zeros((self.worlds, self.guidance_count), dtype=float, device=self.device)
        job.auxiliary = auxiliary
        job.guidance_enabled = 0
        job.query_enabled = 0
        job.penetration_enabled = 1
        job.reference_scale = float(reference_scale)
        root = RootTracking()
        root.qpos = wp.array(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), dtype=float, device=self.device)
        root.position_weight = self.root_pos_weight
        root.rotation_weight = self.root_rot_weight
        loss = wp.empty(self.worlds, dtype=float, device=self.device, requires_grad=True)
        seed = wp.full(self.worlds, value=LOSS_GRADIENT_SCALE, dtype=float, device=self.device)
        tape = wp.Tape()
        tape.record_launch(_humanoid_loss, self.worlds, 0, [job, root], [loss],
                           wp.get_device(self.device), block_dim=256)
        workspace = mpc_loss.LossWorkspace(job.target, auxiliary, job, loss, seed, tape)
        self._root_jobs[id(workspace)] = root
        return workspace

    def _allocate_training(self, guidance_count):
        self.guidance_count = guidance_count
        immediate = tuple(self._workspace(runtime, reference_scale=(0.0 if i == self.dense_steps - 1 else 1.0))
                          for i, runtime in enumerate(self.steps))
        observation = self.terminal_observation
        terminal = self._workspace(observation, qpos=self.steps[-1].recorded.result.qpos,
                                   reference_scale=self.cfg.loss.terminal,
                                   contact_active=observation.workspace.collision.contacts.active)
        return mpc_loss.TrainingLossRuntime(
            immediate=immediate, terminal=terminal,
            trajectory_terms=wp.empty((self.worlds, self.dense_steps + 1), dtype=float, device=self.device),
            regularization=self.regularizer.runtime.total,
            total=wp.empty(self.worlds, dtype=float, device=self.device),
        )

    def prepare_window(self, targets, previous_action, *, previous_previous_action=None):
        root = np.asarray(getattr(targets, "root_qpos", None), dtype=np.float32)
        if root.shape != (self.dense_steps, 7) or not np.isfinite(root).all():
            raise ValueError("DexForge targets.root_qpos must be finite (dense_steps, 7) WXYZ poses")
        if np.any(np.linalg.norm(root[:, 3:], axis=-1) < 1e-8):
            raise ValueError("DexForge root reference quaternions must be nonzero")
        super().prepare_window(targets, previous_action, previous_previous_action=previous_previous_action)
        for step, workspace in enumerate(self.training.immediate):
            self._root_jobs[id(workspace)].qpos.assign(root[step])
        self._root_jobs[id(self.training.terminal)].qpos.assign(root[-1])

    def _evaluate_trajectory(self, runtime):
        for column, workspace in enumerate((*runtime.immediate, runtime.terminal)):
            wp.launch(_humanoid_loss, dim=self.worlds,
                      inputs=[workspace.job, self._root_jobs[id(workspace)]], outputs=[workspace.loss], device=self.device)
            wp.launch(mpc_loss._store_trajectory_loss, dim=self.worlds, inputs=[workspace.loss, column],
                      outputs=[runtime.trajectory_terms], device=self.device)


def create_optimizer(cpu_model, device_model, index, *, cfg, settings,
                     root_pos_weight=10.0, root_rot_weight=3.0,
                     penetration_output_indices=None, guidance_count=4):
    if not isinstance(guidance_count, int) or guidance_count < 1:
        raise ValueError("guidance_count must be positive")
    if index.action_dim != 29 or cpu_model.nq != 43 or cpu_model.nv != 41:
        raise ValueError("DexForge humanoid adapter requires 29 actuated joints and full qpos43/qvel41")
    if np.any(np.asarray(index.hand_qpos) < 7) or np.any(np.asarray(index.hand_qpos) >= index.object_qpos):
        raise ValueError("DexForge joint controls must exclude floating root and object qpos")

    class HumanoidWindowOptimizer(ServoPriorOptimizer):
        # The pinned upstream workflow imports ServoPriorOptimizer as its
        # WindowOptimizer. This is the sole solver-method override.
        def _make_objective(self, cpu_model, index, *, cfg, guidance_count):
            return HumanoidObjective(
                cpu_model, self.compiled, index, self.steps, self.terminal_observation,
                cfg=cfg, guidance_count=guidance_count, worlds=self.settings.worlds,
                settings=settings, root_pos_weight=root_pos_weight, root_rot_weight=root_rot_weight,
                penetration_output_indices=penetration_output_indices,
            )

    return HumanoidWindowOptimizer(cpu_model, device_model, index, guidance_count=guidance_count, cfg=cfg)

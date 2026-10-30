"""Forward-only ComFree rollout and original dex objective evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from mujoco_warp._src import smooth

from retarget.config.dex import RetargetConfig
from retarget.env.dex.control import action_knot_interpolation_map
from retarget.env.mjwarp import ComFreeEnvironment
from retarget.env.surface import SurfaceQueries
from retarget.task.dex.cost import MpcObjective
from retarget.task.dex.reference import ModelIndex

BASE_ACTION_DIM = 6


@wp.struct
class ControlModel:
    hand_qpos: wp.array(dtype=int)
    hand_ctrl: wp.array(dtype=int)
    init_bound: wp.array(dtype=float)
    action_dim: int
    qpos_dim: int
    qvel_dim: int


@wp.struct
class InitialStateJob:
    model: ControlModel
    raw: wp.array2d(dtype=float)
    base_qpos: wp.array(dtype=float)
    base_qvel: wp.array(dtype=float)
    hand_reference: wp.array(dtype=float)
    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    allow_init: float


@wp.struct
class ControlJob:
    model: ControlModel
    raw: wp.array2d(dtype=float)
    target: wp.array(dtype=float)
    action: wp.array2d(dtype=float)
    step: int


@wp.struct
class DenseControlJob:
    model: ControlModel
    left_action: wp.array2d(dtype=float)
    right_action: wp.array2d(dtype=float)
    control: wp.array2d(dtype=float)
    fraction: float


@wp.kernel
def _copy_initial(job: InitialStateJob):
    world, index = wp.tid()
    if index < job.model.qpos_dim:
        job.qpos[world, index] = job.base_qpos[index]
    if index < job.model.qvel_dim:
        job.qvel[world, index] = job.base_qvel[index]


@wp.kernel
def _apply_initial_delta(job: InitialStateJob):
    world, action = wp.tid()
    delta = job.model.init_bound[action] * wp.tanh(job.raw[world, action])
    qpos = job.model.hand_qpos[action]
    job.qpos[world, qpos] = job.hand_reference[action] + job.allow_init * delta


@wp.kernel
def _build_control_kernel(job: ControlJob):
    world, action = wp.tid()
    offset = job.model.action_dim * (1 + job.step) + action
    job.action[world, action] = job.target[action] + job.raw[world, offset]


@wp.kernel(enable_backward=False)
def _build_dense_control_kernel(job: DenseControlJob):
    world, action = wp.tid()
    left = job.left_action[world, action]
    right = job.right_action[world, action]
    job.control[world, job.model.hand_ctrl[action]] = left + job.fraction * (
        right - left
    )


def _grouped_action_values(
    action_dim: int,
    values: tuple[float, float, float],
) -> np.ndarray:
    output = np.full(action_dim, values[2], np.float32)
    output[:3] = values[0]
    output[3:BASE_ACTION_DIM] = values[1]
    return output


def _control_model(
    cpu_model, index: ModelIndex, device, *, cfg: RetargetConfig
) -> ControlModel:
    bounds = cfg.bounds
    model = ControlModel()
    model.hand_qpos = wp.array(index.hand_qpos, dtype=int, device=device)
    model.hand_ctrl = wp.array(index.hand_ctrl, dtype=int, device=device)
    model.init_bound = wp.array(
        _grouped_action_values(
            index.action_dim,
            (bounds.init_base_pos, bounds.init_base_rot, bounds.init_finger),
        ),
        dtype=float,
        device=device,
    )
    model.action_dim = index.action_dim
    model.qpos_dim = cpu_model.nq
    model.qvel_dim = cpu_model.nv
    return model


def _build_initial(control: SimpleNamespace, *, allow_init: bool) -> None:
    job = InitialStateJob()
    job.model = control.model
    job.raw = control.raw
    job.base_qpos = control.base_qpos
    job.base_qvel = control.base_qvel
    job.hand_reference = control.hand_reference
    job.qpos = control.initial_qpos
    job.qvel = control.initial_qvel
    job.allow_init = float(allow_init)
    worlds = control.raw.shape[0]
    width = max(control.model.qpos_dim, control.model.qvel_dim)
    wp.launch(_copy_initial, dim=(worlds, width), inputs=[job])
    wp.launch(
        _apply_initial_delta, dim=(worlds, control.model.action_dim), inputs=[job]
    )


def _build_control(control: SimpleNamespace, step: int) -> None:
    job = ControlJob()
    job.model = control.model
    job.raw = control.raw
    job.target = control.target_actions[step]
    job.action = control.actions[step]
    job.step = step
    wp.launch(
        _build_control_kernel,
        dim=(control.raw.shape[0], control.model.action_dim),
        inputs=[job],
    )


def _build_dense_control(
    control: SimpleNamespace, dense_step: int, knot_map: tuple
) -> None:
    left, right, fraction = knot_map
    control.controls[dense_step].zero_()
    job = DenseControlJob()
    job.model = control.model
    job.left_action = control.previous_action if left < 0 else control.actions[left]
    job.right_action = control.actions[right]
    job.control = control.controls[dense_step]
    job.fraction = fraction
    wp.launch(
        _build_dense_control_kernel,
        dim=(control.raw.shape[0], control.model.action_dim),
        inputs=[job],
    )


class ForwardRollout:
    def __init__(
        self,
        cpu_model,
        device_model,
        index,
        cfg,
        guidance_count,
        worlds,
        *,
        surfaces=None,
    ):
        self.cpu_model, self.index, self.cfg = cpu_model, index, cfg
        grid = cfg.simulator.time_grid
        self.horizon = cfg.prediction.horizon
        self.worlds = worlds
        self.device = device_model.qpos0.device
        self.env = ComFreeEnvironment(
            cpu_model,
            device_model,
            worlds,
            stiffness=cfg.simulator.stiffness,
            damping=cfg.simulator.damping,
        )
        self.surfaces = surfaces or SurfaceQueries(
            cpu_model, index.object_body, device=self.device
        )
        matrix = lambda width: wp.zeros(
            (worlds, width), dtype=float, device=self.device
        )
        self.control = c = SimpleNamespace(
            model=_control_model(cpu_model, index, self.device, cfg=cfg),
            raw=matrix(index.action_dim * (1 + self.horizon)),
            initial_qpos=matrix(cpu_model.nq),
            initial_qvel=matrix(cpu_model.nv),
            base_qpos=wp.zeros(cpu_model.nq, device=self.device),
            base_qvel=wp.zeros(cpu_model.nv, device=self.device),
            hand_reference=wp.zeros(index.action_dim, device=self.device),
            previous_action=matrix(index.action_dim),
            target_actions=tuple(
                wp.zeros(index.action_dim, device=self.device)
                for _ in range(self.horizon)
            ),
            actions=tuple(matrix(index.action_dim) for _ in range(self.horizon)),
            controls=tuple(
                matrix(cpu_model.nu) for _ in range(self.horizon * grid.knot_substeps)
            ),
        )
        self.dense_map = action_knot_interpolation_map(self.horizon, grid.knot_substeps)
        self.steps = []
        self.initial_time = 0.0
        for control in c.controls:
            result = self._allocate_record()
            self.steps.append(
                SimpleNamespace(
                    inputs=SimpleNamespace(dynamics=SimpleNamespace(ctrl=control)),
                    recorded=SimpleNamespace(result=result),
                )
            )
        self.observation = SimpleNamespace(
            recorded=SimpleNamespace(result=self._allocate_record())
        )
        self.objective = MpcObjective(
            cpu_model,
            SimpleNamespace(device=self.device, surfaces=self.surfaces),
            index,
            self.steps,
            self.observation,
            cfg=cfg,
            guidance_count=guidance_count,
            worlds=worlds,
        )
        self.loss = self.objective.training

    def prepare(self, qpos, qvel, targets, previous_action, time=0.0):
        c = self.control
        c.base_qpos.assign(np.asarray(qpos, np.float32))
        c.base_qvel.assign(np.asarray(qvel, np.float32))
        c.hand_reference.assign(np.asarray(qpos, np.float32)[self.index.hand_qpos])
        c.previous_action.assign(
            np.broadcast_to(
                previous_action, (self.worlds, self.index.action_dim)
            ).astype(np.float32)
        )
        self.initial_time = time
        for a, t in zip(c.target_actions, targets.actions):
            a.assign(t)
        self.objective.prepare_window(targets, previous_action)

    def forward(self):
        _build_initial(self.control, allow_init=False)
        for step in range(self.horizon):
            _build_control(self.control, step)
        for step, knot_map in enumerate(self.dense_map):
            _build_dense_control(self.control, step, knot_map)
        data = self.env.data
        wp.copy(data.qpos, self.control.initial_qpos)
        wp.copy(data.qvel, self.control.initial_qvel)
        data.time.fill_(self.initial_time)
        data.qacc_warmstart.zero_()
        for runtime in self.steps:
            wp.copy(data.ctrl, runtime.inputs.dynamics.ctrl)
            self.env.evaluate_dynamics()
            # Preserve the original objective's pre-step geometry/post-step
            # tracking timing, without retaining an entire engine per step.
            self._record_geometry(runtime.recorded.result)
            self.env.integrate()
            result = runtime.recorded.result
            wp.copy(result.qpos, data.qpos)
            wp.copy(result.qvel, data.qvel)
            wp.copy(result.qacc, data.qacc)
        smooth.kinematics(self.env.model, data)
        self._record_geometry(self.observation.recorded.result)

    def _allocate_record(self):
        result = self.surfaces.allocate(self.worlds)
        for name, width in (
            ("qpos", self.cpu_model.nq),
            ("qvel", self.cpu_model.nv),
            ("qacc", self.cpu_model.nv),
        ):
            setattr(result, name, wp.zeros((self.worlds, width), device=self.device))
        result.body_position = wp.zeros(
            (self.worlds, self.cpu_model.nbody), dtype=wp.vec3, device=self.device
        )
        result.body_matrix = wp.zeros(
            (self.worlds, self.cpu_model.nbody), dtype=wp.mat33, device=self.device
        )
        return result

    def _record_geometry(self, result):
        data = self.env.data
        wp.copy(result.body_position, data.xpos)
        wp.copy(result.body_matrix, data.xmat)
        self.surfaces.measure(result.body_position, result.body_matrix, result)

    def score(self, values):
        costs = []
        for chunk in values.split(self.worlds):
            x = wp.to_torch(self.control.raw)
            x.zero_()
            x[: len(chunk), self.index.action_dim :] = chunk
            if len(chunk) < self.worlds:
                x[len(chunk) :] = x[0]
            self.forward()
            self.objective.evaluate(self.control, allow_init=False)
            scores = wp.to_torch(self.loss.total).clone()
            finite = torch.isfinite(wp.to_torch(self.env.data.qpos)).all(1)
            finite &= torch.isfinite(wp.to_torch(self.env.data.qvel)).all(1)
            scores[~finite] = torch.inf
            costs.append(scores[: len(chunk)])
        return torch.cat(costs)

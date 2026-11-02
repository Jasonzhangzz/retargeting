"""Native MuJoCo-Warp batch/state adapter for SPIDER sampling rollouts.

No contact response, mass, inertia, material or actuator parameter is replaced.
Like SPIDER, ``step`` leaves body/site transforms at the engine's pre-integration
observation while qpos/qvel contain the post-integration state. Call the explicit
``refresh_kinematics`` only when an observation needs the latter poses.
"""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .capture import capture_guard


# mjSTATE_INTEGRATION inputs plus derivatives copied by the SPIDER adapter.
# Contact/constraint arrays are derived and rebuilt by the next native step.
STATE_FIELDS = (
    "time", "qpos", "qvel", "act", "history", "qacc_warmstart", "ctrl",
    "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
    "userdata", "act_dot", "qacc",
)


def control_bounds(cpu):
    """Optional admissible input projection; the native engine clamps by itself."""
    bounds = np.asarray(cpu.actuator_ctrlrange, np.float32).copy()
    limited = np.asarray(cpu.actuator_ctrllimited, bool)
    bounds[~limited] = (-np.inf, np.inf)
    return bounds[:, 0], bounds[:, 1]


class BatchedPhysics:
    """One native model and independent physical states for all MPPI candidates.

    Use a separate one-world instance for execution. ``warmup`` uses a second
    data buffer, so graph preparation never advances the measured initial state.
    All allocations and state buffers live on one device. ``reset``/``restore``
    and ``step(use_graph=False)`` are device-only for already-device Tensor inputs,
    allowing a caller to capture an entire prediction horizon after ``warmup``.
    """

    def __init__(self, cpu, worlds, device="cuda:0", *, nconmax=256, njmax=2048,
                 nccdmax=None, capture=True):
        if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds < 1:
            raise ValueError("Native batch size must be a positive integer")
        if cpu.nplugin or cpu.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
            raise ValueError("Sampling state restore does not support plugins or sleeping bodies")
        self.cpu, self.worlds, self.device = cpu, worlds, str(device)
        self.capture_enabled = bool(capture)
        self.graph = None
        capacities = dict(nworld=worlds, nconmax=int(nconmax), njmax=int(njmax),
                          nccdmax=int(nconmax if nccdmax is None else nccdmax))
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(cpu)
            # Ordinary fixed solver loops are equivalent to conditional nodes;
            # older CUDA drivers cannot capture the latter (upstream SPIDER fix).
            if (hasattr(self.model.opt, "graph_conditional")
                    and wp.get_cuda_driver_version() < (12, 4)):
                self.model.opt.graph_conditional = False
            self.data = mjw.make_data(cpu, **capacities)
            self._warm_data = mjw.make_data(cpu, **capacities)
            self._capture_stream = wp.Stream(self.device)
        self._defaults = self._snapshot_data(self.data)
        low, high = control_bounds(cpu)
        self.control_low = torch.as_tensor(low, device=self.device)
        self.control_high = torch.as_tensor(high, device=self.device)

    @staticmethod
    def _snapshot_data(data, world=None):
        state = {}
        for name in STATE_FIELDS:
            value = wp.to_torch(getattr(data, name))
            state[name] = (value if world is None else value[world:world+1]).clone()
        return state

    def snapshot(self, world=0):
        """Clone the full integration state of one world, not just qpos/qvel."""
        if not 0 <= world < self.worlds:
            raise IndexError("Snapshot world is outside the native batch")
        return self._snapshot_data(self.data, world)

    def _copy_field(self, target, name, value):
        destination = wp.to_torch(getattr(target, name))
        source = torch.as_tensor(value, dtype=destination.dtype, device=destination.device)
        if source.shape == destination.shape[1:]:
            source = source.unsqueeze(0)
        if source.shape != destination.shape:
            if source.shape != (1, *destination.shape[1:]):
                raise ValueError(f"Invalid {name} batch: {tuple(source.shape)} for {tuple(destination.shape)}")
            source = source.expand_as(destination)
        destination.copy_(source)

    def restore(self, state):
        """Broadcast a measured one-world snapshot across candidates.

        Every persistent field is required: partial snapshots must not leak
        actuator/history/force/warm-start state from the previous candidate.
        """
        if not isinstance(state, Mapping) or set(state) != set(STATE_FIELDS):
            raise ValueError("Native restore requires a complete integration-state snapshot")
        for name in STATE_FIELDS:
            self._copy_field(self.data, name, state[name])
        self.data.overflow.zero_()

    def reset(self, qpos, qvel, *, time=0., control=None, state=None):
        """Reset to defaults or a full snapshot, then set explicit state inputs."""
        self.restore(self._defaults if state is None else state)
        self._copy_field(self.data, "qpos", qpos)
        self._copy_field(self.data, "qvel", qvel)
        if time is not None:
            wp.to_torch(self.data.time).fill_(float(time))
        if control is not None:
            self._copy_field(self.data, "ctrl", control)

    def project_ctrl(self, controls):
        """Explicit optional input clipping; disabled in the source-parity step."""
        values = torch.as_tensor(controls, dtype=torch.float32, device=self.device)
        return torch.maximum(torch.minimum(values, self.control_high), self.control_low)

    def warmup(self):
        """Warm native kernels and prepare a one-step graph without moving data."""
        if self.graph is not None:
            return
        # A caller can be on Torch's legacy default stream (CUDA stream 0),
        # which CUDA cannot capture. Prepare the child graph on our own stream;
        # graph launches later use the caller's current Warp stream as usual.
        wp.synchronize_device(self.device)
        with wp.ScopedDevice(self.device), wp.ScopedStream(self._capture_stream):
            for name in STATE_FIELDS:
                wp.copy(getattr(self._warm_data, name), getattr(self.data, name))
            mjw.step(self.model, self._warm_data)
            wp.synchronize_device(self.device)
            if self.capture_enabled:
                with capture_guard(), wp.ScopedCapture(stream=self._capture_stream) as capture:
                    mjw.step(self.model, self.data)
                self.graph = capture.graph
                wp.synchronize_device(self.device)

    def step(self, controls, *, use_graph=True, clip=False):
        """Advance real batched dynamics; no surrogate forces or state projection."""
        if use_graph and self.capture_enabled and self.graph is None:
            # Run after the first control copy so warmup sees the actual input.
            self._copy_field(self.data, "ctrl", self.project_ctrl(controls) if clip else controls)
            self.warmup()
        else:
            self._copy_field(self.data, "ctrl", self.project_ctrl(controls) if clip else controls)
        if use_graph and self.capture_enabled:
            wp.capture_launch(self.graph)
        else:
            mjw.step(self.model, self.data)

    def refresh_kinematics(self):
        """Opt-in post-state body/site poses; never called implicitly by step."""
        from mujoco_warp._src import smooth
        smooth.kinematics(self.model, self.data)

    @property
    def qpos(self):
        return wp.to_torch(self.data.qpos)

    @property
    def qvel(self):
        return wp.to_torch(self.data.qvel)

    @property
    def xpos(self):
        return wp.to_torch(self.data.xpos)

    @property
    def xmat(self):
        return wp.to_torch(self.data.xmat).reshape(self.worlds, self.cpu.nbody, 3, 3)

    def check(self):
        """Synchronizing audit for outside a capture; reject capacity overflows."""
        overflow = wp.to_torch(self.data.overflow).clone()
        finite = torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1)
        constraints = wp.to_torch(self.data.nefc).clone()
        valid = finite & (overflow == 0) & (constraints <= self.data.njmax)
        return dict(valid=valid, finite=finite, overflow=overflow,
                    constraint_count=constraints,
                    total_contacts=int(self.data.nacon.numpy()[0]),
                    contact_capacity=self.data.naconmax,
                    constraints_per_world_capacity=self.data.njmax)


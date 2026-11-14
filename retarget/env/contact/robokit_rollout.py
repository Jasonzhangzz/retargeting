"""Named-state/control-sequence adapter around original Robokit SimMPC rollouts."""
from dataclasses import replace
import numpy as np
import warp as wp
from robokit.helpers.sim_mpc import SimMPC, SimMPCStage
from robokit.sim import MjWarpSim
from .native import snapshot


class StatefulSimMPC(SimMPC):
    """Extend source reset graphs with the remaining named integration fields."""
    def __init__(self, config, sim, ctrl_ref):
        self.named_state = snapshot(sim._args['data']) if sim._args['data'] is not None else {}
        self.extra_buffers = {}
        super().__init__(config, sim, ctrl_ref)

    def _build(self, worlds: str, nodes: wp.array, spans: set) -> None:
        extensions = [term for term, _, _ in self.config.terms if hasattr(term, 'register')]
        for term in extensions:
            term.register(self.sims[worlds])
        super()._build(worlds, nodes, spans)
        sim = self.sims[worlds]
        buffers = []
        for name, value in self.named_state.items():
            if name in self.state or name == 'ctrl' or not np.size(value) or not hasattr(sim._d, name):
                continue
            field = getattr(sim._d, name)
            if not isinstance(field, wp.array) or not field.size:
                continue
            buffer = wp.empty_like(field)
            shape = buffer.numpy().shape
            buffer.assign(np.broadcast_to(value, shape).copy())
            buffers.append((name, field, buffer, shape))
        self.extra_buffers[worlds] = buffers
        original_reset = self.graphs[worlds + '_reset']
        def reset() -> None:
            original_reset()
            for _, field, buffer, _ in buffers:
                wp.copy(field, buffer)
            if hasattr(sim._d, 'overflow'):
                sim._d.overflow.zero_()
            for term in extensions:
                term.reset(sim)
        self.graphs[worlds + '_reset'] = reset

    def set_named_state(self, state: dict) -> None:
        self.named_state = state
        for name, field in self.state.items():
            field.assign(np.asarray(state[name]).reshape(field.shape))
        for buffers in self.extra_buffers.values():
            for name, _, buffer, shape in buffers:
                buffer.assign(np.broadcast_to(state[name], shape).copy())


class GPURollout(StatefulSimMPC):
    def __init__(self, task, population: int, device: str = 'cuda:0', extra_term=None):
        self.task, self.population, self.trajectories, self.calls = task, population, 0, 0
        self.extra_term, self.cold_rollouts = extra_term, 0
        config = replace(task.config, num_samples=population + 1, node_steps=1,
            interpolation='zero', ctrl_std=0., init_iterations=0,
            stages=[SimMPCStage(1)], min_improvement=-np.inf)
        if extra_term is not None:
            config = replace(config, terms=[*config.terms, (extra_term, 1., 1.)])
        with wp.ScopedDevice(device):
            ncon, nj = {'sharpa_pen': (48, 300), 'sharpa_coin': (64, 400), 'h1_jog': (120, 500)}[task.name]
            sim = MjWarpSim(task.plan_scene.compile(), 1, task.state, nconmax=ncon, njmax=nj, device=device)
            super().__init__(config, sim, np.zeros_like(task.context['ctrl_ref']))
            self.cost_buffer = wp.zeros(population, dtype=float, device=device)
        self.stream = sim.stream
        self.device_name = device
        self.physics = self.sims['rollout']

    def compute_weighted_cost(self, values, *, out_cost: wp.array) -> None:
        self.cold_rollouts += values.batch_size
        super().compute_weighted_cost(values, out_cost=out_cost)

    def evaluate(self, controls: np.ndarray, state: dict, packet: dict, events: list,
                 extra: bool = True, coarse: tuple = (1., 1)) -> np.ndarray:
        if coarse != (1., 1):
            raise ValueError('Original SimMPC owns coarse source-baseline schedules; CEM rollouts use the original fine model')
        count = len(controls)
        if not 1 < count <= self.population:
            raise ValueError('CEM proposal batches require 2..population rows')
        finite = np.isfinite(controls).all(axis=(1, 2))
        controls = np.nan_to_num(controls, nan=0., posinf=0., neginf=0.)
        padded = np.repeat(controls[:1], self.population, axis=0)
        padded[:count] = controls
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self.set_named_state(state)
            self.ctrl_ref.zero_()
            if self.extra_term is not None:
                self.extra_term.update(packet, events, count)
            self.clock.assign([self.task.clock, 0, self.task.profile.horizon])
            self.iteration = 0
            self.cost_buffer.zero_()
            self.sampler.draws.get(self.var_key).x.assign(padded.astype(np.float32).reshape(self.population, -1))
            super().compute_weighted_cost(self.sampler.draws, out_cost=self.cost_buffer)
            wp.synchronize_device(self.device)
            cost = self.cost_buffer.numpy()[:count].astype(float)
            q, v = self.physics.data.qpos.numpy()[:count], self.physics.data.qvel.numpy()[:count]
            valid = np.isfinite(q).all(1) & np.isfinite(v).all(1)
            if hasattr(self.physics._d, 'overflow'):
                import mujoco_warp as mjw
                overflow = np.asarray(self.physics._d.overflow.numpy()).reshape(self.population, -1)
                ignored = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS) if hasattr(mjw, 'OverflowType') else 0
                valid &= np.all((overflow.astype(np.int64) & ~ignored) == 0, axis=1)[:count]
        cost[~valid | ~finite | ~np.isfinite(cost)] = np.inf
        self.trajectories += count
        self.calls += 1
        return cost

"""Original SimMPC, with CPU4 review; full-joint CEM is its sampler extension."""
from dataclasses import replace
import gc
import time
import numpy as np
import torch
import warp as wp
from robokit.helpers.sim_mpc import SimMPCStage
from robokit.helpers.sim_mpc.sim_mpc import _compute_ctrl_kernel
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage
from robokit.sim import MjWarpSim
from retarget.solver.base import Result, NoValidCandidate
from retarget.solver.contact_cem import DEFAULTS
from retarget.solver.robokit_cem import CEMSamplerConfig, CEMPopulationSolver
from .robokit_rollout import StatefulSimMPC


class ControlPlanner(StatefulSimMPC):
    def __init__(self, task, method: str, device: str, extra_term=None):
        if method not in ('source_sampling', 'cem'):
            raise ValueError('ControlPlanner supports original SimMPC and its joint-CEM extension')
        self.task, self.method, self.recording = task, method, False
        self.archive, self.trajectories, self.cold_rollouts, self.gpu_seconds = [], 0, 0, 0.
        self.extra_term = extra_term
        config = task.config
        if method == 'cem':
            config = replace(config, num_samples=task.profile.samples + 1,
                stages=[SimMPCStage(task.profile.iterations)], min_improvement=-np.inf)
        if extra_term is not None:
            config = replace(config, terms=[*config.terms, (extra_term, 1., 1.)])
        with wp.ScopedDevice(device):
            ncon, nj = {'sharpa_pen': (48, 300), 'sharpa_coin': (64, 400), 'h1_jog': (120, 500)}[task.name]
            sim = MjWarpSim(task.plan_scene.compile(), 1, task.state, nconmax=ncon, njmax=nj, device=device)
            super().__init__(config, sim, task.context['ctrl_ref'])
            if method == 'cem':
                mask = self.sampler.active_dof_mask
                self.graphs.clear(); self.terms.clear(); self.extra_buffers.clear()
                if extra_term is not None:
                    extra_term.bindings.clear()
                gc.collect()
                sampler = CEMSamplerConfig(var_key=self.var_key, num_draws=task.profile.samples,
                    branch_sigma=self.node_std, seed=task.seed,
                    cem_elite_fraction=DEFAULTS.elite_fraction, cem_refit=DEFAULTS.refit,
                    cem_min_sigma_fraction=DEFAULTS.min_sigma_fraction)
                self.solver = CEMPopulationSolver([self], PopulationSolverConfig(
                    stages=[Stage(task.profile.iterations, sample_every=1)], samplers=[sampler],
                    cuda_graph_mode='none'), device, active_dof_mask=mask)
                self.solver.setup(self.plan_values)
                self.sampler = self.solver.samplers[0]
        self.stream = sim.stream
        self.previous_exists = False

    def compute_weighted_cost(self, values, *, out_cost: wp.array) -> None:
        before = time.perf_counter()
        super().compute_weighted_cost(values, out_cost=out_cost)
        nodes = values.get(self.var_key).x
        count = nodes.shape[0]
        if not self.recording:
            self.cold_rollouts += count
            return
        worlds = 'rollout' if count > 1 else 'validation'
        sim = self.sims[worlds]
        valid = np.isfinite(sim.data.qpos.numpy()).all(1) & np.isfinite(sim.data.qvel.numpy()).all(1)
        if hasattr(sim._d, 'overflow'):
            import mujoco_warp as mjw
            ignored = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS) if hasattr(mjw, 'OverflowType') else 0
            overflow = np.asarray(sim._d.overflow.numpy()).reshape(count, -1).astype(np.int64)
            valid &= np.all((overflow & ~ignored) == 0, axis=1)
        costs, params = out_cost.numpy(), nodes.numpy()
        costs[~valid | ~np.isfinite(costs)] = np.inf
        out_cost.assign(costs)
        ids = np.argsort(costs, kind='stable')[:DEFAULTS.cpu_candidates]
        self.archive.extend((float(costs[i]), params[i].copy()) for i in ids if np.isfinite(costs[i]))
        self.trajectories += count
        self.gpu_seconds += time.perf_counter() - before

    def controls(self, nodes: np.ndarray, clock: int) -> np.ndarray:
        nodes = np.asarray(nodes, np.float32).reshape(-1, self.num_nodes * self.nu)
        out = wp.zeros((self.task.profile.horizon + 1, len(nodes), self.nu), dtype=float, device=self.device)
        clock_array = wp.array([clock, 0, 0], dtype=int, device=self.device)
        wp.launch(_compute_ctrl_kernel, dim=(len(nodes), out.shape[0], self.nu),
            inputs=[wp.array(nodes, dtype=float, device=self.device), self.interpolation,
                    self.ctrl_ref, clock_array, 1, len(nodes), self.command_bounds], outputs=[out], device=self.device)
        return out.numpy().transpose(1, 0, 2).copy()

    def _optimize(self, t0: int) -> list:
        previous = self.execution.numpy().copy() if self.previous_exists else None
        self.archive.clear()
        self.recording = True
        try:
            if self.method == 'source_sampling':
                infos = super()._optimize(t0)
            else:
                # Source SimMPC's state, node-bound and epoch setup; CEM changes
                # only the sampler and ends on the last scored proposal batch.
                cfg = self.config
                iterations = cfg.init_iterations if t0 == 0 and cfg.init_iterations else cfg.max_iterations
                self.clock.assign([t0, 0, cfg.horizon_steps])
                self.solver.config.stages = [Stage(iterations, sample_every=1)]
                self.t0, self.iteration = t0, 0
                self.sampler.epoch.fill_(t0 // cfg.ctrl_steps)
                node_ref = self._host['ctrl_ref'][:, np.minimum(
                    t0 + np.arange(self.num_nodes) * cfg.node_steps, self._host['ctrl_ref'].shape[1] - 1)]
                bounds = self._host['bounds']
                self.sampler.lower = wp.array((bounds[:, 0] - node_ref).reshape(1, -1), dtype=float, device=self.device)
                self.sampler.upper = wp.array((bounds[:, 1] - node_ref).reshape(1, -1), dtype=float, device=self.device)
                self.solver.solve(self.plan_values)
                infos = [dict(t0=t0, iterations=iterations,
                    optimizer_backend='robokit.PopulationSolver/CEMSampler(GaussianSampler)',
                    history=self.history.numpy()[:self.iteration].tolist())]
        finally:
            self.recording = False
        if not self.archive:
            raise NoValidCandidate(infos)
        selected, seen = [], set()
        for cost, nodes in sorted(self.archive, key=lambda item: item[0]):
            key = nodes.tobytes()
            if key not in seen:
                selected.append(nodes); seen.add(key)
            if len(selected) == DEFAULTS.cpu_candidates:
                break
        if previous is not None:
            selected.append(previous.ravel())
        controls = self.controls(np.asarray(selected), t0)
        checked = self.validate(controls, [self.events] * len(controls))
        cost = np.array([row['cost'] for row in checked]); cost[~np.isfinite(cost)] = np.inf
        if not np.isfinite(cost).any():
            raise NoValidCandidate(infos)
        winner = int(cost.argmin())
        self.execution.assign(np.asarray(selected[winner]).reshape(1, -1))
        self.last_result = Result(torch.as_tensor(selected[winner]), float(cost[winner]), infos,
            self.trajectories - self.window_before, self.trajectories - self.window_before + len(checked),
            dict(controls=controls[winner], events=self.events, cpu4=checked[winner],
                cpu4_evaluated=len(checked), optimizer_backend='original_robokit_SimMPC' if self.method == 'source_sampling'
                else 'robokit_SimMPC_with_CEMSampler', cold_gpu_rollouts=self.cold_rollouts))
        return infos

    def solve(self, state: dict, clock: int, packet: dict, events: list, validate) -> Result:
        self.validate, self.events = validate, events
        self.window_before = self.trajectories
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self.set_named_state(state)
            reference = self.task.context['ctrl_ref']
            self.ctrl_ref.assign(reference if reference.ndim == 3 else reference[None])
            self._host['ctrl_ref'] = reference if reference.ndim == 3 else reference[None]
            if self.extra_term is not None:
                self.extra_term.update(packet, [events] * self.num_samples, self.num_samples)
            self.plan(self.task.mirror.probe, clock)
            self.previous_exists = True
        return self.last_result

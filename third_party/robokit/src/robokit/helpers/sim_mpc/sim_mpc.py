# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Receding-horizon MPPI (`PopulationSolver` with a `GaussianSampler`) scored by batched simulator rollouts.

Each `SimMPC.plan`, from every instance's given state:
1. `_optimize`: draw node offsets around each plan (draw zero is the plan) and score each by its rollout.
2. Move each plan to the draws' cost-weighted mean and redraw, until the best draws stop beating the plans.
3. Score the moved plans; an instance whose moved plan costs over `best_draw_margin` temperatures more keeps its best draw.
4. Return each plan's first `ctrl_steps` commands, then shift the plans forward.

`robokit.sim.Sim`s resized from the given one hold the worlds: "rollout" (`batch_size * num_samples * num_replicas`) and "validation" (`batch_size * num_replicas`), instance-major. Coarse to fine (`config.stages`): a stage's rollouts step at its `dt`, or along its dt schedule over the horizon, each step's costs weighted by the sim steps it spans. `SimMPC.execute` runs the planner in closed loop on a simulator of its own.
"""

import copy
import dataclasses
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import warp as wp

from robokit.helpers.sim_mpc.config import SimMPCConfig, SimMPCStage
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage
from robokit.opt.sampler import GaussianSampler, GaussianSamplerConfig
from robokit.opt.var_values import VarValues, VectorVar
from robokit.terms.simulation import SimContext
from robokit.terms.simulation.sim_context import sim_context
from robokit.terms.task import Task
from robokit.utils.warp_utils import masked_copy


if TYPE_CHECKING:
    from robokit.sim import Sim


class _Key(Enum):
    NODES = 0


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _compute_ctrl_kernel(
    nodes: wp.array2d(dtype=wp.float32),
    interpolation: wp.array2d(dtype=float),
    ctrl_ref: wp.array3d(dtype=float),
    clock: wp.array(dtype=int),
    row_worlds: int,
    instance_worlds: int,
    bounds: wp.array2d(dtype=float),
    ctrl: wp.array3d(dtype=float),
):
    """Each world's commands at steps `clock[1] + k`, into `ctrl[k]`: its reference plus its node row weighted by that step's `interpolation` row."""
    world, k, command = wp.tid()  # pyright: ignore[reportAssignmentType]
    step = clock[1] + k
    nu = ctrl.shape[2]
    row = wp.min(world // row_worlds, nodes.shape[0] - 1)
    offset = float(0.0)
    for node in range(interpolation.shape[1]):
        offset += interpolation[step, node] * nodes[row, node * nu + command]
    reference = ctrl_ref[world // instance_worlds, wp.min(clock[0] + step, ctrl_ref.shape[1] - 1), command]
    ctrl[k, world, command] = wp.clamp(reference + offset, bounds[command, 0], bounds[command, 1])


@wp.kernel
def _set_clipped_ctrl_kernel(limits: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float)):
    world, command = wp.tid()  # pyright: ignore[reportAssignmentType]
    ctrl[world, command] = wp.clamp(ctrl[world, command], limits[command, 0], limits[command, 1])


@wp.kernel
def _set_broadcast_state_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qpos_std: wp.array(dtype=float),
    qvel_std: wp.array(dtype=float),
    num_replicas: int,
    instance_worlds: int,
    seed: int,
    clock: wp.array(dtype=int),
    iteration: wp.array(dtype=int),
    out_qpos: wp.array2d(dtype=float),
    out_qvel: wp.array2d(dtype=float),
):
    """Each world's instance's executed state plus its replica's noise, redrawn per plan start and iteration."""
    world = wp.tid()
    instance = world // instance_worlds
    rng = wp.rand_init(seed + clock[0], world % num_replicas + num_replicas * iteration[0])
    for i in range(qpos.shape[1]):
        out_qpos[world, i] = qpos[instance, i] + qpos_std[i] * wp.randn(rng)
    for i in range(qvel.shape[1]):
        out_qvel[world, i] = qvel[instance, i] + qvel_std[i] * wp.randn(rng)


@wp.kernel
def _set_value_kernel(value: float, out: wp.array(dtype=float)):
    out[wp.tid()] = value


@wp.kernel
def _set_broadcast_kernel(x: wp.array2d(dtype=float), instance_worlds: int, out: wp.array2d(dtype=float)):
    world, i = wp.tid()  # pyright: ignore[reportAssignmentType]
    out[world, i] = x[world // instance_worlds, i]


@wp.kernel
def _compute_step_cost_kernel(
    term_cost: wp.array2d(dtype=float),
    scales: wp.array2d(dtype=float),
    clock: wp.array(dtype=int),
    ticks: int,
    cost: wp.array(dtype=float),
):
    """Add every term's step cost at its running scale times the `ticks` sim steps this step stands for; thread 0 ticks the clock, which no thread here reads."""
    world = wp.tid()
    for term in range(term_cost.shape[0]):
        cost[world] += float(ticks) * scales[term, 0] * term_cost[term, world]
    if world == 0:
        clock[1] = clock[1] + ticks


@wp.kernel
def _compute_draw_cost_kernel(
    term_cost: wp.array2d(dtype=float),
    scales: wp.array2d(dtype=float),
    horizon: int,
    ticks: int,
    num_rows: int,
    num_replicas: int,
    iteration: int,
    rollout_cost: wp.array(dtype=float),
    cost: wp.array(dtype=float),
    history: wp.array3d(dtype=float),
):
    """Per instance: finish each world's cost, add each row's replica mean, and record history."""
    instance = wp.tid()
    best = float(0.0)
    for row in range(instance * num_rows, (instance + 1) * num_rows):
        total = float(0.0)
        for world in range(row * num_replicas, (row + 1) * num_replicas):
            value = rollout_cost[world]
            for term in range(term_cost.shape[0]):
                value += (scales[term, 1] - float(ticks) * scales[term, 0]) * term_cost[term, world]
            value = value / float(horizon)
            value = wp.where(wp.isfinite(value), value, 1.0e6)
            rollout_cost[world] = value
            total += value
        cost[row] += total / float(num_replicas)
        best = wp.where(row == instance * num_rows, cost[row], wp.min(best, cost[row]))
    if iteration >= 0:
        history[iteration, instance, 0] = cost[instance * num_rows]
        history[iteration, instance, 1] = best


@wp.kernel
def _compute_shifted_nodes_kernel(
    nodes: wp.array2d(dtype=wp.float32), shift: wp.array2d(dtype=float), shifted: wp.array2d(dtype=wp.float32)
):
    """Each plan's nodes `ctrl_steps` later: per command, node `j` becomes `sum_k shift[j, k] * node_k`."""
    instance, column = wp.tid()  # pyright: ignore[reportAssignmentType]
    nu = nodes.shape[1] // shift.shape[0]
    node = column // nu
    command = column % nu
    value = float(0.0)
    for k in range(shift.shape[1]):
        value += shift[node, k] * nodes[instance, k * nu + command]
    shifted[instance, column] = value


# --- planner ----------------------------------------------------------------
class SimMPC(Task):
    """Receding-horizon MPPI over simulated rollouts, for `config.batch_size` instances.

    Lifecycle:
        1. `__init__` resizes `sim` into the rollout and validation worlds and captures their graphs on the first plan.
        2. `plan(data, t0)` refines every instance's plan from its state and returns the next `ctrl_steps` commands.
        3. The caller executes them on its own simulator (or `execute` does on one), then plans again at `t0 + ctrl_steps`.

    Args:
        config: The settings, with the cost terms (`config.add`), the node noise (`ctrl_std`) and command bounds.
        sim: The planning simulator, e.g. `MjWarpSim(model, 1)`; for scenes that differ, one scene per instance.
        ctrl_ref: The reference commands the plans offset, `(T, nu)` or `(batch_size, T, nu)`.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import mujoco
        >>> from robokit.helpers.sim_mpc import SimMPCConfig
        >>> from robokit.sim import MjWarpSim
        >>> from robokit.terms import SensorTask
        >>> xml = (
        ...     '<mujoco><option timestep="0.01"/><worldbody><body><joint name="x" type="slide" damping="5"/>'
        ...     '<geom size="0.05" mass="1"/></body></worldbody><actuator><position joint="x" kp="50"/></actuator>'
        ...     '<sensor><jointpos name="x" joint="x"/></sensor></mujoco>'
        ... )
        >>> model = mujoco.MjModel.from_xml_string(xml)
        >>> config = SimMPCConfig(num_samples=32, horizon_steps=20, node_steps=10, ctrl_steps=10, max_iterations=2)
        >>> config = config.add(SensorTask(np.full((60, 1), 0.3), {"x": 100.0}))
        >>> mpc = SimMPC(config, MjWarpSim(model, 1), np.zeros((60, 1)))
        >>> commands, infos = mpc.plan(mujoco.MjData(model), 0)  # execute these, then plan again at step 10
        >>> commands.shape, mpc.sims["rollout"].nworld
        ((10, 1, 1), 31)
        >>> (rollout,), (plans,) = mpc.execute(MjWarpSim(model, 1), 40)  # or run the loop on a simulator
        >>> rollout["qpos"].shape, len(plans)
        ((40, 1), 4)
        >>> # "auto" picked 4x on the CPU (2x on an RTX 3090): every rollout's first quarter steps at 10 ms, the rest at 40 ms
        >>> mpc.stages  # xdoctest: +IGNORE_WANT
        [SimMPCStage(iters=2, dt=[(0.25, None), (1.0, 0.04)])]
    """

    var_key = _Key.NODES

    def __init__(self, config: SimMPCConfig, sim: "Sim", ctrl_ref: np.ndarray):
        import mujoco  # pyright: ignore[reportMissingImports]  # here, so that the module imports without MuJoCo

        batch_size = config.batch_size
        num_replicas = config.num_replicas
        self.config = config
        self.batch_size = batch_size
        self.device = wp.get_device()
        self.num_samples = config.num_samples - 1  # one draw's budget scores the moved plan
        if batch_size < 1 or num_replicas < 1 or self.num_samples < 2:
            raise ValueError("batch_size and num_replicas must be positive; num_samples needs three draws")
        self.instance_worlds = dict(rollout=self.num_samples * num_replicas, validation=num_replicas)
        mppi = config.planner == "mppi"
        with wp.ScopedDevice(self.device):  # each instance's worlds consecutive
            self.sims = {name: sim.resized(batch_size * n) for name, n in self.instance_worlds.items() if mppi}
        self.dt = sim.mjm.opt.timestep
        self.stages, self.check = config.stages or [SimMPCStage(config.max_iterations)], False  # auto: fine at first
        dts = [dt for stage in self.stages for _, dt in self._schedule(stage.dt)]
        ratios = np.array([(dt or self.dt) / self.dt for dt in dts])
        if np.any(ratios < 1) or not np.allclose(ratios, np.round(ratios)):
            raise ValueError("a stage's dt must be a whole multiple of the simulator's timestep")
        model = self.model = (self.sims["rollout"] if mppi else sim).mjm  # its layout: for several scenes, their merge
        nu = model.nu
        self.nu = nu
        ctrl_ref = np.asarray(ctrl_ref)
        if ctrl_ref.ndim == 2:  # one reference for every instance
            ctrl_ref = np.broadcast_to(ctrl_ref, (batch_size, *ctrl_ref.shape))
        if ctrl_ref.shape[-1] != nu or ctrl_ref.ndim != 3 or len(ctrl_ref) != batch_size:
            raise ValueError("ctrl_ref must be (T, nu) or (batch_size, T, nu)")
        self.ctrl_ref = wp.array(ctrl_ref, dtype=float)
        # --- interpolation: every mode is linear in the nodes, so one row of node weights per sim step ---
        num_nodes = -(-config.horizon_steps // config.node_steps) + 1
        self.num_nodes = num_nodes
        steps = np.arange(config.ctrl_steps + (num_nodes - 1) * config.node_steps + 1)  # past the last shifted node
        basis = config.interpolation
        if isinstance(basis, str):
            node = np.minimum(steps // config.node_steps, num_nodes - 1)  # each step's segment start
            phase = np.minimum(steps / config.node_steps - node, 1.0)  # and how far into the segment it is
            powers = dict(  # weights of nodes `node - 1 .. node + 2` per power of `phase`, clamped to the ends
                linear=[[0, 1, 0, 0], [0, -1, 1, 0]],
                cubic=[[0, 1, 0, 0], [-0.5, 0, 0.5, 0], [1, -2.5, 2, -0.5], [-0.5, 1.5, -1.5, 0.5]],
                zero=[[0, 1, 0, 0]],
            )[basis]
            weights = phase[:, None] ** np.arange(len(powers)) @ np.array(powers)
            interpolation = np.zeros((len(steps), num_nodes))
            for neighbor in range(4):
                np.add.at(interpolation, (steps, np.clip(node + neighbor - 1, 0, num_nodes - 1)), weights[:, neighbor])
        else:
            if np.ndim(basis) != 2 or np.shape(basis)[1] != num_nodes:
                raise ValueError("an interpolation basis needs one column per node")
            interpolation = np.asarray(basis)[np.minimum(steps, len(basis) - 1)]
        self.interpolation = wp.array(interpolation, dtype=float)
        # a shift moves node j to the plan's time ctrl_steps later, fading to zero past the horizon
        shifted_steps = steps[config.ctrl_steps :: config.node_steps]
        tail = np.clip(1 - (shifted_steps - config.horizon_steps) / config.ctrl_steps, 0, 1)
        shift = tail[:, None] * interpolation[config.ctrl_steps :: config.node_steps]
        self.shift = wp.array(shift, dtype=float)
        clamp = not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL  # MuJoCo clamps ctrl to ctrlrange
        limits = np.where(model.actuator_ctrllimited[:, None] * clamp, model.actuator_ctrlrange, [-np.inf, np.inf])
        self.ctrl_limits = wp.array(limits, dtype=float)
        bounds = limits if config.ctrl_range is None else np.asarray(config.ctrl_range)
        if bounds.shape != (nu, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("ctrl_range must contain ordered bounds for every command")
        self.command_bounds = wp.array(bounds, dtype=float)
        self._host = dict(ctrl_ref=ctrl_ref, bounds=bounds)  # for the sampler's per-plan bounds
        self.qpos_std = wp.array(np.broadcast_to(config.qpos_std, model.nq), dtype=float)
        self.qvel_std = wp.array(np.broadcast_to(config.qvel_std, model.nv), dtype=float)
        # --- buffers ---
        self.state = dict(  # each instance's state, which `plan` copies in and every rollout starts from
            time=wp.zeros((batch_size, 1), dtype=float),
            qpos=wp.array(np.tile(model.qpos0, (batch_size, 1)), dtype=float),
            qvel=wp.zeros((batch_size, model.nv), dtype=float),
            act=wp.zeros((batch_size, model.na), dtype=float),
            qacc_warmstart=wp.zeros((batch_size, model.nv), dtype=float),
        )
        self.commands = wp.zeros((config.ctrl_steps, batch_size, nu), dtype=float)
        self.clock = wp.array([0, 0, config.horizon_steps], dtype=int)  # [t0, t, horizon]
        self.rollout_cost = wp.zeros(batch_size * self.instance_worlds["rollout"], dtype=float)
        self.counter = wp.zeros(1, dtype=int)  # the iteration, which seeds replica noise
        self.history = wp.zeros((max(config.max_iterations, config.init_iterations, 1), batch_size, 2), dtype=float)
        self.plan_values = VarValues({_Key.NODES: VectorVar(wp.zeros((batch_size, num_nodes * nu), dtype=wp.float32))})
        self.execution = self.plan_values.get(_Key.NODES).x
        self.shifted = wp.zeros_like(self.execution)
        self.best_draw_mask = wp.zeros(batch_size, dtype=wp.int32)
        self.graphs = {}
        self.terms = {}
        self.t0 = 0
        self.iteration = 0
        self.scales = np.reshape([[r, config.terminal_scale if t is None else t] for _, r, t in config.terms], (-1, 2))
        if not mppi:  # iLQR on commands held for `node_steps`, first the reference's, from the instances' states
            from robokit.opt.ilqr_optimizer import ILQROptimizer, ILQROptimizerConfig
            from robokit.sim.dynamics import SimDynamics

            if config.control_map is not None:
                raise ValueError("planner 'ilqr' takes commands as controls: no control_map")
            dynamics = SimDynamics(sim, config.node_steps)
            dynamics.ctrl_range = bounds
            ilqr_config = config.ilqr or ILQROptimizerConfig(improve=True, feedback_cap=0.1, min_regularization=30.0)
            ilqr_config = dataclasses.replace(ilqr_config, max_iter=config.max_iterations)
            self.ilqr = ILQROptimizer(dynamics, [term for term, _, _ in config.terms], ilqr_config, scales=self.scales)
            self.ilqr_start = dynamics.build_state(self.state["qpos"], self.state["qvel"], self.state["act"])
            ref = ctrl_ref[:, np.minimum(np.arange(0, config.horizon_steps, config.node_steps), ctrl_ref.shape[1] - 1)]
            self.ilqr_plan = VectorVar(wp.array(np.clip(ref, *bounds.T).reshape(batch_size, -1), dtype=float))
            return
        # --- MPPI: node noise grows from ctrl_std at the first free node to std_growth times it at the last ---
        ctrl_std = np.broadcast_to(config.ctrl_std, nu)
        ramp = np.minimum(np.arange(num_nodes - 1) / max(config.horizon_steps // config.node_steps - 1, 1), 1.0)
        node_std = np.vstack([np.zeros(nu), config.std_growth ** ramp[:, None] * ctrl_std])
        if isinstance(basis, str) and basis == "zero" or config.draw_first_node:  # else it holds the last command
            node_std[0] = ctrl_std
        mask = wp.array(node_std.reshape(-1) > 0, dtype=wp.float32)
        sampler_config = GaussianSamplerConfig(
            var_key=_Key.NODES, num_draws=self.num_samples, init_std=0.0, min_std=config.min_std,
            gamma=config.std_gamma, seed=config.seed, elite_ratio=config.elite_ratio, mean_elites=True,
            noise_decay=config.noise_decay, wide_ratio=config.wide_ratio, wide_scale=config.wide_scale,
        )  # fmt: skip
        solver_config = PopulationSolverConfig(  # SimMPC captures its own step graphs, per `config.cuda_graph_mode`
            stages=[Stage(config.max_iterations)], samplers=[sampler_config], cuda_graph_mode="none",
            min_improvement=config.min_improvement, patience=config.patience,
        )  # fmt: skip
        self.solver = PopulationSolver([self], solver_config, self.device, active_dof_mask=mask)
        self.solver.setup(self.plan_values)  # scores the plans once, which captures the rollout graphs
        self.check = config.stages is None  # on the first plan's rollouts, from a given state
        self.sampler = cast(GaussianSampler, self.solver.samplers[0])
        self.node_std = np.tile(node_std.reshape(1, -1), (batch_size, 1))  # the configured noise
        self.sampler.std_reset.assign(self.node_std)
        self.sampler.reset()
        self.sampler.lower = wp.zeros_like(self.execution)  # the sampler clamps the draws to the command bounds
        self.sampler.upper = wp.zeros_like(self.execution)

    def _build(self, worlds: str, nodes: wp.array, spans: set):
        """Resolve the costs on a world set and capture its reset and one-step graphs, each run once first: the reset copies the instances' states in with replica noise and settles them with zero controls; the step writes, maps and clips the commands, steps and adds the costs."""
        config = self.config
        sim = self.sims[worlds]
        data = sim.data
        instance_worlds = self.instance_worlds[worlds]
        context = sim_context(sim)
        context[SimContext.CLOCK] = self.clock
        context[SimContext.WORLDS_PER_INSTANCE] = instance_worlds
        context[SimContext.NUM_INSTANCES] = self.batch_size
        terms = [copy.copy(term) for term, _, _ in config.terms]
        for term in terms:
            term.resolve(None, context)
        scales = wp.array(self.scales, dtype=float)  # each term's running and terminal scales
        term_cost = wp.zeros((len(terms), sim.nworld), dtype=float)
        self.terms[worlds] = terms, term_cost, scales  # kept alive for the graphs
        state = self.state

        def reset():
            self.clock[1:2].zero_()
            self.rollout_cost.zero_()
            inputs = [state["qpos"], state["qvel"], self.qpos_std, self.qvel_std, config.num_replicas, instance_worlds]
            inputs += [config.seed, self.clock, self.counter]
            wp.launch(_set_broadcast_state_kernel, dim=sim.nworld, inputs=inputs, outputs=[data.qpos, data.qvel])
            data.ctrl.zero_()
            wp.launch(_set_value_kernel, dim=sim.opt.timestep.shape, inputs=[self.dt], outputs=[sim.opt.timestep])
            for name, out in (("act", data.act), ("qacc_warmstart", data.qacc_warmstart), ("time", data.time)):
                if name == "time":  # after settling: the rollouts start at the state's time
                    for _ in range(config.settle_steps):
                        sim.step()
                out = out.reshape((sim.nworld, -1))
                wp.launch(_set_broadcast_kernel, dim=out.shape, inputs=[state[name], instance_worlds], outputs=[out])

        def step(ticks: int):
            timestep = sim.opt.timestep
            wp.launch(_set_value_kernel, dim=timestep.shape, inputs=[ticks * self.dt], outputs=[timestep])
            inputs = [nodes, self.interpolation, self.ctrl_ref, self.clock]
            inputs += [config.num_replicas, instance_worlds, self.command_bounds]
            ctrl = data.ctrl.reshape((1, *data.ctrl.shape))
            wp.launch(_compute_ctrl_kernel, dim=(sim.nworld, 1, self.nu), inputs=inputs, outputs=[ctrl])
            if config.control_map is not None:
                config.control_map(sim, self.clock)
            wp.launch(_set_clipped_ctrl_kernel, dim=data.ctrl.shape, inputs=[self.ctrl_limits], outputs=[data.ctrl])
            sim.step()
            term_cost.zero_()
            for i, term in enumerate(terms):
                term.compute_weighted_cost(None, out_cost=term_cost[i])
            inputs = [term_cost, scales, self.clock, ticks]
            wp.launch(_compute_step_cost_kernel, dim=sim.nworld, inputs=inputs, outputs=[self.rollout_cost])

        bodies = {worlds + "_reset": reset, **{f"{worlds}_{t}": partial(step, t) for t in spans}}
        for name, body in bodies.items():
            body()
            if self.device.is_cuda and self.config.cuda_graph_mode == "step":
                with wp.ScopedCapture() as capture:
                    body()
                self.graphs[name] = partial(wp.capture_launch, capture.graph)
            else:
                self.graphs[name] = body

    @staticmethod
    def _schedule(dt: Any) -> List[Tuple[float, Optional[float]]]:
        """A stage's dt as `(fraction of the horizon it runs to, dt)` pairs."""
        return list(dt) if isinstance(dt, (list, tuple)) else [(1.0, dt)]

    def _segments(self, dt: Any) -> List[Tuple[int, int]]:
        """Runs of (sim steps per physics step, physics steps) along the horizon for a stage's dt; each part's remainder first, at the timestep."""
        horizon, start, runs = self.config.horizon_steps, 0, []
        for until, part_dt in self._schedule(dt):
            k, end = round((part_dt or self.dt) / self.dt), round(until * horizon)
            runs += [(1, (end - start) % k), (k, (end - start) // k)]
            start = end
        return [(t, n) for t, n in runs if n]

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        """Add each node row's rollout cost: draws in the rollout worlds, moved plans in the validation worlds."""
        nodes = cast(VectorVar, cast(VarValues, var_values).get(_Key.NODES)).x
        is_rollout = nodes.shape[0] > self.batch_size
        worlds = "rollout" if is_rollout else "validation"
        plans = [self._segments(stage.dt) for stage in self.stages for _ in range(stage.iters)]
        if worlds + "_reset" not in self.graphs:
            self._build(worlds, nodes, {1, 2, 4, *(t for plan in plans for t, _ in plan)})
        if is_rollout:
            self.iteration += 1
        self.counter.fill_(self.iteration - 1)
        # this iteration's stage: a plan of other than `max_iterations` runs the stages' last iterations, the first stage longer or cut; the moved plans step at the timestep
        index = max(self.iteration - 1 + len(plans) - self.solver.config.num_iters, 0)
        ends, horizon, check = {}, self.config.horizon_steps, self.check and is_rollout
        split = {k: [(0.25, None), (1.0, k * self.dt)] for k in (4, 2)}  # auto's: the first quarter at the timestep
        for k in (4, 2, 1) if check else (None,):  # auto, once: the same draws at each k
            plan = self._segments(split.get(k)) if check else plans[index] if is_rollout else [(1, horizon)]
            self.graphs[worlds + "_reset"]()
            for t, steps in plan:
                for _ in range(steps):
                    self.graphs[f"{worlds}_{t}"]()
            if check:
                ends[k] = self.sims[worlds].data.qpos.numpy().reshape(self.batch_size, -1, self.model.nq)
        if check:  # the coarsest whose draws end within 11% of their spread of the fine ones
            spread = np.linalg.norm(ends[1] - ends[1].mean(1, keepdims=True), axis=-1).mean(1)
            error = {k: np.median(np.linalg.norm(ends[k] - ends[1], axis=-1), axis=1) / spread for k in (4, 2)}
            coarse = next((k for k in (4, 2) if np.all(error[k] <= 0.11)), 1)
            self.stages, self.check = [SimMPCStage(self.config.max_iterations, split.get(coarse))], False
        last = plan[-1][0]  # the last step's sim steps, whose running cost the terminal one replaces
        _, term_cost, scales = self.terms[worlds]
        num_rows = self.num_samples if is_rollout else 1
        iteration = self.iteration - 1 if is_rollout else -1  # validation records no history
        inputs = [term_cost, scales, horizon, last, num_rows]
        inputs += [self.config.num_replicas, iteration]
        outputs = [self.rollout_cost, out_cost, self.history]
        wp.launch(_compute_draw_cost_kernel, dim=self.batch_size, inputs=inputs, outputs=outputs)

    def _optimize(self, t0: int) -> List[dict]:
        """Refine every instance's plan from sim step `t0`; return each instance's iterations and costs."""
        config = self.config
        num_replicas = config.num_replicas
        self.clock.assign([t0, 0, config.horizon_steps])
        first_plan = t0 == 0 and config.init_iterations > 0
        self.solver.config.stages = [Stage(config.init_iterations if first_plan else config.max_iterations)]
        self.t0 = t0
        self.iteration = 0
        self.sampler.epoch.fill_(t0 // config.ctrl_steps)
        if t0 == 0:  # an earlier episode's carried noise
            self.sampler.std_reset.assign(self.node_std)
        ctrl_ref, bounds = self._host["ctrl_ref"], self._host["bounds"]  # the bounds as offsets from each node's ref
        node_ref = ctrl_ref[:, np.minimum(t0 + np.arange(self.num_nodes) * config.node_steps, ctrl_ref.shape[1] - 1)]
        self.sampler.lower.assign((bounds[:, 0] - node_ref).reshape(self.batch_size, -1))
        self.sampler.upper.assign((bounds[:, 1] - node_ref).reshape(self.batch_size, -1))
        self.solver.solve(self.plan_values)
        history = self.history.numpy()[: self.iteration].transpose(1, 0, 2)
        # the solver scored the moved plans last; execute one unless it costs `best_draw_margin` temperatures more
        validation_cost = self.rollout_cost.numpy()[: self.batch_size * num_replicas]
        mean_costs = validation_cost.reshape(self.batch_size, num_replicas).mean(axis=1)
        temperatures = self.sampler.temperatures.numpy().astype(np.float64)
        best_costs = history[:, -1, 1].astype(np.float64)
        executed_means = mean_costs.astype(np.float64) <= best_costs + config.best_draw_margin * temperatures
        best_draws = cast(VectorVar, cast(VarValues, self.sampler.best_draws).get(_Key.NODES)).x
        self.best_draw_mask.assign(~executed_means)
        masked_copy(self.best_draw_mask, best_draws, self.execution)
        scales = np.ones(self.batch_size)
        if config.std_range is not None and self.solver.config.num_iters == 1:  # carry the refit spread, one scale
            drawn = self.node_std[0] > 0
            ratios = self.sampler.std.numpy()[:, drawn] / self.node_std[:, drawn]
            scales = np.clip(np.sqrt(np.mean(ratios**2, axis=1)), *config.std_range)
            self.sampler.std_reset.assign(scales[:, None] * self.node_std)
        return [
            dict(t0=t0, iterations=self.iteration, plan_cost=float(costs[-1, 0]), best_cost=float(costs[-1, 1]),
                 history=[(float(plan), float(best)) for plan, best in costs], mean_cost=float(mean_cost),
                 executed_mean=bool(executed_mean), rollout_evaluations=self.iteration * self.num_samples + 1,
                 temperature=float(temperature), std_scale=float(scale))
            for costs, mean_cost, executed_mean, temperature, scale in zip(history, mean_costs, executed_means,
                                                                           temperatures, scales)
        ]  # fmt: skip

    def plan(self, data: Any, t0: int) -> Tuple[wp.array, List[dict]]:
        """Refine every instance's plan from its state at sim step `t0`; return its next `ctrl_steps` commands.

        Args:
            data: Each instance's `time`, `qpos`, `qvel`, `act` and `qacc_warmstart`, `(batch_size, ...)` arrays: e.g. the executing simulator's `data`, or a `mujoco.MjData` for one instance.
            t0: The state's sim step, which indexes `ctrl_ref` and the costs' references.

        Returns:
            The commands, `(ctrl_steps, batch_size, nu)` before any `control_map`, in a buffer the next call reuses, and each instance's plan info (`t0`, `iterations`, `plan_cost`, `best_cost`, ...).
        """
        for out, value in zip(self.state.values(), (data.time, data.qpos, data.qvel, data.act, data.qacc_warmstart)):
            if isinstance(value, wp.array):
                wp.copy(out, value.reshape(out.shape))
            else:
                out.assign(np.reshape(value, out.shape))
        if self.config.planner == "ilqr":  # from the last plan shifted to `t0` (after the first); then its first knots
            steps, ref, bounds = self.config.node_steps, self._host["ctrl_ref"], self._host["bounds"]
            shift = (t0 - self.t0) / steps if self.t0 < t0 else None
            if shift and self.config.ctrl_steps > steps:  # several knots a plan, as mjbatch: open loop, reference after
                plan, new = self.ilqr_plan.x.numpy().reshape(self.batch_size, -1, self.nu), int(shift)
                times = np.minimum(t0 + (plan.shape[1] - new + np.arange(new)) * steps, ref.shape[1] - 1)
                plan, shift = np.concatenate([plan[:, new:], np.clip(ref[:, times], *bounds.T)], axis=1), None
                self.ilqr_plan.x.assign(plan.reshape(self.batch_size, -1))
            self.ilqr.set_start(self.ilqr_start, t0)
            _, costs = self.ilqr.solve(self.ilqr_plan, shift=shift)
            self.t0 = t0
            plan = self.ilqr_plan.x.numpy().reshape(self.batch_size, -1, self.nu)
            self.commands.assign(plan[:, np.arange(self.config.ctrl_steps) // steps].transpose(1, 0, 2))
            return self.commands, [dict(t0=t0, plan_cost=float(c) / self.config.horizon_steps) for c in costs.numpy()]
        infos = self._optimize(t0)
        self.clock.assign([t0, 0, 0])
        inputs = [self.execution, self.interpolation, self.ctrl_ref, self.clock, 1, 1, self.command_bounds]
        dim = (self.batch_size, self.config.ctrl_steps, self.nu)
        wp.launch(_compute_ctrl_kernel, dim=dim, inputs=inputs, outputs=[self.commands])
        inputs = [self.execution, self.shift]  # the plans from `t0 + ctrl_steps` on
        wp.launch(_compute_shifted_nodes_kernel, dim=self.shifted.shape, inputs=inputs, outputs=[self.shifted])
        wp.copy(self.execution, self.shifted)
        return self.commands, infos

    def execute(
        self, sim: "Sim", num_steps: int, callback: Optional[Callable[[List[dict]], None]] = None
    ) -> Tuple[List[Dict[str, np.ndarray]], List[List[dict]]]:
        """Run the planner in closed loop on `sim`, one world per instance, for `num_steps` sim steps: each control period plans from `sim`'s state, then steps `sim` through the commands, mapped by the config's `control_map` and clipped to the controls MuJoCo applies, and calls `callback` with every instance's plan info.

        Returns:
            Each instance's rows, `time`, `qpos`, `qvel`, `ctrl` and `sensordata` (after the step), each `(num_steps, ...)`, and its plan info per control period.
        """
        data, control_map = sim.data, self.config.control_map
        clock = wp.zeros(3, dtype=int)  # [t0, k, 0] for `control_map`
        names, plans = ("time", "qpos", "qvel", "ctrl", "sensordata"), []
        rows = {name: wp.empty((num_steps, *getattr(data, name).shape), dtype=float) for name in names}  # no syncs
        for t0 in range(0, num_steps, self.config.ctrl_steps):
            commands, infos = self.plan(data, t0)
            for k in range(min(self.config.ctrl_steps, num_steps - t0)):
                wp.copy(data.ctrl, commands[k])
                if control_map is not None:
                    clock.assign([t0, k, 0])
                    control_map(sim, clock)
                wp.launch(_set_clipped_ctrl_kernel, dim=data.ctrl.shape, inputs=[self.ctrl_limits], outputs=[data.ctrl])
                for name in names:
                    if name == "sensordata":  # after the step
                        sim.step()
                    if rows[name].size:  # no sensors: no sensordata
                        wp.copy(rows[name][t0 + k], getattr(data, name))
            plans.append(infos)
            if callback is not None:
                callback(infos)
        rows = {name: np.moveaxis(x.numpy(), 0, 1) for name, x in rows.items()}  # (nworld, num_steps, ...)
        return [{name: x[i] for name, x in rows.items()} for i in range(sim.nworld)], [list(p) for p in zip(*plans)]


__all__ = ["SimMPC", "sim_context"]

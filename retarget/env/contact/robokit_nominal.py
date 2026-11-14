"""Original Robokit SimMPC supplies a whole-body nominal for contact residuals."""
import numpy as np
import warp as wp
from dataclasses import replace
from robokit.sim import MjWarpSim
from robokit.helpers.sim_mpc import SimMPCStage
from .robokit_rollout import StatefulSimMPC
from .robokit_planner import ControlPlanner
from .contact_terms import ContactDynamicsTask


class SourceNominalPlanner(StatefulSimMPC):
    def __init__(self, task, device: str):
        self.task, self.recording = task, False
        self.trajectories, self.cold_rollouts = 0, 0
        self.best_cost, self.best_nodes = np.inf, None
        self.reference_term = ContactDynamicsTask(task.model, task.profile.horizon, tracking=True) if task.reference else None
        config = task.config
        if self.reference_term is not None:
            # A frozen control trace was executed at the fine timestep. The
            # source jog's 0.04 s coarse stages skip its intervening commands
            # and can mis-rank even the exact feedforward trace as infeasible.
            config = replace(config, stages=[SimMPCStage(task.profile.iterations)],
                terms=[*config.terms, (self.reference_term, 1., 1.)])
        with wp.ScopedDevice(device):
            sim = MjWarpSim(task.plan_scene.compile(), 1, task.state,
                nconmax=120, njmax=500, device=device)
            super().__init__(config, sim, task.context['ctrl_ref'])
        self.stream = sim.stream

    def compute_weighted_cost(self, values, *, out_cost):
        super().compute_weighted_cost(values, out_cost=out_cost)
        if self.recording:
            self.trajectories += values.batch_size
            if self.reference_term is not None:
                # Preserve the best already-scored incumbent across source
                # iterations. A moved mean must not erase a feasible frozen
                # feedforward plan just because the last draw batch drifted.
                scores = out_cost.numpy()
                j = int(np.argmin(scores))
                if np.isfinite(scores[j]) and scores[j] < self.best_cost:
                    self.best_cost = float(scores[j])
                    self.best_nodes = values.get(self.var_key).x.numpy()[j:j + 1].copy()
        else:
            self.cold_rollouts += values.batch_size

    def _optimize(self, t0):
        infos = super()._optimize(t0)
        if self.reference_term is not None and self.best_nodes is not None:
            self.execution.assign(self.best_nodes)
        # Capture before the original plan() shifts its execution nodes.
        self.controls = ControlPlanner.controls(self, self.execution.numpy(), t0)[0]
        return infos

    def propose(self, state: dict, clock: int, packet: dict) -> np.ndarray:
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self.set_named_state(state)
            if self.reference_term is not None:
                self.reference_term.update(packet, [[]] * self.num_samples, self.num_samples)
            self.best_cost, self.best_nodes = np.inf, None
            self.recording = True
            try:
                self.plan(self.task.mirror.probe, clock)
            finally:
                self.recording = False
        return self.controls.copy()

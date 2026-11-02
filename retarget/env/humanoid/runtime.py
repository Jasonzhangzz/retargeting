"""Humanoid execution with SPIDER or cached-contact trajectory scoring."""

import json
import time
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

from retarget.task.humanoid import cost

from . import physics
from .config import PhysicsConfig
from .interp import interp
from .reference import load_data
from .rollout import make_rollout_fn


def contact_settings(dynamics, options=None):
    """Validate the opt-in scoring mode without initializing a simulator."""
    if dynamics not in ("mjwarp", "frozen_contact", "scsp"):
        raise ValueError("environment.dynamics must be mjwarp, frozen_contact or scsp")
    if dynamics == "scsp":
        if options is not None:
            raise ValueError("SCSP contact options belong in environment.scsp_contact")
        return None
    if dynamics == "mjwarp":
        if options is not None:
            raise ValueError("environment.contact_cost requires dynamics: frozen_contact")
        return None
    from retarget.task.humanoid.cost_contact import ContactCostOptions

    if options is None:
        return ContactCostOptions()
    if isinstance(options, ContactCostOptions):
        return options
    if not isinstance(options, Mapping):
        raise ValueError("environment.contact_cost must be a mapping")
    try:
        return ContactCostOptions(**options)
    except TypeError as exc:
        raise ValueError(f"Invalid environment.contact_cost: {exc}") from exc


def make_config(data, device, batch):
    values = yaml.safe_load((data / "source_config.yaml").read_text())
    known = {f.name for f in fields(PhysicsConfig)}
    c = PhysicsConfig(**{k: v for k, v in values.items() if k in known})
    c.device = device
    c.num_samples = batch
    c.model_path = str(data / "scene.xml")
    # SPIDER computes the integer grid before applying task_info's exact ref_dt.
    for name, dt in [
        ("horizon", c.horizon),
        ("knot", c.knot_dt),
        ("ref", c.ref_dt),
        ("ctrl", c.ctrl_dt),
    ]:
        setattr(c, name + "_steps", int(np.round(dt / c.sim_dt)))
    info = (
        json.loads((data / "task_info.json").read_text())
        if (data / "task_info.json").exists()
        else {}
    )
    c.ref_dt = info.get("ref_dt", c.ref_dt)
    c.contact_site_ids = info.get("contact_site_ids", [])
    if c.contact_rew_scale > 0 and not c.contact_site_ids:
        raise ValueError("Missing contact_site_ids in task_info.json")
    m = physics.setup_mj_model(c)
    c.nq, c.nv, c.nu, c.npair = m.nq, m.nv, m.nu, m.npair
    c.nq_obj = 0
    return c


class HumanoidEnvironment:
    """MJWarp execution and an optional frozen-contact MPC prediction model.

    Frozen scoring prepares a contact schedule along ``self.nominal`` once per
    MPC horizon. Candidate controls and successive optimizer rounds reuse that
    schedule; only execution and horizon shifting invalidate it. Execution still
    uses the original MJWarp simulator. The frozen prediction mode disables the
    upstream domain-randomization ensemble explicitly.
    """

    def __init__(
        self,
        task,
        data_root,
        device="cuda:0",
        batch=64,
        scene=None,
        reference=None,
        record_predictions=True,
        dynamics="mjwarp",
        contact_options=None,
        contact_graph=False,
        scsp_options=None,
    ):
        self.dynamics = dynamics
        self.contact_options = contact_settings(dynamics, contact_options)
        from .scsp import contact_options as scsp_contact_options

        if scsp_options is not None and dynamics != "scsp":
            raise ValueError("environment.scsp_contact requires dynamics: scsp")
        self.scsp_options = scsp_contact_options(scsp_options) if dynamics == "scsp" else None
        if not isinstance(contact_graph, bool):
            raise ValueError("contact_graph must be a boolean")
        if contact_graph and dynamics != "frozen_contact":
            raise ValueError("contact_graph requires dynamics: frozen_contact")
        if contact_graph and record_predictions:
            raise ValueError("contact_graph requires record_predictions=False")
        self.contact_graph = contact_graph
        self._contact_graphs = {}
        self.contact_graph_builds = 0
        self.contact_graph_seconds = 0.0
        self._contact_cache = None
        self.contact_cache_builds = 0
        self.config = c = make_config(Path(data_root), device, batch)
        self.source_num_dr = c.num_dr
        if dynamics in ("frozen_contact", "scsp"):
            c.num_dr = 0
        if dynamics == "scsp":
            c.point_contact_sites = self.scsp_options.hand_sites + self.scsp_options.foot_sites
            c.point_contact_radius = self.scsp_options.point_radius
            c.point_contact_friction = self.scsp_options.friction
        if scene is not None:
            c.model_path = str(scene)
        self.refs = load_data(
            c,
            str(Path(data_root) / "reference.npz") if reference is None else reference,
        )
        self.device, self.batch = device, batch
        wp.init()
        self.stream = torch.cuda.ExternalStream(
            wp.get_stream(device).cuda_stream, device=device
        )
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            self.env = physics.setup_env(c, self.refs)
            c.npair = self.env.model_cpu.npair
            if dynamics == "scsp":
                c.contact_site_ids = [self.env.model_cpu.site(name).id for name in c.point_contact_sites]
            self.q = torch.empty((c.horizon_steps + 1, batch, c.nq), device=device)
            self.v = torch.empty((c.horizon_steps + 1, batch, c.nv), device=device)
        torch.cuda.current_stream(device).wait_stream(self.stream)
        self.index = 0
        self.record_predictions = record_predictions
        mj = physics
        self.rollout = make_rollout_fn(
            self._step,
            mj.save_state,
            mj.load_state,
            cost.get_reward,
            cost.get_terminal_reward,
            cost.get_terminate,
            mj.get_trace,
            mj.save_env_params,
            mj.load_env_params,
            None,
        )
        if c.num_dr == 0:
            self.env_params = [dict(pair_margin=0.0, xy_offset=0.0)]
        else:
            self.env_params = [
                dict(pair_margin=float(m), xy_offset=float(x))
                for m, x in zip(
                    np.linspace(*c.pair_margin_range, c.num_dr),
                    np.linspace(*c.xy_offset_range, c.num_dr),
                )
            ]
        self.first_step = 0
        self.nominal = self.refs[2][: c.horizon_steps].clone()
        self.knots = int(round(c.horizon / c.knot_dt))
        self.control_scale = 0.05
        self.calls = self.trajectories = 0

    def _step(self, config, env, control):
        physics.step_env(config, env, control)
        self.index += 1
        if self.record_predictions:
            self.q[self.index].copy_(wp.to_torch(env.data_wp.qpos))
            self.v[self.index].copy_(wp.to_torch(env.data_wp.qvel))

    def controls(self, x):
        c = self.config
        knots = x.float().reshape(-1, self.knots, c.nu) * self.control_scale
        # Same dense-reference plus align_corners=True knot interpolation as
        # upstream sample_ctrls. No extra actuator clamp is added here.
        return self.nominal + interp(knots, c.knot_steps)

    def project_parameters(self, x):
        """Keep candidate controls inside the cached model's local trust region."""
        if self.contact_options is None or self.contact_options.action_trust_radius is None:
            return x
        limit = self.contact_options.action_trust_radius / self.control_scale
        return x.clamp(-limit, limit)

    def score(self, x):
        if self.dynamics == "scsp":
            raise ValueError("SCSP uses prepare_scsp_problem and two coupled objectives")
        count = len(x)
        if count > self.batch:
            return torch.cat([self.score(chunk) for chunk in x.split(self.batch)])
        return self.score_controls(self.controls(x))

    def score_controls(self, controls):
        """Evaluate dense controls using the selected horizon prediction model."""
        c = self.config
        if len(controls) > self.batch:
            return torch.cat(
                [self.score_controls(chunk) for chunk in controls.split(self.batch)]
            )
        count = len(controls)
        refs = tuple(
            r[self.first_step + 1 : self.first_step + c.horizon_steps + 1]
            for r in self.refs
        )
        if self.dynamics == "frozen_contact":
            from retarget.task.humanoid.cost_contact import get_trajectory_cost

            cache = self.prepare_contact_rollout()
            if self.contact_graph:
                trajectory_cost = self._graphed_contact_score(cache, controls, refs)
            else:
                prediction = cache.rollout(controls)
                trajectory_cost, _ = get_trajectory_cost(
                    c, prediction, refs, options=self.contact_options
                )
                if self.record_predictions:
                    with torch.no_grad():
                        self.q[: c.horizon_steps + 1, :count].copy_(
                            prediction.qpos.detach().transpose(0, 1)
                        )
                        self.v[: c.horizon_steps + 1, :count].copy_(
                            prediction.qvel.detach().transpose(0, 1)
                        )
            self.calls += 1
            self.trajectories += count
            return trajectory_cost.double().clone()
        worst_cost = torch.full((self.batch,), -torch.inf, device=self.device)
        for params in self.env_params:
            _, reward, _, _ = self.rollout_batch(controls, refs, params)
            worst_cost = torch.maximum(worst_cost, -reward)
        self.calls += 1
        self.trajectories += count
        return worst_cost[:count].double().clone()

    def _graphed_contact_score(self, cache, controls, refs):
        """Capture one static horizon per input shape and autograd mode.

        Gradients are first order, as required by ``make_graphed_callables``.
        A graph's intermediate buffers must survive until its backward finishes.
        Additional outstanding forwards of that shape therefore run eagerly.
        """
        from retarget.task.humanoid.cost_contact import get_trajectory_cost

        def objective(actions):
            prediction = cache.rollout(actions)
            return get_trajectory_cost(
                self.config, prediction, refs, options=self.contact_options
            )[0]

        requires_grad = torch.is_grad_enabled() and controls.requires_grad
        key = (tuple(controls.shape), controls.dtype, requires_grad)
        if key not in self._contact_graphs:
            sample = controls.detach().clone().requires_grad_(requires_grad)
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.set_grad_enabled(requires_grad):
                graphed = torch.cuda.make_graphed_callables(
                    objective, (sample,), num_warmup_iters=1,
                )
            torch.cuda.synchronize(self.device)
            self.contact_graph_seconds += time.perf_counter() - started
            self.contact_graph_builds += 1
            self._contact_graphs[key] = dict(callable=graphed, pending=False)
        entry = self._contact_graphs[key]
        if requires_grad and entry["pending"]:
            return objective(controls)
        # Captured outputs occupy reusable graph storage. Clone before the next
        # population chunk or line-search trial overwrites that storage.
        graph_input = controls.view_as(controls) if requires_grad else controls.detach()
        output = entry["callable"](graph_input).clone()
        if requires_grad:
            entry["pending"] = True

            def release(gradient):
                entry["pending"] = False
                # Captured backward also returns reusable storage. Preserve
                # gradients retained by autograd.grad across later replays.
                return gradient.clone()

            graph_input.register_hook(release)
        return output

    def prepare_contact_rollout(self):
        """Cache nominal-horizon contacts, never candidate-dependent contacts.

        Preparation uses the engine's isolated scratch world and does not alter
        the execution state. Call this before optimization to account for cache
        preparation separately; scoring also prepares it lazily when needed.
        """
        if self.dynamics != "frozen_contact":
            raise ValueError("prepare_contact_rollout requires frozen_contact dynamics")
        if self._contact_cache is None:
            from .frozen_contact import FrozenContactRollout

            self._contact_graphs.clear()
            self._contact_cache = FrozenContactRollout.from_env(
                self.config, self.env, self.nominal, options=self.contact_options,
                anchor_to_mjwarp=self.contact_options.nominal_correction,
            )
            self.contact_cache_builds += 1
        return self._contact_cache

    def prepare_scsp_problem(self, mean, *, seed=0):
        """Project the dataset hand points and cache this window's dynamics."""
        if self.dynamics != "scsp":
            raise ValueError("prepare_scsp_problem requires scsp dynamics")
        from .scsp import HumanoidSCSP

        self.scsp_model = HumanoidSCSP(self, options=self.scsp_options, seed=seed)
        self.contact_cache_builds += 1
        return self.scsp_model.problem(mean)

    def rollout_batch(self, controls, refs, params):
        count = len(controls)
        padded = controls[:1].expand(self.batch, -1, -1).clone()
        padded[:count] = controls
        self.index = 0
        if self.record_predictions:
            self.q[0].copy_(wp.to_torch(self.env.data_wp.qpos))
            self.v[0].copy_(wp.to_torch(self.env.data_wp.qvel))
        return self.rollout(self.config, self.env, padded, refs, params)

    def execute(self, controls):
        self._contact_cache = None
        self._contact_graphs.clear()
        states, velocities, warmstarts = [], [], []
        for control in controls:
            physics.step_env(self.config, self.env, control)
            d = self.env.data_wp
            states.append(d.qpos.numpy()[0].copy())
            velocities.append(d.qvel.numpy()[0].copy())
            warmstarts.append(d.qacc_warmstart.numpy()[0].copy())
        physics.sync_env(self.config, self.env, None)
        return states, velocities, warmstarts

    def shift(self, controls, executed):
        self._contact_cache = None
        self._contact_graphs.clear()
        self.first_step += executed
        retained = controls[executed:]
        self.nominal = torch.cat(
            (
                retained,
                self.refs[2][
                    self.first_step + len(retained) : self.first_step
                    + self.config.horizon_steps
                ],
            )
        )

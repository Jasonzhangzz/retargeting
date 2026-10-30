"""Humanoid execution with SPIDER or cached-contact trajectory scoring."""

import json
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
    if dynamics not in ("mjwarp", "frozen_contact"):
        raise ValueError("environment.dynamics must be mjwarp or frozen_contact")
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
    ):
        self.dynamics = dynamics
        self.contact_options = contact_settings(dynamics, contact_options)
        self._contact_cache = None
        self.contact_cache_builds = 0
        self.config = c = make_config(Path(data_root), device, batch)
        self.source_num_dr = c.num_dr
        if dynamics == "frozen_contact":
            c.num_dr = 0
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

    def score(self, x):
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

            prediction = self.prepare_contact_rollout().rollout(controls)
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

            self._contact_cache = FrozenContactRollout.from_env(
                self.config, self.env, self.nominal, options=self.contact_options
            )
            self.contact_cache_builds += 1
        return self._contact_cache

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

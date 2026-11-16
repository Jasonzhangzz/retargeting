#!/usr/bin/env python3
"""Stability-gated primitive grasp search with virtual gravity by default.

Run in Dex-Supermimic's environment, beside refine_primitives_supermimic.py.
Preparation support is removed before scoring the unsupported grasp. Virtual
wrench feasibility and actual physical perturbation validation are separate.
A grasp that never passes the gate is a failure even if support holds it up.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

import refine_primitives_supermimic as original


def phase_observation(env):
    controller = getattr(env, 'gravity_controller', None)
    if controller is None:
        return torch.zeros((env.num_envs, 7), device=env.device)
    return controller.observation()


def reset_controller(env, env_ids):
    controller = getattr(env, 'gravity_controller', None)
    if controller is not None:
        controller.reset(env_ids)


def phased_reward(env):
    controller = env.gravity_controller
    released = controller.support_weight < 1e-6
    quality, distance, count, penetration, _ = original.grasp_quantities(env)
    # Supported preparation cannot collect the full unassisted grasp reward.
    reward = original.grasp_reward(env) * torch.where(released, 1., .2)
    reward += 3 * released.float() + 2 * controller.stable.float()
    reward -= 5 * controller.timed_out.float()
    if controller.args.gravity_mode == 'virtual':
        reward += 4 * virtual_wrench_proxy(env)
    return reward


def virtual_wrench_system(env):
    """Finite friction cones with measured preload budgets; forces in newtons.

    These are contact-level feasibility tests, not joint-torque certificates.
    Twelve signed force/torque targets accompany the gravity-only target.
    """
    sensor = env.scene['grasp_contacts'].data
    normal, tangent = sensor.normal, sensor.tangent
    theta = torch.arange(8, device=env.device) * math.pi / 4
    side = torch.cross(normal, tangent, dim=-1)
    force = normal[:, :, None, :] + .8 * (
        theta.cos()[None, None, :, None] * tangent[:, :, None, :]
        + theta.sin()[None, None, :, None] * side[:, :, None, :])
    lever = ((sensor.pos - env.scene['object'].data.root_link_pos_w[:, None, :])
             / env.primitive_radius)
    torque = torch.cross(lever[:, :, None, :].expand_as(force), force, dim=-1)
    G = torch.cat((force, torque), -1).flatten(1, 2).transpose(1, 2)
    found = (sensor.found > 0) & (sensor.dist < .001)
    budget = sensor.force[:, :, 0].abs().clamp_max(1.) * found
    targets = torch.zeros((13, 6), device=env.device)
    targets[:, 2] = env.primitive_mass * 9.81  # counteract virtual downward gravity
    for i in range(12):
        axis = (i // 2) % 3
        component = axis if i < 6 else axis + 3
        magnitude = env.primitive_mass * 9.81 * env.gravity_controller.args.disturbance_g
        targets[i + 1, component] += magnitude * (1 if i % 2 == 0 else -1) * (1 if i < 6 else .5)
    return G, budget, targets


def virtual_wrench_proxy(env):
    G, budget, targets = virtual_wrench_system(env)
    # Positive cone coefficients, projected to measured per-contact normal
    # preload. A few GPU NNLS iterations provide a training proxy; CPU LP below
    # decides exact final feasibility for this finite cone approximation.
    n, contacts = budget.shape
    weights = (budget[:, :, None, None] / 16).expand(n, contacts, 8, 13).clone()
    lipschitz = G.square().sum((1, 2)).clamp_min(1)
    for _ in range(12):
        flat = weights.flatten(1, 2)
        residual = G @ flat - targets.T[None]
        gradient = (G.transpose(1, 2) @ residual).reshape_as(weights)
        weights = (weights - gradient / lipschitz[:, None, None, None]).clamp_min(0)
        scale = (budget[:, :, None] / weights.sum(2).clamp_min(1e-8)).clamp_max(1)
        weights *= scale[:, :, None, :]
    residual = (G @ weights.flatten(1, 2) - targets.T[None]).norm(dim=1)
    return torch.exp(-residual.amax(-1) / (env.primitive_mass * 9.81 * .25))


def virtual_wrench_lp(env):
    from scipy.optimize import linprog
    G, budget, targets = virtual_wrench_system(env)
    G, budget, targets = G.cpu().numpy(), budget.cpu().numpy(), targets.cpu().numpy()
    feasible = np.zeros((env.num_envs, 13), dtype=bool)
    cap = np.kron(np.eye(budget.shape[1]), np.ones((1, 8)))
    for row in range(env.num_envs):
        for index, target in enumerate(targets):
            result = linprog(np.zeros(G.shape[-1]), A_ub=cap, b_ub=budget[row],
                             A_eq=G[row], b_eq=target, bounds=(0, None), method='highs')
            feasible[row, index] = result.success
    return feasible


class StabilityGravityController:
    def __init__(self, env, args):
        self.env, self.args = env, args
        self.body = int(env.scene['object'].indexing.body_ids[0])
        n, device = env.num_envs, env.device
        self.release_step = torch.full((n,), -1, device=device, dtype=torch.long)
        self.stable_steps = torch.zeros(n, device=device, dtype=torch.long)
        self.stable = torch.zeros(n, device=device, dtype=torch.bool)
        self.timed_out = self.stable.clone()
        self.support_weight = torch.ones(n, device=device)
        self.previous_q = env.scene['robot'].data.joint_pos.clone()
        self.previous_contacts = env.scene['grasp_contacts'].data.found.squeeze(-1).bool().clone()
        sensor = env.scene['grasp_contacts']
        slot_digits = [int(match.group(1)) if (match := re.search(r'finger([1-5])_', name)) else 0
                       for name in sensor.primary_names for _ in range(sensor.cfg.num_slots)]
        self.slot_digits = torch.tensor(slot_digits, device=device)
        self.max_probe_displacement = torch.zeros(n, device=device)
        self.probe_steps = torch.zeros(n, device=device, dtype=torch.long)
        self.contact_probe_steps = self.probe_steps.clone()
        self.max_probe_rotation = torch.zeros(n, device=device)
        self.peak_disturbance_force = torch.zeros(n, device=device)
        self.peak_disturbance_torque = torch.zeros(n, device=device)
        self.max_gate_run = torch.zeros(n, device=device, dtype=torch.long)
        self.last_gate_checks = torch.zeros((n, 7), device=device, dtype=torch.bool)
        self.probe_reference = env.scene['object'].data.default_root_state[:, :3].clone()
        self.probe_reference += env.scene.env_origins
        self.probe_quaternion = env.scene['object'].data.default_root_state[:, 3:7].clone()
        self.original_physics_step = env.sim.step
        env.sim.step = self.physics_step

    def reset(self, ids):
        if ids is None:
            ids = slice(None)
        self.release_step[ids] = -1
        self.stable_steps[ids] = 0
        self.stable[ids] = False
        self.timed_out[ids] = False
        self.support_weight[ids] = 1
        self.previous_q[ids] = self.env.scene['robot'].data.default_joint_pos[ids]
        self.previous_contacts[ids] = False
        self.max_probe_displacement[ids] = 0
        self.probe_steps[ids] = 0
        self.contact_probe_steps[ids] = 0
        self.max_probe_rotation[ids] = 0
        self.peak_disturbance_force[ids] = 0
        self.peak_disturbance_torque[ids] = 0
        self.max_gate_run[ids] = 0
        self.last_gate_checks[ids] = False
        self.probe_reference[ids] = (self.env.scene['object'].data.default_root_state[ids, :3]
                                     + self.env.scene.env_origins[ids])
        self.probe_quaternion[ids] = self.env.scene['object'].data.default_root_state[ids, 3:7]

    def contact_state(self):
        data = self.env.scene['grasp_contacts'].data
        found = data.found.squeeze(-1).bool() & (data.dist.squeeze(-1) < .001)
        digits = torch.stack([(found & (self.slot_digits == digit)).any(-1)
                              for digit in range(1, 6)], -1).sum(-1)
        return found, digits

    def elapsed(self):
        return (self.env.episode_length_buf - self.release_step).float() * self.env.step_dt

    def update_gate(self):
        env, args = self.env, self.args
        robot, obj = env.scene['robot'], env.scene['object']
        contact = env.scene['grasp_contacts'].data
        found, digits = self.contact_state()
        intersection = (found & self.previous_contacts).sum(-1)
        union = (found | self.previous_contacts).sum(-1).clamp_min(1)
        similarity = intersection / union
        dq = (robot.data.joint_pos - self.previous_q).abs().amax(-1)
        penetration = (-contact.dist.squeeze(-1)).clamp_min(0).amax(-1)
        checks = torch.stack((
            (robot.data.joint_vel.square().mean(-1).sqrt() < args.joint_speed) & (dq < .005),
            torch.linalg.vector_norm(obj.data.root_link_lin_vel_w, dim=-1) < .02,
            torch.linalg.vector_norm(obj.data.root_link_ang_vel_w, dim=-1) < .5,
            found.sum(-1) >= 3,
            digits >= 2,
            similarity >= .8,
            penetration < .003,
        ), -1)
        preparing = self.release_step < 0
        self.last_gate_checks = checks
        self.stable = checks.all(-1)
        self.stable_steps = torch.where(preparing & self.stable, self.stable_steps + 1,
                                        torch.zeros_like(self.stable_steps))
        self.max_gate_run = torch.maximum(self.max_gate_run, self.stable_steps)
        now = env.episode_length_buf
        ready = (preparing & (self.stable_steps >= math.ceil(args.stable_s / env.step_dt))
                 & (now * env.step_dt >= args.min_prepare_s)
                 & (now * env.step_dt <= args.max_prepare_s))
        self.release_step[ready] = now[ready]
        self.probe_reference[ready] = obj.data.root_link_pos_w[ready]
        self.probe_quaternion[ready] = obj.data.root_link_pose_w[ready, 3:7]
        self.timed_out |= preparing & (now * env.step_dt > args.max_prepare_s)
        self.support_weight = torch.where(self.release_step >= 0,
            (1 - self.elapsed() / args.ramp_s).clamp(0, 1), torch.ones_like(self.support_weight))
        self.previous_q.copy_(robot.data.joint_pos)
        self.previous_contacts.copy_(found)

    def observation(self):
        elapsed = (self.elapsed() - self.args.ramp_s).clamp_min(0)
        return torch.stack((self.support_weight,
                            (self.stable_steps * self.env.step_dt / self.args.stable_s).clamp_max(1),
                            (self.release_step >= 0).float(), elapsed / self.args.probe_s,
                            self.stable.float(), self.timed_out.float(),
                            torch.full_like(elapsed, float(self.args.gravity_mode == 'staged'))), -1)

    def physics_step(self):
        env, args = self.env, self.args
        obj = env.scene['object']
        root = obj.data.root_link_pose_w
        reference = obj.data.default_root_state.clone()
        reference[:, :3] += env.scene.env_origins
        w = self.support_weight[:, None]
        # A damped six-dimensional support is present only during preparation
        # and its removal ramp. Gravity compensation acts at the object's COM.
        force = w * (80 * (reference[:, :3] - root[:, :3])
                     - 3.6 * obj.data.root_link_lin_vel_w)
        q0, q = reference[:, 3:7], root[:, 3:7]
        error_vec = (-q0[:, :1] * q[:, 1:] + q[:, :1] * q0[:, 1:]
                     - torch.cross(q0[:, 1:], q[:, 1:], dim=-1))
        scalar = (q0 * q).sum(-1)
        error_vec *= torch.where(scalar >= 0, 1., -1.)[:, None]
        torque = w * (.1 * error_vec - .003 * obj.data.root_link_ang_vel_w)
        if args.gravity_mode == 'staged':
            force[:, 2] += w[:, 0] * args.mass * 9.81
        else:
            # Cancel physical gravity throughout. Gravity-sized loads become
            # short probes after the object is completely unsupported.
            force[:, 2] += args.mass * 9.81
        t = self.elapsed() - args.ramp_s
        probing = (self.release_step >= 0) & (t >= 0)
        if args.gravity_mode == 'disturbance':
            axis_id = torch.arange(env.num_envs, device=env.device) % 12
            axis = torch.eye(3, device=env.device)[(axis_id // 2) % 3]
            axis *= torch.where(axis_id % 2 == 0, 1., -1.)[:, None]
            envelope = torch.sin(((t - .25) / args.pulse_s).clamp(0, 1) * math.pi)
            envelope *= (probing & (t >= .25) & (t < .25 + args.pulse_s))
            load_force = axis * (envelope * args.mass * 9.81 * args.disturbance_g
                                 * (axis_id < 6))[:, None]
            load_torque = axis * (envelope * args.mass * 9.81 * env.primitive_radius
                                  * .5 * args.disturbance_g * (axis_id >= 6))[:, None]
            force += load_force
            torque += load_torque
            self.peak_disturbance_force = torch.maximum(self.peak_disturbance_force,
                torch.linalg.vector_norm(load_force, dim=-1))
            self.peak_disturbance_torque = torch.maximum(self.peak_disturbance_torque,
                torch.linalg.vector_norm(load_torque, dim=-1))
        env.sim.data.xfrc_applied[:, self.body, :3] = force
        env.sim.data.xfrc_applied[:, self.body, 3:] = torque
        self.original_physics_step()

    def record_probe(self):
        probing = (self.release_step >= 0) & (self.elapsed() >= self.args.ramp_s)
        distance = torch.linalg.vector_norm(
            self.env.scene['object'].data.root_link_pos_w - self.probe_reference, dim=-1)
        self.max_probe_displacement = torch.maximum(self.max_probe_displacement,
                                                     torch.where(probing, distance, 0.))
        self.probe_steps += probing.long()
        _, digits = self.contact_state()
        self.contact_probe_steps += (probing & (digits >= 2)).long()
        dot = (self.env.scene['object'].data.root_link_pose_w[:, 3:7]
               * self.probe_quaternion).sum(-1).abs().clamp(0, 1)
        rotation = 2 * torch.acos(dot)
        self.max_probe_rotation = torch.maximum(self.max_probe_rotation,
                                                torch.where(probing, rotation, 0.))


def make_config(args, z):
    from mjlab.managers import EventTermCfg, RewardTermCfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    cfg = original.make_config(args, z)
    cfg.episode_length_s = args.max_prepare_s + args.ramp_s + args.probe_s + .2
    del cfg.events['training_force_pulse']
    cfg.events['reset_gravity_controller'] = EventTermCfg(func=reset_controller, mode='reset')
    cfg.rewards['grasp_quality'] = RewardTermCfg(func=phased_reward, weight=1.)
    if args.gravity_mode == 'virtual':
        for sensor in cfg.scene.sensors:
            if sensor.name == 'grasp_contacts':
                sensor.fields = (*sensor.fields, 'force')
    for group in cfg.observations.values():
        group.terms['grasp_phase'] = ObservationTermCfg(func=phase_observation)
    # The generic cube drop counter is not appropriate during supported search.
    cfg.terminations['cage_drop'].params['max_outside_steps'] = 1000000
    return cfg


@torch.no_grad()
def evaluate(env, policy, args):
    env.cfg.auto_reset = False
    obs = env.reset()[0]['policy']
    env._manual_reset_pending.zero_()
    controller = env.gravity_controller
    finite = True
    trajectory = []
    for _ in range(math.ceil(env.cfg.episode_length_s / env.step_dt) - 1):
        action = torch.zeros((env.num_envs, 20), device=env.device) if policy is None else policy(obs)
        controller.update_gate()
        obs = env.step(action)[0]['policy']
        controller.record_probe()
        trajectory.append(torch.stack((controller.support_weight,
            controller.release_step.float(), controller.stable_steps.float(),
            controller.max_probe_displacement), -1).cpu().numpy())
        finite = bool(torch.isfinite(obs).all())
        if not finite:
            break
    enough_probe = controller.probe_steps * env.step_dt >= args.probe_s - .04
    pulse_exposed = ((controller.peak_disturbance_force > .001)
                     | (controller.peak_disturbance_torque > .00001))
    if args.gravity_mode in ('staged', 'virtual'):
        pulse_exposed = enough_probe
    robot, obj = env.scene['robot'], env.scene['object']
    _, final_digits = controller.contact_state()
    contact_fraction = controller.contact_probe_steps / controller.probe_steps.clamp_min(1)
    retained = (contact_fraction >= .8) & (final_digits >= 2)
    settled = ((obj.data.root_link_lin_vel_w.norm(dim=-1) < .03)
               & (obj.data.root_link_ang_vel_w.norm(dim=-1) < .5))
    success = (enough_probe & pulse_exposed & retained & settled
               & (controller.max_probe_displacement < .03)
               & (controller.max_probe_rotation < .5) & finite)
    virtual_feasible = virtual_wrench_lp(env) if args.gravity_mode == 'virtual' and finite else None
    if virtual_feasible is not None:
        success &= torch.as_tensor(virtual_feasible.all(-1), device=env.device)
    base = robot.data.root_link_pose_w.clone()
    pose = obj.data.root_link_pose_w.clone()
    for value in (base, pose):
        value[:, :3] -= env.scene.env_origins
        value[:, 2] -= 1
    result = dict(finite=finite, released_count=int((controller.release_step >= 0).sum()),
                  fully_probed_count=int((enough_probe & pulse_exposed).sum()),
                  success_count=int(success.sum()), success_rate=float(success.float().mean()),
                  gate_timeout_count=int(controller.timed_out.sum()),
                  contact_retained_count=int(retained.sum()), final_settled_count=int(settled.sum()),
                  max_consecutive_stable_s=float(controller.max_gate_run.max() * env.step_dt),
                  final_gate_checks_pass_count=controller.last_gate_checks.sum(0).tolist(),
                  gate_check_names=['joint_speed_and_change', 'object_linear_speed',
                                    'object_angular_speed', 'three_contacts', 'two_fingers',
                                    'contact_persistence', 'penetration_under_3mm'])
    arrays = dict(target_qpos=robot.data.joint_pos.cpu().numpy(), target_base_pose=base.cpu().numpy(),
                  object_pose_world=pose.cpu().numpy(),
                  control_target_qpos=env.action_manager.get_term('joint_pos').processed_action.cpu().numpy(),
                  actuator_force=robot.data.actuator_force.cpu().numpy(),
                  hold_pass=success.cpu().numpy(), max_displacement_m=controller.max_probe_displacement.cpu().numpy(),
                  release_step=controller.release_step.cpu().numpy(), probe_steps=controller.probe_steps.cpu().numpy(),
                  peak_disturbance_force_n=controller.peak_disturbance_force.cpu().numpy(),
                  peak_disturbance_torque_nm=controller.peak_disturbance_torque.cpu().numpy(),
                  contact_retention_fraction=contact_fraction.cpu().numpy(),
                  max_rotation_rad=controller.max_probe_rotation.cpu().numpy(),
                  phase_trajectory=np.stack(trajectory), joint_names=np.array(robot.joint_names))
    contact = env.scene['grasp_contacts'].data
    arrays.update(contact_pos=contact.pos.cpu().numpy() - env.scene.env_origins.cpu().numpy()[:, None, :]
                  - np.array([0, 0, 1]), contact_normal=contact.normal.cpu().numpy(),
                  contact_tangent=contact.tangent.cpu().numpy(), contact_found=contact.found.cpu().numpy())
    if virtual_feasible is not None:
        result.update(virtual_gravity_feasible_count=int(virtual_feasible[:, 0].sum()),
                      virtual_all_wrenches_feasible_count=int(virtual_feasible.all(-1).sum()),
                      fully_probed_count=0, fully_assessed_count=int(enough_probe.sum()),
                      physically_probed_count=0,
                      qualification='virtual contact-wrench feasibility; not physical disturbance validation')
        arrays['virtual_wrench_feasible'] = virtual_feasible
        arrays['contact_force'] = contact.force.cpu().numpy()
    return result, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pose-file', type=Path, required=True)
    ap.add_argument('--row', type=int, default=27)
    ap.add_argument('--kind', choices=('sphere', 'disk', 'cylinder'), default='disk')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--num-envs', type=int, default=128)
    ap.add_argument('--iterations', type=int, default=200)
    ap.add_argument('--mass', type=float, default=.04)
    ap.add_argument('--native-collision', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--gravity-mode', choices=('virtual', 'disturbance', 'staged'), default='virtual')
    ap.add_argument('--stable-s', type=float, default=.3)
    ap.add_argument('--min-prepare-s', type=float, default=.5)
    ap.add_argument('--max-prepare-s', type=float, default=3.)
    ap.add_argument('--ramp-s', type=float, default=.5)
    ap.add_argument('--probe-s', type=float, default=2.)
    ap.add_argument('--pulse-s', type=float, default=.4)
    ap.add_argument('--disturbance-g', type=float, default=1.)
    ap.add_argument('--joint-speed', type=float, default=.3)
    ap.add_argument('--evaluate-only', action='store_true')
    ap.add_argument('--actor-checkpoint', type=Path)
    args = ap.parse_args()
    if min(args.stable_s, args.ramp_s, args.probe_s, args.pulse_s) <= 0:
        ap.error('durations must be positive')
    if args.max_prepare_s < max(args.stable_s, args.min_prepare_s):
        ap.error('max-prepare-s must cover the minimum preparation and stability interval')
    if args.probe_s < .25 + args.pulse_s:
        ap.error('probe-s must cover the entire disturbance pulse')
    args.output.mkdir(parents=True, exist_ok=False)
    z = np.load(args.pose_file, allow_pickle=False)
    args.radius, args.height = float(z['radius_m'][args.row]), float(z['height_m'][args.row])
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    wp.init()
    wp.set_device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)
    env = ManagerBasedRlEnv(make_config(args, z), device='cuda:0')
    env.primitive_radius, env.primitive_mass = args.radius, args.mass
    env.gravity_controller = StabilityGravityController(env, args)
    cfg = DexSuperMimicCfg(task='hand2-reorient', algorithm='ppo', num_envs=args.num_envs,
        max_iterations=args.iterations, logger='none', compile=False, compile_policy=False, amp=False,
        wm_hidden=(32, 32), ppo_rollout_steps=32, ppo_epochs=3, ppo_minibatches=4,
        ppo_init_std=.05, ppo_std_min=.01, ppo_lr=3e-4, log_interval=10,
        save_interval=max(1, args.iterations // 2))
    env.max_common_steps = args.iterations * cfg.ppo_rollout_steps
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(privileged_simulation_policy=True, world_model_training=False,
        preparation_support='damped 6D pose spring and gravity compensation; removed before scoring',
        release_gate='continuous joint/object/contact stability, no forced release on timeout',
        success='released, no support, complete probe >=2s, displacement <30mm, rotation <.5rad, '
                'two-finger contacts >=80% and at end, final linear speed <.03m/s and angular speed <.5rad/s',
        disturbance_axes='12 signed world-frame force/torque directions across environments',
        support_translation_k_n_per_m=80., support_translation_d_ns_per_m=3.6,
        support_rotation_k_nm_per_rad=.05, support_rotation_d_nms_per_rad=.003,
        native_collision_exclusions='restored from wuji_description, alongside native collision meshes',
        virtual_preload_budget='measured contact normal preload, capped at 1N/contact; no joint torque feasibility proof',
        unsupported_gravity=(9.81 if args.gravity_mode == 'staged' else 0.))
    if args.gravity_mode == 'virtual':
        metadata['success'] = metadata['success'].replace('complete probe >=2s',
            'unsupported zero-gravity hold >=2s and all 13 virtual wrench LPs feasible')
    (args.output / 'experiment.json').write_text(json.dumps(metadata, indent=2))
    runner = None
    try:
        baseline, baseline_arrays = evaluate(env, None, args)
        print('BASELINE', json.dumps(baseline), flush=True)
        np.savez_compressed(args.output / 'baseline-evaluation.npz', **baseline_arrays)
        if not baseline['finite']:
            raise RuntimeError('Non-finite baseline')
        if not args.evaluate_only or args.actor_checkpoint:
            env.cfg.auto_reset = True
            # Training and evaluation use exactly the same stability gate.
            ordinary_step = env.step
            def training_step(action):
                env.gravity_controller.update_gate()
                return ordinary_step(action)
            env.step = training_step
            runner = DexSuperMimicRunner(env, cfg, args.output / 'training')
            with torch.no_grad():
                runner.actor.net[-1].weight[:20].zero_()
                runner.actor.net[-1].bias[:20].zero_()
            if args.actor_checkpoint:
                state = torch.load(args.actor_checkpoint, map_location='cuda:0', weights_only=False)
                runner.actor.load_state_dict(state['modules']['actor'])
            if not args.evaluate_only:
                runner.learn()
            env.step = ordinary_step
            learned, learned_arrays = evaluate(env, runner.actor.mean, args)
            np.savez_compressed(args.output / 'learned-evaluation.npz', **learned_arrays)
        else:
            learned = None
        summary = dict(baseline=baseline, learned=learned)
        (args.output / 'evaluation.json').write_text(json.dumps(summary, indent=2))
        print('EVALUATION', json.dumps(summary), flush=True)
    finally:
        if runner is not None:
            runner.close()
        env.close()


if __name__ == '__main__':
    main()

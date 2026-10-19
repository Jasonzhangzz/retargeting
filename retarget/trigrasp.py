"""Legacy lightweight NumPy MPC seed used by the AdamW branch.

This is an approximation, not an exact port of the external trigrasp MPC.
SQP uses retarget.fa.trigrasp_comfree_sqp; surface contact planning uses
retarget.contact_exploration.LocalContactExplorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(value: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, eps)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = _unit(np.asarray(quat, dtype=float).reshape(1, 4))[0]
    w, x, y, z = q
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=float,
    )


def _quat_integrate(quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q
    wx, wy, wz = np.asarray(omega, dtype=float)
    derivative = np.array(
        [-x * wx - y * wy - z * wz, w * wx + y * wz - z * wy,
         w * wy + z * wx - x * wz, w * wz + x * wy - y * wx]
    )
    return _unit((q + 0.5 * dt * derivative).reshape(1, 4))[0]


@dataclass
class TrigraspMPCParams:
    h: float = 0.01
    horizon: int = 5
    n_cmd: int = 26
    num_contacts: int = 4
    object_mass: float = 0.1
    contact_stiffness: float = 200.0
    object_weight: float = 1.0
    quaternion_weight: float = 1.0
    tip_weight: float = 1.0
    posture_weight: float = 0.1
    action_weight: float = 0.01
    terminal_weight: float = 2.0
    command_limit: float = 0.05

    @property
    def n_qvel(self):
        return 6 + self.n_cmd

    @property
    def pose_size(self):
        return 7 + self.n_cmd

    @property
    def state_size(self):
        return self.pose_size + self.n_qvel + self.n_cmd

    # Names used by the reference planner are kept as read-only aliases so
    # existing controller code can be migrated without changing dimensions.
    @property
    def h_(self):
        return self.h

    @property
    def n_cmd_(self):
        return self.n_cmd

    @property
    def n_qvel_(self):
        return self.n_qvel

    @property
    def n_qpos_(self):
        return self.state_size


class TrigraspContactModel:
    """Explicit quasi-dynamic contact step used by :class:`TrigraspMPC`."""

    def __init__(self, params: TrigraspMPCParams, *, object_inertia=None,
                 robot_stiffness=None, robot_damping=None):
        self.params = params
        self.object_mass = np.eye(6) * float(params.object_mass)
        self.object_mass[3:, 3:] *= 0.1
        if object_inertia is not None:
            matrix = np.asarray(object_inertia, float).reshape(6, 6)
            self.object_mass = 0.5 * (matrix + matrix.T)
        self.object_inverse = np.linalg.pinv(self.object_mass)
        self.robot_stiffness = np.full(params.n_cmd, 50.0) if robot_stiffness is None else np.asarray(robot_stiffness, float).reshape(params.n_cmd)
        self.robot_damping = np.full(params.n_cmd, 1.0) if robot_damping is None else np.asarray(robot_damping, float).reshape(params.n_cmd)
        self.external_wrench = np.zeros(6)
        self.generalized_inverse = np.zeros((params.n_qvel, params.n_qvel))
        self.generalized_inverse[:6, :6] = self.object_inverse
        self.generalized_inverse[6:, 6:] = 1.0 / np.maximum(self.robot_stiffness, 1e-8)

    def step(self, state, command, phi=None, jac=None, *, target=None):
        p = self.params
        state = np.asarray(state, float).reshape(p.state_size)
        command = np.asarray(command, float).reshape(p.n_cmd)
        pose = state[:p.pose_size].copy()
        velocity = state[p.pose_size:p.pose_size + p.n_qvel].copy()
        servo = state[-p.n_cmd:].copy()
        next_servo = np.clip(servo + command, -np.inf, np.inf)
        hand_q = pose[7:7 + p.n_cmd]
        hand_velocity = velocity[6:]
        free = velocity.copy()
        free[:6] += p.h * self.object_inverse @ self.external_wrench
        free[6:] += p.h * (self.robot_stiffness * (next_servo - hand_q) - self.robot_damping * hand_velocity)
        corrected = free.copy()
        if phi is not None and jac is not None:
            gaps = np.asarray(phi, float).reshape(-1)
            jacobian = np.asarray(jac, float)
            if jacobian.ndim != 2 or jacobian.shape[1] != p.n_qvel:
                raise ValueError("contact Jacobian must have shape (3*n_contacts, 6+n_cmd)")
            contacts = min(p.num_contacts, gaps.size // 4, jacobian.shape[0] // 4)
            if contacts:
                rows, contact_gaps = [], []
                for i in range(contacts):
                    block = jacobian[4 * i:4 * i + 4]
                    row = np.mean(block, axis=0)
                    gap = float(np.mean(gaps[4 * i:4 * i + 4]))
                    rows.append(row)
                    contact_gaps.append(gap)
                if rows:
                    j = np.asarray(rows)
                    # Generalized contact impulses.  The compact expression
                    # matches the explicit model's J.T contact response.
                    impulse = np.maximum(0.0, -np.sum(j * free[None, :], axis=1) - np.asarray(contact_gaps) / p.h)
                    corrected += self.generalized_inverse @ j.T @ impulse
        next_pose = pose.copy()
        next_pose[:3] += p.h * corrected[:3]
        next_pose[3:7] = _quat_integrate(pose[3:7], corrected[3:6], p.h)
        next_pose[7:7 + p.n_cmd] += p.h * corrected[6:]
        return np.r_[next_pose, corrected, next_servo]


class TrigraspMPC:
    """Receding-horizon planner following the reference trigrasp flow."""

    def __init__(self, params: TrigraspMPCParams, model: TrigraspContactModel | None = None):
        self.params = params
        self.model = TrigraspContactModel(params) if model is None else model
        self.sol_guess = None

    def _tip_targets(self, object_pose, anchors):
        pose = np.asarray(object_pose, float).reshape(7)
        return pose[:3][None, :] + np.asarray(anchors, float) @ quat_wxyz_to_matrix(pose[3:7]).T

    def _rollout(self, state, controls, phi, jac, anchors):
        current = np.asarray(state, float).copy()
        trajectory, cost = [], 0.0
        p = self.params
        target_pose = self._target_pose
        target_hand = self._target_hand
        tip_positions = self._tip_positions
        tip_jacobian = self._tip_jacobian
        hand_reference = self._hand_reference
        gap = float(np.min(np.asarray(phi, float))) if np.asarray(phi).size else .02
        attraction = float(np.clip((gap - .002) / .018, 0., 1.) ** 2)
        for command in controls:
            current = self.model.step(current, command, phi, jac, target=anchors)
            trajectory.append(current.copy())
            pose = current[:p.pose_size]
            pos_error = pose[:3] - target_pose[:3]
            quat_error = 1.0 - float(np.dot(pose[3:7], target_pose[3:7]) ** 2)
            hand_error = pose[7:7 + p.n_cmd] - target_hand
            predicted_tips = tip_positions + (tip_jacobian @ (pose[7:7 + p.n_cmd] - hand_reference)).reshape(-1, 3)
            target_tips = self._tip_targets(pose[:7], anchors)
            tip_error = predicted_tips - target_tips
            cost += p.object_weight * float(pos_error @ pos_error) + p.quaternion_weight * quat_error
            cost += attraction * p.posture_weight * float(hand_error @ hand_error) + p.action_weight * float(command @ command)
            cost += p.tip_weight * float(np.sum(tip_error * tip_error))
            if not np.isfinite(current).all():
                return np.inf, trajectory
        final = current[:p.pose_size]
        cost += p.terminal_weight * p.object_weight * float(np.sum((final[:3] - target_pose[:3]) ** 2))
        return float(cost), trajectory

    def plan_once(self, *, state, target_object_pose, target_hand_q, phi, jac,
                  anchors, tip_positions=None, tip_jacobian=None, sol_guess=None):
        p = self.params
        state = np.asarray(state, float).reshape(p.state_size)
        target_pose = np.asarray(target_object_pose, float).reshape(7)
        target_hand = np.asarray(target_hand_q, float).reshape(p.n_cmd)
        anchors = np.asarray(anchors, float).reshape(-1, 3)
        self._target_pose, self._target_hand = target_pose, target_hand
        self._tip_positions = np.asarray(tip_positions, float).reshape(-1, 3)
        self._tip_jacobian = tip_jacobian
        self._hand_reference = state[7:7 + p.n_cmd].copy()
        controls = []
        current = state.copy()
        if tip_positions is None:
            tip_positions = self._tip_targets(current[:7], anchors)
        if tip_jacobian is None:
            tip_jacobian = np.zeros((3 * len(anchors), p.n_cmd))
        tip_jacobian = np.asarray(tip_jacobian, float).reshape(3 * len(anchors), p.n_cmd)
        for _ in range(p.horizon):
            predicted = np.asarray(tip_positions) + (tip_jacobian @ (current[7:7 + p.n_cmd] - state[7:7 + p.n_cmd])).reshape(-1, 3)
            error = (self._tip_targets(current[:7], anchors) - predicted).reshape(-1)
            if error.size and tip_jacobian.size:
                delta = tip_jacobian.T @ np.linalg.solve(tip_jacobian @ tip_jacobian.T + 1e-5 * np.eye(error.size), error)
            else:
                delta = np.zeros(p.n_cmd)
            command = np.clip(0.25 * (target_hand + delta - current[-p.n_cmd:]), -p.command_limit, p.command_limit)
            controls.append(command)
            current = self.model.step(current, command, phi, jac, target=anchors)
        alignment = np.asarray(controls)
        if isinstance(sol_guess, dict):
            sol_guess = sol_guess.get("u_traj")
        candidates = [alignment, np.zeros_like(alignment)]
        if sol_guess is not None and np.asarray(sol_guess).shape == alignment.shape:
            candidates.append(np.asarray(sol_guess, float))
        best_cost, best_traj, best = np.inf, [], candidates[0]
        for candidate in candidates:
            value, trajectory = self._rollout(state, np.clip(candidate, -p.command_limit, p.command_limit), phi, jac, anchors)
            if np.isfinite(value) and value < best_cost:
                best_cost, best_traj, best = value, trajectory, np.asarray(candidate).copy()
        self.sol_guess = best.copy()
        return {"action": best[0].copy(), "u_traj": best, "rollout_q": np.asarray(best_traj),
                "solve_status": "rollout_selected", "solver_backend": "explicit-rollout",
                "cost": float(best_cost), "sol_guess": best.copy()}

    def plan_contacts(self, state, phi, jac, *, tips, tip_jacobian, anchors,
                      goal, hand_posture, sol_guess=None, **_ignored):
        """Compatibility entry point matching ``trigrasp.py``'s live MPC."""
        state = np.asarray(state, float).reshape(self.params.state_size)
        goal = np.asarray(goal, float).reshape(-1)
        target_pose = np.r_[goal[:3], state[3:7]] if goal.size < 7 else goal[:7]
        return self.plan_once(
            state=state, target_object_pose=target_pose,
            target_hand_q=np.asarray(hand_posture, float), phi=phi, jac=jac,
            anchors=anchors, tip_positions=tips, tip_jacobian=tip_jacobian,
            sol_guess=sol_guess,
        )


# Keep the migration name explicit for users that previously imported the
# reference controller's base class.
MPCExplicit = TrigraspMPC
AllegroOnlineMPCParams = TrigraspMPCParams
AllegroOnlineContactModel = TrigraspContactModel
AllegroOnlineMPC = TrigraspMPC


__all__ = ["TrigraspMPCParams", "TrigraspContactModel", "TrigraspMPC", "MPCExplicit",
           "AllegroOnlineMPCParams", "AllegroOnlineContactModel", "AllegroOnlineMPC"]

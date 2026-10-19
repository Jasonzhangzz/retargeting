"""Rollout recording used by the standalone workflow."""
from __future__ import annotations
import numpy as np
QUATERNION_EPSILON=1.e-12

class RolloutRecorder:
    def __init__(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        start_frame: float,
        warm: np.ndarray,
    ):
        self.qpos = [np.asarray(qpos, np.float32)]
        self.qvel = [np.asarray(qvel, np.float32)]
        self.qpos_frames = [float(start_frame)]
        self.raw_warm = [np.asarray(warm, np.float32)]
        self.ctrl: list[np.ndarray] = []
        self.target_frames: list[float] = []
        self.object_velocity_reference: list[np.ndarray] = []
        self.drift: list[float] = []
        self.rotation: list[float] = []
        self.step_seconds: list[float] = []
        self.step_sizes: list[int] = []
        self.scores: list[np.ndarray] = []
        self.candidate_drifts: list[np.ndarray] = []
        self.has_finite_best: list[np.ndarray] = []
        self.candidate_valid: list[np.ndarray] = []
        self.training_failure_iteration: list[np.ndarray] = []
        self.training_failure_code: list[np.ndarray] = []
        self.rejection_code: list[np.ndarray] = []
        self.selected: list[int] = []

    def replace_initial(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.qpos[0] = np.asarray(qpos, np.float32)
        self.qvel[0] = np.asarray(qvel, np.float32)

    def record(
        self,
        plan,
        targets,
        *,
        qpos_sequence: np.ndarray,
        qvel_sequence: np.ndarray,
        index,
        warm: np.ndarray,
        display_frames: np.ndarray | None = None,
    ) -> None:
        self.step_seconds.append(plan.seconds)
        count = qpos_sequence.shape[0]
        self.step_sizes.append(count)
        frames = targets.frames[:count] if display_frames is None else display_frames
        for step in range(count):
            target = targets.object_qpos[step]
            qpos = np.asarray(qpos_sequence[step], np.float32)
            actual = qpos[index.object_qpos : index.object_qpos + 7]
            self.qpos.append(qpos)
            self.qvel.append(np.asarray(qvel_sequence[step], np.float32))
            self.qpos_frames.append(float(frames[step]))
            self.raw_warm.append(np.asarray(warm, np.float32))
            self.ctrl.append(np.asarray(plan.control_sequence[step], np.float32))
            self.target_frames.append(float(targets.frames[step]))
            velocity=getattr(targets,'object_velocity_world',None)
            self.object_velocity_reference.append(np.zeros(6,np.float32) if velocity is None else velocity[step].copy())
            self.drift.append(float(np.linalg.norm(actual[:3] - target[:3])))
            self.rotation.append(quaternion_angle_deg(actual[3:7], target[3:7]))
            self.scores.append(np.asarray(plan.diag["scores"], np.float32))
            self.candidate_drifts.append(np.asarray(plan.diag["drifts"], np.float32))
            self.has_finite_best.append(np.asarray(plan.diag["has_finite_best"], bool))
            self.candidate_valid.append(np.asarray(plan.diag["valid"], bool))
            self.training_failure_iteration.append(
                np.asarray(plan.diag["training_failure_iteration"], np.int32)
            )
            self.training_failure_code.append(
                np.asarray(plan.diag["training_failure_code"], np.int32)
            )
            self.rejection_code.append(
                np.asarray(plan.diag["rejection_code"], np.int32)
            )
            self.selected.append(int(plan.diag["selected"]))

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "qpos_traj": np.stack(self.qpos),
            "qvel_traj": np.stack(self.qvel),
            "qpos_frames": np.asarray(self.qpos_frames, np.float32),
            "ctrl_traj": np.stack(self.ctrl),
            "raw_params": np.asarray(self.raw_warm[-1], np.float32),
            "raw_warm_history": np.stack(self.raw_warm),
            "multistart_scores": np.stack(self.scores),
            "multistart_drifts": np.stack(self.candidate_drifts),
            "multistart_has_finite_best": np.stack(self.has_finite_best),
            "multistart_valid": np.stack(self.candidate_valid),
            "multistart_training_failure_iteration": np.stack(
                self.training_failure_iteration
            ),
            "multistart_training_failure_code": np.stack(self.training_failure_code),
            "multistart_rejection_code": np.stack(self.rejection_code),
            "multistart_selected": np.asarray(self.selected, np.int32),
            "multistart_step_sec": np.repeat(
                np.asarray(self.step_seconds, np.float32),
                np.asarray(self.step_sizes, np.int32),
            ),
            "mpc_step_sec": np.asarray(self.step_seconds, np.float32),
            "mpc_execution_steps": np.asarray(self.step_sizes, np.int32),
            "drift": np.asarray(self.drift, np.float32),
            "rot_err_deg": np.asarray(self.rotation, np.float32),
            "target_frames": np.asarray(self.target_frames, np.float32),
            "object_reference_velocity_world": np.asarray(self.object_velocity_reference,np.float32),
        }
        return arrays

def quaternion_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = left / (float(np.linalg.norm(left)) + QUATERNION_EPSILON)
    right = right / (float(np.linalg.norm(right)) + QUATERNION_EPSILON)
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot) * 180.0 / np.pi)

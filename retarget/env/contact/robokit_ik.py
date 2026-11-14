"""Contact-frame targets solved by Robokit's existing batched IK/LM stack."""
import mujoco
import numpy as np
import warp as wp
from robokit.helpers.ik import IKSolver, IKSolverConfig
from robokit.helpers.ik.config import IKContext
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.robo.robot import Robot
from robokit.terms import PositionTask, RotationTask, PositionLimit, ConfigurationTask, SmoothnessTask
from robokit.xform.numpy import quaternion_to_matrix, matrix_to_quaternion, rotation_between_vectors
from robokit.utils.warp_utils import wp_vec7
from retarget.solver.contact_cem import DEFAULTS


class ContactIK:
    def __init__(self, task, device: str):
        self.task, self.model, self.device, self.solvers = task, task.model, device, {}
        scene = task.scene.copy()
        for key in list(scene.keys):
            scene.delete(key)
        if task.name != 'h1_jog':
            scene.delete(scene.body('pen' if task.name == 'sharpa_pen' else 'coin'))
        root_name = 'pelvis' if task.name == 'h1_jog' else 'right_hand_C_MC'
        root = scene.body(root_name)
        for joint in list(scene.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                scene.delete(joint)
        root.pos, root.quat = [0., 0., 0.], [1., 0., 0., 0.]
        self.robot = Robot.load(scene.to_xml(), load_meshes=False)
        self.root = task.model.body(root_name).id
        self.qadr = np.array([task.model.jnt_qposadr[task.model.joint(name).id]
                             for name in self.robot.spec.actuated_joint_names])
        self.control_indices = np.array([self.robot.spec.actuated_joint_names.index(
            task.model.joint(int(j)).name) for j in task.model.actuator_trnid[:, 0]])
        self.probe = mujoco.MjData(task.model)

    def state(self, qpos: np.ndarray, reference: np.ndarray):
        qpos = np.broadcast_to(qpos, (len(reference), self.model.nq)).copy()
        qpos[:, self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]] = reference
        bases = []
        for q in qpos:
            self.probe.qpos[:] = q
            mujoco.mj_kinematics(self.model, self.probe)
            bases.append(np.r_[self.probe.xpos[self.root], self.probe.xquat[self.root]])
        return self.robot.state(q=wp.array(qpos[:, self.qadr].astype(np.float32), dtype=float, device=self.device),
            T_world_base=wp.array(np.asarray(bases, np.float32), dtype=wp_vec7, device=self.device))

    def solve(self, geom: int, local: np.ndarray, normal: np.ndarray, target: np.ndarray,
              desired_normal: np.ndarray, reference: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        body = int(self.model.geom_bodyid[geom])
        link_name = self.model.body(body).name
        key = (body, len(reference))
        if key not in self.solvers:
            config = IKSolverConfig(enable_T_world_base=True, base_active_mask=(0.,) * 6,
                solver=PopulationSolverConfig(stages=[Stage(DEFAULTS.lm_iterations, 1,
                    LMOptimizerConfig(lm_lambda=DEFAULTS.lm_damping**2))], cuda_graph_mode='full'))
            config.add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.))
            config.add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=.01))
            config.add(PositionLimit(weight=50.))
            config.add(ConfigurationTask(rest_q=IKContext.REST_Q, weight=.03))
            config.add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=.01))
            self.solvers[key] = IKSolver(config, self.robot, link_name, device=self.device)
        posture = (qpos[:, self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]]
                   if np.ndim(qpos) == 2 else reference)
        reference_state = self.state(qpos, posture)
        self.robot.forward_kinematics(reference_state)
        link = self.robot.spec.link_names.index(link_name)
        frame = reference_state.T_world_link.numpy()[:, link]
        rotation = quaternion_to_matrix(frame[:, 3:])
        geom_rotation = quaternion_to_matrix(self.model.geom_quat[geom])
        body_point = self.model.geom_pos[geom] + geom_rotation @ local
        body_normal = geom_rotation @ normal
        current_normal = np.einsum('bij,j->bi', rotation, body_normal)
        rotation = rotation_between_vectors(current_normal, desired_normal) @ rotation
        translation = target - np.einsum('bij,j->bi', rotation, body_point)
        frame_target = np.c_[translation, matrix_to_quaternion(rotation)].astype(np.float32)
        solution = self.solvers[key].solve(
            wp.array(frame_target[:, None], dtype=wp_vec7, device=self.device),
            init_state=reference_state, rest_state=reference_state, prev_state=reference_state,
            init_sample_range=0., base_init_sample_range=(0., 0., 0.))
        q = solution.q.numpy()[:, self.control_indices]
        return np.clip(reference + q - posture, *self.task.ctrl_range.T) - reference

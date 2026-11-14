# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false
"""`CartesianControl`: a `SimMPC` control map from world-frame body twists to joint velocity commands."""

from typing import TYPE_CHECKING, Sequence

import numpy as np
import warp as wp


if TYPE_CHECKING:
    import mujoco  # pyright: ignore[reportMissingImports]

    from robokit.sim import Sim


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _compute_cartesian_ctrl_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    dofs: wp.array(dtype=int),
    ancestors: wp.array(dtype=float),
    gear: wp.array(dtype=float),
    body: int,
    root: int,
    target_position: wp.vec3,
    target_rotation: wp.mat33,
    position_gain: wp.vec3,
    rotation_gain: wp.vec3,
    damping: float,
    ctrl: wp.array2d(dtype=float),
):
    """Each world's commands `gear * J^T (J J^T + damping)^-1 twist`, the twist being the commanded one plus the feedback toward the target; `J`'s columns are the actuated dofs' world-frame twists at the body."""
    world = wp.tid()
    offset = xpos[world, body] - subtree_com[world, root]
    q_xyzw_error = wp.quat_from_matrix(target_rotation * wp.transpose(xmat[world, body]))
    if q_xyzw_error[3] < 0.0:
        q_xyzw_error = -q_xyzw_error
    angular = wp.vec3(q_xyzw_error[0], q_xyzw_error[1], q_xyzw_error[2])
    length = wp.length(angular)
    angular *= 2.0 * wp.atan2(length, q_xyzw_error[3]) / wp.max(length, 1.0e-9)
    linear = target_position - xpos[world, body]
    twist = wp.spatial_vector()
    for i in range(3):
        twist[i] = position_gain[i] * linear[i]
        twist[i + 3] = rotation_gain[i] * angular[i]
    for i in range(wp.min(6, ctrl.shape[1])):
        twist[i] += ctrl[world, i]
    matrix = wp.spatial_matrix()  # J J^T
    for actuator in range(dofs.shape[0]):
        motion = cdof[world, dofs[actuator]]
        omega = wp.spatial_top(motion) * ancestors[actuator]
        velocity = (wp.spatial_bottom(motion) + wp.cross(omega, offset)) * ancestors[actuator]
        column = wp.spatial_vector(velocity[0], velocity[1], velocity[2], omega[0], omega[1], omega[2])
        for i in range(6):
            for j in range(6):
                matrix[i, j] += column[i] * column[j]
    # damped 6x6 Cholesky solve; no fixed robot size or matrix inverse is needed
    lower = wp.spatial_matrix()
    for j in range(6):
        diagonal = matrix[j, j] + damping
        for k in range(j):
            diagonal -= lower[j, k] * lower[j, k]
        lower[j, j] = wp.sqrt(wp.max(diagonal, 1.0e-12))
        for i in range(j + 1, 6):
            value = matrix[i, j]
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            lower[i, j] = value / lower[j, j]
    forward = wp.spatial_vector()
    solution = wp.spatial_vector()
    for i in range(6):
        value = twist[i]
        for j in range(i):
            value -= lower[i, j] * forward[j]
        forward[i] = value / lower[i, i]
    for reverse in range(6):
        i = 5 - reverse
        value = forward[i]
        for j in range(i + 1, 6):
            value -= lower[j, i] * solution[j]
        solution[i] = value / lower[i, i]
    for actuator in range(dofs.shape[0]):
        motion = cdof[world, dofs[actuator]]
        omega = wp.spatial_top(motion) * ancestors[actuator]
        velocity = (wp.spatial_bottom(motion) + wp.cross(omega, offset)) * ancestors[actuator]
        column = wp.spatial_vector(velocity[0], velocity[1], velocity[2], omega[0], omega[1], omega[2])
        ctrl[world, actuator] = wp.dot(column, solution) * gear[actuator]


# --- control map ------------------------------------------------------------
class CartesianControl:
    """A `SimMPC` control map: world-frame body twists (the first six commands) to joint velocities by damped least squares, with feedback toward `position_ref` and `q_wxyz_ref`; needs scalar joint actuators.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco)
        >>> import mujoco
        >>> model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><joint name="j" type="slide"/>'
        ... '<geom size="0.1"/></body></worldbody><actuator><velocity joint="j"/></actuator></mujoco>')
        >>> control = CartesianControl(model, "b", position_gain=[0, 0, 1])
        >>> control.dofs.shape
        (1,)
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        body: str,
        position_ref: Sequence[float] = (0, 0, 0),
        q_wxyz_ref: Sequence[float] = (1, 0, 0, 0),
        position_gain: Sequence[float] = (0, 0, 0),
        rotation_gain: Sequence[float] = (0, 0, 0),
        damping: float = 1e-3,
    ):
        import mujoco  # pyright: ignore[reportMissingImports]

        joints = model.actuator_trnid[:, 0]
        q_wxyz = np.asarray(q_wxyz_ref, float)
        if damping <= 0 or np.any(model.actuator_trntype != 0) or np.any(model.jnt_type[joints] < 2):
            raise ValueError("CartesianControl requires scalar joint actuators and positive damping")
        if q_wxyz.shape != (4,) or not np.isfinite(q_wxyz).all() or np.linalg.norm(q_wxyz) == 0:
            raise ValueError("q_wxyz_ref must be a finite nonzero quaternion")
        self.body = model.body(body).id
        self.root = int(model.body_rootid[self.body])
        chain = [self.body]  # the body and its ancestors, which the joints that move it sit on
        while chain[-1]:
            chain.append(int(model.body_parentid[chain[-1]]))
        self.ancestors = wp.array(np.isin(model.jnt_bodyid[joints], chain).astype(np.float32), dtype=float)
        self.dofs = wp.array(model.jnt_dofadr[joints].astype(np.int32), dtype=int)
        self.gear = wp.array(model.actuator_gear[:, 0].astype(np.float32), dtype=float)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, q_wxyz / np.linalg.norm(q_wxyz))
        self.position = wp.vec3(*position_ref)
        self.rotation = wp.mat33(*rotation)
        self.position_gain = wp.vec3(*position_gain)
        self.rotation_gain = wp.vec3(*rotation_gain)
        self.damping = damping

    def __call__(self, sim: "Sim", clock: wp.array):
        sim.kinematics()  # `xpos` and `xmat` of the current state
        sim.com_pos()  # and `subtree_com` and `cdof`
        data = sim.data
        inputs = [data.xpos, data.xmat, data.subtree_com, data.cdof, self.dofs, self.ancestors, self.gear, self.body]
        inputs += [self.root, self.position, self.rotation, self.position_gain, self.rotation_gain, self.damping]
        wp.launch(_compute_cartesian_ctrl_kernel, dim=sim.nworld, inputs=inputs, outputs=[data.ctrl])


__all__ = ["CartesianControl"]

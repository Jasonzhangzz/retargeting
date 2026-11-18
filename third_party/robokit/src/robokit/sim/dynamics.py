# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
"""`SimDynamics`: one step of a `robokit.sim.Sim` as `Dynamics`, on MuJoCo states made of `QposVar` joint positions."""

import copy
from enum import Enum
from typing import Any, Dict, Optional, Tuple, cast

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np
import warp as wp

from robokit.opt.ilqr_optimizer import Dynamics
from robokit.opt.var_values import Var, VarValues, VectorVar
from robokit.terms.simulation.sim_context import sim_context
from robokit.terms.task import Context
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather, masked_copy, wp_device_type
from robokit.xform.warp.rotation_conversions import (
    axis_angle_to_quaternion_func,
    quaternion_invert_func,
    quaternion_multiply_func,
    quaternion_to_axis_angle_func,
)


f64 = wp.float64


class SimKey(Enum):
    """`SimDynamics` states' parts."""

    QPOS = 0
    QVEL = 1
    ACT = 2


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _integrate_qpos_kernel(
    q: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    lin_q: wp.array(dtype=int),  # linear coordinates' qpos indices
    lin_v: wp.array(dtype=int),  # ... and dof indices
    quat_q: wp.array(dtype=int),  # quaternions' qpos addresses
    quat_v: wp.array(dtype=int),  # ... and their rotations' dof addresses
    mask: wp.array(dtype=float),
    has_mask: bool,
    out: wp.array2d(dtype=float),
):
    """Row b moved by `v`: linear coordinates add, quaternions turn by the rotation vector in their own frame."""
    b = wp.tid()
    for i in range(lin_q.shape[0]):
        m = float(1.0)
        if has_mask:
            m = mask[lin_v[i]]
        out[b, lin_q[i]] = q[b, lin_q[i]] + m * v[b, lin_v[i]]
    for i in range(quat_q.shape[0]):
        a, d = quat_q[i], quat_v[i]
        w = wp.vec3(v[b, d], v[b, d + 1], v[b, d + 2])
        if has_mask:
            w = wp.cw_mul(w, wp.vec3(mask[d], mask[d + 1], mask[d + 2]))
        quat = wp.vec4(q[b, a], q[b, a + 1], q[b, a + 2], q[b, a + 3])
        p = wp.normalize(quaternion_multiply_func(quat, axis_angle_to_quaternion_func(w)))
        for j in range(4):
            out[b, a + j] = p[j]


@wp.kernel
def _difference_qpos_kernel(
    q: wp.array2d(dtype=float),
    p: wp.array2d(dtype=float),
    lin_q: wp.array(dtype=int),
    lin_v: wp.array(dtype=int),
    quat_q: wp.array(dtype=int),
    quat_v: wp.array(dtype=int),
    out: wp.array2d(dtype=float),
):
    """Row b's velocity from `q` to `p`; quaternions' rotation vector of conj(q) p, the shorter way, in float64."""
    b = wp.tid()
    for i in range(lin_q.shape[0]):
        out[b, lin_v[i]] = p[b, lin_q[i]] - q[b, lin_q[i]]
    for i in range(quat_q.shape[0]):
        a, d = quat_q[i], quat_v[i]
        qa = wp.vec4d(f64(q[b, a]), f64(q[b, a + 1]), f64(q[b, a + 2]), f64(q[b, a + 3]))
        pa = wp.vec4d(f64(p[b, a]), f64(p[b, a + 1]), f64(p[b, a + 2]), f64(p[b, a + 3]))
        turn = quaternion_to_axis_angle_func(quaternion_multiply_func(quaternion_invert_func(qa), pa))
        for j in range(3):
            out[b, d + j] = float(turn[j])


# --- states -----------------------------------------------------------------
class QposVar(Var):
    """MuJoCo joint positions `(B, nq)`, float32, with `qvel`'s tangent space (nv).

    Example:
        >>> import mujoco
        >>> model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='1'/></body>"
        ...                                        "</worldbody></mujoco>")
        >>> q = QposVar(wp.array([[0, 0, 0, 1, 0, 0, 0]], dtype=float), model)
        >>> v = wp.array([[1, 0, 0, 0, 0, np.pi / 2]], dtype=float)  # 1 m along x, a quarter turn about z
        >>> p = q.integrate(v)
        >>> p.qpos.numpy().astype(float).round(4).tolist()
        [[1.0, 0.0, 0.0, 0.7071, 0.0, 0.0, 0.7071]]
        >>> q.difference(p).numpy().astype(float).round(4).tolist()
        [[1.0, 0.0, 0.0, 0.0, 0.0, 1.5708]]
    """

    def __init__(self, qpos: wp.array, model: mujoco.MjModel):
        lin, quat = [], []  # (qpos index, dof index) of each linear coordinate, each quaternion's first
        for j in range(model.njnt):
            a, d, kind = model.jnt_qposadr[j], model.jnt_dofadr[j], model.jnt_type[j]
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                lin, quat = lin + [(a + i, d + i) for i in range(3)], quat + [(a + 3, d + 3)]
            elif kind == mujoco.mjtJoint.mjJNT_BALL:
                quat.append((a, d))
            else:
                lin.append((a, d))
        self.qpos, self.nv = qpos, model.nv
        self._joints = [wp.array(np.array(p, np.int32).reshape(-1, 2)[:, i].copy(), dtype=int, device=qpos.device)
                        for p in (lin, quat) for i in range(2)]  # fmt: skip

    def _like(self, qpos: wp.array) -> "QposVar":
        var = copy.copy(self)
        var.qpos = qpos
        return var

    @property
    def tangent_dim(self) -> int:
        return self.nv

    @property
    def batch_size(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def device(self) -> wp_device_type:
        return self.qpos.device

    def clone(self) -> "QposVar":
        return self._like(wp.clone(self.qpos))

    def view(self, batch_size: int, offset: int = 0) -> "QposVar":
        return self._like(self.qpos[offset : offset + batch_size])

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["QposVar"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> "QposVar":
        """As `mj_integratePos`: free and ball joints' quaternions turn by `velocity` in their own frame, others add."""
        if weight_decay != 0.0:
            raise ValueError("QposVar does not support weight_decay")
        out = out if out is not None else self._like(wp.empty_like(self.qpos))
        ins = [self.qpos, velocity, *self._joints, tangent_mask, tangent_mask is not None]
        wp.launch(_integrate_qpos_kernel, self.batch_size, inputs=ins, outputs=[out.qpos], device=self.device)
        return out

    def difference(self, other: "QposVar", out: Optional[ArrayLike] = None) -> ArrayLike:
        """The velocity from these positions to `other`'s, as `mj_differentiatePos`."""
        out = out if out is not None else wp.empty((self.batch_size, self.nv), dtype=float, device=self.device)
        ins = [self.qpos, other.qpos, *self._joints]
        wp.launch(_difference_qpos_kernel, self.batch_size, inputs=ins, outputs=[out], device=self.device)
        return out

    def gather(self, indices: ArrayLike, out: Optional["QposVar"] = None) -> "QposVar":
        return self._like(gather(self.qpos, indices, out.qpos if out is not None else None))

    def accept(self, accept_mask: ArrayLike, proposed: "QposVar") -> "QposVar":
        masked_copy(accept_mask, proposed.qpos, self.qpos)
        return self


# --- dynamics ---------------------------------------------------------------
class SimDynamics(Dynamics):
    """`Dynamics` for `robokit.opt.ilqr_optimizer.ILQROptimizer`: one physics step of `(qpos, qvel, act)` on a `Sim`.

    Args:
        sim: A `Sim`; `resized` gives one per batch size stepped and finite-difference setting, as `SimMPC` does.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import mujoco, numpy as np
        >>> from robokit.sim import MjWarpSim
        >>> xml = '<mujoco><worldbody><body><joint name="x" type="slide" axis="1 0 0"/>'
        >>> xml += '<geom size=".1"/></body></worldbody>'
        >>> model = mujoco.MjModel.from_xml_string(xml + '<actuator><motor joint="x"/></actuator></mujoco>')
        >>> dynamics = SimDynamics(MjWarpSim(model, 1))
        >>> x = dynamics.build_state(np.zeros((2, 1)), np.array([[0.0], [1.0]]))
        >>> x_next = x.clone()
        >>> dynamics.step(x, wp.zeros((2, 1), dtype=float), x_next)
        >>> x_next.get(SimKey.QPOS).qpos.numpy()[:, 0].astype(float).round(4).tolist()  # 1 m/s for one 2 ms step
        [0.0, 0.002]
    """

    def __init__(self, sim: Any, steps_per_knot: int = 1):
        from robokit.sim import _mjwarp_sort  # here, so that the module imports without MuJoCo Warp

        model = sim.mjm
        limited = model.actuator_ctrllimited.astype(bool)[:, None]
        super().__init__(model.nu, np.where(limited, model.actuator_ctrlrange, [-np.inf, np.inf]), steps_per_knot)
        self.model, self._base = model, sim
        self._sims: Dict[Tuple[int, bool], Any] = {}
        self._sim: Any = None  # the last stepped
        self._last: Optional[Var] = None  # the states it stepped to
        _mjwarp_sort.patch()

    def _get_sim(self, batch_size: int, finite_differences: bool) -> Any:
        """The simulator that steps `batch_size` states, made on first use."""
        key = (batch_size, finite_differences)
        if key not in self._sims:
            sim = self._sims[key] = self._base.resized(batch_size)
            if finite_differences and hasattr(sim, "opt"):  # MuJoCo Warp: sorted rows (costly, so not elsewhere)
                import mujoco_warp as mjw  # pyright: ignore[reportMissingImports]

                sim.opt.deterministic = int(mjw.DeterminismType.CONSTRAINT)  # so neighboring states' differences agree
                sim.data.contact.geomcollisionid.zero_()
        return self._sims[key]

    def get_context(self, batch_size: int, finite_differences: bool = False) -> Dict[Context, Any]:
        """The simulation terms' context (`sim_context`) of the worlds that step `batch_size` states."""
        return sim_context(self._get_sim(batch_size, finite_differences))

    def build_state(self, qpos: ArrayLike, qvel: ArrayLike, act: Optional[ArrayLike] = None) -> VarValues:
        """A `VarValues` state keyed by `SimKey`.

        Args:
            qpos: `(B, nq)` joint positions.
            qvel: `(B, nv)` velocities.
            act: `(B, na)` activations, if the model has any.
        """
        array = lambda a: a if isinstance(a, wp.array) else wp.array(np.asarray(a, np.float32), dtype=float)
        parts: Dict[Enum, Var] = {SimKey.QPOS: QposVar(array(qpos), self.model), SimKey.QVEL: VectorVar(array(qvel))}
        if self.model.na:
            parts[SimKey.ACT] = VectorVar(array(act))
        return VarValues(parts)

    def step(self, x: Var, u: wp.array, x_next: Var, finite_differences: bool = False):
        """`Dynamics.step`; then the worlds' `sensordata` is that of `x`, the rest that of `x_next`."""
        assert isinstance(x, VarValues) and isinstance(x_next, VarValues)
        sim = self._get_sim(x.batch_size, finite_differences)
        fresh = x is not self._last or sim is not self._sim  # a step that continues the last keeps its warm start
        for key in [k for k in SimKey if k in x] if fresh else ():  # a fresh start: its states, a cold solver
            wp.copy(getattr(sim.data, key.name.lower()), _get_array(x.get(key)))
        if fresh:
            sim.data.qacc_warmstart.zero_()
        wp.copy(sim.data.ctrl, u)
        for _ in range(self.steps_per_knot):  # under the same controls
            sim.step()
        for key in [k for k in SimKey if k in x_next]:
            wp.copy(_get_array(x_next.get(key)), getattr(sim.data, key.name.lower()))
        self._sim, self._last = sim, x_next


def _get_array(var: Var) -> wp.array:
    return var.qpos if isinstance(var, QposVar) else cast(VectorVar, var).x


__all__ = ["QposVar", "SimDynamics", "SimKey"]

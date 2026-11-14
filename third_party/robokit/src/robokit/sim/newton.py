# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false
"""`NewtonSim`: a MuJoCo model's worlds on Newton's Featherstone solver, read and written as MuJoCo's data fields."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Optional

import mujoco
import newton
import numpy as np
import warp as wp

from robokit.sim.mjwarp import _reset_kernel


# --- kernels ---------------------------------------------------------------
@wp.kernel
def _compute_derived_kernel(
    body_q: wp.array2d(dtype=wp.transform),
    body_qd: wp.array2d(dtype=wp.spatial_vector),
    link: wp.array(dtype=int),
    kind: wp.array(dtype=int),
    body: wp.array(dtype=int),
    adr: wp.array(dtype=int),
    subtree_mass: wp.array2d(dtype=float),
    limits: wp.array2d(dtype=float),
    gain: wp.array(dtype=float),
    dof: wp.array(dtype=int),
    ctrl: wp.array2d(dtype=float),
    dt: float,
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    sensordata: wp.array2d(dtype=float),
    joint_f: wp.array2d(dtype=float),
    time: wp.array(dtype=float),
):
    """Per world: body poses (body `i` is Newton's `link[i]`), sensors, joint forces from `ctrl`, and `time` += `dt`."""
    w, i = wp.tid()  # pyright: ignore[reportAssignmentType]
    if i == 0:
        time[w] = time[w] + dt
    if i > 0 and i < xpos.shape[1]:
        T_world_body = body_q[w, link[i]]
        xpos[w, i] = wp.transform_get_translation(T_world_body)
        xmat[w, i] = wp.quat_to_matrix(wp.transform_get_rotation(T_world_body))
    if i < ctrl.shape[1]:
        joint_f[w, dof[i]] = gain[i] * wp.clamp(ctrl[w, i], limits[i, 0], limits[i, 1])
    if i < kind.shape[0]:
        T_world_body = body_q[w, link[body[i]]]
        value = wp.transform_get_translation(T_world_body)  # framepos
        if kind[i] == 1:  # framezaxis
            value = wp.quat_rotate(wp.transform_get_rotation(T_world_body), wp.vec3(0.0, 0.0, 1.0))
        if kind[i] == 2:  # subtreelinvel: the subtree's mass-weighted centre-of-mass velocity
            momentum, mass = wp.vec3(0.0), float(0.0)  # pyright: ignore[reportCallIssue]
            for b in range(1, subtree_mass.shape[1]):
                momentum += subtree_mass[i, b] * wp.spatial_top(body_qd[w, link[b]])
                mass += subtree_mass[i, b]
            value = momentum / mass  # pyright: ignore[reportOperatorIssue]
        for k in range(3):
            sensordata[w, adr[i] + k] = value[k]


# --- simulator -------------------------------------------------------------
_LINK_MASS = 1e-6  # kg: each nested link that carries a body's joint after its first


class NewtonSim:
    """Step `nworld` copies of `model` on Newton's `SolverFeatherstone`, `substeps` steps per `step()`, as MuJoCo data.

    Args:
        data: The state `reset()` restores (`qpos`, `qvel`, `ctrl`, `time`), else keyframe `key`'s, else the defaults.
        contact_ke: Penalty contacts' stiffness.
        contact_kd: Penalty contacts' damping.
        contact_kf: Penalty contacts' friction stiffness.
        limit_ke: Penalty joint limits' stiffness.
        limit_kd: Penalty joint limits' damping.

    Example:
        >>> # xdoctest: +REQUIRES(module:newton)
        >>> model = mujoco.MjModel.from_xml_string(
        ...     '<mujoco><worldbody><body name="b"><joint name="z" type="slide" axis="0 0 1"/><joint name="y"/>'
        ...     '<geom size="0.1" mass="1"/></body></worldbody><actuator><motor joint="y" ctrlrange="-1 1"/>'
        ...     '</actuator><sensor><framepos name="p" objtype="xbody" objname="b"/></sensor></mujoco>'
        ... )
        >>> sim = NewtonSim(model, nworld=4, substeps=2)
        >>> sim.step()  # every world falls
        >>> sim.data.qpos.shape, sim.data.xpos.shape, sim.data.substep_sensordata.shape
        ((4, 2), (4, 2), (2, 4, 3))
        >>> sim.reset(wp.array(np.arange(4) < 2))  # the first two worlds back to the start
        >>> (sim.data.qpos.numpy()[:, 0] < 0).tolist()
        [False, False, True, True]
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        nworld: int,
        data: Optional[mujoco.MjData] = None,
        substeps: int = 1,
        key: Optional[str] = None,
        device=None,
        contact_ke: float = 2e4,
        contact_kd: float = 200.0,
        contact_kf: float = 200.0,
        limit_ke: float = 2e3,
        limit_kd: float = 5.0,
    ):
        self._args = {k: v for k, v in locals().items() if k not in ("self", "nworld")}  # for `resized`
        m, sensor = model, mujoco.mjtSensor
        kinds = {int(sensor.mjSENS_FRAMEPOS): 0, int(sensor.mjSENS_FRAMEZAXIS): 1, int(sensor.mjSENS_SUBTREELINVEL): 2}
        frame = m.sensor_type != sensor.mjSENS_SUBTREELINVEL
        unsupported = ~np.isin(m.sensor_type, list(kinds)) | frame & (m.sensor_objtype != mujoco.mjtObj.mjOBJ_XBODY)
        actuators = np.stack([m.actuator_trntype, m.actuator_dyntype, m.actuator_gaintype, m.actuator_biastype])
        if np.any(m.jnt_type < 2) or np.any(actuators != 0) or np.any(unsupported | (m.sensor_refid >= 0)):
            raise ValueError("NewtonSim supports slide/hinge joints, gain-only joint actuators and the listed sensors")

        # `model` as MJCF for Newton: each body's joints after its first on nested links.
        s = lambda x: " ".join(map(str, np.ravel(x)))
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        asset, links = ET.SubElement(root, "asset"), [ET.SubElement(root, "worldbody")]  # each body's innermost link
        for k in range(m.nmesh):
            vert = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            face = m.mesh_face[m.mesh_faceadr[k] : m.mesh_faceadr[k] + m.mesh_facenum[k]]
            ET.SubElement(asset, "mesh", name=str(k), vertex=s(vert), face=s(face))
        for b in range(1, m.nbody):
            link = ET.SubElement(links[m.body_parentid[b]], "body", pos=s(m.body_pos[b]), quat=s(m.body_quat[b]))
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                if j > m.body_jntadr[b]:  # a later joint: on a nested link
                    ET.SubElement(link, "inertial", pos="0 0 0", mass=str(_LINK_MASS), diaginertia=s([_LINK_MASS] * 3))
                    link = ET.SubElement(link, "body")
                dof, limit = m.jnt_dofadr[j], dict(range=s(m.jnt_range[j])) if m.jnt_limited[j] else {}
                ET.SubElement(
                    link, "joint", type=("slide", "hinge")[m.jnt_type[j] - 2], pos=s(m.jnt_pos[j]),
                    axis=s(m.jnt_axis[j]), damping=str(m.dof_damping[dof]), armature=str(m.dof_armature[dof]), **limit,
                )  # fmt: skip
            ET.SubElement(
                link, "inertial", pos=s(m.body_ipos[b]), quat=s(m.body_iquat[b]), mass=str(m.body_mass[b]),
                diaginertia=s(m.body_inertia[b]),
            )  # fmt: skip
            links.append(link)
        for g in range(m.ngeom):
            mesh = dict(mesh=str(m.geom_dataid[g])) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else {}
            ET.SubElement(
                links[m.geom_bodyid[g]], "geom", type=mujoco.mjtGeom(m.geom_type[g]).name[7:].lower(),
                size=s(m.geom_size[g]), pos=s(m.geom_pos[g]), quat=s(m.geom_quat[g]), friction=s(m.geom_friction[g]),
                contype=str(m.geom_contype[g]), conaffinity=str(m.geom_conaffinity[g]), **mesh,
            )  # fmt: skip

        # Newton's model of `nworld` copies, its collisions and its solver.
        b = newton.ModelBuilder()
        b.add_mjcf(ET.tostring(root, encoding="unicode"))
        b.shape_material_ke[:] = [contact_ke] * len(b.shape_material_ke)
        b.shape_material_kd[:] = [contact_kd] * len(b.shape_material_kd)
        b.shape_material_kf[:] = [contact_kf] * len(b.shape_material_kf)
        b.joint_limit_ke[:] = [limit_ke] * len(b.joint_limit_ke)
        b.joint_limit_kd[:] = [limit_kd] * len(b.joint_limit_kd)
        scene = newton.ModelBuilder()
        scene.replicate(b, nworld)
        self.mjm, self.nworld, self.substeps, self.device = m, nworld, substeps, wp.get_device(device)
        self.stream = wp.get_stream(self.device) if self.device.is_cuda else None
        newton_model = self.newton_model = scene.finalize(device=self.device)  # `model`'s bodies, joints and geoms
        state, control = self._state, self._control = newton_model.state(), newton_model.control()
        self._pipeline = newton.CollisionPipeline(newton_model)
        self._contacts = self._pipeline.contacts()
        self._solver = newton.solvers.SolverFeatherstone(newton_model, angular_damping=0.0)

        # MuJoCo's data fields over Newton's state, `_compute_derived_kernel`'s arguments and `reset`'s initial rows.
        ancestor = np.eye(m.nbody, dtype=bool)  # ancestor[c, a]: body a is c or holds it
        for c in range(1, m.nbody):
            ancestor[c] |= ancestor[m.body_parentid[c]]
        if data is None:
            data = mujoco.MjData(m)
            if key is not None:
                mujoco.mj_resetDataKeyframe(m, data, m.key(key).id)
        with wp.ScopedDevice(self.device):
            limited = m.actuator_ctrllimited.astype(bool)[:, None]
            d = self.data = SimpleNamespace(  # as `mujoco.MjData`'s; `qpos`, `qvel`: Newton's joint state
                time=wp.zeros(nworld, dtype=float),
                qpos=state.joint_q.reshape((nworld, -1)),
                qvel=state.joint_qd.reshape((nworld, -1)),
                act=wp.zeros((nworld, 0), dtype=float),
                qacc_warmstart=wp.zeros((nworld, m.nv), dtype=float),
                ctrl=wp.zeros((nworld, m.nu), dtype=float),  # clamped to `ctrlrange`, times gear and gain: joint forces
                sensordata=wp.zeros((nworld, m.nsensordata), dtype=float),
                xpos=wp.zeros((nworld, m.nbody), dtype=wp.vec3),
                xmat=wp.array(np.tile(np.eye(3), (nworld, m.nbody, 1, 1)), dtype=wp.mat33),
                nacon=wp.zeros(1, dtype=int),
                contact=SimpleNamespace(
                    worldid=wp.zeros(0, dtype=int),
                    geom=wp.zeros(0, dtype=wp.vec2i),
                    dist=wp.zeros(0, dtype=float),
                    pos=wp.zeros(0, dtype=wp.vec3),
                ),
            )
            self.model = SimpleNamespace()  # no MuJoCo fields
            self.data.substep_sensordata = wp.zeros((substeps, nworld, m.nsensordata), dtype=float)
            self._inputs = [  # of `_compute_derived_kernel`, before `dt`
                state.body_q.reshape((nworld, -1)),
                state.body_qd.reshape((nworld, -1)),
                wp.array(np.arange(m.nbody) - 1 + np.cumsum(np.maximum(m.body_jntnum - 1, 0)), dtype=int),
                wp.array([kinds[int(t)] for t in m.sensor_type], dtype=int),
                wp.array(m.sensor_objid, dtype=int),
                wp.array(m.sensor_adr, dtype=int),
                wp.array(ancestor[:, m.sensor_objid].T * m.body_mass, dtype=float),
                wp.array(np.where(limited, m.actuator_ctrlrange, [-np.inf, np.inf]), dtype=float),
                wp.array(m.actuator_gear[:, 0] * m.actuator_gainprm[:, 0], dtype=float),
                wp.array(m.jnt_dofadr[m.actuator_trnid[:, 0]], dtype=int),
                d.ctrl,
            ]
            self._outputs = [d.xpos, d.xmat, d.sensordata, control.joint_f.reshape((nworld, -1)), d.time]
            self._initial = [  # (world-major data array, each world's initial row) pairs for `reset`
                (getattr(d, f).reshape((nworld, -1)), wp.array(np.tile(getattr(data, f), (nworld, 1)), dtype=float))
                for f in ("qpos", "qvel", "ctrl", "time")
            ]
        self._dim = (nworld, max(m.nbody, m.nsensor, m.nu))
        self.step()  # allocates the solver's buffers before any graph capture
        self.reset(wp.full(nworld, 1, dtype=int, device=self.device))

    def resized(self, nworld: int) -> "NewtonSim":
        """A new simulator of this one's model and settings with `nworld` worlds."""
        return NewtonSim(nworld=nworld, **self._args)

    def step(self):
        """Advance `substeps` steps from `qpos`, `qvel` as they are; derived fields lag a substep, as MuJoCo's."""
        dt, s = self.mjm.opt.timestep, self._state
        with wp.ScopedDevice(self.device):
            newton.eval_fk(self.newton_model, s.joint_q, s.joint_qd, s)  # the body poses of `qpos`, also if written
            for k in range(self.substeps):  # `substep_sensordata[k]`: substep `k`'s start
                wp.launch(_compute_derived_kernel, dim=self._dim, inputs=[*self._inputs, dt], outputs=self._outputs)
                wp.copy(self.data.substep_sensordata, self.data.sensordata, dest_offset=k * self.data.sensordata.size)
                self._pipeline.collide(s, self._contacts)
                self._solver.step(s, s, self._control, self._contacts, dt)

    def forward(self):
        """Recompute `xpos`, `xmat`, `sensordata` (body framepos, framezaxis; subtreelinvel) from `qpos` and `qvel`."""
        s = self._state
        with wp.ScopedDevice(self.device):
            newton.eval_fk(self.newton_model, s.joint_q, s.joint_qd, s)
            wp.launch(_compute_derived_kernel, dim=self._dim, inputs=[*self._inputs, 0.0], outputs=self._outputs)

    def reset(self, mask: wp.array):
        """Reset the worlds where `mask` (nworld,), bool or int, is set, then run `forward()`; no host syncs."""
        with wp.ScopedDevice(self.device):
            for out, value in self._initial:
                wp.launch(_reset_kernel, dim=out.shape, inputs=[mask, value], outputs=[out])
        self.forward()


__all__ = ["NewtonSim"]

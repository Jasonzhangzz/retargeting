"""Thin environment adapters around the installed MuJoCo Warp engine."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import warp as wp
from mujoco_warp._src import forward, smooth

from retarget.dynamics.comfree import ContactResponse


def load_model(path, *, device):
    cpu = mujoco.MjModel.from_xml_path(str(path))
    # The original ComFree response consumes dense constraint rows. This only
    # changes storage, not the contact Jacobian or collision geometry.
    cpu.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    if cpu.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError("ComFree requires pyramidal friction cones")
    with wp.ScopedDevice(device):
        model = mjw.put_model(cpu)
    return model, cpu


class KinematicModel:
    def __init__(self, cpu, model):
        self.cpu, self.model = cpu, model
        self.device = model.qpos0.device

    def allocate(self, worlds):
        return allocate_output(self, worlds)

    def evaluate(self, qpos, pose):
        kinematics(self, qpos, pose)


def allocate_output(model, worlds, *, requires_grad=False):
    if requires_grad:
        raise ValueError("Sampling environments are forward-only")
    # Kinematic-only calls need none of the collision/constraint capacity.
    with wp.ScopedDevice(model.device):
        data = mjw.make_data(model.cpu, nworld=worlds, nconmax=1, njmax=1, nccdmax=1)
    return SimpleNamespace(
        data=data,
        body_position=data.xpos,
        body_matrix=data.xmat,
        body_quaternion=data.xquat,
        joint_anchor=data.xanchor,
        joint_axis=data.xaxis,
    )


def kinematics(model, qpos, pose):
    wp.copy(pose.data.qpos[: qpos.shape[0]], qpos)
    smooth.kinematics(model.model, pose.data)
    smooth.com_pos(model.model, pose.data)


class ComFreeEnvironment:
    """MJWarp forward pipeline with only its constraint solve replaced."""

    def __init__(self, cpu, model, worlds, *, stiffness=0.2, damping=0.001):
        if cpu.opt.integrator not in (
            mujoco.mjtIntegrator.mjINT_EULER,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        ):
            raise ValueError("ComFree environment supports Euler/implicitfast only")
        if cpu.neq or (cpu.dof_frictionloss != 0).any():
            raise ValueError(
                "Unilateral ComFree rows do not support equality or frictionloss constraints"
            )
        self.cpu, self.model = cpu, model
        # Upstream _efc_row hardcodes width=.01. Let MJWarp assemble its usual
        # rows with exactly that width, rather than maintaining a row builder.
        for name in (
            "geom_solimp",
            "pair_solimp",
            "jnt_solimp",
            "dof_solimp",
            "tendon_solimp_fri",
            "tendon_solimp_lim",
            "flex_solimp",
            "eq_solimp",
        ):
            array = getattr(model, name)
            if array.size:
                values = array.numpy()
                values[..., 2] = 0.01
                array.assign(values)
        with wp.ScopedDevice(model.qpos0.device):
            self.data = mjw.make_data(
                cpu, nworld=worlds, nconmax=512, njmax=4096, nccdmax=512
            )
        self.response = ContactResponse(
            worlds,
            cpu.nv,
            device=model.qpos0.device,
            stiffness=stiffness,
            damping=damping,
        )

    def evaluate_dynamics(self):
        m, d = self.model, self.data
        forward.fwd_position(m, d)
        forward.fwd_velocity(m, d)
        forward.fwd_actuation(m, d)
        forward.fwd_acceleration(m, d)
        force = self.response.compute(
            m.opt.timestep,
            d.qvel,
            d.qacc_smooth,
            d.efc.J,
            d.efc.pos,
            d.efc.D,
            d.nefc,
            d.qfrc_smooth,
            d.efc.force,
            d.qfrc_constraint,
        )
        smooth.solve_m(m, d, d.qacc, force)
        # Both installed MJWarp integrators read this total-force scratch.
        wp.copy(d.efc.Ma, force)

    def integrate(self):
        if self.cpu.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            forward.euler(self.model, self.data)
        else:
            forward.implicit(self.model, self.data)

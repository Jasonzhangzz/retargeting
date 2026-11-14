# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
import math
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Optional, Tuple, cast

import warp as wp

from robokit.lie.se3 import se3_identity
from robokit.opt.var_values import Var
from robokit.robo.robot_kernels import (
    accept_state_forward_kinematics_kernel,
    clamp_joint_positions_kernel,
    compute_link_jacobian_kernel,
    integrate_floating_base_kernel,
    integrate_joint_positions_kernel,
)
from robokit.robo.robot_spec import RobotSpec
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather as warp_gather
from robokit.utils.warp_utils import wp_device_type, wp_vec7


# lazily allocated buffers, copied in clone()/gather() only when materialized
_GATHER_BUFFERS = (
    "T_world_joint",
    "T_world_link",
    "S_world",
    "collision_sphere_centers_world",
    "link_bounding_sphere_centers_world",
    "link_capsule_endpoint_a_world",
    "link_capsule_endpoint_b_world",
)
_CLONE_BUFFERS = _GATHER_BUFFERS + ("com_world",)


if TYPE_CHECKING:
    from robokit.robo.robot import Robot


class RobotState(Var):
    spec: RobotSpec
    q: wp.array  # [batch_size, num_dofs] or [batch_size, num_frames, num_dofs]
    T_world_base: wp.array  # wp_vec7

    has_floating_base: bool
    is_fk_computed: bool
    is_motion_subspace_computed: bool
    is_collision_spheres_computed: bool
    is_capsules_computed: bool
    is_com_computed: bool

    def __init__(self, robot: "Robot", q: wp.array, T_world_base: Optional[wp.array] = None):
        self.robot = robot
        self.spec = robot.spec
        self.q = self._validate_q_shape(q).contiguous()
        requires_grad = self.q.requires_grad
        self.base_step_scale = 1e-4
        self.clamp_q_on_integrate = False  # clamp q to the joint limits after every integrate
        self.spec_tensors = robot.spec.get_tensors(str(self.q.device))
        self.clamp_limits = self.spec_tensors.actuated_joint_limits  # (dofs, 2) bounds clamp_q uses

        if T_world_base is None:
            self.T_world_base = se3_identity(shape=self.shape, device=self.device, requires_grad=requires_grad)
            self.has_floating_base = False
        else:
            self.T_world_base = T_world_base.contiguous()
            self.has_floating_base = True

        self.invalidate()

    def __repr__(self) -> str:
        return f"RobotState(q={self.q.numpy()}, T_world_base={self.T_world_base}, is_fk_computed={self.is_fk_computed})"

    # --- lazily allocated buffers (safe with CUDA graphs: capture sites warm up first) ---
    @cached_property
    def T_world_joint(self) -> wp.array:
        source = self.__dict__.get("_view_source")
        if source is not None:
            return source.T_world_joint.reshape((source.num_elements, self.spec.num_joints))[
                : self.num_elements
            ].reshape(self.shape + (self.spec.num_joints,))
        return se3_identity(
            shape=self.shape + (self.spec.num_joints,), device=self.device, requires_grad=self.q.requires_grad
        )

    @cached_property
    def T_world_link(self) -> wp.array:
        source = self.__dict__.get("_view_source")
        if source is not None:
            return source.T_world_link.reshape((source.num_elements, self.spec.num_links))[: self.num_elements].reshape(
                self.shape + (self.spec.num_links,)
            )
        return se3_identity(
            shape=self.shape + (self.spec.num_links,), device=self.device, requires_grad=self.q.requires_grad
        )

    @cached_property
    def X_local(self) -> wp.array:
        return wp.zeros(shape=self.shape + (self.spec.num_joints,), dtype=wp_vec7, device=self.device)

    @cached_property
    def S_world(self) -> wp.array:
        return wp.zeros(
            shape=self.shape + (6, self.spec.num_joints),
            dtype=wp.float32,
            device=self.device,
            requires_grad=self.q.requires_grad,
        )

    @cached_property
    def com_world(self) -> wp.array:
        return wp.zeros(shape=(self.num_elements,), dtype=wp.vec3, device=self.device)

    @cached_property
    def collision_sphere_centers_world(self) -> wp.array:
        num_spheres = len(self.spec.local_collision_sphere_centers) if self.spec.has_collision_spheres else 0
        return wp.zeros(
            shape=self.shape + (num_spheres,), dtype=wp.vec3, device=self.device, requires_grad=self.q.requires_grad
        )

    @cached_property
    def link_bounding_sphere_centers_world(self) -> wp.array:
        num_bounding = self.spec.num_links if self.spec.has_collision_spheres else 0
        return wp.zeros(
            shape=self.shape + (num_bounding,), dtype=wp.vec3, device=self.device, requires_grad=self.q.requires_grad
        )

    @cached_property
    def link_capsule_endpoint_a_world(self) -> wp.array:
        return wp.zeros(
            shape=self.shape + (self.spec.num_links,),
            dtype=wp.vec3,
            device=self.device,
            requires_grad=self.q.requires_grad,
        )

    @cached_property
    def link_capsule_endpoint_b_world(self) -> wp.array:
        return wp.zeros(
            shape=self.shape + (self.spec.num_links,),
            dtype=wp.vec3,
            device=self.device,
            requires_grad=self.q.requires_grad,
        )

    # --- shape and device properties ---
    def _validate_q_shape(self, q: wp.array) -> wp.array:
        if q.ndim == 1:
            q = q.reshape((1, q.shape[0]))
        elif q.ndim in {2, 3}:
            if q.shape[-1] != self.spec.num_actuated_joints:
                raise ValueError(f"Expected q to have {self.spec.num_actuated_joints} DOFs, but got {q.shape[1]}")
        else:
            raise ValueError(f"Expected q to have 1, 2, or 3 dimensions, but got {q.ndim}")
        return q

    @property
    def is_trajectory(self) -> bool:
        return self.q.ndim == 3  # [batch_size, num_frames, num_dofs]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.q.shape[:-1]

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def device(self) -> wp_device_type:
        return cast(wp_device_type, self.q.device)

    @property
    def tangent_dim(self) -> int:
        single_tangent_dim = (6 if self.has_floating_base else 0) + self.spec.num_actuated_joints
        return single_tangent_dim * self.shape[1] if self.is_trajectory else single_tangent_dim

    # --- computed results access ---
    def get_T_world_link(self, link_index: int) -> wp.array:
        if not self.is_fk_computed:
            raise RuntimeError("Forward kinematics has not been computed. Call robot.forward_kinematics() first.")
        xyz_wxyz = self.T_world_link
        index = tuple([slice(None)] * (xyz_wxyz.ndim - 1) + [link_index])
        # slicing out one link yields a strided view; downstream se3_* kernels flatten
        return cast(wp.array, xyz_wxyz[index]).contiguous()

    def get_link_jacobian(self, link_index: int, reference_frame: Literal["body", "spatial"] = "body") -> wp.array:
        if not self.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(self)

        wp.init()
        device = self.S_world.device
        requires_grad = self.S_world.requires_grad
        num_actuated = self.spec.num_actuated_joints
        ne = self.num_elements

        total_dofs = (6 if self.has_floating_base else 0) + num_actuated
        J_out = wp.zeros((*self.shape, 6, total_dofs), dtype=wp.float32, device=device, requires_grad=requires_grad)
        spec_t = self.spec.get_tensors(str(device))
        reference_frame_int = 1 if reference_frame == "body" else 0

        wp.launch(
            kernel=compute_link_jacobian_kernel,
            dim=(ne, total_dofs),
            inputs=[
                self.S_world.reshape((ne, 6, self.spec.num_joints)),
                self.T_world_link.reshape((ne, self.spec.num_links)),
                self.T_world_base.flatten(),
                spec_t.link_ancestor_joints_mask[link_index],
                spec_t.joints_to_actuated_mapping,
                reference_frame_int,
                self.has_floating_base,
                link_index,
                J_out.reshape((ne, 6, total_dofs)),
            ],
            device=device,
        )

        return J_out

    # --- solver interface (Var) ---
    def invalidate(self):
        self.is_fk_computed = False
        self.is_motion_subspace_computed = False
        self.is_collision_spheres_computed = False
        self.is_capsules_computed = False
        self.is_com_computed = False

    def precompute(self):
        if not self.is_fk_computed:
            self.robot.forward_kinematics(self)

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["RobotState"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> "RobotState":
        """Integrate the state by a tangent-space velocity, optionally reporting the move the limits let through."""
        if velocity.shape != (self.batch_size, self.tangent_dim):
            raise ValueError(f"Expected velocity shape {(self.batch_size, self.tangent_dim)}, but got {velocity.shape}")

        requires_grad = velocity.requires_grad or self.q.requires_grad
        if out is None:
            out = type(self)(
                robot=self.robot,
                q=wp.empty_like(self.q, requires_grad=requires_grad),
                T_world_base=wp.empty_like(self.T_world_base, requires_grad=requires_grad)
                if self.has_floating_base
                else None,
            )

        ne = self.num_elements
        na = self.spec.num_actuated_joints
        num_frames = self.shape[1] if self.is_trajectory else 1
        single_tangent_dim = (6 if self.has_floating_base else 0) + na
        velocity_offset = 0
        if self.has_floating_base:
            wp.launch(
                kernel=integrate_floating_base_kernel,
                dim=ne,
                inputs=[
                    self.T_world_base.flatten(),
                    velocity,
                    tangent_mask,
                    tangent_mask is not None,
                    num_frames,
                    single_tangent_dim,
                    wp.float32(self.base_step_scale),
                ],
                outputs=[out.T_world_base.flatten()],
                device=self.device,
            )
            velocity_offset = 6
        elif out.T_world_base.ptr != self.T_world_base.ptr:
            wp.copy(out.T_world_base, self.T_world_base)

        # the projection onto the limits is not part of the gradient: differentiable inputs keep it off any tape
        clamp = self.clamp_q_on_integrate and not requires_grad
        wp.launch(
            kernel=integrate_joint_positions_kernel,
            dim=(ne, na),
            inputs=[
                self.q.reshape((ne, na)),
                velocity,
                velocity_offset,
                num_frames,
                single_tangent_dim,
                tangent_mask,
                tangent_mask is not None,
                weight_decay,
                self.clamp_limits if clamp else None,
                clamp,
                out_velocity is not None,
            ],
            outputs=[out.q.reshape((ne, na)), out_velocity],
            device=self.device,
        )

        out.base_step_scale, out.clamp_q_on_integrate = self.base_step_scale, self.clamp_q_on_integrate
        out.clamp_limits = self.clamp_limits
        out.invalidate()
        if self.clamp_q_on_integrate and not clamp:
            out.clamp_q(record_tape=False)  # as benchmarked: the projection ignores the tangent mask
        return out

    def clamp_q(self, tangent_mask: Optional[wp.array] = None, record_tape: bool = True) -> "RobotState":
        """Clamp joints while preserving masked coordinates; `record_tape=False` excludes projection from tape."""
        ne, na = self.num_elements, self.spec.num_actuated_joints
        wp.launch(
            clamp_joint_positions_kernel,
            dim=(ne, na),
            inputs=[
                self.q.reshape((ne, na)),
                self.clamp_limits,
                self.shape[1] if self.is_trajectory else 1,
                self.spec.num_actuated_joints + (6 if self.has_floating_base else 0),
                6 if self.has_floating_base else 0,
                tangent_mask,
                tangent_mask is not None,
            ],
            device=self.device,
            record_tape=record_tape,
        )
        self.invalidate()
        return self

    def clone(self) -> "RobotState":
        state = object.__new__(type(self))
        state.spec = self.spec
        state.q = wp.clone(self.q)
        state.T_world_base = wp.clone(self.T_world_base)
        for name in _CLONE_BUFFERS:
            if name in self.__dict__:
                setattr(state, name, wp.clone(self.__dict__[name]))
        state.has_floating_base = self.has_floating_base
        state.is_fk_computed = self.is_fk_computed
        state.is_motion_subspace_computed = self.is_motion_subspace_computed
        state.is_collision_spheres_computed = self.is_collision_spheres_computed
        state.is_capsules_computed = self.is_capsules_computed
        state.is_com_computed = self.is_com_computed
        state.base_step_scale = self.base_step_scale
        state.clamp_q_on_integrate = self.clamp_q_on_integrate
        state.clamp_limits = self.clamp_limits
        state.robot = self.robot
        state.spec_tensors = self.spec_tensors
        return state

    def view(self, batch_size: int, offset: int = 0) -> "RobotState":
        """Elements `offset` to `offset + batch_size` of every buffer, sharing storage."""
        state = object.__new__(type(self))
        state.spec = self.spec
        state.q = self.q[offset : offset + batch_size]
        state.T_world_base = self.T_world_base[offset : offset + batch_size]
        state._view_source = self
        for name in _CLONE_BUFFERS:
            if name in self.__dict__:
                buf = self.__dict__[name]  # com_world is already flat; everything else leads with the batch
                k = buf.shape[0] // self.batch_size
                setattr(state, name, buf[offset * k : (offset + batch_size) * k])
        state.has_floating_base = self.has_floating_base
        state.is_fk_computed = self.is_fk_computed
        state.is_motion_subspace_computed = self.is_motion_subspace_computed
        state.is_collision_spheres_computed = self.is_collision_spheres_computed
        state.is_capsules_computed = self.is_capsules_computed
        state.is_com_computed = self.is_com_computed
        state.base_step_scale = self.base_step_scale
        state.clamp_q_on_integrate = self.clamp_q_on_integrate
        state.clamp_limits = self.clamp_limits
        state.robot = self.robot
        state.spec_tensors = self.spec_tensors
        return state

    def flatten(self) -> "RobotState":
        """Zero-copy view folding the frame axis into the batch axis so per-pose kernels run unchanged."""
        if not self.is_trajectory:
            return self
        ne = self.num_elements
        state = object.__new__(type(self))
        state.spec = self.spec
        state.q = self.q.reshape((ne, self.q.shape[-1]))
        state.T_world_base = self.T_world_base.flatten()
        state._view_source = self
        for name in _CLONE_BUFFERS + ("X_local",):
            buf = self.__dict__.get(name)
            if buf is not None:
                # com_world is already flat; everything else leads with (batch, frames)
                shape = (ne,) + tuple(buf.shape[2:])
                setattr(state, name, buf.reshape(shape) if tuple(buf.shape[:2]) == self.shape else buf)
        state.has_floating_base = self.has_floating_base
        state.is_fk_computed = self.is_fk_computed
        state.is_motion_subspace_computed = self.is_motion_subspace_computed
        state.is_collision_spheres_computed = self.is_collision_spheres_computed
        state.is_capsules_computed = self.is_capsules_computed
        state.is_com_computed = self.is_com_computed
        state.base_step_scale = self.base_step_scale
        state.clamp_q_on_integrate = self.clamp_q_on_integrate
        state.clamp_limits = self.clamp_limits
        state.robot = self.robot
        state.spec_tensors = self.spec_tensors
        return state

    def gather(self, indices: ArrayLike, out: Optional["RobotState"] = None, q_only: bool = True) -> "RobotState":
        num_gather = indices.shape[0]
        device = self.q.device
        requires_grad = self.q.requires_grad

        if out is None:
            new_shape = (num_gather,) + self.q.shape[1:]
            q_placeholder = wp.empty(new_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self.has_floating_base:
                T_world_base_placeholder = wp.empty(
                    new_shape[:-1], dtype=wp_vec7, device=device, requires_grad=requires_grad
                )
            else:
                T_world_base_placeholder = None
            out = type(self)(robot=self.robot, q=q_placeholder, T_world_base=T_world_base_placeholder)

        warp_gather(self.q, indices, out.q)
        if self.has_floating_base:
            warp_gather(self.T_world_base, indices, out.T_world_base)
        if not q_only:
            if not self.has_floating_base:
                warp_gather(self.T_world_base, indices, out.T_world_base)
            for name in _GATHER_BUFFERS:
                if name in ("T_world_joint", "T_world_link") or name in self.__dict__:
                    warp_gather(getattr(self, name), indices, getattr(out, name))

        out.has_floating_base = self.has_floating_base
        out.is_fk_computed = False if q_only else self.is_fk_computed
        out.is_motion_subspace_computed = False if q_only else self.is_motion_subspace_computed
        out.is_collision_spheres_computed = False if q_only else self.is_collision_spheres_computed
        out.is_capsules_computed = False if q_only else self.is_capsules_computed
        out.is_com_computed = False  # com_world is never gathered
        out.base_step_scale, out.clamp_q_on_integrate = self.base_step_scale, self.clamp_q_on_integrate
        out.clamp_limits = self.clamp_limits
        out.robot = self.robot
        out.spec_tensors = self.spec_tensors
        return out

    def accept(self, accept_mask: ArrayLike, proposed: "RobotState") -> "RobotState":
        """Optimized accept: copies q, and the forward kinematics this state already had (1 kernel)."""
        ne = self.num_elements
        na, nj, nl = self.spec.num_actuated_joints, self.spec.num_joints, self.spec.num_links
        epb = ne // self.batch_size
        floating = self.has_floating_base
        fk = self.is_fk_computed  # a stale state has none to keep fresh, so the copy is q (and a floating base)
        wp.launch(
            accept_state_forward_kinematics_kernel,
            dim=(ne, max(na, nj, nl) if fk else max(na, 1)),
            inputs=[
                accept_mask,
                proposed.q.reshape((ne, na)),
                self.q.reshape((ne, na)),
                proposed.T_world_base.flatten() if floating else None,
                self.T_world_base.flatten() if floating else None,
                proposed.T_world_joint.reshape((ne, nj)) if fk else None,
                self.T_world_joint.reshape((ne, nj)) if fk else None,
                proposed.T_world_link.reshape((ne, nl)) if fk else None,
                self.T_world_link.reshape((ne, nl)) if fk else None,
                epb,
            ],
            device=self.device,
        )
        self.is_motion_subspace_computed = False
        self.is_collision_spheres_computed = False
        self.is_capsules_computed = False
        self.is_com_computed = False
        return self


RobotCollisionState = RobotState

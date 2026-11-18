from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import warp as wp
import yourdfpy


if TYPE_CHECKING:
    import torch
    from jaxtyping import Float

from robokit.lie.se3 import se3_from_matrix, se3_to_matrix  # noqa: F401 (se3_to_matrix used in doctests)
from robokit.robo.robot_kernels import (
    compute_com_kernel,
    compute_forward_kinematics_accum_kernel,
    compute_forward_kinematics_local_kernel,
    compute_forward_kinematics_sequential_kernel,
    compute_forward_kinematics_warp_kernel,
    compute_motion_subspace_kernel,
    transform_collision_points_kernel,
    transform_link_points_kernel,
)
from robokit.robo.robot_spec import RobotSpec
from robokit.robo.robot_state import RobotState
from robokit.utils.warp_utils import wp_device_type, wp_vec7  # noqa: F401 (used in doctests)


class Robot:
    """
    Robot model backed by Warp.

    Examples:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> q = wp.from_numpy(np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0]), dtype=wp.float32)
        >>> state = robot.state(q=q)
        >>> state = robot.forward_kinematics(state)
        >>> hand_idx = robot.spec.link_names.index("panda_hand")
        >>> link_pose = state.T_world_link[:, hand_idx]
        >>> expected = np.array([6.0860e-02, -4.7547e-12, 7.6373e-01, 0.0382, 0.9193, 0.3808, 0.0922], dtype=np.float32)
        >>> expected = wp.from_numpy(expected, dtype=wp_vec7)
        >>> np.allclose(se3_to_matrix(link_pose).numpy(), se3_to_matrix(expected).numpy(), atol=1e-4)
        True
    """

    spec: RobotSpec

    def __init__(self, spec: RobotSpec):
        self.spec = spec

    def __repr__(self) -> str:
        return (
            f"Robot(name={self.spec.name}, num_actuated_joints={self.spec.num_actuated_joints}, num_links={self.spec.num_links})\n"
            + self.spec.kinematic_tree_str
        )

    @staticmethod
    def load(
        robot_description_or_path: Union[str, Path, yourdfpy.URDF],
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
        load_collision_spheres: bool = False,
        collision_spheres_path: Optional[Union[str, Path]] = None,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[Union[Sequence[str], str]] = None,
        self_collision_ignore_path: Optional[Union[str, Path]] = None,
    ) -> "Robot":
        """
        Load a robot model from a URDF/MJCF path, description string, or parsed `yourdfpy.URDF`.

        Args:
            robot_description_or_path: URDF/MJCF file path, description string, or already-parsed `yourdfpy.URDF`.
            load_meshes: Load link visual/collision meshes.
            mesh_dir: Directory to resolve mesh files against.
            load_collision_spheres: Load collision spheres from `collision_spheres_path`.
            collision_spheres_path: YAML file with per-link collision spheres and optional `self_collision_link_buffers`.
            base_link_name: Base link; defaults to the URDF root.
            ee_link_names: End-effector link name(s).
            self_collision_ignore_path: YAML file with link pairs self-collision terms skip.

        Returns:
            The loaded robot model.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> robot.spec.num_actuated_joints
            8
        """
        return Robot(
            spec=RobotSpec.parse(
                robot_description_or_path=robot_description_or_path,
                load_meshes=load_meshes,
                mesh_dir=mesh_dir,
                load_collision_spheres=load_collision_spheres,
                collision_spheres_path=collision_spheres_path,
                base_link_name=base_link_name,
                ee_link_names=ee_link_names,
                self_collision_ignore_path=self_collision_ignore_path,
            ),
        )

    # --- state and kinematics ---
    def state(
        self,
        q: Optional[Union[wp.array, np.ndarray]] = None,
        T_world_base: Optional[wp.array] = None,
    ) -> RobotState:
        """
        Build a robot state for the given joint positions (defaults to `zero_q`).

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> robot.state().q.shape
            (1, 8)
        """
        if q is not None and isinstance(q, np.ndarray):
            if T_world_base is not None:
                device = T_world_base.device
            else:
                device = wp.get_device("cpu")
            q = wp.from_numpy(q.astype(np.float32), dtype=wp.float32, device=device)
        elif q is None:
            q = self.spec.get_tensors().zero_q
        elif not isinstance(q, wp.array):
            raise TypeError(f"Expected q to be a wp.array or np.ndarray, but got {type(q)}")
        return RobotState(robot=self, q=q, T_world_base=T_world_base)

    def forward_kinematics(self, state: RobotState, use_sequential: Optional[bool] = None) -> RobotState:
        """
        Compute forward kinematics for given joint positions; returns a state with all link transforms.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> q = wp.from_numpy(np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0]), dtype=wp.float32)
            >>> state = robot.state(q=q)
            >>> state = robot.forward_kinematics(state)
            >>> hand_idx = robot.spec.link_names.index("panda_hand")
            >>> link_pose = state.T_world_link[:, hand_idx]
            >>> expected = np.array([6.0860e-02, -4.7547e-12, 7.6373e-01, 0.0382, 0.9193, 0.3808, 0.0922], dtype=np.float32)
            >>> expected = wp.from_numpy(expected, dtype=wp_vec7)
            >>> np.allclose(se3_to_matrix(link_pose).numpy(), se3_to_matrix(expected).numpy(), atol=1e-4)
            True
            >>> T_world_base = wp.from_numpy(np.array([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0]), dtype=wp_vec7)
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.forward_kinematics(state)
            >>> link_pose = state.T_world_link[:, hand_idx]
            >>> expected = np.array([1.76371783, 2.0, 2.93915576, -0.24222432, 0.71524162, 0.29626278, -0.58478125], dtype=np.float32)
            >>> expected = wp.from_numpy(expected, dtype=wp_vec7)
            >>> np.allclose(se3_to_matrix(link_pose).numpy(), se3_to_matrix(expected).numpy(), atol=1e-4)
            True
        """
        ne, st = state.num_elements, state.spec_tensors
        q_2d = state.q.reshape((ne, self.spec.num_actuated_joints))
        base_flat = state.T_world_base.flatten()
        outputs = [
            state.T_world_joint.reshape((ne, self.spec.num_joints)),
            state.T_world_link.reshape((ne, self.spec.num_links)),
        ]
        sequential = state.q.requires_grad if use_sequential is None else use_sequential
        # the motion subspace comes along once its buffer exists: allocating is illegal in graph capture
        with_subspace = not sequential and "S_world" in state.__dict__
        # the collision spheres likewise, so the collision terms find them without a launch of their own
        with_points = not sequential and self.spec.has_collision_spheres
        with_points = with_points and "collision_sphere_centers_world" in state.__dict__
        if sequential:
            wp.launch(
                kernel=compute_forward_kinematics_sequential_kernel,
                dim=ne,
                inputs=[
                    q_2d,
                    base_flat,
                    st.actuated_joint_indices,
                    st.mimic_actuated_joint_indices,
                    st.mimic_multipliers,
                    st.mimic_offsets,
                    st.joint_twists,
                    st.parent_joint_transforms,
                    st.topological_order_joint_indices,
                    st.parent_joint_indices,
                    st.link_parent_joint_indices,
                ],
                outputs=outputs,
                device=state.device,
            )
        elif state.device.is_cuda and self.spec.num_joints <= 32:
            num_spheres = len(self.spec.local_collision_sphere_centers) if with_points else 0
            wp.launch(
                compute_forward_kinematics_warp_kernel,
                dim=ne * 32,
                inputs=[
                    q_2d,
                    base_flat,
                    st.actuated_joint_indices,
                    st.mimic_actuated_joint_indices,
                    st.mimic_multipliers,
                    st.mimic_offsets,
                    st.joint_types,
                    st.joint_axes,
                    st.parent_joint_transforms,
                    st.parent_joint_indices,
                    st.link_parent_joint_indices,
                    st.joint_twists,
                    int(with_subspace),
                    st.local_collision_sphere_centers if with_points else None,
                    st.collision_spheres_link_indices if with_points else None,
                    st.local_link_bounding_sphere_centers if with_points else None,
                ],
                outputs=[
                    *outputs,
                    state.S_world.reshape((ne, 6, self.spec.num_joints)) if with_subspace else None,
                    state.collision_sphere_centers_world.reshape((ne, num_spheres)) if with_points else None,
                    state.link_bounding_sphere_centers_world.reshape((ne, self.spec.num_links))
                    if with_points
                    else None,
                ],
                block_dim=128,
                device=state.device,
            )
        else:
            num_spheres = len(self.spec.local_collision_sphere_centers) if with_points else 0
            X_local_2d = state.X_local.reshape((ne, self.spec.num_joints))
            wp.launch(
                compute_forward_kinematics_local_kernel,
                dim=(ne, self.spec.num_joints),
                inputs=[
                    q_2d,
                    st.actuated_joint_indices,
                    st.mimic_actuated_joint_indices,
                    st.mimic_multipliers,
                    st.mimic_offsets,
                    st.joint_types,
                    st.joint_axes,
                    st.parent_joint_transforms,
                ],
                outputs=[X_local_2d],
                device=state.device,
            )
            wp.launch(
                compute_forward_kinematics_accum_kernel,
                dim=(ne, self.spec.num_links + (num_spheres + self.spec.num_links if with_points else 0)),
                inputs=[
                    X_local_2d,
                    base_flat,
                    st.parent_joint_indices,
                    st.link_parent_joint_indices,
                    st.joint_twists,
                    int(with_subspace),
                    st.local_collision_sphere_centers if with_points else None,
                    st.collision_spheres_link_indices if with_points else None,
                    st.local_link_bounding_sphere_centers if with_points else None,
                ],
                outputs=[
                    *outputs,
                    state.S_world.reshape((ne, 6, self.spec.num_joints)) if with_subspace else None,
                    state.collision_sphere_centers_world.reshape((ne, num_spheres)) if with_points else None,
                    state.link_bounding_sphere_centers_world.reshape((ne, self.spec.num_links))
                    if with_points
                    else None,
                ],
                device=state.device,
            )
        state.is_fk_computed = True
        state.is_motion_subspace_computed = with_subspace
        state.is_collision_spheres_computed = with_points
        state.is_capsules_computed = False
        state.is_com_computed = False
        return state

    def forward_kinematics_via_matrix_numpy(
        self,
        q: "Float[np.ndarray, '... num_dofs']",
        T_world_base: "Optional[Float[np.ndarray, '... 4 4']]" = None,
        device: Optional[str] = None,
    ) -> "Float[np.ndarray, '... num_links 4 4']":
        """
        Compute forward kinematics as `4x4` matrices for NumPy arrays, without an autograd round trip.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> q = np.zeros((1, robot.spec.num_actuated_joints), dtype=np.float32)
            >>> robot.forward_kinematics_via_matrix_numpy(q).shape
            (1, 13, 4, 4)
        """
        wp_device = wp.get_device(device)
        base = None
        if T_world_base is not None:
            base = se3_from_matrix(
                wp.from_numpy(np.ascontiguousarray(T_world_base, dtype=np.float32), dtype=wp.mat44, device=wp_device)
            )
        state = self.state(
            q=wp.from_numpy(np.ascontiguousarray(q, dtype=np.float32), dtype=wp.float32, device=wp_device),
            T_world_base=base,
        )
        return se3_to_matrix(self.forward_kinematics(state).T_world_link).numpy()

    def _state_from_numpy(
        self,
        q: "Float[np.ndarray, '... num_dofs']",
        T_world_base: "Optional[Float[np.ndarray, '... 7']]" = None,
        device: Optional[str] = None,
    ) -> RobotState:
        """Upload NumPy joint positions, with an optional 7D base pose, into a robot state."""
        wp_device = wp.get_device(device)
        base = None
        if T_world_base is not None:
            base = wp.array(
                np.ascontiguousarray(T_world_base, dtype=np.float32).reshape(-1, 7), dtype=wp_vec7, device=wp_device
            )
        return self.state(
            q=wp.from_numpy(np.ascontiguousarray(q, dtype=np.float32), dtype=wp.float32, device=wp_device),
            T_world_base=base,
        )

    def forward_kinematics_numpy(
        self,
        q: "Float[np.ndarray, '... num_dofs']",
        T_world_base: "Optional[Float[np.ndarray, '... 7']]" = None,
        device: Optional[str] = None,
    ) -> "Float[np.ndarray, '... num_links 7']":
        """
        Compute world link poses as translation-first 7D poses for NumPy arrays.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> q = np.zeros((1, robot.spec.num_actuated_joints), dtype=np.float32)
            >>> robot.forward_kinematics_numpy(q).shape
            (1, 13, 7)
        """
        return self.forward_kinematics(self._state_from_numpy(q, T_world_base, device)).T_world_link.numpy()

    def compute_center_of_mass_numpy(
        self,
        q: "Float[np.ndarray, '... num_dofs']",
        T_world_base: "Optional[Float[np.ndarray, '... 7']]" = None,
        device: Optional[str] = None,
    ) -> "Float[np.ndarray, '... 3']":
        """
        Compute the whole-body center of mass in world frame for NumPy arrays.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> q = np.zeros((1, robot.spec.num_actuated_joints), dtype=np.float32)
            >>> np.allclose(robot.compute_center_of_mass_numpy(q)[0], [0.0232, 0.0061, 0.6062], atol=1e-4)
            True
        """
        return self.compute_center_of_mass(self._state_from_numpy(q, T_world_base, device)).com_world.numpy()

    def compute_motion_subspace(self, state: RobotState) -> RobotState:
        """Store robot joint Jacobians in state for `state.get_link_jacobian()`.

        Base layouts:
        - Fixed base: joint columns only.
        - Floating base: prepend J_base, giving shape [batch, 6, 6 + num_dofs].

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> T_world_base = wp.from_numpy(np.array([1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0]), dtype=wp_vec7)
            >>> q = wp.from_numpy(np.array([0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0]), dtype=wp.float32)
            >>> state = robot.state(q=q, T_world_base=T_world_base)
            >>> state = robot.compute_motion_subspace(state)
            >>> J = state.get_link_jacobian(robot.spec.link_names.index("panda_hand"), "body")
            >>> np.allclose(J.numpy()[0, 0, :6], np.array([0.693, 0.7071, 0.1405, -0.54, 0.5207, 0.043], dtype=np.float32), atol=1e-4)
            True
            >>> J = state.get_link_jacobian(robot.spec.link_names.index("panda_hand"), "spatial")
            >>> np.allclose(J.numpy()[0, 0, 6:], np.array([0.0001, -3., 1.8641, 3.2647, -1.4346, 3.0467, -0.3974, 0.], dtype=np.float32), atol=1e-3)
            True
            >>> np.allclose(J.numpy()[0, 0, :6], np.array([0.0001, 0., 1., -2., -3., 0.0001], dtype=np.float32), atol=1e-3)
            True
        """
        if not state.q.requires_grad:  # the parallel FK writes the subspace with the links once its buffer exists
            state.S_world
            return self.forward_kinematics(state)
        if not state.is_fk_computed:
            state = self.forward_kinematics(state)
        ne = state.num_elements
        wp.launch(
            kernel=compute_motion_subspace_kernel,
            dim=(ne, self.spec.num_joints),
            inputs=[
                state.T_world_joint.reshape((ne, self.spec.num_joints)),
                state.spec_tensors.joint_twists,
            ],
            outputs=[state.S_world.reshape((ne, 6, self.spec.num_joints))],
            device=state.device,
        )
        state.is_motion_subspace_computed = True
        return state

    def compute_center_of_mass(self, state: RobotState) -> RobotState:
        """
        Compute the whole-body center of mass in world frame into `state.com_world`.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> state = robot.compute_center_of_mass(robot.state())
            >>> np.allclose(state.com_world.numpy()[0], [0.0232, 0.0061, 0.6062], atol=1e-4)
            True
        """
        if not state.is_fk_computed:
            self.forward_kinematics(state)
        spec_t = self.spec.get_tensors(str(state.q.device))
        total_mass_inv = 1.0 / self.spec.total_mass if self.spec.total_mass > 0 else 0.0
        wp.launch(
            kernel=compute_com_kernel,
            dim=state.num_elements,
            inputs=[
                state.T_world_link.reshape((state.num_elements, self.spec.num_links)),
                spec_t.link_masses,
                spec_t.link_local_com_positions,
                wp.float32(total_mass_inv),  # pyright: ignore[reportArgumentType]
            ],
            outputs=[state.com_world],
            device=state.q.device,
        )
        state.is_com_computed = True
        return state

    # --- collision transforms ---
    def transform_collision_spheres(self, state: RobotState) -> RobotState:
        """
        Compute the collision sphere centers in world frame.

        Args:
            state: The robot state.

        Returns:
            The robot state with updated collision sphere centers.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> state = robot.state()
            >>> robot.transform_collision_spheres(state) is state  # no collision spheres loaded
            True
        """
        if not self.spec.has_collision_spheres:
            return state
        if not state.q.requires_grad:  # the parallel FK writes the spheres with the links once their buffers exist
            state.collision_sphere_centers_world
            state.link_bounding_sphere_centers_world
            return self.forward_kinematics(state)

        if not state.is_fk_computed:
            self.forward_kinematics(state)

        spec_t = self.spec.get_tensors(str(state.q.device))
        num_spheres = len(self.spec.local_collision_sphere_centers)

        num_links = self.spec.num_links
        wp.launch(
            kernel=transform_collision_points_kernel,
            dim=(state.num_elements, num_spheres + num_links),
            inputs=[
                state.T_world_link.reshape((state.num_elements, num_links)),
                spec_t.local_collision_sphere_centers,
                spec_t.collision_spheres_link_indices,
                spec_t.local_link_bounding_sphere_centers,
            ],
            outputs=[
                state.collision_sphere_centers_world.reshape((state.num_elements, num_spheres)),
                state.link_bounding_sphere_centers_world.reshape((state.num_elements, num_links)),
            ],
            device=state.q.device,
        )

        state.is_collision_spheres_computed = True
        return state

    def transform_collision_capsules(self, state: RobotState) -> RobotState:
        """
        Compute the per-link capsule segment endpoints (a, b) in world frame.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> state = robot.state()
            >>> robot.transform_collision_capsules(state) is state  # no link capsules loaded
            True
        """
        if not self.spec.has_link_capsules:
            return state

        if not state.is_fk_computed:
            self.forward_kinematics(state)

        spec_t = self.spec.get_tensors(str(state.q.device))
        num_links = self.spec.num_links
        T_world_link = state.T_world_link.reshape((state.num_elements, num_links))

        wp.launch(
            kernel=transform_link_points_kernel,
            dim=(state.num_elements, num_links),
            inputs=[
                T_world_link,
                spec_t.local_link_capsule_endpoint_a,
                spec_t.link_identity_indices,
                state.link_capsule_endpoint_a_world.reshape((state.num_elements, num_links)),
            ],
            device=state.q.device,
        )
        wp.launch(
            kernel=transform_link_points_kernel,
            dim=(state.num_elements, num_links),
            inputs=[
                T_world_link,
                spec_t.local_link_capsule_endpoint_b,
                spec_t.link_identity_indices,
                state.link_capsule_endpoint_b_world.reshape((state.num_elements, num_links)),
            ],
            device=state.q.device,
        )

        state.is_capsules_computed = True
        return state

    # --- torch interface ---
    def forward_kinematics_via_matrix_torch(
        self,
        q: "Float[torch.Tensor, '... num_dofs']",
        T_world_base: "Optional[Float[torch.Tensor, '... 4 4']]" = None,
        analytical_backward: bool = True,
        record_cmd: bool = False,
    ) -> "Float[torch.Tensor, '... num_links 4 4']":
        """
        Compute differentiable forward kinematics as `4x4` matrices for torch tensors.

        Example:
            >>> import torch
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> T_world_link = robot.forward_kinematics_via_matrix_torch(torch.zeros(robot.spec.num_actuated_joints))
            >>> T_world_link.shape
            torch.Size([13, 4, 4])
        """
        import torch

        if T_world_base is None:
            base = torch.eye(4, device=q.device, dtype=torch.float32)
            base = torch.broadcast_to(base, q.shape[:-1] + (4, 4))
        else:
            base = T_world_base.to(device=q.device, dtype=torch.float32)
            base = torch.broadcast_to(base, q.shape[:-1] + (4, 4))

        if analytical_backward:
            from robokit.robo.robot_torch_wrappers import WarpForwardKinematicsMatrix

            return WarpForwardKinematicsMatrix.apply(self.spec, q, base, record_cmd)  # type: ignore
        else:
            from robokit.robo.robot_torch_wrappers import WarpForwardKinematicsMatrixAutodiff

            return WarpForwardKinematicsMatrixAutodiff.apply(self.spec, q, base)  # type: ignore

    def transform_link_points_torch(
        self,
        T_world_link: "Float[torch.Tensor, '... num_links 4 4']",
        local_points: "Float[torch.Tensor, 'num_points 3']",
        point_link_indices: "Float[torch.Tensor, 'num_points']",
    ) -> "Float[torch.Tensor, '... num_points 3']":
        """Transform local points attached to links to world frame.

        Args:
            T_world_link: Link transforms in world frame [..., num_links, 4, 4]
            local_points: Local points in link frames [num_points, 3]
            point_link_indices: Link index for each point [num_points]

        Returns:
            World-frame points [..., num_points, 3]

        Example:
            >>> import torch
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> T_world_link = robot.forward_kinematics_via_matrix_torch(torch.zeros(robot.spec.num_actuated_joints))
            >>> points = robot.transform_link_points_torch(T_world_link, torch.zeros(2, 3), torch.tensor([0, 1]))
            >>> points.shape
            torch.Size([2, 3])
        """
        from robokit.robo.robot_torch_wrappers import WarpTransformLinkPoints

        return WarpTransformLinkPoints.apply(T_world_link, local_points, point_link_indices)  # type: ignore

    def map_to_full_joint_values_torch(
        self, actuated_values: "Float[torch.Tensor, '... num_actuated_joints']"
    ) -> "Float[torch.Tensor, '... num_joints']":
        """
        Expand actuated joint values to all joints (mimic joints follow their controlling joint).

        Example:
            >>> import torch
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> robot.map_to_full_joint_values_torch(torch.zeros(robot.spec.num_actuated_joints)).shape
            torch.Size([12])
        """
        import torch

        spec_t = self.spec.get_tensors_torch(device=actuated_values.device)
        is_mimic = spec_t.mimic_actuated_joint_indices != -1
        controlling_indices = torch.where(is_mimic, spec_t.mimic_actuated_joint_indices, spec_t.actuated_joint_indices)
        padded_actuated_values = torch.cat([actuated_values, torch.zeros_like(actuated_values[..., :1])], dim=-1)
        safe_controlling_indices = torch.where(
            controlling_indices == -1, self.spec.num_actuated_joints, controlling_indices
        )
        controlling_values = padded_actuated_values[..., safe_controlling_indices]
        return controlling_values * spec_t.mimic_multipliers + spec_t.mimic_offsets

    @property
    def zero_q(self) -> wp.array:
        return self.spec.get_tensors().zero_q

    @property
    def midrange_q(self) -> wp.array:
        return self.spec.get_tensors().midrange_q

    @property
    def num_actuated_joints(self) -> int:
        return self.spec.num_actuated_joints

    @property
    def link_names(self) -> List[str]:
        return self.spec.link_names

    # --- joint sampling ---
    def sample_q(
        self,
        num_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
        device: wp_device_type = None,
    ) -> wp.array:
        """
        Sample joint positions uniformly within the actuated joint limits.

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> robot = Robot.load(load_robot_description("panda_description"))
            >>> robot.sample_q(num_samples=3, rng=np.random.default_rng(0)).shape
            (3, 8)
        """
        joint_limits = self.spec.actuated_joint_limits
        if rng is None:
            rng = np.random.default_rng()
        q_np = rng.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(num_samples, joint_limits.shape[0]))
        return wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32, device=device)

# pyright: reportMissingImports=false
"""Hand-eye camera extrinsic calibration: mask/depth alignment via coarse-to-fine multi-candidate LM."""

import copy
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import warp as wp
from jaxtyping import Float, Int

from robokit.helpers.hand_eye_calibration.config import HandEyeCalibratorConfig
from robokit.helpers.hand_eye_calibration.keypoint_alignment_task import KeypointAlignmentTask
from robokit.helpers.hand_eye_calibration.mask_alignment_task import (
    HandEyeContext,
    MaskAlignmentNormalEqTask,
    MaskAlignmentOffsetTask,
    _precompute_merged_mesh,
    _render_batch,
    dr,
)
from robokit.lie.se3 import SE3Var, se3_from_matrix, se3_to_matrix
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSampler
from robokit.opt.var_values import Var, VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7


class _MaskCandidateScore(ResidualTask):
    """Rank accepted poses by soft IoU; local LM keeps its original objective."""

    residual_weight = None

    def __init__(self, terms: List[ResidualTask], reuse_accepted: bool = True):
        self.terms = terms
        self.reuse_accepted = reuse_accepted

    @property
    def residual_dim(self) -> int:
        return 1

    @property
    def parallel_safe(self) -> bool:
        return False  # Reads the mask term's accepted render.

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        cost = torch.zeros(var_values.batch_size, device=str(var_values.device))
        for term in self.terms:
            if isinstance(term, MaskAlignmentNormalEqTask):
                if not self.reuse_accepted:
                    term._curr_is_fresh = False
                if term.batch_size != var_values.batch_size or not term._curr_is_fresh:
                    term.prepare(var_values, False, 1)
                target = cast(torch.Tensor, term.target_masks)[None]
                prediction = term._rendered_curr.reshape(var_values.batch_size, term.N, *target.shape[-2:]).clamp(0, 1)
                intersection = torch.minimum(prediction, target).sum((-2, -1))
                union = torch.maximum(prediction, target).sum((-2, -1))
                mismatch = torch.where(union > 0, 1 - intersection / union.clamp(min=1e-8), 0.0)
                area = target.sum((-2, -1))
                # Empty targets have IoU zero for every false-positive mask; rank those candidates by their actual false-positive pixels.
                mismatch = torch.where(
                    area > 0, mismatch, prediction.square().sum((-2, -1)) / (target.shape[-2] * target.shape[-1] * 0.05)
                )
                area = torch.where(area > 0, area, target.shape[-2] * target.shape[-1] * 0.05)
                cost += 0.5 * (mismatch * area * cast(torch.Tensor, term.observation_weights)).sum(-1)
            else:
                term.precompute(var_values, need_gradient=False)
                residual = wp.to_torch(term.compute_weighted_residual(var_values))
                cost += 0.5 * residual.square().sum(-1)
        residual = (2 * cost.clamp(min=0)).sqrt()[:, None]
        if out_residual is None:
            return wp.from_torch(residual)
        wp.to_torch(out_residual)[:, row_offset : row_offset + 1].copy_(residual)
        return out_residual


@contextmanager
def _torch_warp_stream(device: torch.device) -> Iterator[None]:
    """Keep Torch and Warp operations ordered, including external caller streams."""
    if device.type != "cuda":
        yield
        return
    incoming = torch.cuda.current_stream(device)
    stream = cast(torch.cuda.Stream, wp.stream_to_torch(str(device)))
    stream.wait_stream(incoming)
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        incoming.wait_stream(stream)


# --- rendering --------------------------------------------------------------
def build_robot_render_data(
    robot: Robot,
    q: Float[torch.Tensor, "n_obs num_dofs"],
    T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]] = None,
    link_names: Optional[List[str]] = None,
) -> Tuple[
    Float[torch.Tensor, "n_obs num_links 4 4"],
    List[Float[torch.Tensor, "num_vertices 3"]],
    List[Int[torch.Tensor, "num_faces 3"]],
]:
    """Build render inputs (FK link poses + GPU mesh buffers) for a robot.

    Args:
        robot: Robot loaded with `load_meshes=True`.
        q: Joint positions `[N, num_dofs]`.
        T_mount_base: Mount poses `[N,4,4]`; supplied poses produce `T_mount_link` rather than `T_base_link`.
        link_names: Links to render; `None` selects all links with visual meshes.

    Returns:
        `(T_ref_link [N,L,4,4], link_vertices, link_faces)` over the selected links.
    """
    mesh_dict = robot.spec.get_link_meshes(mode="visual")
    if link_names is None:
        names = [n for n in robot.spec.link_names if n in mesh_dict]
    else:
        missing = [n for n in link_names if n not in mesh_dict]
        if missing:
            raise ValueError(f"links without visual meshes: {missing}")
        names = list(link_names)
    if not names:
        raise ValueError("robot has no visual meshes; load it with load_meshes=True")
    device = q.device
    link_vertices: List[torch.Tensor] = []
    link_faces: List[torch.Tensor] = []
    for n in names:
        mesh = cast(trimesh.Trimesh, mesh_dict[n])
        # Merge repeated OBJ vertices; silhouette rendering needs positions only.
        vertices, inverse = np.unique(np.asarray(mesh.vertices, dtype=np.float32), axis=0, return_inverse=True)
        link_vertices.append(torch.from_numpy(vertices).to(device))
        link_faces.append(torch.from_numpy(inverse[mesh.faces].astype(np.int32)).to(device))
    idx = [robot.spec.link_names.index(n) for n in names]
    T_ref_link = robot.forward_kinematics_via_matrix_torch(q)[:, idx]
    if T_mount_base is not None:
        T_ref_link = T_mount_base.to(T_ref_link)[:, None] @ T_ref_link
    return T_ref_link, link_vertices, link_faces


def compute_robot_masks(
    robot: Robot,
    q: Float[torch.Tensor, "n_obs num_dofs"],
    camera_intrinsic: Float[torch.Tensor, "3 3"],
    extrinsic: Float[torch.Tensor, "4 4"],
    height: int,
    width: int,
    T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]] = None,
    link_names: Optional[List[str]] = None,
    mount_link: Optional[str] = None,
    T_mount_offset: Optional[Float[torch.Tensor, "4 4"]] = None,
) -> Float[torch.Tensor, "n_obs height width"]:
    """Render robot silhouette masks (visualization; same renderer as the cost term).

    Args:
        robot: Robot loaded with `load_meshes=True`.
        q: Joint positions `[N, num_dofs]`.
        camera_intrinsic: Camera intrinsic `[3, 3]`.
        extrinsic: `[4,4]` transform `T_camera_base`, or `T_camera_mount` with `T_mount_base`.
        height: Image height in pixels.
        width: Image width in pixels.
        T_mount_base: Per-sample mount pose `[N, 4, 4]` (mounted camera).
        link_names: Links to render; `None` selects all links with visual meshes.
        mount_link: Root link of a mounted subtree whose attachment carries an offset.
        T_mount_offset: `[4,4]` adapter transform right-composed onto `mount_link`'s nominal pose.

    Returns:
        Anti-aliased silhouette masks `[N, H, W]` in `[0, 1]`.
    """
    T_ref_link, link_vertices, link_faces = build_robot_render_data(robot, q, T_mount_base, link_names)
    if T_mount_offset is not None:
        if mount_link is None:
            raise ValueError("T_mount_offset needs mount_link")
        names = list(
            link_names
            if link_names is not None
            else [n for n in robot.spec.link_names if n in robot.spec.get_link_meshes(mode="visual")]
        )
        # the mounted subtree: links below mount_link, plus mount_link itself
        mount_idx = robot.spec.link_names.index(mount_link)
        subtree = robot.spec.link_ancestor_links_mask[:, mount_idx].copy()
        subtree[mount_idx] = True
        indices = [robot.spec.link_names.index(n) for n in names]
        moved = torch.from_numpy(subtree[indices]).to(T_ref_link.device)
        T_ref_mount = T_ref_link[:, names.index(mount_link)]
        shift = T_ref_mount @ T_mount_offset.to(T_ref_link) @ torch.linalg.inv(T_ref_mount)
        T_ref_link = torch.where(moved[None, :, None, None], shift[:, None] @ T_ref_link, T_ref_link)
    merged_faces, v_base = _precompute_merged_mesh(link_vertices, link_faces, T_ref_link)
    with torch.no_grad():
        masks, _ = _render_batch(
            dr.RasterizeCudaContext(), extrinsic, camera_intrinsic, merged_faces, v_base, height, width, False
        )
    return masks


class HandEyeCalibrator:
    """Solve the camera extrinsic from robot silhouette masks and/or depth maps."""

    def __init__(
        self,
        config: HandEyeCalibratorConfig,
        robot: Robot,
        camera_intrinsic: Float[torch.Tensor, "*obs 3 3"],
        height: int,
        width: int,
        render_link_names: Optional[List[str]] = None,
        target_link_names: Optional[Union[List[str], List[List[str]]]] = None,
    ):
        wp.init()
        self.camera_intrinsic = camera_intrinsic
        self.height = height
        self.width = width
        self.config = config
        self.robot = robot
        self.render_link_names = render_link_names
        self.target_link_names = target_link_names
        self._wp_device = str(camera_intrinsic.device)
        self._initialized = False
        self.last_q: Optional[torch.Tensor] = None
        self._render_meshes: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None

    def _build(
        self,
        target_masks: Optional[Float[torch.Tensor, "n_obs height width"]],
        target_depths: Optional[Float[torch.Tensor, "n_obs height width"]],
        observation_weights: Optional[Float[torch.Tensor, "n_obs"]],
        q: Float[torch.Tensor, "n_obs num_dofs"],
        T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]],
        keypoint_links: Optional[List[str]] = None,
        target_keypoints: Optional[torch.Tensor] = None,
        keypoint_weights: Optional[torch.Tensor] = None,
    ):
        """Build image-pyramid tasks and the native multi-candidate solver."""
        # --- rendered geometry ---
        mesh_dict = self.robot.spec.get_link_meshes(mode="visual", return_mesh=False)
        names = (
            list(self.render_link_names)
            if self.render_link_names is not None
            else [n for n in self.robot.spec.link_names if n in mesh_dict]
        )
        if names != getattr(self, "_render_names", None):
            self._render_meshes = None
        self._render_names = names
        target_link_mask = None
        if self.target_link_names is not None:
            if not self.target_link_names:
                raise ValueError("target_link_names cannot be empty")
            groups = (
                [cast(List[str], self.target_link_names)]
                if isinstance(self.target_link_names[0], str)
                else cast(List[List[str]], self.target_link_names)
            )
            if len(groups) > 1 and len(groups) != len(q):
                raise ValueError(f"got {len(groups)} target link groups for {len(q)} observations")
            missing = set().union(*map(set, groups)) - set(names)
            if missing:
                raise ValueError(f"target links are not rendered: {sorted(missing)}")
            target_link_mask = torch.tensor(
                [[False, *(name in set(group) for name in names)] for group in groups], device=q.device
            )
            if len(groups) == 1:
                target_link_mask = target_link_mask[0]
        link_indices = np.array([self.robot.spec.link_names.index(n) for n in names], dtype=np.int64)
        if self._render_meshes is None:
            T_ref_link, link_vertices, link_faces = build_robot_render_data(self.robot, q, T_mount_base, names)
            self._render_meshes = link_vertices, link_faces
        else:
            link_vertices, link_faces = self._render_meshes
            T_ref_link = self.robot.forward_kinematics_via_matrix_torch(q)[:, link_indices]
            if T_mount_base is not None:
                T_ref_link = T_mount_base.to(T_ref_link)[:, None] @ T_ref_link
        # --- optimization variables ---
        levels = list(self.config.levels)
        if self.config.candidate_metric == "iou" and (target_masks is None or any(l.use_depth for l in levels)):
            raise ValueError("IoU candidate ranking requires RGB masks without depth levels")
        first_num_candidates = self.config.solver.num_candidates[0]
        init_v7 = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (first_num_candidates, 1))
        named: Dict[Enum, Var] = {VarKey.ROBOT: SE3Var(wp.from_numpy(init_v7, dtype=wp_vec7, device=self._wp_device))}
        templates = list(self.config.terms)
        if self.config.optimize_q:
            if any(isinstance(t, MaskAlignmentOffsetTask) for t in templates):
                raise ValueError("optimize_q cannot be combined with shared model-offset tasks")
            for i, template in enumerate(templates):
                if type(template) is MaskAlignmentNormalEqTask:
                    task = MaskAlignmentOffsetTask(
                        offset_joints=range(self.robot.num_actuated_joints), robot=self.robot, optimize_q=True
                    )
                    # Preserve the configured mask/depth fields, changing only its q parameterization.
                    task.__dict__.update(template.__dict__)
                    templates[i] = task
            if not any(isinstance(t, MaskAlignmentOffsetTask) for t in templates):
                raise ValueError("optimize_q needs a MaskAlignmentNormalEqTask")
        for template in templates:  # model offsets are variables their mask term declares
            named.update(template.initial_vars(first_num_candidates, self._wp_device))
        self._initial_var = VarValues(named)
        self._nominal_var = self._initial_var.clone()  # each restart starts every variable from here
        stage_terms: List[List[ResidualTask]] = []
        self._keypoint_solver = None
        if target_keypoints is not None and (not self.config.optimize_q or not keypoint_links):
            raise ValueError("keypoints require optimize_q and keypoint_links")

        # --- rank observations ---
        # rank by robot-mask area (= information content); coarse stages slice the top-K
        if target_masks is not None:
            areas = target_masks.sum(dim=(1, 2))
        else:
            assert target_depths is not None
            areas = (target_depths > 0).sum(dim=(1, 2)).float()
        N = areas.shape[0]
        self._mask_area = float(areas.sum() if observation_weights is None else (areas * observation_weights).sum())
        area_order = torch.argsort(areas, descending=True)

        # --- per-level terms ---
        for level in levels:
            if level.use_depth and target_depths is None:
                raise ValueError("level has use_depth but no target_depths were given")
            obs_idx = slice(None)
            if level.obs_subset_size is not None and level.obs_subset_size < N:
                obs_idx = area_order[: level.obs_subset_size]
            masks_full = None if target_masks is None else target_masks[obs_idx].contiguous()
            depths_full = None if target_depths is None else target_depths[obs_idx].contiguous()
            weights_l = None if observation_weights is None else observation_weights[obs_idx].contiguous()
            T_ref_link_obs = T_ref_link[obs_idx].contiguous()
            target_link_mask_l = (
                target_link_mask
                if target_link_mask is None or target_link_mask.ndim == 1
                else target_link_mask[obs_idx].contiguous()
            )

            # Keep the coarse image large enough to rank pose candidates.
            ds = min(level.downscale, max(1, min(self.height, self.width) // 32))
            intrinsic = self.camera_intrinsic
            if intrinsic.ndim == 3:
                intrinsic = intrinsic[obs_idx].contiguous()
            H_l, W_l, K_l = self.height, self.width, intrinsic
            masks_l, depths_l = masks_full, depths_full
            if ds > 1:
                H_l, W_l = self.height // ds, self.width // ds
                K_l = intrinsic.clone()
                K_l[..., :2, :] /= ds
                weights_l = (q.new_ones((T_ref_link_obs.shape[0],)) if weights_l is None else weights_l) * ds**2
                masks_l = None if masks_full is None else F.avg_pool2d(masks_full.unsqueeze(1), ds).squeeze(1)
                # stride subsample at avg_pool pixel centers (averaging bleeds hole-zeros into depths)
                if depths_full is not None:
                    depths_l = depths_full[:, ds // 2 : H_l * ds : ds, ds // 2 : W_l * ds : ds].contiguous()
            if masks_l is None:
                # depth-only: keep a silhouette at every level, even those without the depth term
                assert depths_l is not None
                masks_l = (depths_l > 0).float()
            ctx = HandEyeContext
            context: Dict[Context, Any] = {
                ctx.CAMERA_INTRINSIC: K_l,
                ctx.TARGET_MASKS: masks_l.contiguous(),
                ctx.TARGET_DEPTHS: depths_l if level.use_depth else None,
                ctx.TARGET_LINK_MASK: target_link_mask_l,
                ctx.OBSERVATION_WEIGHTS: weights_l,
                ctx.LINK_POSES: T_ref_link_obs,
                ctx.LINK_VERTICES: link_vertices,
                ctx.LINK_FACES: link_faces,
                ctx.HEIGHT: H_l,
                ctx.WIDTH: W_l,
                ctx.BLUR_SIGMA: level.blur_sigma,
                ctx.PATIENCE: level.patience,
                ctx.ROBOT: self.robot,
                ctx.Q: q[obs_idx],
                ctx.LINK_INDICES: link_indices,
                ctx.T_MOUNT_BASE: None if T_mount_base is None else T_mount_base[obs_idx],
            }
            terms: List[ResidualTask] = []
            for template in templates:
                if template.active:
                    task = copy.copy(template)
                    if isinstance(task, MaskAlignmentNormalEqTask):
                        task.normalize_by_union = self.config.candidate_metric == "iou" and ds > 1
                        task._optimize_camera = self.config.optimize_camera
                        task._contour_geometry = ds == 1
                    task.resolve(self._initial_var, context)
                    if task.active:
                        terms.append(task)
            if target_keypoints is not None:
                terms.append(
                    KeypointAlignmentTask(
                        self.robot,
                        cast(List[str], keypoint_links),
                        target_keypoints[obs_idx] / ds,
                        K_l,
                        None if keypoint_weights is None else keypoint_weights[obs_idx],
                        None if T_mount_base is None else T_mount_base[obs_idx],
                        self.config.keypoint_weight,
                        self.config.keypoint_sigma / ds,
                        self.config.keypoint_huber_delta / ds,
                    )
                )
            stage_terms.append(terms)

        # --- keypoint initialization ---
        if target_keypoints is not None and self.config.keypoint_warmup_iters:
            # Guide half the global hypotheses; retain raw image modes for uncertain or missing keypoints.
            term = KeypointAlignmentTask(
                self.robot,
                cast(List[str], keypoint_links),
                target_keypoints,
                self.camera_intrinsic,
                keypoint_weights,
                T_mount_base,
                self.config.keypoint_weight,
                self.config.keypoint_sigma,
                self.config.keypoint_huber_delta,
            )
            active_q = np.r_[np.zeros(6), np.ones(self.robot.num_actuated_joints)].astype(np.float32)
            self._keypoint_solver = PopulationSolver(
                [term],
                PopulationSolverConfig(
                    stages=[
                        Stage(
                            self.config.keypoint_warmup_iters,
                            first_num_candidates,
                            LMOptimizerConfig(lm_lambda=0.01, early_stopping_interval=0),
                        )
                    ],
                    use_early_stopping=False,
                    cuda_graph_mode="none",
                ),
                device=self._wp_device,
                active_dof_mask=wp.from_numpy(active_q, device=self._wp_device),
            )

        # --- solver: one segment per pyramid level ---
        active = None
        if not self.config.optimize_camera:
            active = np.ones(self._initial_var.tangent_dim, dtype=np.float32)
            active[:6] = 0
            if not active.any():
                raise ValueError("enable optimize_q, optimize_camera or a model-offset task")
        self._solver = PopulationSolver(
            terms=stage_terms,
            config=self.config.solver,
            device=self._wp_device,
            score_terms=[
                _MaskCandidateScore(
                    terms,
                    reuse_accepted=not self.config.solver.samplers
                    and self.config.solver.num_iters > 0
                    and all(i > 0 for i in self.config.solver.select_at),
                )
                for terms in stage_terms
            ]
            if self.config.candidate_metric == "iou"
            else None,
            active_dof_mask=None if active is None else wp.from_numpy(active, device=self._wp_device),
        )
        self._solver.setup(self._initial_var)
        self._rng = np.random.default_rng(self.config.seed)
        self._initialized = True

    def solve(
        self,
        initial_extrinsic: Float[torch.Tensor, "4 4"],
        target_masks: Optional[Float[torch.Tensor, "n_obs height width"]] = None,
        target_depths: Optional[Float[torch.Tensor, "n_obs height width"]] = None,
        observation_weights: Optional[Float[torch.Tensor, "n_obs"]] = None,
        q: Optional[Float[torch.Tensor, "*batch num_dofs"]] = None,
        T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]] = None,
        keypoint_links: Optional[List[str]] = None,
        target_keypoints: Optional[torch.Tensor] = None,
        keypoint_weights: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "4 4"]:
        """Solve camera extrinsics from `initial_extrinsic`.

        Lifecycle:
            1. Build the solver for new observations, or reuse stored data.
            2. Reset variables and sample camera or joint candidates.
            3. Optionally initialize q from keypoints, then run native population LM.
            4. Retain the best result and restart if mask mismatch remains high.

        Data:
        - Any target supplied: replace data and rebuild the solver.
        - No targets supplied: reuse stored data.

        Args:
            initial_extrinsic: Initial OpenCV `T_camera_base` or mounted-camera `T_camera_mount`, `[4,4]`.
            target_masks: Robot silhouette masks `[N, H, W]` in `[0, 1]`.
            target_depths: Robot-only depth maps `[N, H, W]` in meters, 0 = invalid.
            observation_weights: Optional nonnegative objective weights `[N]`.
            q: Known `[N,num_dofs]`, or one initial `[num_dofs]` shared across views with `optimize_q`; None uses midpoints.
            T_mount_base: Per-sample mount pose `[N, 4, 4]` (mounted camera).
            keypoint_links: Optional link-frame origins to project, with optimize_q.
            target_keypoints: Their observed pixels `[N,P,2]`. NaNs mark missing points.
            keypoint_weights: Optional nonnegative confidence weights `[N,P]`.

        Returns:
            Optimized extrinsic `[4,4]` on the input device; `last_q` stores fitted q and `last_var` includes model offsets.
        """

        with _torch_warp_stream(initial_extrinsic.device):
            # --- prepare observations ---
            if target_masks is not None or target_depths is not None:
                if self.config.optimize_q:
                    if q is None:
                        q = torch.as_tensor(self.robot.spec.midrange_q, device=initial_extrinsic.device)
                    q = q.detach().to(device=initial_extrinsic.device, dtype=torch.float32)
                    limits = torch.as_tensor(self.robot.spec.actuated_joint_limits, device=q.device, dtype=q.dtype)
                    if q.shape != (self.robot.num_actuated_joints,) or not torch.isfinite(q).all():
                        raise ValueError("optimize_q expects one finite initial q [num_dofs]")
                    if not torch.isfinite(limits).all():
                        raise ValueError("optimize_q requires finite joint limits")
                    if ((q < limits[:, 0]) | (q > limits[:, 1])).any():
                        raise ValueError("initial q must lie within joint limits")
                    self._q_initial = q.detach().to(dtype=torch.float32).clone()
                    count = len(target_masks) if target_masks is not None else len(cast(torch.Tensor, target_depths))
                    q = q[None].expand(count, -1).contiguous()
                elif q is None:
                    raise ValueError("pass q [N, num_dofs] with target data")
                self._build(
                    target_masks,
                    target_depths,
                    observation_weights,
                    q,
                    T_mount_base,
                    keypoint_links,
                    target_keypoints,
                    keypoint_weights,
                )
            elif not self._initialized:
                raise ValueError("no data stored: pass target_masks/target_depths with q")
            dev = initial_extrinsic.device
            first_num_candidates = self.config.solver.num_candidates[0]

            mat_np = initial_extrinsic.detach().cpu().numpy().astype(np.float32)
            self.last_final_cost = float("inf")
            for _ in range(1 + self.config.max_restarts):
                self._initial_var.accept(
                    wp.ones(first_num_candidates, dtype=wp.int32, device=self._wp_device), self._nominal_var
                )

                # --- sample camera candidates ---
                T_init = se3_from_matrix(wp.from_numpy(mat_np.reshape(1, 4, 4), dtype=wp.mat44, device=self._wp_device))
                init_v7 = T_init.numpy()  # [1, 7]

                twists_np = np.zeros((first_num_candidates, 6), dtype=np.float32)
                if first_num_candidates > 1 and self.config.optimize_camera:
                    # Sample direction and [0.5, 1.0] × noise magnitude; Gaussian magnitudes cluster near zero.
                    n = first_num_candidates - 1
                    t_dirs = self._rng.standard_normal((n, 3)).astype(np.float32)
                    t_dirs /= np.linalg.norm(t_dirs, axis=1, keepdims=True) + 1e-12
                    t_mags = self._rng.uniform(0.5, 1.0, size=(n, 1)).astype(np.float32) * self.config.candidate_noise_t
                    r_dirs = self._rng.standard_normal((n, 3)).astype(np.float32)
                    r_dirs /= np.linalg.norm(r_dirs, axis=1, keepdims=True) + 1e-12
                    r_mags = self._rng.uniform(0.5, 1.0, size=(n, 1)).astype(np.float32) * self.config.candidate_noise_r
                    twists_np[1:, :3] = t_dirs * t_mags
                    twists_np[1:, 3:] = r_dirs * r_mags
                # Map camera-frame noise through Ad(T^-1) so body-frame integration matches `exp(xi_cam) * T`.
                rot, trans = mat_np[:3, :3], mat_np[:3, 3]
                twists_np[:, :3] = (twists_np[:, :3] - np.cross(trans, twists_np[:, 3:])) @ rot
                twists_np[:, 3:] = twists_np[:, 3:] @ rot
                var = cast(SE3Var, self._initial_var.get(VarKey.ROBOT))
                wp.copy(
                    var.xyz_wxyz,
                    wp.from_numpy(np.tile(init_v7, (first_num_candidates, 1)), dtype=wp_vec7, device=self._wp_device),
                )
                var.integrate(wp.from_numpy(twists_np, dtype=wp_vec6, device=self._wp_device), out=var)
                # --- sample joint candidates and initialize keypoints ---
                if self.config.optimize_q:
                    lower, upper = self.robot.spec.actuated_joint_limits.T
                    radius = self.config.candidate_noise_q
                    scale = upper - lower if radius is None else np.full_like(lower, 2 * radius)
                    sampler = RobertsSampler(
                        first_num_candidates, scale, seed=int(self._rng.integers(10000)), device=self._wp_device
                    )
                    state = cast(RobotState, self._initial_var.get(VarKey.QPOS_OFFSET))
                    noise = wp.empty_like(state.q)
                    sampler.fill(noise)
                    center = torch.as_tensor((lower + upper) / 2, device=dev) if radius is None else self._q_initial
                    sampled = (center + wp.to_torch(noise)).clamp(
                        torch.as_tensor(lower, device=dev), torch.as_tensor(upper, device=dev)
                    )
                    sampled[0] = self._q_initial
                    wp.to_torch(state.q).copy_(sampled)
                    state.invalidate()
                    if self._keypoint_solver is not None:
                        original = wp.to_torch(state.q).clone()
                        self._keypoint_solver.solve(self._initial_var)
                        wp.to_torch(state.q)[1::2].copy_(original[1::2])
                        state.invalidate()

                # --- solve and retain the best candidate ---
                best_var, best_costs = self._solver.solve(self._initial_var)
                best_state = cast(SE3Var, best_var.get(VarKey.ROBOT))
                cost = float(best_costs.numpy()[0])
                if cost < self.last_final_cost:
                    best_mat = se3_to_matrix(best_state.xyz_wxyz).numpy()[0]
                    self.last_final_cost, self.last_var = cost, best_var.clone()
                    if self.config.optimize_q:
                        self.last_q = wp.to_torch(cast(RobotState, self.last_var.get(VarKey.QPOS_OFFSET)).q)[0].clone()
                        self.last_candidates = wp.to_torch(
                            cast(RobotState, self._solver.final_var.get(VarKey.QPOS_OFFSET)).q
                        ).clone()
                        assert self._solver.candidate_scores is not None
                        self.last_candidate_costs = wp.to_torch(self._solver.candidate_scores).clone()
                # --- retry unmatched observations ---
                if 2 * cost <= self.config.restart_mismatch * self._mask_area:
                    break
            return torch.as_tensor(best_mat, device=dev)

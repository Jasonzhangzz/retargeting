"""Inspire Hand retargeting presets."""

from dataclasses import replace

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandRetargeterOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    FIVE_PINCH_PAIRS,
    FIVE_ROOT_TIP_PAIRS,
    FIVE_TIP_PAIRS,
    TARGET_CHAINS,
    TARGET_NAMES,
    correspondences,
    wrist_config,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec


ctx = HandContext  # the values the retargeters give their templates


_ONLINE_LINKS = {
    "wrist": "base",
    "thumb_mcp": "thumb_proximal",
    "thumb_tip": "thumb_tip",
    "index_mcp": "index_proximal",
    "index_tip": "index_tip",
    "middle_mcp": "middle_proximal",
    "middle_tip": "middle_tip",
    "ring_mcp": "ring_proximal",
    "ring_tip": "ring_tip",
    "pinky_mcp": "pinky_proximal",
    "pinky_tip": "pinky_tip",
}
_OFFLINE_LINKS = {
    "wrist": "hand_base_link",
    "thumb_mcp": "thumb_intermediate",
    "thumb_tip": "thumb_tip",
    "index_pip": "index_intermediate",
    "index_tip": "index_tip",
    "middle_pip": "middle_intermediate",
    "middle_tip": "middle_tip",
    "ring_pip": "ring_intermediate",
    "ring_tip": "ring_tip",
    "pinky_pip": "pinky_intermediate",
    "pinky_tip": "pinky_tip",
}
_DIRECTION_PAIRS = (
    ("thumb_mcp", "thumb_tip"),
    ("index_mcp", "index_tip"),
    ("middle_mcp", "middle_tip"),
    ("ring_mcp", "ring_tip"),
    ("pinky_mcp", "pinky_tip"),
)


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-x", four_fingers_up_axis="-y"),
)
online = HandRetargeterOnlineConfig(  # tuned on DexYCB subjects 01/02 (RetargetingKit benchmark)
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight=correspondences(_ONLINE_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0),
            weight_mode="cost",
            huber_on_norm=True,
            soft_gate_start_distance=0.138,
            soft_gate_full_distance=0.0345,
            name="vectors",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.03, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.001, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
        SelfCollisionTask(weight=10.0, margin=0.002, max_active_pairs=32, name="self_collision"),
    ],
    target_scale=1.2075,
    output_ema_alpha=0.4,
    solver=PopulationSolverConfig(
        stages=[Stage(30, optimizer=LMOptimizerConfig(lm_lambda=0.01))], cuda_graph_mode="full"
    ),
)
offline = HandRetargeterOfflineConfig(target_scale=1.15)
offline.task("positions").weight = 26.0

online_wrist = wrist_config(spec, online=online, fingertip_weight=26.0)


spec_left = replace(spec, target_link_names=_ONLINE_LINKS)
online_left = HandRetargeterOnlineConfig(
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_ONLINE_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
                **correspondences(_ONLINE_LINKS, FIVE_TIP_PAIRS, 0.2),
            },
            weight_mode="cost",
            huber_on_norm=True,
            huber_delta=0.02,
            soft_gate_start_distance=0.046,
            soft_gate_full_distance=0.0115,
            name="vectors",
        ),
        FrameVectorTask(
            targets=ctx.DIRECTIONS,
            target_names=ctx.TARGET_NAMES,
            weight=correspondences(_ONLINE_LINKS, _DIRECTION_PAIRS),
            weight_mode="normalized",
            direction_only=True,
            name="directions",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.05, name="smoothness"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    pinch_correspondences=tuple(correspondences(_ONLINE_LINKS, FIVE_PINCH_PAIRS)),
    target_scale=1.15,
)


__all__ = ["offline", "online", "online_left", "online_wrist", "spec", "spec_left"]

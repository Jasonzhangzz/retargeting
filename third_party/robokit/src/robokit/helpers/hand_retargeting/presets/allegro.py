"""Allegro Hand retargeting presets."""

from dataclasses import replace

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandRetargeterOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FOUR_DIRECTION_PAIRS,
    FOUR_FINGERTIPS,
    FOUR_PINCH_PAIRS,
    FOUR_ROOT_TIP_PAIRS,
    FOUR_TIP_PAIRS,
    TARGET_CHAINS,
    TARGET_NAMES,
    correspondences,
    wrist_config,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec


ctx = HandContext  # the values the retargeters give their templates


_LINKS = {
    "wrist": "wrist",
    "thumb_cmc": "link_13.0",
    "thumb_mcp": "link_14.0",
    "thumb_ip": "link_15.0",
    "thumb_tip": "link_15.0_tip",
    "index_mcp": "link_1.0",
    "index_pip": "link_2.0",
    "index_dip": "link_3.0",
    "index_tip": "link_3.0_tip",
    "middle_mcp": "link_5.0",
    "middle_pip": "link_6.0",
    "middle_dip": "link_7.0",
    "middle_tip": "link_7.0_tip",
    "ring_mcp": "link_9.0",
    "ring_pip": "link_10.0",
    "ring_dip": "link_11.0",
    "ring_tip": "link_11.0_tip",
}
_LINKS_LEFT = {
    **{name: link for name, link in _LINKS.items() if name == "wrist" or name.startswith("thumb")},
    "index_mcp": "link_9.0",
    "index_pip": "link_10.0",
    "index_dip": "link_11.0",
    "index_tip": "link_11.0_tip",
    "middle_mcp": "link_5.0",
    "middle_pip": "link_6.0",
    "middle_dip": "link_7.0",
    "middle_tip": "link_7.0_tip",
    "ring_mcp": "link_1.0",
    "ring_pip": "link_2.0",
    "ring_dip": "link_3.0",
    "ring_tip": "link_3.0_tip",
}
_OFFLINE_LINKS = {
    name: _LINKS[name]
    for name in (
        "wrist",
        "thumb_mcp",
        "thumb_tip",
        "index_pip",
        "index_tip",
        "middle_pip",
        "middle_tip",
        "ring_pip",
        "ring_tip",
    )
}
spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FOUR_FINGERTIPS,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y"),
)
online = HandRetargeterOnlineConfig(  # tuned on DexYCB subjects 01/02 (RetargetingKit benchmark)
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_LINKS, FOUR_ROOT_TIP_PAIRS, 1.0),
                **correspondences(_LINKS, FOUR_PINCH_PAIRS, 2.0),
                **correspondences(_LINKS, FOUR_TIP_PAIRS[3:], 2.0 / 15.0),
            },
            weight_mode="cost",
            huber_on_norm=True,
            soft_gate_start_distance=0.032,
            soft_gate_full_distance=0.008,
            name="vectors",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.03, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.001, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    pinch_correspondences=tuple(correspondences(_LINKS, FOUR_PINCH_PAIRS)),
    target_scale=1.36,
    output_ema_alpha=0.4,
    solver=PopulationSolverConfig(
        stages=[Stage(10, optimizer=LMOptimizerConfig(lm_lambda=0.01))], cuda_graph_mode="full"
    ),
    sampler=RobertsSamplerConfig(keep_first=True, scale=0.0),
)
offline = HandRetargeterOfflineConfig(target_scale=1.6)
offline.task("positions").weight = 26.0

online_wrist = wrist_config(spec, online=online, fingertip_weight=26.0)


spec_left = replace(spec, target_link_names=_LINKS_LEFT)
online_left = HandRetargeterOnlineConfig(
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_LINKS_LEFT, FOUR_ROOT_TIP_PAIRS, 0.1),
                **correspondences(_LINKS_LEFT, FOUR_TIP_PAIRS, 0.3),
            },
            weight_mode="cost",
            huber_on_norm=True,
            huber_delta=0.02,
            soft_gate_start_distance=0.064,
            soft_gate_full_distance=0.016,
            name="vectors",
        ),
        FrameVectorTask(
            targets=ctx.DIRECTIONS,
            target_names=ctx.TARGET_NAMES,
            weight=correspondences(_LINKS_LEFT, FOUR_DIRECTION_PAIRS),
            weight_mode="normalized",
            direction_only=True,
            name="directions",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.05, name="smoothness"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    pinch_correspondences=tuple(correspondences(_LINKS_LEFT, FOUR_PINCH_PAIRS)),
    target_scale=1.6,
)


__all__ = ["offline", "online", "online_left", "online_wrist", "spec", "spec_left"]

"""Paxini DexH13 retargeting presets."""

from dataclasses import replace

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandRetargeterOnlineConfig,
    HandSpec,
    _online_solver,
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
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec


ctx = HandContext  # the values the retargeters give their templates


_LINKS = {
    "wrist": "right_palm_link",
    "thumb_cmc": "right_thumb_link_1",
    "thumb_mcp": "right_thumb_link_2",
    "thumb_ip": "right_thumb_link_3",
    "thumb_tip": "thumb_tip",
    "index_mcp": "right_index_link_1",
    "index_pip": "right_index_link_2",
    "index_dip": "right_index_link_3",
    "index_tip": "index_tip",
    "middle_mcp": "right_middle_link_1",
    "middle_pip": "right_middle_link_2",
    "middle_dip": "right_middle_link_3",
    "middle_tip": "middle_tip",
    "ring_mcp": "right_ring_link_1",
    "ring_pip": "right_ring_link_2",
    "ring_dip": "right_ring_link_3",
    "ring_tip": "ring_tip",
}
_LINKS_LEFT = {name: link.replace("right_", "left_") for name, link in _LINKS.items()}
_OFFLINE_LINKS = {
    "wrist": "right_palm_link",
    "thumb_mcp": "right_thumb_link_1",
    "thumb_tip": "thumb_tip",
    "index_pip": "right_index_link_1",
    "index_tip": "index_tip",
    "middle_pip": "right_middle_link_1",
    "middle_tip": "middle_tip",
    "ring_pip": "right_ring_link_1",
    "ring_tip": "ring_tip",
}
spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FOUR_FINGERTIPS,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="z", four_fingers_up_axis="x"),
)
online = HandRetargeterOnlineConfig(  # tuned on DexYCB subjects 01/02 (RetargetingKit benchmark)
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight=correspondences(_LINKS, FOUR_ROOT_TIP_PAIRS, 1.0),
            weight_mode="cost",
            huber_on_norm=True,
            soft_gate_start_distance=0.04352,
            soft_gate_full_distance=0.01088,
            name="vectors",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.02, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.001, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    target_scale=1.306305170059204,
    output_ema_alpha=0.8,
    solver=_online_solver((6, 10, 14)),
    base_init_sample_range=(0.0, 0.0, 0.0),
)
offline = HandRetargeterOfflineConfig(target_scale=1.36)
offline.task("positions").weight = 24.0

online_wrist = wrist_config(
    spec,
    online=online,
    fingertip_weight=24.0,
    wrist_weight=1.0,
    direction_weight=5.0,
    wrist_direction_weight=5.0,
)


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
            soft_gate_start_distance=0.068,
            soft_gate_full_distance=0.017,
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
    target_scale=1.7,
)


__all__ = ["offline", "online", "online_left", "online_wrist", "spec", "spec_left"]

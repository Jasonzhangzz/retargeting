"""Barrett Hand online retargeting preset."""

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    TARGET_CHAINS,
    TARGET_COORD_SPEC,
    TARGET_NAMES,
    THREE_FINGERTIPS,
    THREE_PINCH_PAIRS,
    THREE_ROOT_TIP_PAIRS,
    correspondences,
    wrist_config,
)
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask


ctx = HandContext  # the values the retargeters give their templates


_LINKS = {
    "wrist": "base_link",
    "thumb_tip": "finger_3_dist_link",
    "index_mcp": "finger_2_med_link",
    "index_tip": "finger_2_dist_link",
    "middle_mcp": "finger_1_med_liink",
    "middle_tip": "finger_1_dist_link",
}
_DIRECTION_PAIRS = (("index_mcp", "index_tip"), ("middle_mcp", "middle_tip"))


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=THREE_FINGERTIPS,
    root_link_coord_spec=TARGET_COORD_SPEC,
)
online = HandRetargeterOnlineConfig(
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_LINKS, THREE_ROOT_TIP_PAIRS, 0.2),
                **correspondences(_LINKS, THREE_PINCH_PAIRS, 0.6),
            },
            weight_mode="cost",
            huber_on_norm=True,
            huber_delta=0.02,
            soft_gate_start_distance=0.06,
            soft_gate_full_distance=0.015,
            name="vectors",
        ),
        FrameVectorTask(
            targets=ctx.DIRECTIONS,
            target_names=ctx.TARGET_NAMES,
            weight=correspondences(_LINKS, _DIRECTION_PAIRS),
            weight_mode="normalized",
            direction_only=True,
            name="directions",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.05, name="smoothness"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    pinch_correspondences=tuple(correspondences(_LINKS, THREE_PINCH_PAIRS)),
    target_scale=1.5,
)

online_wrist = wrist_config(
    spec, online=online, fingertip_weight=24.0, direction_weight=20.0, wrist_direction_weight=2.0
)


__all__ = ["online", "online_wrist", "spec"]

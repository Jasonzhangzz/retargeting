"""LEAP Hand retargeting presets."""

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
    "wrist": "base",
    "thumb_cmc": "thumb_pip",
    "thumb_mcp": "thumb_dip",
    "thumb_ip": "thumb_fingertip",
    "thumb_tip": "thumb_tip_head",
    "index_mcp": "pip",
    "index_pip": "dip",
    "index_dip": "fingertip",
    "index_tip": "index_tip_head",
    "middle_mcp": "pip_2",
    "middle_pip": "dip_2",
    "middle_dip": "fingertip_2",
    "middle_tip": "middle_tip_head",
    "ring_mcp": "pip_3",
    "ring_pip": "dip_3",
    "ring_dip": "fingertip_3",
    "ring_tip": "ring_tip_head",
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
            weight=correspondences(_LINKS, FOUR_ROOT_TIP_PAIRS, 1.0),
            weight_mode="cost",
            huber_on_norm=True,
            soft_gate_start_distance=0.04096,
            soft_gate_full_distance=0.01024,
            name="vectors",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.02, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.001, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    target_scale=1.2797727584838867,
    solver=_online_solver((6, 10, 14)),
    base_init_sample_range=(0.0, 0.0, 0.0),
)
offline = HandRetargeterOfflineConfig(target_scale=1.28)
offline.task("positions").weight = 26.0

online_wrist = wrist_config(spec, online=online, fingertip_weight=26.0)


spec_left = replace(spec, target_link_names=_LINKS)
online_left = HandRetargeterOnlineConfig(
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_LINKS, FOUR_ROOT_TIP_PAIRS, 0.1),
                **correspondences(_LINKS, FOUR_TIP_PAIRS, 0.3),
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
            weight=correspondences(_LINKS, FOUR_DIRECTION_PAIRS),
            weight_mode="normalized",
            direction_only=True,
            name="directions",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.05, name="smoothness"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    pinch_correspondences=tuple(correspondences(_LINKS, FOUR_PINCH_PAIRS)),
    target_scale=1.6,
)


__all__ = ["offline", "online", "online_left", "online_wrist", "spec", "spec_left"]

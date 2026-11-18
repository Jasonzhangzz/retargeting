"""Sharpa Wave Hand retargeting presets."""

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandRetargeterOnlineConfig,
    HandSpec,
    _online_solver,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    FIVE_ROOT_TIP_PAIRS,
    TARGET_CHAINS,
    TARGET_NAMES,
    correspondences,
    wrist_config,
)
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec


ctx = HandContext  # the values the retargeters give their templates


_LINKS = {
    "wrist": "right_hand_C_MC",
    "thumb_cmc": "right_thumb_MC",
    "thumb_mcp": "right_thumb_PP",
    "thumb_ip": "right_thumb_DP",
    "thumb_tip": "right_thumb_fingertip",
    "index_mcp": "right_index_PP",
    "index_pip": "right_index_MP",
    "index_dip": "right_index_DP",
    "index_tip": "right_index_fingertip",
    "middle_mcp": "right_middle_PP",
    "middle_pip": "right_middle_MP",
    "middle_dip": "right_middle_DP",
    "middle_tip": "right_middle_fingertip",
    "ring_mcp": "right_ring_PP",
    "ring_pip": "right_ring_MP",
    "ring_dip": "right_ring_DP",
    "ring_tip": "right_ring_fingertip",
    "pinky_mcp": "right_pinky_PP",
    "pinky_pip": "right_pinky_MP",
    "pinky_dip": "right_pinky_DP",
    "pinky_tip": "right_pinky_fingertip",
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
        "pinky_pip",
        "pinky_tip",
    )
}


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-z", four_fingers_up_axis="y"),
)
online = HandRetargeterOnlineConfig(  # tuned on DexYCB subjects 01/02 (RetargetingKit benchmark)
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight=correspondences(_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0),
            weight_mode="cost",
            huber_on_norm=True,
            soft_gate_start_distance=0.024,
            soft_gate_full_distance=0.006,
            name="vectors",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.02, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.001, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
        SelfCollisionTask(weight=10.0, margin=0.002, max_active_pairs=32, name="self_collision"),
    ],
    target_scale=1.1541533470153809,
    solver=_online_solver((6, 10, 14)),
    base_init_sample_range=(0.0, 0.0, 0.0),
)
offline = HandRetargeterOfflineConfig(target_scale=1.2)
offline.task("positions").weight = 26.0

online_wrist = wrist_config(spec, online=online, fingertip_weight=26.0)
online_wrist.task("self_collision").max_active_pairs = 256


__all__ = ["offline", "online", "online_wrist", "spec"]

"""XHand retargeting presets."""

from dataclasses import replace

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandRetargeterOnlineConfig,
    HandSpec,
    _online_solver,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    FIVE_PINCH_PAIRS,
    FIVE_ROOT_JOINT_PAIRS,
    FIVE_ROOT_TIP_PAIRS,
    FIVE_TIP_PAIRS,
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
    "wrist": "right_hand_link",
    "thumb_mcp": "right_hand_thumb_rota_link1",
    "thumb_ip": "right_hand_thumb_rota_link2",
    "thumb_tip": "right_hand_thumb_rota_tip",
    "index_mcp": "right_hand_index_rota_link1",
    "index_pip": "right_hand_index_rota_link2",
    "index_tip": "right_hand_index_rota_tip",
    "middle_mcp": "right_hand_mid_link1",
    "middle_pip": "right_hand_mid_link2",
    "middle_tip": "right_hand_mid_tip",
    "ring_mcp": "right_hand_ring_link1",
    "ring_pip": "right_hand_ring_link2",
    "ring_tip": "right_hand_ring_tip",
    "pinky_mcp": "right_hand_pinky_link1",
    "pinky_pip": "right_hand_pinky_link2",
    "pinky_tip": "right_hand_pinky_tip",
}
_LINKS_LEFT = {name: link.replace("right_hand_", "left_hand_") for name, link in _LINKS.items()}
_OFFLINE_LINKS = {
    "wrist": "right_hand_link",
    "thumb_mcp": "right_hand_thumb_rota_link1",
    "thumb_tip": "right_hand_thumb_rota_tip",
    "index_pip": "right_hand_index_rota_link1",
    "index_tip": "right_hand_index_rota_tip",
    "middle_pip": "right_hand_mid_link1",
    "middle_tip": "right_hand_mid_tip",
    "ring_pip": "right_hand_ring_link1",
    "ring_tip": "right_hand_ring_tip",
    "pinky_pip": "right_hand_pinky_link1",
    "pinky_tip": "right_hand_pinky_tip",
}
_VECTOR_LINKS = {**_LINKS, **_OFFLINE_LINKS}
_VECTOR_LINKS_LEFT = {name: link.replace("right_hand_", "left_hand_") for name, link in _VECTOR_LINKS.items()}
_DIRECTION_PAIRS = (
    ("thumb_mcp", "thumb_ip"),
    ("thumb_ip", "thumb_tip"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_tip"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_tip"),
    ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_tip"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_tip"),
)


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z"),
)
online = HandRetargeterOnlineConfig(  # tuned on DexYCB subjects 01/02 (RetargetingKit benchmark)
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight=correspondences(_VECTOR_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0),
            weight_mode="cost",
            huber_on_norm=True,
            soft_gate_start_distance=0.04032,
            soft_gate_full_distance=0.01008,
            name="vectors",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.02, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.001, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
        SelfCollisionTask(weight=10.0, margin=0.002, max_active_pairs=32, name="self_collision"),
    ],
    target_scale=1.1761616468429565,
    output_ema_alpha=0.4,
    solver=_online_solver((6, 10, 14)),
    base_init_sample_range=(0.0, 0.0, 0.0),
)
offline = HandRetargeterOfflineConfig(target_scale=1.12)
offline.task("positions").weight = 48.0
online_wrist = wrist_config(spec, online=online, fingertip_weight=48.0, q_smoothness_weight=1.2)
online_wrist.task("self_collision").max_active_pairs = 256


spec_left = replace(spec, target_link_names=_LINKS_LEFT)
online_left = HandRetargeterOnlineConfig(
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_VECTOR_LINKS_LEFT, FIVE_ROOT_TIP_PAIRS + FIVE_ROOT_JOINT_PAIRS, 0.05),
                **correspondences(_VECTOR_LINKS_LEFT, FIVE_TIP_PAIRS, 0.15),
            },
            weight_mode="cost",
            huber_on_norm=True,
            huber_delta=0.02,
            soft_gate_start_distance=0.056,
            soft_gate_full_distance=0.014,
            name="vectors",
        ),
        FrameVectorTask(
            targets=ctx.DIRECTIONS,
            target_names=ctx.TARGET_NAMES,
            weight=correspondences(_LINKS_LEFT, _DIRECTION_PAIRS),
            weight_mode="normalized",
            direction_only=True,
            name="directions",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.05, name="smoothness"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    pinch_correspondences=tuple(correspondences(_VECTOR_LINKS_LEFT, FIVE_PINCH_PAIRS)),
    target_scale=1.4,
)


__all__ = ["offline", "online", "online_left", "online_wrist", "spec", "spec_left"]

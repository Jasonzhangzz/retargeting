"""xArm7 with XHand retargeting presets."""

from dataclasses import replace

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandRetargeterOnlineConfig,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    FIVE_ROOT_JOINT_PAIRS,
    FIVE_ROOT_TIP_PAIRS,
    FIVE_TIP_PAIRS,
    XARM7_INIT_Q,
    correspondences,
)
from robokit.helpers.hand_retargeting.presets.xhand import (
    _DIRECTION_PAIRS,
    _LINKS,
    _VECTOR_LINKS,
)
from robokit.helpers.hand_retargeting.presets.xhand import (
    spec as _XHAND_SPEC,
)
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask


ctx = HandContext  # the values the retargeters give their templates


spec = replace(
    _XHAND_SPEC,
    floating_base=False,
    init_q_by_name=dict(XARM7_INIT_Q),
    rest_q_by_name=dict(XARM7_INIT_Q),
)
online = HandRetargeterOnlineConfig(
    terms=[
        FrameVectorTask(
            targets=ctx.POINTS,
            target_names=ctx.TARGET_NAMES,
            target_links=ctx.TARGET_LINK_NAMES,
            weight={name: 3.0 for name in FIVE_FINGERTIPS},
            weight_mode="normalized",
            huber_delta=0.05,
            name="positions",
        ),
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_VECTOR_LINKS, FIVE_ROOT_TIP_PAIRS + FIVE_ROOT_JOINT_PAIRS, 0.05),
                **correspondences(_VECTOR_LINKS, FIVE_TIP_PAIRS, 0.1),
            },
            weight_mode="cost",
            huber_on_norm=True,
            huber_delta=0.02,
            soft_gate_start_distance=0.04032,
            soft_gate_full_distance=0.01008,
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
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.02, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.005, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    output_ema_alpha=0.8,
)
offline = HandRetargeterOfflineConfig()
offline.task("positions").weight = 32.0


__all__ = ["offline", "online", "spec"]

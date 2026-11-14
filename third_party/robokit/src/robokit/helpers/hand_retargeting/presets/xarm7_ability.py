"""xArm7 with Ability Hand offline retargeting presets."""

from copy import deepcopy
from dataclasses import replace

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets import ability
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    TARGET_CHAINS,
    TARGET_NAMES,
    XARM7_INIT_Q,
)
from robokit.terms.dense.frame_vector_task import FrameVectorTask


_LINKS = {
    "wrist": "base",
    "thumb_tip": "thumb_tip",
    "index_tip": "index_tip",
    "middle_tip": "middle_tip",
    "ring_tip": "ring_tip",
    "pinky_tip": "pinky_tip",
}


spec = HandSpec(
    floating_base=False,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    root_link_coord_spec=ability.spec.root_link_coord_spec,
    init_q_by_name=dict(XARM7_INIT_Q),
)
offline = HandRetargeterOfflineConfig(target_scale=0.99)
offline.task("positions").weight = 24.0


spec_no_mimic = spec
offline_no_mimic = HandRetargeterOfflineConfig(target_scale=0.99)
offline_no_mimic.task("positions").weight = 26.0

online = replace(  # Ability's online tuning plus fingertip positions
    ability.online,
    terms=[
        FrameVectorTask(
            targets=HandContext.POINTS,
            target_names=HandContext.TARGET_NAMES,
            target_links=HandContext.TARGET_LINK_NAMES,
            weight={name: 3.0 for name in FIVE_FINGERTIPS},
            weight_mode="normalized",
            huber_delta=0.05,
            name="positions",
        ),
        *deepcopy(ability.online.terms),
    ],
)

__all__ = ["offline", "offline_no_mimic", "online", "spec", "spec_no_mimic"]

"""xArm7 with MANO-kinematic hand offline retargeting preset."""

from robokit.helpers.hand_retargeting.config import (
    HandRetargeterOfflineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    TARGET_CHAINS,
    TARGET_COORD_SPEC,
    TARGET_NAMES,
    XARM7_INIT_Q,
)


_LINKS = {
    "wrist": "mano_link_0",
    "thumb_cmc": "mano_link_1",
    "thumb_mcp": "mano_link_2",
    "thumb_ip": "mano_link_3",
    "thumb_tip": "mano_link_4",
    "index_mcp": "mano_link_5",
    "index_pip": "mano_link_6",
    "index_dip": "mano_link_7",
    "index_tip": "mano_link_8",
    "middle_mcp": "mano_link_9",
    "middle_pip": "mano_link_10",
    "middle_dip": "mano_link_11",
    "middle_tip": "mano_link_12",
    "ring_mcp": "mano_link_13",
    "ring_pip": "mano_link_14",
    "ring_dip": "mano_link_15",
    "ring_tip": "mano_link_16_tip",
    "pinky_mcp": "mano_link_17_tip",
    "pinky_pip": "mano_link_18_tip",
    "pinky_dip": "mano_link_19_tip",
    "pinky_tip": "mano_link_20_tip",
}
spec = HandSpec(
    floating_base=False,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    root_link_coord_spec=TARGET_COORD_SPEC,
    init_q_by_name=dict(XARM7_INIT_Q),
)
offline = HandRetargeterOfflineConfig()
offline.task("positions").weight = 26.0


__all__ = ["offline", "spec"]

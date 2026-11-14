"""Wuji Hand 2.2 right-hand offline retargeting preset (20 independent joints)."""

from robokit.helpers.hand_retargeting.config import HandRetargeterOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.presets._common import FIVE_FINGERTIPS, TARGET_CHAINS, TARGET_NAMES
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_LINKS = {
    "wrist": "r_mount",
    "thumb_cmc": "r_thumb_proximal_abd",
    "thumb_mcp": "r_thumb_middle",
    "thumb_ip": "r_thumb_distal",
    "thumb_tip": "r_thumb_tip",
}
for _finger in ("index", "middle", "ring", "pinky"):
    _prefix = f"r_{_finger}" + ("_finger" if _finger != "pinky" else "")
    _LINKS.update(
        zip(
            (f"{_finger}_{joint}" for joint in ("mcp", "pip", "dip", "tip")),
            (f"{_prefix}_{link}" for link in ("proximal_abd", "middle", "distal", "tip")),
        )
    )

spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z"),
)
offline = HandRetargeterOfflineConfig(target_scale=1.0)
offline.task("positions").weight = 26.0

__all__ = ["offline", "spec"]

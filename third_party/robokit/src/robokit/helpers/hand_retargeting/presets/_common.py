"""Shared data and helpers for built-in hand presets."""

from copy import deepcopy
from dataclasses import replace
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from robokit.helpers.hand_retargeting.config import HandContext, HandRetargeterOnlineConfig, HandSpec, _online_solver
from robokit.helpers.hand_retargeting.kinematics_utils import build_retargeting_pairs
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import MANOPTH_HAND_COORD_SPEC


TARGET_COORD_SPEC = MANOPTH_HAND_COORD_SPEC
TARGET_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
TARGET_CHAINS = (
    ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
    ("index_mcp", "index_pip", "index_dip", "index_tip"),
    ("middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
    ("ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
)
FIVE_FINGERTIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
FOUR_FINGERTIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip")
THREE_FINGERTIPS = ("thumb_tip", "index_tip", "middle_tip")

FIVE_ROOT_TIP_PAIRS = tuple(("wrist", tip) for tip in FIVE_FINGERTIPS)
FOUR_ROOT_TIP_PAIRS = tuple(("wrist", tip) for tip in FOUR_FINGERTIPS)
THREE_ROOT_TIP_PAIRS = tuple(("wrist", tip) for tip in THREE_FINGERTIPS)
FIVE_TIP_PAIRS = tuple((origin, task) for i, origin in enumerate(FIVE_FINGERTIPS) for task in FIVE_FINGERTIPS[i + 1 :])
FOUR_TIP_PAIRS = tuple((origin, task) for i, origin in enumerate(FOUR_FINGERTIPS) for task in FOUR_FINGERTIPS[i + 1 :])
THREE_PINCH_PAIRS = (("thumb_tip", "index_tip"), ("thumb_tip", "middle_tip"))
FOUR_PINCH_PAIRS = tuple(("thumb_tip", tip) for tip in FOUR_FINGERTIPS[1:])
FIVE_PINCH_PAIRS = tuple(("thumb_tip", tip) for tip in FIVE_FINGERTIPS[1:])
FOUR_DIRECTION_PAIRS = tuple((chain[i], chain[i + 1]) for chain in TARGET_CHAINS[:4] for i in range(len(chain) - 1))
FIVE_ROOT_JOINT_PAIRS = (
    ("wrist", "thumb_mcp"),
    ("wrist", "index_pip"),
    ("wrist", "middle_pip"),
    ("wrist", "ring_pip"),
    ("wrist", "pinky_pip"),
)
XARM7_INIT_Q: Dict[str, float] = dict(
    zip(
        ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
        np.deg2rad((-1.8, 7.7, 1.8, 13.7, 180.0, 83.9, -0.2)).tolist(),
    )
)


def correspondences(
    target_link_names: Mapping[str, str], target_pairs: Sequence[Tuple[str, str]], weight: float = 1.0
) -> Dict[Tuple[str, str, str, str], float]:
    """Build weighted correspondences from semantic target pairs."""
    return {(target_link_names[origin], target_link_names[task], origin, task): weight for origin, task in target_pairs}


def wrist_config(
    spec: HandSpec,
    online: HandRetargeterOnlineConfig,
    *,
    fingertip_weight: float = 24.0,
    wrist_weight: float = 0.1,
    direction_weight: float = 10.0,
    wrist_direction_weight: float = 10.0,
    q_smoothness_weight: float = 0.5,
) -> HandRetargeterOnlineConfig:
    """Build wrist-tracking settings while retaining the hand's vector and collision terms.

    Example:
        >>> from robokit.helpers.hand_retargeting.presets import shadow
        >>> config = wrist_config(shadow.spec, shadow.online, fingertip_weight=48.0)
        >>> config.reference_origin, config.task("positions").weight["thumb_tip"], config.solver.num_candidates
        ('wrist', 48.0, [64, 8, 1])
    """
    ctx = HandContext
    names = [name for name in spec.target_names if name in spec.target_link_names]
    pairs = [(names[i], names[j]) for i, j in build_retargeting_pairs(spec, names, include_root=True)]
    return replace(
        online,
        terms=[
            FrameVectorTask(
                targets=ctx.POINTS,
                target_names=ctx.TARGET_NAMES,
                target_links=ctx.TARGET_LINK_NAMES,
                weight={
                    spec.root_target_name: wrist_weight,
                    **dict.fromkeys(spec.contact_target_names, fingertip_weight),
                },
                weight_mode="normalized",
                huber_delta=1.0,
                name="positions",
            ),
            *[deepcopy(term) for term in online.terms if term.name == "vectors"],  # the hand's own
            FrameVectorTask(
                targets=ctx.DIRECTIONS,
                target_names=ctx.TARGET_NAMES,
                weight={
                    (spec.target_link_names[a], spec.target_link_names[b], a, b): (
                        wrist_direction_weight if spec.root_target_name in (a, b) else direction_weight
                    )
                    for a, b in pairs
                },
                weight_mode="normalized",
                direction_only=True,
                direction_metric="cosine",
                name="directions",
            ),
            SmoothnessTask(
                prev_var=ctx.PREVIOUS_STATE,
                warmup_frames=1,
                frame=ctx.FRAME,
                weight=q_smoothness_weight,
                base_weight=0.3 if spec.floating_base else None,
                name="smoothness",
            ),
            ConfigurationTask(rest_q=ctx.REST_Q, weight=0.05, name="rest"),
            PositionLimit(weight=100.0, name="position_limit"),
            *[deepcopy(term) for term in online.terms if term.name == "self_collision"],  # the hand's own
        ],
        reference_origin="wrist",
        output_ema_alpha=1.0,
        base_init_sample_range=(0.0, 0.0, 0.0),
        sampler=RobertsSamplerConfig(scale=0.05, keep_first=True),
        solver=_online_solver((24, 40, 0), (64, 8, 1)),
    )

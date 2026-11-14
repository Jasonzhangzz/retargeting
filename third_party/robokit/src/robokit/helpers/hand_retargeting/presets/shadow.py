"""Shadow Hand retargeting presets."""

from dataclasses import replace

from robokit.helpers.hand_retargeting.config import (
    HandContext,
    HandRetargeterOfflineConfig,
    HandRetargeterOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import (
    FIVE_FINGERTIPS,
    FIVE_PINCH_PAIRS,
    FIVE_ROOT_TIP_PAIRS,
    FIVE_TIP_PAIRS,
    TARGET_CHAINS,
    TARGET_NAMES,
    correspondences,
    wrist_config,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec


ctx = HandContext  # the values the retargeters give their templates


_LINKS = {
    "wrist": "palm",
    "thumb_cmc": "thproximal",
    "thumb_mcp": "thhub",
    "thumb_ip": "thmiddle",
    "thumb_tip": "thtip",
    "index_mcp": "ffproximal",
    "index_pip": "ffmiddle",
    "index_dip": "ffdistal",
    "index_tip": "fftip",
    "middle_mcp": "mfproximal",
    "middle_pip": "mfmiddle",
    "middle_dip": "mfdistal",
    "middle_tip": "mftip",
    "ring_mcp": "rfproximal",
    "ring_pip": "rfmiddle",
    "ring_dip": "rfdistal",
    "ring_tip": "rftip",
    "pinky_mcp": "lfproximal",
    "pinky_pip": "lfmiddle",
    "pinky_dip": "lfdistal",
    "pinky_tip": "lftip",
}
_OFFLINE_LINKS = {
    "wrist": "palm",
    "thumb_mcp": "thmiddle",
    "thumb_tip": "thtip",
    "index_pip": "ffmiddle",
    "index_tip": "fftip",
    "middle_pip": "mfmiddle",
    "middle_tip": "mftip",
    "ring_pip": "rfmiddle",
    "ring_tip": "rftip",
    "pinky_pip": "lfmiddle",
    "pinky_tip": "lftip",
}
_DIRECTION_PAIRS = (
    ("thumb_cmc", "thumb_mcp"),
    ("thumb_ip", "thumb_tip"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_dip"),
    ("index_dip", "index_tip"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_dip"),
    ("middle_dip", "middle_tip"),
    ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_dip"),
    ("ring_dip", "ring_tip"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_dip"),
    ("pinky_dip", "pinky_tip"),
)


spec = HandSpec(
    floating_base=True,
    target_names=TARGET_NAMES,
    target_link_names=_OFFLINE_LINKS,
    target_chains=TARGET_CHAINS,
    root_target_name="wrist",
    contact_target_names=FIVE_FINGERTIPS,
    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="-z"),
)
online = HandRetargeterOnlineConfig(  # tuned on DexYCB subjects 01/02 (RetargetingKit benchmark)
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0),
                **correspondences(_LINKS, FIVE_PINCH_PAIRS, 2.0),
                **correspondences(_LINKS, FIVE_TIP_PAIRS[4:], 2.0 / 15.0),
            },
            weight_mode="cost",
            huber_on_norm=True,
            soft_gate_start_distance=0.1404,
            soft_gate_full_distance=0.0351,
            name="vectors",
        ),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.03, name="smoothness"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=0.001, name="rest"),
        PositionLimit(weight=1.0, name="position_limit"),
    ],
    pinch_correspondences=tuple(correspondences(_LINKS, FIVE_PINCH_PAIRS)),
    target_scale=1.1115,
    output_ema_alpha=0.4,
    solver=PopulationSolverConfig(
        stages=[Stage(10, optimizer=LMOptimizerConfig(lm_lambda=0.01))], cuda_graph_mode="full"
    ),
)
offline = HandRetargeterOfflineConfig(target_scale=1.17)
offline.task("positions").weight = 24.0

online_wrist = wrist_config(spec, online=online, fingertip_weight=24.0)


spec_left = replace(spec, target_link_names=_LINKS)
online_left = HandRetargeterOnlineConfig(
    terms=[
        FrameVectorTask(
            targets=ctx.VECTORS,
            target_names=ctx.TARGET_NAMES,
            soft_gate_targets=ctx.CONTACT_TARGETS,
            weight={
                **correspondences(_LINKS, FIVE_ROOT_TIP_PAIRS, 1.0 / 15.0),
                **correspondences(_LINKS, FIVE_TIP_PAIRS, 0.2),
            },
            weight_mode="cost",
            huber_on_norm=True,
            huber_delta=0.02,
            soft_gate_start_distance=0.048,
            soft_gate_full_distance=0.012,
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
    pinch_correspondences=tuple(correspondences(_LINKS, FIVE_PINCH_PAIRS)),
    target_scale=1.2,
)


__all__ = ["offline", "online", "online_left", "online_wrist", "spec", "spec_left"]

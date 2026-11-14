"""G1 offline mapping with lower pelvis weights, paired with the `g1_offline` optimizer preset."""

import dataclasses
from copy import deepcopy

from robokit.helpers.humanoid_retargeting.config import HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned
from robokit.terms.dense.frame_task import FrameTask


g1_offline_mapping = dataclasses.replace(
    g1_aligned,
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={**g1_aligned.task("target").position_weight, "pelvis": 100},
            orientation_weight={**g1_aligned.task("target").orientation_weight, "pelvis": 20},
            name="target",
        ),
        *deepcopy(g1_aligned.terms[1:]),
    ],
)


__all__ = ["g1_offline_mapping"]

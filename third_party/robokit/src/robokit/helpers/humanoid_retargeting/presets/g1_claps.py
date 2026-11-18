"""G1 aligned preset with a correspondence edge that transfers human wrist-to-wrist motion."""

import dataclasses
from copy import deepcopy

from robokit.helpers.humanoid_retargeting.config import CorrespondenceEdge, HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned
from robokit.terms.dense.frame_vector_task import FrameVectorTask


g1_claps = dataclasses.replace(
    g1_aligned,
    terms=[
        *deepcopy(g1_aligned.terms),
        FrameVectorTask(
            origin_link_indices=HumanoidContext.CORRESPONDENCE_ORIGINS,
            task_link_indices=HumanoidContext.CORRESPONDENCE_LINKS,
            targets=HumanoidContext.VECTORS,
            weight=120.0,
            weight_mode="cost",
            name="correspondence",
        ),
    ],
    correspondence_edges=[
        CorrespondenceEdge(
            origin_link="left_wrist_yaw_link",
            task_link="right_wrist_yaw_link",
            origin_joint="left_wrist",
            task_joint="right_wrist",
        ),
    ],
)


__all__ = ["g1_claps"]

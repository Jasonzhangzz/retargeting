# MANO 五步响应入口

当前抓取方法见 [MANO patch CEM](mano_solver.md) 与 [切向历史模型](mano_patch_history.md)。

文件名 `solve_mano_comfree.py`、`retarget_mano_comfree_campaign.py` 沿用已有命令名称；实际执行的是当前 Kelvin–Voigt 弹簧阻尼与切向历史模型，不能称作上游 ComFree 原公式。候选姿态重新执行 GPU 几何、接触与五步响应，使用末速度、patch 贴合及穿透验收。

`comfree_rollout.py` 仅保留共用物理参数、坐标转换与 GPU 网格质量属性工具。已移除其旧响应求解器及旧目标选择。

# 当前 MANO patch CEM

入口为 `scripts/solve_mano_comfree.py`（固定输入/单分片）与 `scripts/retarget_mano_comfree_campaign.py`（多抓型、多卡）。两者只允许 `patch_consistent_v5`、五步末速度目标和当前 CEM。初值来自 Shadow FK 与 MANO 骨架 bounded LM，也可精修保存的 MANO 姿态；物体模板目标始终来自原 Shadow 接触映射。

`mano_patch_consistent.PatchConsistentAdapter` 使用同一 patch witness 计算目标和验收：手面 patch 在中性模板点周围 10 mm，且不跨解剖区域；实际 patch 表面间距 ≤2 mm，同一物面 witness 距固定物体目标 ≤5 mm。要求整体覆盖 ≥80%，每个必要手指至少一个合格槽。掌部参与整体覆盖。

手物与自穿透均允许 5 mm，超限继续惩罚。实际碰撞接触输入 `SpringHistoryRollout`，每个候选重置切向形变历史，在五步响应内积累并做库仑回映；默认每步 10 ms、四个子步、最多八次阻尼 Newton。Warp 与 Torch 是同一接触模型的执行/数值对照后端，不是两种抓取目标。

默认末线速度 ≤0.1 m/s、角速度 ≤2 rad/s，同时满足既有无接触参考改善条件。该判据表示短时近似稳定性，不能当作所有扰动下的严格力闭合证明。

唯一 CEM 实现在 `retarget/solver/cem.py`：128 候选 ×24 轮，精英加权均值、原变量范围及噪声尺度、逐 pose RNG、联合验收提前停止、可行候选独立保留。`solver.base.solve` 的通用 CEM 接口也委托这一实现；没有第二份旧 CEM 更新代码。LM 仅用于初始化/标定，不是可选的 grasp pose solver。

MANO 解码、FK/IK、patch 查询、穿透、接触响应及优化的数值运算在 GPU；Python 仍负责调度和内核启动，文件编码与固定资产构建可在 CPU。首次加载、图捕获、优化、最终验收和写盘时间分别报告。

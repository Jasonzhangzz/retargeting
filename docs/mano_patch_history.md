# MANO 五步接触与一致 patch 目标

当前方法是 `solver/cem.py` 的 v5 patch CEM。全部默认参数在同文件 `DEFAULTS`，正式命令行不提供物理参数、权重、阈值或更新预算覆盖。

接触采用 Kelvin–Voigt 弹簧阻尼与切向形变历史。默认法向/切向刚度500 N/m、阻尼5 N·s/m、摩擦0.6；每个候选重置历史，在响应内积累、投影到切平面、做库仑回映并在脱离时清零。五个10 ms外步，每步四个隐式子步；GPU阻尼Newton最多八次。

贴合目标与验收使用同一三角形 patch witness：同解剖区域内10 mm patch，表面距离≤2 mm，同一物面点距固定目标≤5 mm。整体至少80%覆盖，且每个必要手指至少一槽；掌部参与整体覆盖。手物和自穿透均允许5 mm。

默认五步末线速度≤0.1 m/s、角速度≤2 rad/s，并满足无接触参考改善条件。目标由 patch 贴合、穿透和末速度三项组成。CEM独立保留联合通过的候选，并对已通过 pose 提前停止。固定模板点距离仅为几何诊断。

GPU Warp 执行与 Torch 数值参考使用同一模型和参数。加载、首次图准备、优化、最终验收与写盘分开计时。短时稳定判据不代表所有扰动下的严格稳定证明。

固定样本重放入口为 `scripts/experiment_mano_patch_history.py`；数值核对为 `audit_mano_patch_execution.py`、`audit_mano_patch_geometry.py`，网页更新为 `build_mano_patch_history_review.py`。这些工具不恢复生产任务。

# 使用 Shadow 可行抓姿初始化 MANO

入口：`scripts/run_shadow_seeded_mano.py`。独立服务器目录：`/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget_mano_shadow_seed_20261104`。使用现有固定版本环境；全流程在CPU上执行，不挂起生产GPU任务。

## 来源与初始化

从原生 Shadow 固定实验的全部800条中，严格选取六方向物理GSR已观察到通过的490条。逐条保存源结果文件、SHA256、原始索引、最终手面接触状态与GSR来源，不从失败样本补齐，不重新采样。

跨模型映射位于 `retarget/env/grasp/shadow_mano_seed.py`。Shadow与MANO关节轴、数量和骨长不同，不能直接复制角度。先按掌面与五指解剖对应关系传递腕部方向和骨骼方向，以MANO真实骨长构造对应点；用有界位置IK拟合20个手指关节。再按同一手指末节映射四个实际接触槽，用有界位置IK调整腕部与手指，使MANO接触位置贴近Shadow成功姿态的接触位置。

两次IK各最多80次函数评估。IK只使用对应点的位置残差，不引入法向、闭合、碰撞、力或姿态惩罚。它是初始化映射，不改原版ADMM目标和更新规则。保留映射误差、边界内有限状态与失败原因；源Shadow可行不被假定为MANO映射后仍可行。

## 求解与独立验收

- 原版Clarabel QP、50轮ADMM×10子步、MuJoCo虚拟力/碰撞推进、对偶、fallback、squeeze/pregrasp保持上一版。
- 手物与自穿透均允许2mm；物体尺度、场景、摩擦和六外力/六纯扭矩稳定测试保持上一版。
- 评估映射初值与ADMM最终状态的投影闭合、距离/法向覆盖、穿透及联合通过；最终另做原生六外力GSR和扩展12工况评测。
- 法向诊断改为v2：在更新后的实际手部接触位置重新查询MANO碰撞三角形，插值顶点法向。保存旧模板法向作审计，不能将新联合指标与旧版沿用模板法向的计数直接混比。闭合QP、距离±2mm、法向30°、覆盖80%均不放宽。
- 主分母为490条成功Shadow来源。映射失败、ADMM拒绝、异常和缺失输出均保留，不筛掉后再报成功率。

## 产物与运行

阶段为 `prepare → transfer → initial_metrics → quality → final_metrics → wrench → timing → report → export`。通过源/代码/资产/输入指纹及阶段收据接续，STOP文件阻止继续派发。原Shadow基线、旧MANO随机初值实验及四卡CEM均不覆盖。

`source_selection.json` 保存来源；`mapping.json` 与 `mapping/cases/*/mapping.json` 保存逐条映射；`initial_diagnostics.json` / `diagnostics.json` 保存前后指标；`native_evaluation.json` / `wrench_evaluation.json` 保存物理结果；`summary.json` / `report.md` 汇总。初始化IK时间单列，不混入ADMM单条wall time。来源已按Shadow成功筛选，不能将新比例与旧800条随机候选直接解释为无偏质量提升。

Viser数据包含Shadow成功源姿态、MANO映射初值和ADMM最终姿态，支持三阶段对照及来源索引/映射误差核对。

本地完成交接仍由 `deliver_native_mano_results.py` 执行，但绑定此轮独立来源和指纹，端口使用8105；旧8104页面保留旧800条实验。来源完成后自动同步并启动新页面。`check_shadow_seed_viewer.cjs` 最多等待1小时，然后实测HTTP、三阶段对比、样本切换、来源/映射误差显示和法向开关，写入 `viewer/browser_validation.json`；失败记录独立保存，不冒称浏览器已验通。

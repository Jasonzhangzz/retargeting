> 后续更正：此前原生网格动态测试漏恢复相邻连杆碰撞排除规则，结果受虚假自碰撞影响，不能作为抓姿不可行的依据。规则已恢复；稳定门槛、虚拟重力和新的独立物理验证见 [更新报告](../rl-staged-gravity/README.md)。

# Primitive grasp refinement pilot

- 原始数据：31 种抓型、1,247 个尺寸点；MANO、Moxian glove、Wuji2 各有对应姿态。glove 使用新上传的29自由度模型，模型指纹见 summary.json；原始拟合验收标记仍为 false。
- 球/圆柱/圆盘检查了1,141条；106条薄板因冻结数据未明确完整盒体尺寸，没有进行相同物体碰撞检查。多数物体中心处于骨架包围范围内，但这不等于无穿透或有效抓握。
- 初始WuJi2原生碰撞网格检查中628条物体穿透超过3mm；glove蒙皮顶点SDF检查中907条超过3mm。只筛出22条较适合继续测试的WuJi2初态。诊断细节在 ../physics-audit-final/。

## RL小实验

以10_Power_Disk第27行开始，圆盘直径77.5mm、厚15mm、质量40g。使用Dex-Supermimic仓库自带PPO路径，128个并行环境，200次更新（819,200条仿真转移），关节残差范围0.25rad。物体无支撑台、无位姿约束，评估含0.5g方向外力脉冲。

默认胶囊/网格混合碰撞几何下，2秒内物体最大位移不超过30mm的比例由83.6%提高至100%；复测平均最终位移由82.7mm降至8.0mm。测试仅覆盖此初态和此物体，并非跨物体泛化测试。

最新版复测的原生网格有限摩擦锥LP检查得到111/128个几何候选，且物体/手部穿透小于3mm；原始姿态静置对照有107/128个通过，提升幅度较小。以上候选不能称为已验证稳定抓姿。

**原生网格动态复放没有通过。**切换完整原生碰撞网格后，10ms步长出现非有限状态；降低到2ms后数值有限，但原始初态及最优几何候选仍会掉落。因此 verified_stable_target_grasps=0，尚未得到可宣称在原生网格动力学中稳定的一批抓姿。

## 文件

- summary.json：数量、模型指纹、几何诊断和结果边界。
- evaluation.json：默认训练几何的对照评估。
- independent-closure-check.json：逐环境原生网格几何LP检查。
- mesh-closure-candidates.npz：111个训练后几何候选（同一初态的多次rollout，非111种独立抓型）。
- learned-evaluation.npz / baseline-evaluation.npz：实际关节角、手腕、物体姿态、控制目标、执行器力矩和保持标记。实际关节角与PD控制目标不应混淆。
- native-mesh-dynamic-evaluation.json、native-best-seed-dynamic-evaluation.json：原生几何动态失败结果。
- model_final.pt：PPO checkpoint；须使用本实验的场景配置，不能直接用默认立方体play命令。
- mesh-comparison.png：MANO、新Moxian蒙皮、WuJi2初态及RL终态网格对照。

本实验是生成精化候选的特权仿真策略，不是仅输入人手历史的部署映射网络。没有对glove建立动力学模型或执行glove的RL。

## 代码修复

运行PPO时发现原库GAE引导值没有detach，导致多批次反向传播错误。已在Dex-Supermimic的runner.py对final_value添加detach，并用2次小样本更新及200次正式更新验证。默认世界模型训练路径未改变。

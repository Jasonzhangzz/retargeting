# 稳定门槛与虚拟重力扰动

默认模式为 `virtual`：补偿物体实际重力，不施加真实扰动，在接触摩擦锥中计算抵抗虚拟重力及其他方向扳手的能力。保留 `disturbance` 和 `staged` 两种独立物理验证模式。当前只有 Wuji2 单个圆盘初态的试验，没有批量训练全部 1,247 组姿态，也没有 glove 的动力学训练。

## 准备与加载

1. 手与物体直接初始化到 retarget 后的相对姿态。准备期间补偿物体重力，并用带阻尼的六维弹簧维持物体位置和朝向，让手指建立接触。
2. 同时检查：关节速度 RMS <0.3 rad/s、每控制步最大关节变化 <0.005 rad；物体线速度 <0.02 m/s、角速度 <0.5 rad/s；至少 3 个接触槽、至少 2 根手指接触；接触槽集合 Jaccard 相似度 ≥0.8；物体穿透 <3 mm。
3. 上述条件连续满足 0.3 秒，且准备至少 0.5 秒，才触发撤去支撑。最多等待 3 秒，超时不强行加载，记为失败。
4. 支撑在 0.5 秒内撤去。`virtual` 始终抵消物体重力，只计算虚拟扳手可行性；`disturbance` 在撤去支撑后施加短时真实力/力矩；`staged` 在撤去支撑时逐渐启用持续重力。

准备阶段的支撑不计入最终抓握能力。虚拟 LP 使用最终实际接触法向预载，每个接触预算最多 1 N，摩擦系数 0.8、8 条摩擦锥边。检查一个重力抵抗扳手和叠加的 12 个正负轴向力/力矩扳手。训练采用有限次非负最小二乘代理评分，最终评估采用 CPU LP。它不是关节力矩耦合约束下的完整可执行性证明，也不能代替真实扰动验证。

## 当前结果

圆盘直径 77.5 mm、厚 15 mm、质量 40 g；原生碰撞网格，2 ms 物理步长，50 Hz 控制。使用 Dex-Supermimic PPO 路径，128 个并行环境、100 次更新，共 409,600 条仿真转移。策略使用特权仿真观测，并增加准备/加载阶段信息，不是人手历史输入的部署网络。

最终复测：

| 检查 | 初始姿态 | 虚拟模式训练后 |
|---|---:|---:|
| 达到稳定门槛并撤去支撑 | 128/128 | 128/128 |
| 零重力下保持两指接触（≥80%评估时间且末帧保持） | 2/128 | 118/128 |
| 最终接触法向力总和的中位数 | 0.00015 N | 0.05789 N |
| 虚拟重力 LP 可行 | 0/128 | 0/128 |
| 全部虚拟扳手 LP 可行 | 0/128 | 0/128 |
| 独立短时真实扰动通过 | 21/128 | 28/128 |

物体重力约 0.3924 N，当前预载仍不足。短时真实扰动每个环境只测试一个方向，不能将 28 个通过环境解释为 28 个全方向力闭合抓姿。所有环境来自同一个初态；GPU 接触求解重复运行会有少量差异。虚拟模式训练末次评估的接触保持数为 121/128，重新加载 checkpoint 复测为 118/128。

成功还要求至少 2 秒撤去支撑后的评估、最大位移 <30 mm、最大旋转 <0.5 rad、最终线速度 <0.03 m/s、角速度 <0.5 rad/s，以及持续接触；虚拟模式额外要求全部虚拟扳手 LP 可行。当前严格虚拟模式成功数为 0，不宣称已经得到稳定力闭合抓姿。

对照的真实扰动直接训练结果为 25/128 → 11/128，未获得提升，保存在 `physical-training-comparison/`；稳定后开启持续重力的原始姿态对照为 0/12，保存在 `continuous-gravity-baseline/`。

## 原生碰撞修复与此前结论更正

此前撤销胶囊碰撞覆盖时只恢复了网格，漏恢复原模型相邻连杆的碰撞排除规则。覆盖层原本会删除这些规则，L1 网格重新启用后导致掌根与手指产生约 14 mm 的虚假穿透和振荡。现在从原始 Wuji2 模型恢复全部排除规则，并通过回归检查。因此上一版报告的原生动态失败不能作为抓姿不可行的证据。本目录的结果已使用修复配置。

## 运行

在服务器 `/home/zhangzhe/Dex-Supermimic` 下运行，`--output` 必须指向尚不存在的新目录。默认启用原生网格和虚拟模式。

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/refine_primitives_staged_gravity.py \
  --pose-file /home/zhangzhe/retarget/output/primitives_robokit_20261115/wuji2/grids/10_Power_Disk.npz \
  --row 27 --num-envs 128 --iterations 100 \
  --gravity-mode virtual --output logs/new-virtual-run
```

加载本目录 checkpoint 进行独立真实扰动评估：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/refine_primitives_staged_gravity.py \
  --pose-file /home/zhangzhe/retarget/output/primitives_robokit_20261115/wuji2/grids/10_Power_Disk.npz \
  --row 27 --num-envs 128 --gravity-mode disturbance --evaluate-only \
  --actor-checkpoint /home/zhangzhe/retarget/output/primitives_robokit_20261115/rl-staged-gravity/model_final.pt \
  --output logs/new-physical-evaluation
```

将模式改为 `staged` 即可在稳定后逐渐启用持续重力。`--evaluate-only` 不带 checkpoint 时测试原始 retarget 控制目标。

## 文件与验证

- `summary.json`：最终复测、力预算、适用范围。
- `virtual-evaluation/`：纯虚拟扳手最终评估，包含逐方向 LP 标记、接触力、实际关节角、控制目标及准备阶段轨迹。
- `physical-validation/`：同一虚拟训练策略的独立真实扰动验证及原生网格几何检查。
- `model_final.pt`：虚拟模式训练的最终 PPO checkpoint。
- `sources/`：脚本和回归检查快照。另已部署到远端 retarget/scripts 与 Dex-Supermimic/experiments。
- `sha256.json`：文件校验。

5 项检查通过：连续稳定门槛与撤去支撑、丢失接触与超时、持续重力加载、原生相邻连杆排除规则恢复、虚拟模式无真实扰动及 LP 能区分承重与力矩抵抗。并完成 128 环境训练、checkpoint 重载复测与独立真实扰动测试，状态均为有限值。

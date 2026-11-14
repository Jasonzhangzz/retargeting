# Robokit CARD 后端

`moxian_web/robokit_card.py` 在本项目中扩展 Robokit 的 `Task`、`GaussianSampler` 与 `PopulationSolver`；不修改环境内的 Robokit 安装，不使用模拟器或轨迹优化器。机器人沿用现有原生 MJCF 和关节名称映射。

默认 `CardSettings(cem_backend='robokit')` 的流程：

1. 当前 MANO 21 节点 → Robokit 初解。Robokit 从自己的上一帧关节解热启动，腕部初值来自当前 MANO；不读取 CEM 输出。完全相同的float32节点输入复用对应的Robokit解，避免对重复问题反复优化引入微小数值漂移；任意输入变化仍重新求解。`initializer.reused_identical_source` 明确标记复用。断流超过1秒、标定更换、源关节相对腕部跳变超过40 mm时清空初值历史及此缓存。
2. 填入固定16个语义区域的源 CoP、方向、相对压力和上一帧平滑参考。初解与源接触目标不被替换。
3. Warp 计算接触位置、方向对齐、材料点防滑、全手姿态、关节/基座平滑、源动作一致性、胶囊间距和物体 SDF 代价。接触不会关闭该指的姿态项。
4. Robokit 全图捕获执行16轮，每轮1024个候选。5%精英的标准差与温度0.1决定 softmax；噪声固定退火到0.2倍，不拟合新的方差。旋转在本帧初解的轴角图内采样和加权。已有有效解时，默认896个候选沿当前Robokit初解的CEM分布采样，另128个在当前源动作提案附近采样，标准差为对应轮次的0.15倍。所有候选使用相同的当前Robokit搜索边界、代价和端点验收。
5. 保存每轮的全部候选及最后均值。上一帧有效解，以及按当前源腕部刚性运动搬移的版本，若在当前 Robokit 搜索范围内，按当前帧完整代价重新计分，一并参与候选比较。搬移版本保持关节角，仅随源腕部平移和旋转，避免把整手微小位移转化为指关节抖动。保留低代价、初解、时间连续候选与早期候选，最多256个；先作原生网格粗检，再按代价排序以8个一批稠密验收。历史候选同样必须通过当前碰撞验收；未找到可行解时显示当前帧有限候选并标记不可行，不使用未经检查的历史姿态回退。
6. 使用同一 Warp 接触投影结果导出机器人面编号、重心坐标、接触位置和法向，再进行最终原生端点复核。碰撞不检查帧间扫掠轨迹。

胶囊来自 Robokit `RobotSpec.local_link_capsule_endpoints` 和 `link_capsule_radii`；接触曲面仍用语义指腹的原生三角面，通过 BVH 最近点实现式(2)。简化碰撞几何不承担逐税元接触注册的作用。

## 论文对应与实时适配

代价与采样参数依据 [CARD 式(2)、(3)、附录D](https://arxiv.org/html/2610.36539v1)。相对压力仅影响接触优先级；默认方向项不是显式力控制。

保留的项目约束：主CEM分布从本帧Robokit初解开始，所有候选的精化边界均为该初解附近±0.25 rad和20 mm；胶囊间距仅作软代价。推测物体凸包仅作软约束，实测物体网格保留3 mm硬门槛，原生手部自穿透0.5 mm硬验收。没有实现论文的闭式初值、历史混合或实时零相位滤波，因此不能称为作者实现的逐行复现。

`cem_backend='torch'` 为独立参考实现。`coulomb_force` 或 `experimental_self_contact=True` 明确使用参考后端，API 的 `solver.backend_reason` 会说明，不静默忽略实验项。

源 SID 到个性化 MANO 的实测注册状态仍独立报告；提速不代表源接触注册或真实抓握质量已通过验收。

## 计时与验证

- `solver.timing_ms`：整帧分项，CUDA同步后的墙钟耗时。
- `solver.backend_timing_ms`：新后端准备、图执行、原生端点验收；是 `timing_ms.cem` 的子项，不能重复相加。
- `solver.cold_start`：构建、编译和捕获图的首帧，性能统计应与稳态分开。
- 固定1024×16；最终均值另计一次代价，符合当前搜索范围的时间连续候选分别计分，`evaluations=16385 + len(temporal_candidate_costs)`，最多16388。

`E_style` 的指尖、指腹法向和骨段方向三项，以及 `E_smooth` 的关节、腕部平移和旋转三项均进入搜索。默认 `contact_style_weight=1`，全手保留姿态项，符合式(10)显式全指求和；`contact_style_weight=0` 可复现旧版接触指关闭style的选择。接触位置、方向和相对压力目标不变。

`card_wuji2_v9_source_motion` 新增在线运动一致性项：

```
expected_joint = previous_CEM_joint + current_Robokit_joint - previous_Robokit_joint
E_motion = wmotion_joint² * ||joint - expected_joint||²
```

期望值按原生关节限位截断，默认 `wmotion_joint=300`，单位rad。只在有有效参考且源动作连续时启用，标定变化、断流和大幅源姿态跳变时重置。真实Robokit关节运动不受该项惩罚，改变的是CEM修正量；因此没有将手势锁定在上一帧。该期望姿态还作为当前帧候选重新计分并验收。原 `wtheta=0.1` 保留。`motion_joint` 分项进入 `smooth` 组；`joint_motion_reference_rad / joint_motion_reference / source_motion_samples_per_round` 报告实际参考和局部采样数。

运动一致性项和局部候选是本项目的在线防抖扩展，不能宣称为论文原始代价或采样的逐行复现。默认没有输出EMA、未来帧滤波、动作分类或对指闭合约束；新解仍需通过原生0.5 mm自穿透验收。`wmotion_joint=0, contact_style_weight=0, motion_sample_fraction=0` 是旧策略对照。

数值回归检查独立参考代价、精英softmax/退火、源材料坐标、物体SDF、原生穿透门槛、当前有限候选及worker接口：

```bash
.venv-contact/bin/python -m pytest -q tests/mano/test_card_retarget.py tests/mano/test_contact_force.py tests/mano/test_wuji_pinch.py tests/test_moxian_wuji_bridge.py tests/test_moxian_wuji_view.py
```

保存的真实帧对照（输出目录必须新建）：

```bash
.venv-contact/bin/python scripts/benchmark_moxian_robokit_card.py \
  --source output/card_force_ablation_20261109/pinch-source.json \
           output/card_force_ablation_20261109/bottle_cap-source.json \
  --output output/robokit_card_comparison
```

两组输入标签只用于分组评价。每组前半段仅估计相对压力归一化，后半段选帧；各帧独立重置。分别预热后端，再记录性能、源CoP误差、方向和原生穿透。并行实时进程的资源竞争会计入墙钟时间，须同时查看部署后的在线测量。

连续姿态抖动、接触切换及实际动作跟随回放：

```bash
.venv-contact/bin/python scripts/benchmark_moxian_pose_stability.py \
  --source output/card_force_ablation_20261109/pinch-source.json \
           output/card_force_ablation_20261109/bottle_cap-source.json \
  --output output/wuji_pose_stability_comparison
```

加 `--recorded` 重放所有保存的真实帧；默认对固定真实帧施加可重复的微小手指旋转、动作变化、接触开关及50 mm整手位移。标签不输入求解器，统计包含阶段边界的首帧变化；同时报告MANO和Wuji2骨段、指尖变化、接触误差与穿透。

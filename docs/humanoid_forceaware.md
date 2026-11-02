# DexForge Force-Aware 搬箱

当前实现已完成原版流程的 G1 适配和短测；30 步梯度尚未获得认证，默认完整入口会在执行前停止。具体证据见 [验证记录](humanoid_forceaware_validation.md)。

新入口使用 DexForge 提交 `fa6d671948ef03d18514062748b94e445b63cd49` 的正式 FA 求解与执行流程。输入已经包含 G1 关节角，因此跳过 Contact-Aware 和 IK。该入口不运行 ADMM、接触点 CEM、独立接触力采样或 IPOPT。旧模式可通过 `configs/humanoid_move_largebox_scsp.yaml` 显式选择。

## 运行

```bash
bash scripts/setup_forceaware.sh
.venv-forceaware/bin/python run.py --config configs/humanoid_move_largebox_forceaware.yaml
```

根目录 `task.yaml` 也选择此分支。Python 3.10、NumPy 1.26.4、Torch 2.1.2、Warp 1.15.0、MuJoCo 3.6.0 使用独立 `.venv-forceaware`；其他模式继续使用原环境。版本不符会在加载物理模型前报错。

默认先运行真实 native 动力学的单步与 30 步梯度检查；没有通过就不会进入完整搬箱测试。可单独运行并查看报告：

```bash
.venv-forceaware/bin/python scripts/check_dexforge_gradients.py \
  --config configs/humanoid_move_largebox_forceaware.yaml \
  --output output/dexforge_gradient_validation
```

完整参考持续约 4.8 秒，箱体参考平移约 0.6945 m。初始箱体地面侵入以及参考中的原始 root/箱体位姿均保留；不会通过修改输入让验收通过。短测可设置 `environment.frames`，表示源参考的时间间隔数；短测不构成完整任务成功。

## 求解与梯度

`retarget/solver/dexforge.py` 通过目标函数工厂适配 G1。上游正式 `workflow.py` 将 `ServoPriorOptimizer` 作为 `WindowOptimizer` 导入，新类继承这个正式入口类，唯一覆盖的优化器方法是 `_make_objective`。多起点、速度先验初始化、并行 Adam、native adjoint、有限候选筛选、最佳候选恢复和 warm shift 均来自上游原文件。

默认 64 个起点、40 次 Adam、5 个 knots。预测步长 0.01 s、knot 间隔 0.06 s，预测 30 步，每次执行 3 步。直接执行 `Plan.control_sequence`，使用上游 `WarpActionExecutor`；不经过旧 SPIDER 的 0.05 缩放、整数参考插值或初始化预推进。

`retarget/dynamics/dexforge_loss.py` 使用同一 native forward 状态构造 root 位姿、configuration 和物体位姿跟踪的 Warp Tape，并将完整 43 维 qpos cotangent（含四元数四维梯度）交给上游反传。native adjoint 包含状态相关质量矩阵、偏置力、执行器、FK、碰撞、接触行、接触响应和积分；没有每次重算后 detach 的 MJWarp 参数快照。

跟踪项沿用上游平方公式，权重先乘误差再平方；长度换算为毫米，loss 梯度统一缩放 `1e-6`。默认 root 位置/旋转为 `10/3`，29 个关节统一为 `0.03`，物体位置/旋转为 `1/0.3`，物体参考缩放为 `100`，终端为 `2`。这些配置位于 `solver.dexforge`。保留上游初态正则、控制/状态平滑、物理穿透惩罚；全部 29 个关节采用角度单位，避免前六个腿关节被当作平移/旋转手腕。

## 输入、碰撞与初态

保留原始 145 帧 NPZ。root 为 `qpos[:,0:7]`，关节为 `qpos[:,7:36]`，物体为 `qpos[:,36:43]`；关节索引按 actuator transmission 验证，控制仍为 29 维。内部为 MuJoCo WXYZ，仅上游参考接口转换为其规定的物体顺序。参考采样、四元数插值、时间网格和 warm shift 使用官方实现。

源 NPZ 的 qvel 全零，初速度按上游方式由相邻参考位姿经过 `mj_differentiatePos` 得到，并在输出中记录来源。首窗口允许上游有界关节初态修正（每关节上限 0.24 rad），root 和物体初态固定；后续窗口从实际执行状态开始。

四个手心/脚心 site 仅用于观测。原两个手部半径 0.05 m 的球、每脚四个半径 0.005 m 的支撑球按原局部变换和半径转为 source clouds；原质量、惯量、材料和 pair 设置保留。箱体转为保留局部变换的封闭 12 三角形表面，用 mesh hard 接触；地面保留。没有数据接触标签，因此四个 guidance 槽均 inactive，不生成 CA anchors 或吸附约束，也不启用手指专用 capsule 项。

## 来源与结果解释

`third_party/DexForge/UPSTREAM.json` 记录提交和逐文件 SHA-256。加载时验证上游原文件未修改；许可证见该目录及其 bundled ComFree 许可文件。上游求解器与反传核心保持原样，G1 适配在 `retarget` 下。

输出包含实际 FA 执行的轨迹、逐窗口日志、独立验收指标、梯度报告和来源记录。箱体最大位置误差阈值 0.1 m、旋转误差 0.3 rad、root 高度亏损 0.18 m、躯干相对参考倾角 60°。完整执行且全部指标通过才记为任务成功。独立 MuJoCo 控制回放单独标记和评估，不与原版 FA 执行混为一项。单窗口跑通、成本降低或单元测试通过不能证明成功搬箱。

# Force-Aware 实现验证（2026-10-31）

## 后续推箱效果测试

已按用户要求运行 2.5 秒主要推箱阶段，保持 64 起点、40 次 Adam 和 30 步预测预算，共 84 窗口，耗时约 237 秒。该测试显式设为诊断前缀，未通过的完整梯度门槛没有被改成通过。

结果不合格：native 箱体推进约 0.330 m，参考为 0.694 m；1.11 s 后失去手箱接触，1.46 s 箱体误差超过 0.1 m，2.15 s 触发平衡失败，2.5 s 时机器人已倒下。箱体最大误差 0.369 m，root 最大高度亏损 0.309 m。规划与 native 执行 qpos 一致。独立 MuJoCo 控制回放更差，箱体最大误差 1.135 m。

参见 [三栏视频](../output/humanoid_forceaware_push_test_2p5s/video_comparison.mp4)、[详细效果记录](../output/humanoid_forceaware_push_test_2p5s/effect_report.md) 与[指标 JSON](../output/humanoid_forceaware_push_test_2p5s/effect_evaluation.json)。复现配置为 `configs/humanoid_move_largebox_forceaware_diagnostic.yaml`。以下为此前的实现与梯度验证记录。

本次完成了原版 FA 的 G1 适配与短测，完整搬箱验收尚未通过。默认完整入口会在梯度检查后停止，保留失败报告。

## 源码与环境

- 固定 DexForge `fa6d671948ef03d18514062748b94e445b63cd49`，139 个引入文件逐个 SHA-256 一致；求解器、multistart、Adam、候选恢复和 native adjoint 均未修改。
- Python 3.10.20、NumPy 1.26.4、SciPy 1.13.1、Torch 2.1.2、Warp 1.15.0、MuJoCo 3.6.0，独立 `.venv-forceaware`，GPU 为 RTX 4090。
- 四个掌心/脚心是观测点；两个手球、每脚四个球、箱体表面和地面参与物理接触。13 种原 pair 及质量、惯量、servo 参数、摩擦、solref、solimp 已核对。

## 梯度门槛

最终报告：[gradient_validation.json](../output/humanoid_move_largebox_forceaware/gradient_validation.json)。该报告也保存在独立检查目录 `output/dexforge_gradient_validation/`。

| 检查 | 结果 |
| --- | --- |
| 单步 native 完整 qpos/qvel/control 链，3 个方向 | 通过，1% 门槛 |
| 实际双手与箱体接触时，control → 箱体 pose/velocity | 通过，1% 门槛 |
| 30 步 native 完整状态链 | 未认证：没有稳定且保持接触特征一致的差分区间 |
| 30 步实际 G1 FA 目标、正则与 raw 参数链 | 2/3 方向通过；整体未认证 |

手箱检查从原参考第 12 帧（约 0.4 s）复制状态，两手各有一个实际 native 接触，间隙约为 -0.0226/-0.0328 m。只扰动真实 actuator 控制，不扰动物体输入；不使用 IK，不修复原始姿态，不将这个检查状态替换成控制器初态。

30 步检查使用八种差分步长（`1e-2` 至 `3e-6`）。较大的扰动改变中途接触；较小扰动即使保持几何特征，方向差分也未形成满足 5% 要求的稳定区间。实际目标一个方向的 AD 为约 62.56，而差分随步长变化为约 61.49、67.64、62.33、20.60，不能选择其中一个数值作为通过证据。这一结果不能证明完整 30 步梯度已正确，也尚未将问题归因于某个上游导数或仅数值误差。

默认命令已经验证会写入 `GradientValidationError` 状态并在执行前停止；该结果目录没有 `trajectory.npz`。没有运行完整 4.8 s 搬箱，未宣称任务成功。完整入口不能通过设置足够大的 `environment.frames` 来绕过梯度门槛。

## 原预算短测

独立诊断配置显式使用 `environment.frames: 1` 和 `gradient_check: false`。按官方时间网格，一个源时间间隔需要 4 个物理步，执行为 3+1 两个窗口，共 0.04 s。该例仅验证接口和执行一致性，输出 `full_clip_completed: false`、`task_success: false`。

两个窗口均为 64 起点、40 次 Adam、5 knots、30 步预测。求解分别耗时约 3.20/2.99 s，64 个候选均有效。规划和 native 执行的 qpos 最大差为 0。首窗口最大关节初态修正 0.01896 rad，小于 0.24 rad；root/物体未修改，第二窗口初态修正为 0。

| 指标 | native FA 执行 | 独立 MuJoCo 控制回放 |
| --- | --- | --- |
| 完整时长 | 否，0.04 s | 否，0.04 s |
| 箱体最大位置误差 | 0.02330 m | 0.01899 m |
| 箱体最大旋转误差 | 0.09224 rad | 0.10111 rad |
| 数值重置 | 无 | 无 |
| 完整任务成功 | 否 | 否 |

结果见 [native summary](../output/humanoid_forceaware_native_prefix/summary.json)、[独立回放](../output/humanoid_forceaware_native_prefix/independent_mujoco/summary.json)。每窗口保存完整预测 qpos/qvel、46 个物理接触槽的激活/间隙/位置/法向/反力、约束行力、四个观测点、初态修正、分项 tracking cost 和候选有效性。接触数据标记为预测接触；执行前缀与预测一致。

## 测试

```bash
.venv-forceaware/bin/python -m pytest -q tests/test_dexforge_*.py
```

47 项通过，覆盖原版方法继承、source pin、完整 qpos/四元数 loss cotangent、接触间隙梯度、控制尺度、参考采样、原几何/pair、输入隔离、初态限制、warm/execution 接口、独立回放和梯度门槛。

旧环境完整回归为 251 通过、5 跳过、4 项此前存在的失败：两项 SPIDER/ComFree AST 清单不匹配，两项缺少 `data/grasp/assets/wuji/right.xml`。原 `cost.py` 和 `comfree/kernels.py` 与 HEAD 保持逐字节相同，未修改哈希掩盖这些失败。

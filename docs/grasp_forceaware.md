# WUJI2 primitive 抓取的 DexForge ForceAware 接入

此入口把 ContactAware（CA）手姿与接触点交给 DexForge 原版 ForceAware（FA）滚动优化。后续修复及当前运行入口见 [修复记录](grasp_forceaware_repair.md)：新增显式 PD 积分适配、全手内含球云、active-only CA 重解、准确静态预载和独立 mesh 接触力审计。下方保留最初未修复模型的失败实验；这些历史结果不能代表修复后的数值有限性，也不能作为成功抓取证据。

## 原版流程与适配范围

上游固定为 [wmz1226/DexForge 的 `fa6d671948ef03d18514062748b94e445b63cd49`](https://github.com/wmz1226/DexForge/tree/fa6d671948ef03d18514062748b94e445b63cd49)。入口直接调用原版 `forceaware.workflow.rolling_mpc_retarget`，使用原版 `ServoPriorOptimizer`、`MpcObjective`、多起点 Adam、候选有限性筛选、最佳候选恢复、native adjoint、控制插值、warm shift、执行器及输出流程。没有派生优化器或替换目标函数；没有采用 humanoid 适配中关闭 guidance 的配置。

FA 原有的物体轨迹跟踪、手姿先验、初态正则、接触 robustness、marked-query、手指 capsule 自碰撞、穿透和控制/状态平滑均保留。首窗口继续允许原版有界初态修正：腕部平移上限 0.05 m、旋转上限 0.16 rad、手指上限 0.24 rad；物体初态固定，后续窗口从实际执行的 qpos/qvel 接续。执行的是原版生成的 PD 控制序列，每个物理步重新计算碰撞和接触响应。

初始接入的本地适配如下；后续新增模块及运行时积分扩展见修复记录：

- **WUJI2 拓扑**：原版 hand specs 没有 `wuji_right`，因此运行时向原 `HandSpec` 注册右手掌和五个 tip site，并同步消费者的 descriptor。上游源码文件保持原样，注册信息写入 `reference_manifest.json`。
- **语义 CA 输入**：模板只提供 semantic patch 与 seed。输入必须显式标记 `hand_reference_source=dexforge_contactaware`，CA 成功、上游提交一致、所有 required fingers 已验收，并具有显式 active mask。当前三个尺寸各有五个 active slots。FA 读取的是 CA 的接触点优化和手部 IK 结果，不回退到模板关节角或旧 MPPI 最终手姿。
- **参考序列**：26 维手参考为腕部 6 个标量坐标与 20 个手指关节；场景总维度为 `nq=33, nv=32, nu=26`。物体目标始终静止，hold 初速度为零。使用原版 `load_sequence`、`build_index`、`initial_state` 和 `make_window`。桥接按名字核对并映射 body/joint，验证编译后的局部 body frame、joint axis/pivot/type/reference，避免 semantic material points 因场景转换而变帧。
- **物理场景**：保留 WUJI2 的质量、惯量、重力、关节及执行器参数、限力和接触材料。手部全部表面采样（包括刚性 pad/mount）转换为正半径 tangent GS：`center = surface_point - radius * outward_normal`，默认半径 1 mm；物体使用原 primitive 的封闭三角网格。这是有记录的球云近似，不是与原网格碰撞完全等价。物理球云不按 semantic mask 裁剪，因此 active guidance 之外的表面也可能产生真实接触。为满足原版每 guided body 唯一输出的映射，native GS 仅生成手物接触，手内自碰撞使用原版 capsule cost；普通 mesh 场景保留原碰撞掩码。

## 环境、配置与运行

使用隔离的 Python 3.10 环境和 CUDA。数值版本由入口严格检查：NumPy 1.26.4、SciPy 1.13.1、Torch 2.1.2、Warp 1.15.0、MuJoCo 3.6.0。

```bash
bash scripts/setup_forceaware.sh
.venv-forceaware/bin/python run.py --config configs/grasp_contactaware_forceaware.yaml
```

[默认配置](../configs/grasp_contactaware_forceaware.yaml) 从 `output/grasp_contactaware_mppi` 读取已验收 CA 产物，先选中间尺寸圆柱，目标保持 0.4 s。`case_indices` 按已发现输入的排序选择样本；移除此项可处理全部输入。`solver.dexforge: {}` 保留官方默认配置：5 个 knots、40 次 Adam、64 个起点，预测/执行步长均为 0.01 s，action 间隔 0.03 s，knot 间隔 0.06 s，reference 间隔 1/30 s。手姿权重为腕平移 10、腕旋转 3、手指 0.03；其余 loss、bounds 和 smoothing 同官方 YAML。

入口先进行恒定 CA 控制的 native baseline，并保存真实接触载荷与控制梯度诊断；即使 baseline 失败，仍让原版多起点候选筛选判定优化是否可行。缩短 horizon、迭代数、起点数或修改时间网格只能作为显式实验覆盖，必须在结果中记录，不能称为官方默认实验。实际使用的完整配置写入输出目录的 `forceaware_model.yaml`。

上游来源和逐文件 SHA-256 位于 [UPSTREAM.json](../third_party/DexForge/UPSTREAM.json)。`activate_upstream()` 只加载本仓库固定副本，`verify_upstream()` 检查固定提交及 manifest 中全部文件；缺失或字节变化都会报错。可单独复核：

```bash
.venv-forceaware/bin/python -c 'from retarget.dynamics.dexforge_upstream import verify_upstream, require_environment; print(verify_upstream()["commit"]); print(require_environment())'
```

输出另外保存 `upstream_verification.json`、scene/reference manifests 和实际配置。源码 hash 通过只证明加载的核心未改变，不证明物理稳定或梯度正确。

## 默认 10 ms 实测：失败

中间尺寸圆柱使用真实 CA qref、原质量/惯量、原执行器增益/速度项/限力和原 native core。记录如下：

| 实验 | 结果 | 证据 |
| --- | --- | --- |
| 恒定 CA qref 控制，10 ms × 10 步 | 第 10 个 post-step 出现 NaN/非有限状态；五指均未全程保持正法向载荷 | [validation.json](../output/grasp_forceaware_validation/validation.json) |
| 原版默认 5 knots / 40 iterations / 64 starts，独立直接求解探针 | 64/64 候选在 iteration 0 出现 `loss_nonfinite`，`valid=0/64`；原版抛出 `all multistart candidates are non-finite`，未生成成功轨迹 | [solve.log](../output/grasp_forceaware_default_probe/solve.log) |
| native 控制链梯度诊断 | 单步未获得稳定有限差分平台；10 步原链非有限，无法认证导数 | 同一 [validation.json](../output/grasp_forceaware_validation/validation.json) |

当前模型的 26 个执行器均有 position gain 和 velocity bias，关节 `dof_damping` 为零。原 native 核心在当前速度上显式计算 actuator force，形式为 `gain * ctrl + bias0 + bias1 * length + bias2 * velocity`，然后执行 actuator force limit；积分采用 semi-implicit Euler。它对 joint damping 的处理不等同于 MuJoCo `implicitfast` 对执行器速度相关力的处理，设置 XML 的 integrator 名称也不能把该 native step 变成 `implicitfast`。这一离散化与 WUJI2 伺服参数的组合是当前需要处理的数值问题；球云近似、实际接触分配和抓取载荷仍需分别验证。

不能为稳定性直接把 actuator 的速度项搬到 joint damping，同时声称物理模型等价。有执行器限力时，`clip(kp * error - kv * velocity)` 与 `clip(kp * error) - kv * velocity` 不同；搬移后阻尼位于限力之外，也会改变 native 的积分方式和接触耦合。当前适配保留这些物理参数，没有据此改写核心或伺服模型。

## 结果判定

`solver_completed` 仅在请求时长的全部控制步都实际执行、qpos/qvel/control/误差有限且轨迹长度一致时为真。启动原版优化器、单窗口完成、cost 降低或单元测试通过都不足以满足这一条件。

稳定抓取还要独立满足物体最大漂移 2 mm、旋转 5°、穿透 2 mm，以及五个关键手指全程具有正的真实法向接触载荷。CA 的五指几何贴合和 active guidance 只描述初始几何意图，不保证载荷保持，也不保证物理 solver 每个 body 选中的实际最近点仍位于原 semantic patch。

原版输出流程会将保存的控制序列交给独立 MuJoCo mesh 模型回放，默认回放积分步长为 0.002 s，并生成独立 `replay.json`/视频。该回放与 native rollout 分别报告；native 指标通过不能代替独立回放通过。当前 runner 的 `task_success` 同时要求 native hold、五指载荷保持与独立 MuJoCo hold 通过；native 控制链梯度检查也单独报告，不能标作完整 FA 目标函数梯度认证。上述默认实验没有得到稳定成功结果。

控制间隔小于 2 ms 时，原版 workflow 会在完整优化和 native 输出保存后，因默认回放网格不整除而抛出错误。入口仅针对这一确切错误记录 `original_replay_grid_limitation.json`，随后调用原版 `load_rollout/replay_mujoco/replay_errors` 和 renderer，以 `min(0.002, mpc_dt)` 显式设置独立回放步长。报告标记 `independent_replay_time_grid_adapter=true`、`original_default_replay_supported=false`。优化器、native 执行、已保存控制和模型参数均不改变；不宣称这与默认 2 ms 回放数值等价。

## 原控制器的小步长实验

[实验配置](../configs/grasp_contactaware_forceaware_smallstep.yaml)：保留原模型全部动力参数、原版 5 knots / 40 iterations，将 multistart 改为 4，mpc/exec dt 改为 0.0001 s，action/knot 间隔改为 0.003/0.006 s。每个尺寸实际执行 300 个物理步、10 个重规划窗口，共 **0.03 s**；这只是短测试，不能作为 0.4 s 稳定保持的证据。

| 圆柱半径 | native 最大漂移 | 最大穿透 | 不能全程保持正载荷的手指 |
| --- | ---: | ---: | --- |
| 26.25 mm | 25.096 mm | 3.013 mm | 全部五指 |
| 35 mm | 1.973 mm | 3.308 mm | 中指、小指 |
| 43.75 mm | 1.659 mm | 3.305 mm | 小指 |

三例 `solver_completed=true`，但 `task_success=false`。逐物理步独立 native 接触审计的 qpos/qvel 与原版输出完全一致；所有真实法向力来自完整手部物理接触行。接触阈值为 1e-6 N，不能把 active guidance 当作承载。梯度检查均未获得认证，也没有据此替换原版优化算法。

完整 [summary.json](../output/grasp_contactaware_forceaware_smallstep/summary.json)、[诊断图 PNG](../output/grasp_contactaware_forceaware_smallstep/forceaware_diagnostics.png) / [PDF](../output/grasp_contactaware_forceaware_smallstep/forceaware_diagnostics.pdf) 包含 native 与独立 mesh 回放及五指载荷。

35 mm 的另一组 3 knots / 10 iterations / 8 starts 短探针中，五指 30 ms 内均承载，仍漂移 2.671 mm。其平均竖直接触力只有 1.264 N，低于重力 1.962 N，293/300 步支撑不足；净冲量与终态物体速度吻合。[力平衡诊断](../output/grasp_forceaware_smallstep_probe/contact_audit/force_balance_analysis.json) 也验证力解码与 generalized constraint force 一致。五指同时接触不等于静力平衡。

当前固定版本的 `contact_robustness` 是接触深度惩罚；wrench/GWS 指标只在输出阶段计算，没有作为显式力平衡约束加入优化。小尺寸的约 1e9 cost 主要与腕旋转角加速度振荡同量级，而非 wrench 矩阵退化产生的罚项；从执行轨迹重构的 state smoothing 中，腕旋转加速度占约 98.6%。WUJI2 descriptor 得到正确的五条四节手指链。[cost_diagnosis.json](../output/grasp_contactaware_forceaware_smallstep/cost_diagnosis.json) 明确区分执行轨迹分析与未保存的预测 horizon 分解。

## 独立虚拟腕部控制器实验

[可复现配置](../configs/grasp_contactaware_forceaware_wrist_pd.yaml) 仅将三个虚拟腕旋转 servo 从 kp=100、kv=10 调整为 kp=10、kv=0.1，保留质量、惯量、手指执行器与全部限力，显式标记 `controller_changed=true`。这不是原控制器等价重写，也不混入上述原参数结果。FA 核心、损失、bounds 和 smoothing 保持原样。

该变体在 10 ms 仍出现非有限状态；2 ms 的 0.4 s 恒定 CA 控制 baseline 数值有限但物体漂移 408.74 mm。[variant manifest](../output/grasp_forceaware_wrist_pd_variant/variant_manifest.json) 保留完整差异和对照。

同一 35 mm 样本进一步直接执行原版 FA：预测窗口 0.3 s（5 knots、knot 间隔 0.06 s），40 iterations、8 starts，action 间隔 0.03 s，mpc/exec dt=0.002 s，总时长 0.4 s。14 个重规划窗口均有 8/8 有限候选，原版完整 workflow（包括默认 2 ms 独立 MuJoCo replay）正常结束；但物体最大漂移 **237.339 mm**、旋转 **10.103°**、穿透 **2.371 mm**，五指都曾失去载荷。独立 mesh replay 最大漂移 **282.641 mm**。因此仍是 `solver_completed=true`、`task_success=false`；不是成功的稳定抓取。

结果和独立逐步接触审计见 [summary.json](../output/grasp_forceaware_wrist_pd_optimized/summary.json)、[原版求解日志](../output/grasp_forceaware_wrist_pd_optimized/solve.log)。该控制器实验与原控制器三尺寸短测试分开保存，不能混合比较为同一组物理参数。控制梯度检查未获认证。

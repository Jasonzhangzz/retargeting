# WUJI2 静态抓取的 ForceAware 修复记录

本页记录 WUJI2 ForceAware 的输入、几何、积分和诊断修复。当前复测入口为 [fine_samples.yaml](../configs/grasp_contactaware_forceaware_fine_samples.yaml)：保留原 FA 损失/优化器与官方手指 PD，显式使用本地 PD 积分扩展及原 executor 的 2 ms 子步。后文分别保留历史失败、修复后的 10 ms 对照和 2 ms 结果；只在实际通过的测试时长内报告成功。

## 保留的上游与适配边界

DexForge 固定于 `fa6d671948ef03d18514062748b94e445b63cd49`。[来源清单](../third_party/DexForge/UPSTREAM.json) 中的 **139 个文件 SHA-256 保持不变**，由 [dexforge_upstream.py](../retarget/dynamics/dexforge_upstream.py) 校验。仍使用原版 `rolling_mpc_retarget`、`ServoPriorOptimizer`、`MpcObjective`、多起点 Adam、候选筛选、控制插值、warm shift 和执行流程。

FA loss、权重、bounds、smoothing 和优化更新算法没有替换。原版 `contact_robustness` 是接触深度惩罚；GWS/wrench 指标在输出阶段计算，不能把它当作优化中已有的显式静力平衡约束。早期约 `1e9` 的 cost 与腕部角加速度振荡同量级，详见 [cost_diagnosis.json](../output/grasp_contactaware_forceaware_smallstep/cost_diagnosis.json)；该分析明确是执行轨迹重构，并非预测 horizon 的逐项分解。

本地新增能力必须分别记录：积分扩展、梯度诊断、静态控制预紧、可选固定初态，以及可选控制器增益变体。源码哈希通过不代表运行时完全等同原 native Euler，也不代表抓取或全部梯度已经通过验证。

## 保留限力的 PD 隐式积分扩展

原 native 在当前速度上计算 PD 力并限幅，但积分质量矩阵只对 `dof_damping` 加隐式项。WUJI2 的速度阻尼主要在 actuator bias 中，`dof_damping=0`；原 PD 与 10 ms native 网格组合曾在第 10 步出现非有限状态。[原始诊断](../output/grasp_forceaware_validation/validation.json) 和 [默认 64 起点失败日志](../output/grasp_forceaware_default_probe/solve.log) 单独保留。

[dexforge_servo_integration.py](../retarget/dynamics/dexforge_servo_integration.py) 提供显式选择的 `activate_servo_integration()`，profile 为 `local_fixed_pd_implicit_v1`。它在进程内统一接入原 planner、executor 和 native adjoint，适用于固定增益、单位 gear、每个标量 DOF 一个 PD actuator 的模型。原力计算仍为：

```text
force = clip(kp * (clip(control) - q) - kv * v, force_limits)
a_integrate = solve(M + h * (Dpassive + Dact), original_total_force)
Dact = kv    if actuator force is strictly inside its limits
       0     if actuator force is saturated
```

不把速度阻尼移到限力之外：`clip(P-Dv)` 与 `clip(P)-Dv` 不等价。质量、惯量、实际 force clamp 和物理输出 `qacc=M^-1 total_force` 保持原计算；只扩展积分使用的矩阵。固定饱和分支内 `Dact` 为常量，复用原质量求解 VJP，同时保留物理 qacc 的反传路径。边界 mask 变化需要作为不可微分支单独诊断。

这是**本地固定 PD 积分能力扩展，不是完整 MuJoCo implicitfast**；没有加入完整的其他速度相关力导数。每个 worker 在模型编译前激活，`dynamics_adapter.json` 记录 profile、实现哈希及原文件哈希。[测试](../tests/test_dexforge_servo_integration.py) 覆盖未饱和、force 饱和、control 饱和、mask 边界，以及非零物理 qacc cotangent；7 项通过。单 DOF forward 对解析公式及 MuJoCo 对照，qpos/qvel/qacc 的局部 VJP 对解析梯度。

原 PD、10 ms、0.4 s 恒定 CA command 的三尺寸基线均执行 40/40 finite，但物体仍可明显掉落；有限性修复不等于保持成功。[独立基线结果](../output/grasp_forceaware_pd_implicit_validation/summary.json) 未与控制器变体混合。

## FD 假平台与认证范围

曾把不同 hard-contact Newton/projection active sets 上近似相等的 FD 数值误认成平台。修复位于 [forceaware_validation.py](../retarget/env/grasp/forceaware_validation.py)：同时检查碰撞离散 feature、接触行 active、每次 Newton slope、最终非负投影、actuator/control clamp，并拒绝 float32 下已经无法解析的控制扰动。差分先在 host float64 中减分量再归约。

原单步例的 AD 为 `0.338939`；错误平台给出约 `-0.39`，但这些扰动改变了 solver 分支。真正同分支的小扰动 FD 为 `0.340764 / 0.335939`，单步通过。[诊断结论](../output/grasp_forceaware_gradient_audit/diagnosis.json) 已撤回“原 VJP 错误”的推断；**没有因此修改原 VJP 实现**。

[完整 objective 诊断](../retarget/env/grasp/forceaware_objective_validation.py) 保留全部原轨迹、terminal 和 regularization 项，只以一个 candidate 评估，不执行 Adam。预紧中尺寸 `t=0.2 s` 的检查使用实际 qpos/qvel、`raw_warm_history[20]`、`ctrl_traj[19]`，仍为 5 knots / 30 native steps：

| 检查 | 结果 |
| --- | --- |
| 三个随机 raw-control 方向 | 全部 inconclusive；未形成满足所有分支与量化要求的平台 |
| 最后 knot 的 wrist_rz 方向（最大梯度分量） | 同分支通过；AD −3,735,363.01，FD −3,746,288.16 |
| 最后 knot 的 index_dip 方向 | 同分支通过；AD 4,278.64，FD 4,352.59 |
| 最后 knot 的 middle_dip 方向 | inconclusive |

[主报告](../output/grasp_forceaware_preloaded_objective_audit/objective_validation.json) 和 [局部报告](../output/grasp_forceaware_preloaded_objective_audit/localized_joint_validation.json) 均保留 uncertified 范围。fresh 诊断窗口的 previous-previous 正则历史按 API 默认初始化，不宣称精确重演全部 rolling 历史。两条局部通过不等于完整 objective、全部工况或整个轨迹认证；inconclusive 也不能写成同分支 VJP mismatch。

## Viewer、物理审计与穿透阈值

[forceaware_scene.py](../retarget/env/grasp/forceaware_scene.py) 保留普通 mesh viewer/replay 场景的**输入源积分器**；当前 CA 源场景为 implicitfast。native XML 使用原 loader 要求的 Euler schema，实际 PD 积分扩展由进程内 profile 明确记录。二者不再一起强制改成 Euler，也不声称数值轨迹完全一致。独立 MuJoCo mesh replay 继续使用同一初态与已保存控制进行验收。

手部所有皮肤、固定 pad/mount 仍参加手物碰撞，guidance mask 不裁剪物理表面。物体使用封闭 sdfmesh；native 手内自碰通过原 capsule cost，viewer 保留源碰撞掩码。

固定 1 mm tangent sphere 可能从薄片、边角另一侧伸出。当前 `sphere_fit: convex_inside` 通过 [forceaware_geometry.py](../retarget/env/grasp/forceaware_geometry.py) 按样本的 mesh/body provenance 匹配实际碰撞 geom，再缩小正半径，使球满足该 geom 凸包的全部半空间约束 `a·center + b + radius <= 0`。最大请求半径仍为 1 mm；数值角点允许至多 1 μm 的向内 tangent-point 位移，CA material point 和 query index 不变。float32 序列化后重新检查，并在 scene manifest 保存证书、半径分布及实际匹配。

这个证书只证明球位于**对应 MuJoCo per-geom convex collision body** 内，不证明位于原非凸三角网格内部，也不证明球云覆盖整个表面。所有输入采样保留；没有匹配物理 geom 时失败，不启用 visual-only geom，也不按 semantic mask 筛掉手表面。

[runner](../retarget/env/grasp/forceaware_runner.py) 现在直接读取实际 FA contact 配置判定穿透：

| 深度 | 含义 |
| --- | --- |
| `target_phi=-4 mm` | 原 contact robustness 的目标深度 |
| `soft_penetration_limit=4 mm` | `-target_phi`；软穿透罚起点，也是当前 hold 穿透验收线 |
| `penetration_limit=8 mm` | 更深穿透罚的起点；不是允许 8 mm 即算成功，也不是硬约束保证 |

早期独立硬编码的“穿透必须 ≤2 mm”与原目标深度矛盾，已改为从上述配置派生。**2 mm 仍是物体最大漂移阈值**，旋转阈值为 5°。`task_success` 还要求完整时长有限执行、五指每个物理步均具有大于 `1e-6 N` 的实际法向载荷、审计轨迹匹配，以及独立 mesh replay 的物体保持通过。QP 压力、active guidance、GWS pass 或 `solver_completed` 均不能替代这些条件。

[forceaware_mesh_audit.py](../retarget/env/grasp/forceaware_mesh_audit.py) 以官方 replay 的相同初态、控制、mesh 场景及 `sim_dt` 重放，在每次 `mj_step` 后直接读取 `mj_contactForce`，不额外执行 `mj_forward`。固定 sensor/pad 子体按父指链归组，保存逐物理步五指法向力、完整 contact geom/dist/frame/wrench 和物体轨迹。这里按整根手指归组，不等于五个指尖均接触。action 网格上的 qpos/qvel/control/time 必须与已保存的官方 `replay.npz` **逐项 exact 相同**；否则保留差异报告并报错。这些受力只属于独立 MuJoCo mesh replay，不能写成 native GS 载荷或硬件验证。

## CA active-only 与初始接触对齐

[forceaware_ca_refinement.py](../retarget/env/grasp/forceaware_ca_refinement.py) 的 `active_only: true` 只为已接受的 `reference_contact_mask` 活跃槽创建 CA 接触优化变量。释放槽不再参加 contact residual、force closure 或 topology；保存时恢复原 slot 布局，释放槽的 guidance mask 仍为零。完整手表面、query 顺序和裁剪前计算的 forbidden mask 保留。当前 physical 配置用 0.2 mm 接触误差、0.5 mm 穿透容差重新验证 CA 输出，原输入另存。

这修复了“已释放的模板 patch 仍参与重新求解”的问题，尚不保证初始碰撞法向与预期接触方向一致。圆柱端沿/rim 附近最近面可能属于端盖而非侧面；CA 材料点、QP 最近面法向、native GS 接触与实际 mesh 接触仍须逐项核对。初始接触对齐及端沿选择仍在排查，不能把距离较小或 QP 可行当作已解决。

## 精确静态载荷与 command/state 分离

[forceaware_preload.py](../retarget/env/grasp/forceaware_preload.py) 在接受的 CA qpos、零 qvel 下，从完整源 MuJoCo 模型读取 `qfrc_bias-qfrc_passive` 和实际 material-point Jacobian，包含固定附件及真实 gravity-compensation 标记。中尺寸旧近似 bias 与该值最大相差约 **0.849546 N**；使用精确 bias，而不是继续依赖近似重力补偿。

静态 QP 使用实际 CA 手表面点、最近物体网格面法线、摩擦和 actuator control/force bounds，提议满足物体力矩平衡的 command。标量最小压力可复用原有有界 load QP；按手指配置最小压力时，Clarabel 在 CPU 求同一静态输入 QP。它是初始化工具，**不增加或替换 FA loss/约束**。不可行或越界时拒绝输出可用 preload，不静默 clip。QP 压力只是提议，必须由 native 接触行与独立 replay 重新验证。

[forceaware_initialization.py](../retarget/env/grasp/forceaware_initialization.py) 保持两类数据分离：

```text
state / tracking target = accepted CA qref
preload command         = qref + static_torque / kp
initial warm controls   = command - qref
previous_action         = command
```

不把较大的 command 偏移写入 CA qpos 或 hand reference，也不把手指初态修正 bounds 当作 control-offset bounds。原 ServoPrior 保持原样：64 starts 中前半围绕 warm preload，后半围绕原速度 prior；静态时后者为零 residual。

可选 `fixed_initial_state: true` 使用原 solve 参数 `allow_init=False`，避免首窗初态拟合移动静态 QP 所对应的已知 CA 状态；控制优化继续启用。未选该项时，原首窗初态修正仍可发生。初态策略写入 `initial_state_policy.json`。

原 PD 预紧结果保留于 [0.2 N 实验](../output/grasp_contactaware_forceaware_preloaded/summary.json) 和 [0.25 N 实验](../output/grasp_contactaware_forceaware_preloaded_press/summary.json)。其中物体漂移已经较小，但仍有手指瞬时卸载，不能据此报告完整成功。

## 实验性 finger PD 变体

20 个 finger PD 原值来自 [官方 WUJI Hand 2.2 XML](../data/grasp/assets/wuji2/upstream/mjcf/right.xml)，不是本仓库按质量拟合；六个虚拟 wrist servo 才由资产适配脚本添加。原值来源和哈希见 [source.json](../data/grasp/assets/wuji2/source.json)。

可选 `finger_servo_scale` 经 [forceaware_controller.py](../retarget/env/grasp/forceaware_controller.py) 将 20 个 finger 的 `kp` 乘 2、`kv` 乘 √2，同时更新 gainP/biasP。**这改变了官方 finger PD**，属于显式仿真控制器变体，不能标为“原 PD 不变”或硬件校准完成。六个 wrist servo、质量、惯量、armature、关节/控制范围、gear、摩擦和全部 actuator/joint force limits 保持原值，输出 XML 与 manifest 独立保存。

原因是当前约束瓶颈可出现在 command range，而不是 motor torque limit。静态时同一扭矩所需 `command-qref` 在 kp 翻倍后减半；应重新求 preload QP，不能直接复用旧 command。kv 按 √2 缩放只在局部固定惯量、未饱和单关节近似下保持阻尼比，不构成整体接触稳定性证明。所有原 PD 实验继续单独保留，不与该变体合并比较。

该变体仅作为单独控制器实验保留，**不作为当前推荐配置**。当前 physical 结果均为 `controller_changed: false`；不能与 finger PD 变体合并报告。

## 当前诊断复现与三尺寸结果

[physical 配置](../configs/grasp_contactaware_forceaware_physical.yaml) 保留原 FA 5 knots / 40 iterations / 64 starts、10 ms physics、30 ms action、60 ms knot 和全部 loss/bounds/smoothing。显式开启本地 PD 积分扩展、contained GS、active-only CA；使用原 PD，静态 QP 每接触最小压力 0.25 N、总压力下限 6 N。首窗初态修正仍按原 FA 策略启用。

```bash
.venv-forceaware/bin/python run.py --config configs/grasp_contactaware_forceaware_physical.yaml
```

三个圆柱高度均为 120 mm，目标静止，验证时长 0.4 s。mesh 零载荷步数按**拇/食/中/无名/小指**排列，阈值为 `1e-6 N`，2 ms 网格共 200 步：

| 半径 mm | FA 执行 | native 最大漂移 mm | mesh 最大漂移 mm | mesh 五指零载荷步数 | 结论 |
| --- | --- | --- | --- | --- | --- |
| [26.25](../output/grasp_contactaware_forceaware_physical/000_000_1_Large_Diameter_cylinder_r0.0262500_h0.1200000/summary.json) | 40/40 finite | 6.299 | 5.385 | `0 / 0 / 38 / 119 / 63` | 位移与连续载荷均未通过 |
| [35](../output/grasp_contactaware_forceaware_physical/001_001_1_Large_Diameter_cylinder_r0.0350000_h0.1200000/summary.json) | 40/40 finite | 0.858 | 1.008 | `0 / 0 / 0 / 0 / 1` | 物体保持较好，严格五指全程载荷未通过 |
| [43.75](../output/grasp_contactaware_forceaware_physical/002_002_1_Large_Diameter_cylinder_r0.0437500_h0.1200000/control_preload.json) | 未执行 FA | — | — | — | 静态 preload QP 为 `PrimalInfeasible` |

小、中尺寸的独立 mesh 审计都精确复现官方 replay，MuJoCo warning 计数均为零。中尺寸小指唯一零载荷在起始第一个 2 ms 步：存在 mesh 接触，但 `mj_contactForce` 为零；随后 199 步五指均为正载荷。它仍不满足“每步五指承载”的严格验收，两个已求解样本均未报告 `task_success`。native 最大穿透分别为 1.394、1.486 mm，低于当前 4 mm 软验收线；这不抵消位移或失载失败。

[小尺寸 mesh 审计](../output/grasp_contactaware_forceaware_physical/000_000_1_Large_Diameter_cylinder_r0.0262500_h0.1200000/mesh_contact_audit/mesh_contact_audit.json) 与 [中尺寸 mesh 审计](../output/grasp_contactaware_forceaware_physical/001_001_1_Large_Diameter_cylinder_r0.0350000_h0.1200000/mesh_contact_audit/mesh_contact_audit.json) 保存完整逐步数据。大尺寸未得到可用 preload，不能用它的 CA 成功或其他配置结果补作本轮抓取成功。

## 棱边法向一致性修复与最终输入

进一步独立对照发现，CA 与 preload 对圆柱底缘可能各选择不同的等距三角面。投影点虽然相同，任意一个 face normal 却不能代表 edge 的实际接触方向。大尺寸旧食指姿态曾以侧面法向得到约 0.97 的 facing，而最近特征方向的实际 facing 仅约 0.35。

[contactaware.py](../retarget/env/grasp/contactaware.py) 现在对非零距离使用 `sign(phi) * (point - closest) / distance`，仅在距表面 0.1 μm 内回退到面法向。对 guided 点增加 1 μm centered FD 法向 Jacobian，并贯穿原 normal residual、facing 和 force-closure 项；原 SQP、QP 和权重不变。默认 facing 下界从本地 0.25 修正为 0.5。preload 也用同一 signed-feature 约定，并明确只支持 outward-wound closed convex primitive。近表面、inside 符号、rim 面选择和 edge Jacobian 都有回归测试。

新法向下从旧宽松输入直接迭代，小、大尺寸局部 SQP 未通过，已阻止消费。将此前较接近约束的手姿**仅作为数值初值**再次用新约束验收，小尺寸通过，大尺寸 8 轮通过；没有直接继承旧 success 标记。三例最终 accepted 输入位于 [grasp_contactaware_feature_normals_final](../output/grasp_contactaware_feature_normals_final/reference_sources.json)，均保持原语义 material points、全手碰撞、静止物体和无模板角回退。共同的静态初始化下限是 thumb/index/middle 各 0.1 N、ring 0.2 N、pinky 0.3 N、总法向压力 4 N；三例 exact-bias QP 全部可行，这仍不是动态保持证明。

[最终配置](../configs/grasp_contactaware_forceaware_final.yaml) 读取上述新参考，保留官方 WUJI2 finger PD、原 FA loss/weights 和 64 starts，显式将迭代预算设为 200，使用固定已知 CA 初态与 PD 积分扩展。控制和初态策略均写入产物。此前的两个 `loaded` 压力分配试验都在静态 QP 阶段不可行，未运行优化，没有成功结论。

```bash
.venv-forceaware/bin/python run.py --config configs/grasp_contactaware_forceaware_final.yaml
```

## 修复后的动态对照

同一新 CA 参考、common A 预载、官方 finger PD、原 FA 200 iterations / 64 starts，在 10 ms planning/execution 下三个尺寸均完成 0.4 s，物体保持通过，但严格逐步五指承载仍未全部通过：

| 半径 mm | native 最大漂移 mm | mesh 最大漂移 mm | native 五指零载荷步数 / 40 | mesh 五指零载荷步数 / 200 |
| --- | --- | --- | --- | --- |
| 26.25 | 1.380 | 1.927 | `0 / 0 / 3 / 8 / 6` | `0 / 0 / 48 / 52 / 26` |
| 35 | 0.504 | 0.441 | `0 / 1 / 0 / 0 / 1` | `0 / 0 / 0 / 0 / 0` |
| 43.75 | 1.883 | 1.447 | `1 / 1 / 1 / 16 / 4` | `2 / 0 / 0 / 49 / 8` |

[统一索引](../output/grasp_forceaware_repair_report/summary.json) 与 [诊断图](../output/grasp_forceaware_repair_report/forceaware_diagnostics.png) 聚合独立批次的原始产物，没有拼接或修改控制轨迹。`task_success` 保留 false，不能用物体位移通过替代五指承载。

### 原版 executor 的 2 ms 子步

[中尺寸对照](../output/grasp_contactaware_forceaware_fine_executor/000_001_radius_0.03500/summary.json) 保留原 FA 10 ms planning、30 ms action、60 ms knot，只选择原 executor 已支持的 `exec_dt=0.002`。同样 200 iterations / 64 starts 和 PD 积分扩展，没有新损失或控制器增益变化。

该例 **task_success=true**：native/mesh 最大漂移分别 **0.603 / 0.545 mm**，native 最大穿透 **1.119 mm**；两种物理回放各 200 个子步，五指全部持续具有正载荷。实际 native 审计逐样本匹配官方输出，mesh 审计 qpos/qvel/time/control 精确匹配原独立 replay。跨子步的最大位移和旋转也参加验收。

[三尺寸同设置配置](../configs/grasp_contactaware_forceaware_fine_samples.yaml) 已实际完成全部三例，结果如下。每例均为 0.4 s，native 和 mesh 各检查 200 个物理子步；零载荷顺序为拇指、食指、中指、无名指、小指。

| 半径 mm | native / mesh 最大漂移 mm | mesh 最大旋转 ° | native / mesh 五指零载荷步数 | 严格验收 |
| --- | --- | --- | --- | --- |
| 26.25 | 0.663 / 1.371 | 6.087 | `0,0,61,63,0` / `0,0,9,112,93` | 失败：失载及旋转超 5° |
| 35 | 0.603 / 0.545 | 1.494 | `0,0,0,0,0` / `0,0,0,0,0` | **通过** |
| 43.75 | 1.109 / 2.043 | 1.192 | `2,2,0,21,9` / `2,0,0,29,6` | 失败：失载及漂移超 2 mm |

[最终摘要](../output/grasp_forceaware_fine_report/summary.json)、[逐例索引](../output/grasp_forceaware_fine_report/manifest.json) 和 [诊断图](../output/grasp_forceaware_fine_report/forceaware_diagnostics.png) 聚合相同设置下的原始产物；[来源索引](../output/grasp_forceaware_fine_report/source_runs.json) 明确中尺寸与两端尺寸来自不同独立批次，控制轨迹没有拼接或修改。上游 139 个文件哈希复核通过。153 项 FA/适配测试通过、1 项跳过；CA 运行环境的 19 项相关检查通过。

结论是确定的接入错误已修复，中尺寸达到了短程五指承力；两端尺寸仍未全部满足动态保持要求。实际受力集中或阶段性卸载不能由 CA 几何接触、静态 QP 可行或 `solver_completed` 替代。原 FA 是软接触目标，没有逐指正压力硬约束；本轮没有通过添加新 force loss 或放宽 2 mm / 5° / 五指逐步载荷验收来制造成功。中尺寸通过仅证明这一次 0.4 s 窗口，不能外推长期保持。

### 计算耗时

当前只检查同一种圆柱的三个半径，不是遍历物体库。每例 14 个 MPC 窗口、200 iterations、64 starts；RTX 4090 独占时每窗约 13 s，纯优化约 3.1 min/尺寸，包含准备、检查、回放约 3–4 min/尺寸。中尺寸对照曾共享 GPU，每窗约 27.7 s、整体约 6.7 min；这是实际墙钟耗时差异，不能把共享实验当作独占吞吐基准。各例原 `solve.log` 和最终摘要的 `timing` 保留测量值。

### 末端常值指令的独立延长

从原 replay 的实际末状态、末速度继续固定最后 control 1 s，无重新初始化、无 FA 重规划。fine-executor 中尺寸在总时刻 1.192 s 首次发生无名指卸载；总 1.4 s 漂移 2.629 mm。10 ms executor 的中尺寸则漂移 2.490 mm，小尺寸 4.041 mm。前缀与原 replay 精确匹配、均有限、无 MuJoCo 警告；产物见 [延长测试目录](../output/grasp_forceaware_terminal_extension)。这是常值控制延长结果，不能推断持续 MPC 重规划失败，也不能把它写成 1.4 s 优化通过。

完整 objective 的全局梯度仍未认证，不能由保持结果或局部 FD 通过反推；原始 Euler 的失败数据也不与扩展积分后的结果混合。


## 与原版速度比较的纠正

上述 3–4 min 是本次 **200 updates** 的验证配置，不是原版默认性能。固定上游默认是 40 updates、64 starts；本地没有丢失原 `ScopedCapture/capture_launch` CUDA graph 路径。实际对同一个 WUJI 模型、相同参考、预载、64 starts 和 30-step horizon 做独占 GPU 单窗口对照：

| Adam updates | 单窗口 solve 时间 | 含首次 graph 准备的调用时间 |
| --- | --- | --- |
| 40（上游默认预算） | 2.665 s | 3.302 s |
| 200（此前验证配置） | 13.017 s | 13.442 s |

[测量产物及可重跑脚本](../output/grasp_forceaware_speed_audit/budget_benchmark.json) 给出的实测比例为 **4.884×**，与 201/41 次 graph iteration 的工作量相符。按 14 个窗口线性估计，40 updates 约需 37 s 纯优化；这是估算，未声称 40 次整段轨迹仍通过抓取验收。

2 ms executor 只增加实际执行子步，规划仍为 10 ms、30-step horizon，不会让每个优化窗口计算量增加 5 倍。fine_outer 两尺寸的优化共 368.18 s，总墙钟 393.013 s，准备、FD、审计、回放和视频合计约 6.3%，因此不能把分钟级主要耗时归因于额外验证。并行共享 GPU 则确实将窗口时间由约 13 s 增至约 27.7 s。

原版快速示例使用 LEAP/querypoint 手源 + GS 物体；WUJI 是有限半径 GS 手源 + sdfmesh 物体，并多一次固定 PD 隐式积分矩阵求解。原 LEAP 手部采样 12,024，WUJI 16,384，仅约 1.36×；没有证据把剩余速度差距全归因于采样数。几何/normal-frame backward 和积分求解的各自耗时尚未做隔离 A/B profile，不给未经测量的加速倍率。


## 快速预算的后续实测

[性能对照记录](grasp_forceaware_performance.md) 已完成原资产有接触代理基线与三尺寸完整复测：恢复 40 次后每窗 2.66 s，与原资产 2.85 s 同量级，平均每尺寸 37.27 s 纯求解 / 47.70 s 总耗时。额外积分矩阵约占单迭代 0.66%，没有为加速改变物理或几何。快速预算三例五指持续承力均未通过，因此本页 200 次中尺寸成功仅属于原 200 次配置。

论文附录 H 和 Sec.4.2.1 的后续核对见 [性能记录](grasp_forceaware_performance.md)：三尺寸 40 次结果满足论文平均 tracking 公式，但未满足本仓库额外的最大漂移和五指逐步承力条件；两者不互相替代，严格 `task_success` 保持 false。

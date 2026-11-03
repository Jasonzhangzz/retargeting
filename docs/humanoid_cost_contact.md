# Humanoid 固定接触轨迹 cost

原 `retarget/task/humanoid/cost.py` 和默认 SPIDER 评分入口保持不变。新增的 `frozen_contact` 模式在每个 MPC 窗口开始时，沿名义控制轨迹建立一次完整缓存。之后 MPPI/CEM 的所有候选、分批评分和均值更新只执行 Torch 动力学递推，**不调用 MJWarp rollout，也不重新求解 collision**。实际执行动作并进入下一个 MPC 窗口后，才根据新的实测状态重建缓存。

## 运行

```bash
python run.py --config configs/humanoid_move_largebox_contact_mppi.yaml
```

该配置执行完整搬箱参考（49 个 MPC 窗口），使用同一个 MPPI 求解器，预算为每轮 1024 个候选、最多 16 轮、128 个 world，并开启每轮两步梯度更新。需要短测时可显式添加 `environment.frames: 1`。已有 humanoid 配置可增加：

```yaml
environment:
  device: cuda:0
  dynamics: frozen_contact
  contact_graph: true
  contact_cost:
    contact_weight: 1.0
    physical_contact_weight: 10.0
    soft_penetration_weight: 15.0
    penetration_weight: 75.0
    soft_penetration_limit: 0.002
    penetration_limit: 0.010
    object_motion_weight: 1.0
    nominal_correction: true
    action_trust_radius: 0.1
solver:
  type: mppi
  gradient_steps: 2
  gradient_step_size: 1.0
  gradient_backtracks: 6
```

本机已准备匹配 requirements 的独立环境，可用 `.venv-contact/bin/python` 运行上述命令；原 `.venv` 未改动。

MPPI 的候选采样、前 10% 加权更新和噪声退火保持原有算法。开启 `gradient_steps` 后，每轮对加权均值求整个 action sequence 的梯度，在采样标准差坐标中归一化下降方向，并通过回溯只接受有限且严格降低同一轨迹代价的更新。候选评分和梯度更新共用当前 MPC 窗口的接触缓存。`gradient_steps: 0` 是原来的纯采样 MPPI；默认仍为零，CEM 不接受非零值。Humanoid 的梯度更新要求 `dynamics: frozen_contact`。

逐轮日志记录 `gradient_accepted_steps`、`gradient_evaluated` 及更新前后代价，额外梯度/回溯评分计入 `evaluated`。逐窗口记录包含缓存准备时间 `contact_cache_seconds` 和优化时间 `optimization_seconds`；`solve_seconds` 包含两者。初始化预热建立的首个缓存记入初始化时间。

省略 `dynamics` 或设置 `mjwarp` 则继续使用原有 SPIDER 路径。`frozen_contact` 预测不使用原 domain-randomization ensemble；该模式将 `num_dr` 设为零，并在输出中记录所选预测模型、实际执行模型和随机化状态。实际执行仍使用正常 MJWarp 物理仿真。

## 一次缓存，整段递推

`retarget/env/humanoid/frozen_contact.py` 提供 `FrozenContactRollout`。缓存建立使用独立的单 world scratch 数据，读取执行环境的初始状态，不改变执行环境的 qpos/qvel/ctrl：

1. `from_env` 沿名义轨迹缓存每个时间点的接触集合、法向距离、约束 J/D、M、执行器 moment/gain/bias 和平滑动力学。碰撞仅在这一准备阶段运行；启用名义校正时，还会运行下述 scratch 引擎 step。
2. `rollout` 根据候选 action 从初态递推整个 horizon 的 qpos/qvel。它不读取 Warp 数据，不调用引擎，也不重建接触集合。
3. 接触 gap 按相对于名义状态的位姿偏差更新；pyramidal 摩擦使用法向 gap Jacobian 更新距离，使用完整摩擦 J 计算速度和广义力。执行器长度、速度、控制饱和和力饱和随候选状态更新。
4. 半隐式 Euler 积分包括 free/ball joint 的四元数更新。前面 action 的影响会传到后续物体位移、接触对应点和穿透代价。
5. `cost_contact.get_trajectory_cost` 只用预测状态和缓存几何评分，末帧权重与整段平均遵循原 SPIDER 约定。

每个物理接触的几何 Jacobian 来自 Warp 的 `contact.pos/frame/geom` 和当前 `cdof/subtree_com`：分别计算两个碰撞 body 上接触点的 Jacobian，再形成 `normal @ (J_object - J_robot)`（根据 geom 顺序定符号）。完整 `nv` 列包含机器人 root、关节及物体自由关节；摩擦响应使用 Warp 装配的完整约束 Jacobian。终端 site Jacobian 在刷新终端 kinematics 和 COM 后读取，避免沿用前一帧的空间 DOF。

示例开启 `nominal_correction`：在缓存准备阶段额外执行 scratch MJWarp 名义轨迹，把每帧 `MJWarp_next_velocity - ComFree_next_velocity` 作为固定偏差加入候选模型。这样名义动作复现实际引擎的名义 qpos/qvel，候选扰动仍通过 ComFree 接触响应求导，避免两个动力学模型的基准轨迹漂移。偏差本身不求导，也不等于对完整 MJWarp 求导；该选项默认关闭，`from_env` 可通过 options 或显式 `anchor_to_mjwarp=True` 开启。校正路径在准备阶段每个名义时刻还会运行一次引擎 step 的碰撞，候选评分仍不运行碰撞。

`contact_graph: true` 将固定缓存下的 Torch 前向和反向捕获为 CUDA Graph，要求关闭预测记录（runner 已关闭），仅支持一阶梯度。图按 batch、dtype 和是否求导缓存；执行动作、移动窗口时与接触缓存一起失效。输出和梯度拥有独立存储，尚未反传的同形状调用使用 eager 路径，避免覆盖其反向中间量。图捕获与缓存准备都有成本，短窗口或小采样预算不一定加速，日志会记录捕获耗时；默认关闭。

固定碰撞拓扑只适用于名义轨迹附近。`action_trust_radius` 将优化参数投影到每个控制 knot 相对当前名义控制的逐维范围内，单位为 MuJoCo 原始 control（搬箱位置伺服为 rad）；线性插值后的稠密控制也满足此范围。示例使用 0.1，默认 `None` 不限制。采样、均值更新和梯度回溯使用同一投影，进入下一窗口后以更新后的名义控制重新建立局部范围。

如果只希望读取初始时刻一次碰撞，而不准备名义轨迹的接触切换，可用 `from_current(config, env, horizon)`；它在整段中复用单帧局部模型。

这是沿名义轨迹的局部接触动力学模型：M/J/接触集合等随名义时间变化，在候选优化期间固定；候选轨迹的状态和 gap 持续递推。缓存外的新碰撞不会被发现，已有接触可以随 gap 增大而释放。预测偏离名义轨迹较大时，该近似的精度会下降。缓存不随 MPPI 的 action 更新自动重建。

## 可微接口

```python
from retarget.env.humanoid.frozen_contact import FrozenContactRollout
from retarget.task.humanoid import cost_contact

options = cost_contact.ContactCostOptions()
# nominal_actions: (H, nu)。只在进入当前 MPC 窗口时调用。
cache = FrozenContactRollout.from_env(
    config, env, nominal_actions, options=options,
)

# actions: (N, H, nu)，为 MuJoCo 原始 control。
actions = candidate_actions.detach().clone().requires_grad_(True)
prediction = cache.rollout(actions)
cost, info = cost_contact.get_trajectory_cost(
    config, prediction, reference_window, options=options,
    # object_local_points=points_on_object,
    # target_displacement=desired_delta_xyz,  # (H, 3) 或 (N, H, 3)
)
cost.mean().backward()
gradient = actions.grad

# 后续 MPPI 迭代 / 梯度更新重复这两步，继续复用 cache。
prediction = cache.rollout(updated_actions)
cost, info = cost_contact.get_trajectory_cost(config, prediction, reference_window)
```

`reference_window` 沿用原五元组 `(qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref)`，每项带 H 维。位移目标默认是每帧参考物体线速度乘该帧 timestep。`info["predicted_displacement"]` 的形状为 `(N, H, 3)`，状态包含初态，形状为 `(N, H+1, nq/nv)`。

原 `get_reward`、`object_displacement_cost` 保留为单步诊断接口，它们默认刷新碰撞；MPPI 的 `frozen_contact` 评分不调用这些单步接口。

## 代价与点对应

| 每步代价 | 默认计算 | 单位 |
| --- | --- | --- |
| 接触点对应 | `1 * sum(mask * norm(robot_site - object_point))` | m |
| 物理接触点对应 | `physical_contact_weight * sum(norm(point_object - point_robot))`，默认权重 1，示例 10 | m |
| 浅层穿透 | `15 * sum(relu(depth - 0.002)^2)` | m² |
| 深层穿透 | `75 * sum(relu(depth - 0.010)^2)` | m² |
| 物体运动 | `1 * sum((predicted_displacement - target_displacement)^2)` | m² |

超过 10 mm 时两级穿透项同时生效。所有权重、容忍深度和 ComFree stiffness/damping 都可通过 `ContactCostOptions` 调整，接触项不依赖原 `contact_rew_scale` 开关。

`config.contact_site_ids` 指定机器人 site，顺序与 `(H, K)` 接触 mask 和 `(H, K, 3)` 世界坐标参考点一致。实现将目标点转换为参考物体局部坐标，再随预测物体位姿移动。也可直接传入静态 `(K, 3)`、时变 `(H, K, 3)` 或批量 `(N, H, K, 3)` 的 `object_local_points`。没有物体时使用世界坐标目标，例如脚底对应地面点。

`humanoid_object` 默认将最后一个 free joint 作为物体、第一个作为机器人；可用 `object_body_id` 指定物体的 free-joint body。两级穿透仅计算机器人子树与物体子树的已缓存物理接触，不包含地面或自碰撞。动力学仍使用全部支持的接触约束，包括地面支撑。

当前 `move_largebox/reference.npz` 只有 qpos/qvel/ctrl，没有指定点对应标签；使用 site 点对应项需提供 site、mask 和参考点。未配置 site 时仅该项为零。物理接触点对应直接使用缓存中已经接触（`dist <= 0`）的机器人与物体表面点，分别冻结其所属 body 上的点 Jacobian；即使没有标签，也对这些接触面的分离和切向滑动施加代价，并传递整段 action 的梯度。`contact_cost` 为 site 与物理接触两项之和，另分别记录 `site_contact_cost`、`physical_contact_cost`。

接触动力学仍根据物理碰撞工作，不依赖点对应标签。原 SPIDER 搬箱场景的手部物理碰撞使用球形代理，机器人 mesh 用于显示；本实现读取该场景实际启用的碰撞几何。若场景启用 mesh 碰撞，相同接口读取凸 mesh 碰撞的接触点并构建两侧 Jacobian。测试包含真实 mesh 接触；不能将原场景的球形代理结果描述为 mesh 碰撞结果。

`retarget/dynamics/comfree/response.py` 的 `contact_response_torch` 与原 Warp kernel 使用同一闭式响应公式，原 kernel 未改动。支持无 activation 动力学的固定/仿射 gain、无/仿射 bias 执行器，以及 pyramidal 摩擦锥。对不支持的 equality/frictionloss 行、有状态/延迟执行器、自定义控制回调等显式报错。梯度不穿过离散碰撞发现；接触响应截断和执行器饱和均为分段可微。

验证：

```bash
python -m pytest -q
```

有限差分回归覆盖整段早期 action 对末帧机器人 root、关节、物体位移的影响，以及终端接触点 Jacobian。旋转 tracking 在相同四元数处保留解析导数；原 `cost.py` 保持原样。

单窗口对照（同时报告两种模型的交叉评分，避免直接比较不同目标）：

```bash
python scripts/benchmark_contact_mppi.py --contact-graph --nominal-correction --output output/contact_mppi_benchmark.json
```

`--population`、`--iterations`、`--worlds` 可设置与实际求解一致的预算。脚本计入每个窗口的缓存和图捕获成本；单窗口结果不能替代完整轨迹的成功验收。

## 本机验证结果（2026-10-31）

环境为 RTX 4090、CUDA 驱动 12.2、Torch 2.5.1+cu121、MuJoCo 3.10.0、MJWarp 3.10.0.3、Warp 1.16.0。为兼容此驱动，执行图关闭 conditional graph nodes，并在独立缓冲区预热后捕获；不推进执行环境的初态。

最终配置已执行完整 49 个窗口、290 个模拟步，所有状态有限，接受 424 次梯度更新。求解累计 149.82 s，总墙钟约 154 s（含初始化及视频导出）。结果为 **`task_success: false`**：箱体位置 RMS 0.2748 m、最终误差 0.6359 m，机器人出现失衡。因此实现已能完整运行，但本次尚未获得成功搬箱轨迹，也不能据此宣称成功率提高。完整状态、逐轮评分和视频在 `output/humanoid_move_largebox_contact_mppi/{trajectory.npz,iterations.jsonl,summary.json,video.mp4}`。

相同单窗口、seed、1024 候选 × 固定 16 轮、128 world、控制范围 0.1 的独立基准：

| 方法 | 窗口耗时，含接触缓存/图捕获 | 同一 MJWarp 目标交叉评分，越低越好 |
| --- | ---: | ---: |
| 原 MJWarp MPPI | 12.777 s | 0.3320 |
| 固定接触 MPPI | 3.457 s | 0.4928 |
| 固定接触 + 梯度 MPPI | 7.533 s | 0.4383 |

这个预算下梯度方案窗口耗时约为原 MPPI 的 59%（约 1.70 倍速度），并改善了固定接触纯采样的实际目标值；原 MJWarp MPPI 的最终实际目标仍更好。固定模型目标、MJWarp 目标不同，不能直接比较各自记录的 `final_cost`。小预算下缓存与捕获成本可使梯度方案更慢。基准详见 `output/contact_mppi_final_benchmark.json`，可复现命令：

```bash
.venv-contact/bin/python scripts/benchmark_contact_mppi.py \
  --population 1024 --iterations 16 --worlds 128 --gradient-steps 2 \
  --contact-graph --nominal-correction --physical-contact-weight 10 \
  --action-trust-radius 0.1 --output output/contact_mppi_final_benchmark.json
```

相关 solver/config/reference/ComFree/cost/rollout/runtime/CUDA Graph 验证共 105 项通过。另有两项现有 AST 哈希检查在本机未通过，未计入通过数量；`git diff` 确认它们校验的原 `cost.py` 与 ComFree `kernels.py` 没有改动。历史验收使用清理前版本 a24bc7b2 的 humanoid、solver 和 ComFree 测试。此次按近期使用范围移除了这些历史测试，运行代码保留；历史命令涉及的文件可从该提交的 tests/test_*.py 恢复。现行精简检查见 [测试说明](../tests/README.md)。

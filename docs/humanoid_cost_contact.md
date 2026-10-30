# Humanoid 固定接触轨迹 cost

原 `retarget/task/humanoid/cost.py` 和默认 SPIDER 评分入口保持不变。新增的 `frozen_contact` 模式在每个 MPC 窗口开始时，沿名义控制轨迹建立一次完整缓存。之后 MPPI/CEM 的所有候选、分批评分和均值更新只执行 Torch 动力学递推，**不调用 MJWarp rollout，也不重新求解 collision**。实际执行动作并进入下一个 MPC 窗口后，才根据新的实测状态重建缓存。

## 运行

```bash
python run.py --config configs/humanoid_move_largebox_contact_mppi.yaml
```

示例为一个窗口的可运行性短测。已有 humanoid 配置可增加：

```yaml
environment:
  device: cuda:0
  dynamics: frozen_contact
  contact_cost:
    contact_weight: 1.0
    soft_penetration_weight: 15.0
    penetration_weight: 75.0
    soft_penetration_limit: 0.002
    penetration_limit: 0.010
    object_motion_weight: 1.0
```

省略 `dynamics` 或设置 `mjwarp` 则继续使用原有 SPIDER 路径。`frozen_contact` 预测不使用原 domain-randomization ensemble；该模式将 `num_dr` 设为零，并在输出中记录所选预测模型、实际执行模型和随机化状态。实际执行仍使用正常 MJWarp 物理仿真。

## 一次缓存，整段递推

`retarget/env/humanoid/frozen_contact.py` 提供 `FrozenContactRollout`。缓存建立使用独立的单 world scratch 数据，读取执行环境的初始状态，不改变执行环境的 qpos/qvel/ctrl：

1. `from_env` 沿名义轨迹缓存每个时间点的接触集合、法向距离、约束 J/D、M、执行器 moment/gain/bias 和平滑动力学。碰撞仅在这一准备阶段运行，每个名义时间点一次。
2. `rollout` 根据候选 action 从初态递推整个 horizon 的 qpos/qvel。它不读取 Warp 数据，不调用引擎，也不重建接触集合。
3. 接触 gap 按相对于名义状态的位姿偏差更新；pyramidal 摩擦使用法向 gap Jacobian 更新距离，使用完整摩擦 J 计算速度和广义力。执行器长度、速度、控制饱和和力饱和随候选状态更新。
4. 半隐式 Euler 积分包括 free/ball joint 的四元数更新。前面 action 的影响会传到后续物体位移、接触对应点和穿透代价。
5. `cost_contact.get_trajectory_cost` 只用预测状态和缓存几何评分，末帧权重与整段平均遵循原 SPIDER 约定。

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
| 浅层穿透 | `15 * sum(relu(depth - 0.002)^2)` | m² |
| 深层穿透 | `75 * sum(relu(depth - 0.010)^2)` | m² |
| 物体运动 | `1 * sum((predicted_displacement - target_displacement)^2)` | m² |

超过 10 mm 时两级穿透项同时生效。所有权重、容忍深度和 ComFree stiffness/damping 都可通过 `ContactCostOptions` 调整，接触项不依赖原 `contact_rew_scale` 开关。

`config.contact_site_ids` 指定机器人 site，顺序与 `(H, K)` 接触 mask 和 `(H, K, 3)` 世界坐标参考点一致。实现将目标点转换为参考物体局部坐标，再随预测物体位姿移动。也可直接传入静态 `(K, 3)`、时变 `(H, K, 3)` 或批量 `(N, H, K, 3)` 的 `object_local_points`。没有物体时使用世界坐标目标，例如脚底对应地面点。

`humanoid_object` 默认将最后一个 free joint 作为物体、第一个作为机器人；可用 `object_body_id` 指定物体的 free-joint body。两级穿透仅计算机器人子树与物体子树的已缓存物理接触，不包含地面或自碰撞。动力学仍使用全部支持的接触约束，包括地面支撑。

当前 `move_largebox/reference.npz` 只有 qpos/qvel/ctrl，没有指定点对应标签；使用点对应项需提供 site、mask 和参考点。未配置 site 时该项为零。

`retarget/dynamics/comfree/response.py` 的 `contact_response_torch` 与原 Warp kernel 使用同一闭式响应公式，原 kernel 未改动。支持无 activation 动力学的固定/仿射 gain、无/仿射 bias 执行器，以及 pyramidal 摩擦锥。对不支持的 equality/frictionloss 行、有状态/延迟执行器、自定义控制回调等显式报错。梯度不穿过离散碰撞发现；接触响应截断和执行器饱和均为分段可微。

验证：

```bash
python -m pytest -q
```

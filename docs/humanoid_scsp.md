# SCSP：四点真实接触与 121 种手部组合

当前状态：这是尚未通过实际搬运验收的原型。初始未接触时的位置等式已改为按操作阶段和实际接触门控；IPOPT 梯度与刷新后的目标不一致，以及 ComFree 支撑预测和实际执行偏差仍未修复。下文的单元测试和短测结果不表示已能成功搬箱。

`cost_contact1.py` 保留关节、root 和物体跟踪目标；`solver/scsp.py` 负责离散接触 CEM、可微接触力优化、机器人 action 的 IPOPT 和 ADMM 协调。

## 四个机器人接触点

固定采用 `left_palm`、`right_palm`、`left_foot`、`right_foot`，位置分别为手心和脚心 site 中心。`point_model.py` 在 SCSP 专用模型中关闭原机器人碰撞几何及其显式 pairs，增加这四个中心点的微球碰撞体；默认数值半径 `point_radius: 0.0001` m。旧几何保留以保持质量、惯性和可视模型，源 XML 不改写。双手可碰箱子和地面，双脚碰地面，物体与环境的接触保留。

每个机器人点可能对应多个摩擦约束行，箱子也可能产生自己的地面接触；这些行不会增加机器人接触点或 CEM 变量。

## 上层：一次投影、有限组合、梯度求力

1. 每个 MPC 窗口只取起始参考姿态的一组双手手心位置，用 Warp 投影到箱体表面。每手保留最近点和附近另外 10 点，共 11 点。候选保存在物体局部坐标系，整个预测窗口固定，不逐帧生成新候选。
2. 双手组合全集为 `11 × 11 = 121`，脚心不参与该组合搜索。GPU categorical CEM 从该有限全集不放回选取组合，同一轮物体子问题不会重复评估同一个组合。默认 `candidate_population: 24`、`candidate_iterations: 3`，每个 ADMM 物体更新最多访问 72 个不同组合。需要覆盖全部 121 种时可使用 population 121、iterations 1。
3. 每个组合只有一条连续接触力序列，没有随机力粒子。Torch autograd 对物体跟踪和 ADMM 增广目标计算力梯度，GPU 批量执行带独立回溯的投影 Adam 更新。投影满足非负法向力、摩擦锥和力上限；接触力 knots 插值到每个预测步。上层物体使用 ComFree 环境响应递推。

`force_iterations` 是梯度步数上限，`force_step_size` 是力更新步长，`force_backtracks` 控制回溯，`force_tolerance` 控制停止。已删除 `force_population`、`force_sigma` 和 `force_std_floor`。

## 下层：IPOPT 读取真实接触

`real_contact.py` 对每个 IPOPT 目标/梯度评估、每个预测步，将当前机器人状态复制到独立 scratch 环境，刷新 MJWarp 真实碰撞、质量矩阵、接触 Jacobian 和 `efc.D`。随后由 ComFree 计算状态递推和局部可微梯度。实际执行与下层预测使用同一个四点碰撞模型。

CEM 选中的点只作为 ADMM 目标，不会插入下层物理约束。如果真实手物接触不存在，手对物体的预测接触力严格为零；更改上层候选编号不会凭空改变相同 action 的物理轨迹。机器人关节、root、物体轨迹及脚心参考跟踪共同构成下层 cost。双手反力按物体受力方向输出，脚心输出实际地面支持反力。

ADMM 只在两只手的物体接触位置和接触力上协调上下层，脚心维持真实地面接触与参考跟踪，不引入额外离散组合。连续梯度通过当前接触 Jacobian、ComFree 和时间递推传递；接触拓扑本身是离散的，由下一次真实碰撞刷新重新确定。

位置等式现在仅在“请求操作 AND 实际手物接触”时启用。未操作或尚未接触的分量不进入位置惩罚、dual 更新或收敛残差。每轮 ADMM 开始用当前 action 刷新一次 mask，整轮 CEM/IPOPT 共享该固定 mask，下一轮再更新。接触失活或重新激活时清除相应旧 dual。

`environment.scsp_contact.operation_intervals` 可指定全局任务时间区间（秒），如 `[[0.5, 3.0]]`；`[]` 表示不请求操作，`null` 表示整个任务请求操作，但仍须真实接触才施加位置等式。程序接口 `HumanoidSCSP(..., operation_mask=...)` 支持每步或每步每只手的 Boolean mask。未接触时若请求操作，只使用 `approach_weight` 控制的软接近 cost，不施加位置重合等式。力一致性保留，以匹配不存在接触时的零真实力。

## 安装与运行

需要 CUDA Torch、Warp/MJWarp，以及 IPOPT 的 Python 接口 cyipopt。Linux 编译 cyipopt 前需安装系统 IPOPT 开发库、pkg-config 和编译器，例如 `coinor-libipopt-dev pkg-config build-essential`。

```bash
python -m pip install -r requirements-scsp.txt
python run.py --config configs/humanoid_move_largebox_scsp.yaml
```

当前 `.venv-contact` 已安装 cyipopt，可直接用 `.venv-contact/bin/python` 运行。配置同时设置 `solver.type: scsp` 和 `environment.dynamics: scsp`。短测设置 `environment.frames: 1`；IPOPT 每个子问题有 `ipopt_max_iter` 和 `ipopt_max_cpu_time` 预算。达到预算可以保留最佳有限迭代点，同时记录 `ipopt_success: false`，不据此声明 ADMM 收敛。

示例完整参数见 [humanoid_move_largebox_scsp.yaml](../configs/humanoid_move_largebox_scsp.yaml)。`hand_sites` 和 `foot_sites` 各须两个不同名称；附加候选数固定为 10。位置、力 ADMM 残差分别以 `position_scale` 和 `force_scale` 归一化。

## 计数与诊断

`sampled` / `unique_contact_combinations` 表示一个 MPC 求解实际访问过的不同手部组合数，始终不超过 121。同一组合可在后续 ADMM 轮次因 dual 改变而重新优化力，但不增加组合全集。history 的 `contact_combinations` 是该 ADMM 子问题访问的组合数，`contact_universe` 固定为 121。

`force_gradient_steps`、`force_gradient_batches`、`force_evaluations` 和 `force_line_search_evaluations` 独立记录连续力优化的工作量；这些目标/梯度评估不是新增 CEM 接触候选。`evaluated` 仍可高于 121，因为优化一个连续力序列和机器人 action 需要多次目标评估。

逐窗口 `scsp_contacts_*.npz` 保存四个 site ID、中心位置及实际 Jacobian、真实接触反力/间隙/激活状态、选中手部组合与物体局部点、力 knots 和 ADMM dual。`trajectory.npz` 是实际 MJWarp 执行轨迹，history 记录残差、IPOPT 状态和收敛标志。非凸局部优化和有限 CEM 预算不保证全局最优；短测完成不代表完整搬箱成功。

## 修正后的验证（2026-10-31）

四点模型、真实接触、SCSP 适配器、梯度力优化/CEM/IPOPT 和配置测试共 58 项通过。全量回归 184 项通过、1 项跳过，另有 4 项既有失败：两项原始 AST 哈希校验及两项缺少 Wuji 资产的测试。

实际 largebox 单窗口短测使用 `candidate_population: 121`、`candidate_iterations: 1`、3 次力梯度更新、1 轮 ADMM，IPOPT 限时 2 CPU 秒。接触全集和实际访问组合数均为 121，求解约 5.6 秒。四个 site 的接触记录及执行轨迹均有限，未真实接触箱子的双手反力为零。该短测达到 IPOPT 时间预算，ADMM 尚未收敛；结果在 `output/humanoid_scsp_four_point_smoke`，完整搬箱未验证。

## 实际可行性复查

默认优化预算的首个窗口耗时约 270 秒，仅执行 0.1 秒真实轨迹，4 轮 ADMM 均未收敛，最终 primal residual 约 65.08、容差约 0.304。确认下列问题后停止了剩余长测，未把该次验证标记为成功：

- 参考初始手心离箱面约 6.69 cm / 12.28 cm，整个 4.8 秒参考中四点模型都没有自然形成手物接触。复查时的旧 ADMM 对固定初态也施加了重合等式，该条件不可行；此项现已由上述操作/接触 mask 和独立软接近 cost 修正。
- 每次刷新后将质量矩阵、平滑力、Jacobian 等 detach，再对局部模型求导，所得梯度并非刷新后真实目标的完整梯度。实际机器人差分检查中，即使接触 mask 相同，也出现 autograd +2768、有限差分约 -931 的反向结果；简单推物体单元测试未覆盖这一问题。
- 同样的名义控制在 0.8 秒时，下层 ComFree 预测 root 高度约 -0.637 m，实际 MJWarp 四点执行约 +0.184 m，箱体位置相差约 0.197 m。支撑响应和预测/执行一致性需要校准。
- 四点参考初始两脚心分别高于地面约 0.81 mm / 21.32 mm，箱底已穿地约 4.19 cm。直接执行参考控制的对照（没有 SCSP 优化）在四点模型约 0.60 秒失衡，原碰撞模型约 0.83 秒失衡；这说明初始化和参考的物理可行性也需要处理。

诊断保存在 `output/humanoid_scsp_feasibility_02s/diagnostics`，其中 `optimized_prefix.json` 是实际优化后 0.1 秒前缀，`point4_nominal.json` / `original_nominal.json` 是独立名义控制对照，`gradient_diagnosis.json` 是实际任务的梯度检查。完整搬运成功率尚无结果。

## 梯度不一致的具体原因

`real_contact.py::_refresh_stage` 用 detach 后的状态调用 MJWarp，随后读取的质量逆矩阵、平滑偏置力、接触 Jacobian、阻抗等都成为 Torch 常量。前向每次评估仍会随新状态重算这些常量，反向却只对显式的 ComFree、执行器和间隙/site 局部线性项求导。若前向写作 `F(x, theta(x))`，现有反向漏掉了 `dF/dtheta * dtheta/dx`。

遗漏包括质量矩阵随姿态变化、重力/科氏力/被动阻尼随状态与速度变化，以及 Jacobian、接触法向/力方向、阻抗和执行器传动随状态变化。接触集合固定时这些量仍可能变化，所以不能把差异都归于碰撞切换。此前 48 步检查中，同一坐标的 autograd 为 +2768，差分为 -931/-937；8 步对照中，固定整个物理上下文后的差分与 autograd 一致，而每次刷新上下文后出现偏差。

修复需要补齐完整可微 FK/动力学/接触数据的链式项，或采用外层读取真实接触、内层固定同一局部模型的顺序优化，并在外层用真实目标验收。只在反向冻结、前向继续刷新不能得到该刷新目标的完整梯度。这部分尚未修复；操作/接触 mask 的修正不代表梯度问题也已解决。

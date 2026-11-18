# CARD objective 与 CEM

默认算法为 `card_wuji2_v10_paper_objective_solver`，`cem_backend='robokit'` 使用本项目扩展的 Robokit `Task`、`GaussianSampler`、`PopulationSolver` 与完整 CUDA Graph。`cem_backend='torch'` 保留独立参考计算。两者使用同一原生 Wuji2 MJCF、语义接触面、网格拟合胶囊、碰撞对和物体采样点；不修改第三方 Robokit。

## 与论文对齐的部分

依据 [CARD 式(2)、(3)、(10)–(13)及表5](https://arxiv.org/html/2610.36539v1#A4)：

- 四组代价为接触、style、smooth、collision；默认没有源动作一致性项或原生网格穿透软代价。接触点在语义指腹三角面上按式(2)求解，位置、方向与材料点防滑按式(3)计分。所有手指保留式(10)的三项 style。
- 接触优先级 `rho=1+0.5*F/current_frame_max_force`。当前帧力全部为零时 `rho=1`；历史及离线片段的最大压力不参与代价。
- 指间胶囊间距使用6 mm软铰链代价；掌指胶囊间距小于6 mm时硬拒绝。伪物体和实测物体均使用2 mm软代价 allowance、3 mm硬穿透门槛，权重分别3.5与14。
- 原生网格0.5 mm检查只输出诊断，不影响排序、`solution_feasible` 或历史更新。`collision.mesh_hard_rejected` 固定为 false，实际超限见 `mesh_diagnostic_exceeded`；`mesh_role='diagnostic_only'`。论文可行不代表网格无穿透。
- 有上一帧可行解时，均值为 `0.5*previous+0.5*initializer`，并投影到上一输出的关节±0.25 rad与平移20 mm球内；首帧只限制原生关节位置，没有初值附近的局部搜索边界。不增加腕部旋转速度上限。
- 两后端均在相对上一解连续展开的轴角图中直接采样、计算最近图旋转差并加权；不使用 SO(3) 迭代均值替代式(13)。
- 固定16轮、每轮1024个高斯候选，无强制零噪声候选、源动作局部候选或搬移历史候选。精英数为51，有限精英代价按总体标准差标准化，softmax温度0.1。噪声为 `sigma_i=sigma_0*0.2**(i/16)`，`i=0..15`，初始噪声为2 mm／0.016 rad／0.05 rad，不拟合方差。16次更新后的噪声尺度为0.2；最后一批实际使用的尺度为 `0.2**(15/16)`。
- 当前热启动均值、上一帧解和最终均值均重新计分；返回全部已评分可行候选中代价最低者，没有256候选网格验收截断。通常首帧评价16386次，后续有上一解时16387次；最终报告代价和网格诊断的额外计算不计入搜索评价次数。
- 有限精英不足51个时使用所有有限精英；没有有限精英时保持均值、继续固定预算，不优化违约量。全部不可行时显示当前初值在位置／速度限制内的投影，明确标记不可行，不更新有效历史。

`settings.objective_profile` 区分默认 `paper` 和显式实验配置。保留的 `coulomb_force`、对指闭合、非零运动一致性权重和接触指style缩放属于实验；显式力锥和对指实验走 Torch 并报告原因。旧的非零 `motion_sample_fraction` 已移除，会报配置错误。`wmesh` 为兼容保留，默认0，不参与论文目标函数。

## 保留的项目适配

### 轻触输入与阈值

触觉分布的绿色响应来自 ADC；黄色确认接触点由 `moxian_web/contact.py` 的检测器产生，CEM 使用后者。检测器对40 ms EMA后的去皮残差判定：相邻单元达到 `max(0.5 ADC, 2.5σEMA)` 并持续30 ms，或单个单元达到 `max(0.75 ADC, 3σEMA)` 并持续80 ms。单点确认还要求最近25 ms内有原始ADC支持，防止尖峰经EMA拖尾变成接触。释放阈值为 `max(0.25 ADC, 1.25σEMA)`，分区释放防抖80 ms。

新去皮记录测量平滑后的噪声；旧记录按120 Hz EMA方差折算，自动兼容，无需丢弃标定。噪声标准差下限为0.25 ADC。持续1 ADC、只覆盖一个单元的轻触也可形成CEM锚点；已确认接触不会再经过绝对压力门槛。网页“触觉分布”中的 `>5 ADC` 只是显示统计，不控制黄色点或CEM。

接触激活与求解可行性分别验收：CEM仍执行掌指间距和物体穿透硬约束；黄色点已亮而结果不可行时，应检查 `solver.feasible` 和碰撞诊断。

一致性范围为 **objective 和 CEM**，不包含整条源数据与初值流程。Robokit 初解仍由当前源手21节点优化产生，用自己的关节历史热启动并缓存完全相同输入，不读取 CEM 历史；没有替换成论文附录C的闭式初值或拇指仿射映射。

实时路径保留因果接触提取和累计凸包，输出原始 CEM。离线入口保留整段接触凸包、11帧接触图表平滑及经可行性检查的 SG(15,3) 后处理；这是独立于逐帧 CEM 的阶段。离线段最大压力仍可作为预处理元数据保存，但不参与 objective。相对ADC响应未标定为牛顿；默认方向项为法向对齐，不是硬件力控制。

恢复论文硬约束后，现有保守胶囊可能令某些源姿态的初解及整个小噪声分布不可行。程序会报告真实不可行，不扩大噪声、修改初值或放宽门槛。碰撞只检查候选端点，不检查帧间扫掠轨迹。

## 诊断与验证

`solver.consistency_scope='objective_and_cem_only'`；`search_center`、`previous_used_as_search_center`、`empty_elite_iterations`、`output_selection_scope='all_scored_candidates'` 和实际 `evaluations` 说明本帧求解过程。`backend_timing_ms` 是 `timing_ms.cem` 的子项，不应重复相加；首次构建／图捕获由 `cold_start` 标记。

录制片段通过 `scripts/retarget_moxian_wuji2.py --solver cem --input <recording.jsonl> --output <new.jsonl>` 离线重算；输出路径必须新建。

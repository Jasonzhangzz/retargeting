# 固定模板接触的 MPPI 抓取

```bash
.venv-contact/bin/python run.py --config configs/grasp_template_mppi.yaml
```

查看最终抓取姿态：`.venv-contact/bin/python visualize.py`，默认选择最新全尺寸
软接触、GPU FK 批次，请以启动日志中实际监听端口为准。可用
`--batch output目录名 --port 8085` 切换结果与端口。
该入口只展示最终优化姿态，省去阶段选择及接触建立/评测轨迹播放。新结果绿色为
已贴合 patch，亮绿大点为内部接触，橙色为未贴合 patch，蓝色为该接触的物面投影。
历史结果没有 patch 诊断时，仍展示原模板点及投影，并明确标为模板点贴合率。

该入口直接读取 Shadow → WUJI2 的固定骨骼姿态、body-local 接触点和接触法向，
不读取 ADMM 或历史 MPPI 抓姿。物体网格按 `task.options.scale` 缩放后中心化一次，
对象坐标系始终固定；`object_frame.json` 记录输入偏移。
一般类型的接触点直接来自原始标注，不应用 Type 11 的附加掌部点。
按用户更新的接触定义，Type 6、7 在初始化前将每个 distal 锚点固定到指腹中段
（末节长度 30%–70%，实际锚点约 45%–52%，掌侧法向点积大于 0.9），记录策略
`fixed_prismatic_pad_center_v1`；原始映射点与法向保存在 provenance 中，原始标注不改写。
Type 8 保留原始点。`load_template_contacts` 仍提供原始映射用于审计，求解入口使用
`load_solver_template_contacts` 生成版本化固定模板。
因此 Type 11 有 10 个接触。已核验来源哈希的映射检查文件只可复用骨骼转换角度。

初始化先固定模板手指角度及所有 local 接触，在精确 **1×** 物体上求物体相对 wrist
的六维位姿。复用 GraspADMM 使用的 Dexonomy `HandObjMatcher`：从表面采样点和
平面内方向产生多个刚体初值，批量 Adam 优化，再按原 matcher 的全点平均
`1000*距离² + 法向向量差²` 选择最佳有限解。配置中的 `initialization` 控制采样点数、
每点方向数和优化轮数；默认 24×8 个候选、60 轮。该初始化只使用距离与法向。

每个模板及其物体形状只拟合一次标准 object-in-wrist 旋转和平移；不同尺寸以该标准摆放
为参考，平移不随尺寸缩放。随后以 **1× 最终抓姿**的物体接触对应和关节角作为来源，把接触点按实际半径/高度倍率映射并
投影到新物体表面，先以 CUDA 批量有界 LM 求解新的 26 维 hand q，再由 MPPI 探索。
尺寸 IK 允许 root 和全部关节变化，1× 手指角度只作种子，不作姿态约束。即使尺寸
网格没有 1×（例如默认 Type 26），也单独生成精确 1× 参考。厚度实验的 1× 保留
尺寸策略的固定宽度；显式 radius/height 配置以未乘倍率的尺寸为基准。
自定义 `task.object` 网格经 `task.options.scale` 缩放后视作 1×。

实现仍以固定物体坐标系作为 world：把选出的 object-in-wrist 变换取逆作为
hand root 初值，与将物体放在固定 wrist 周围完全等价。`grasp_wrist_rz` 是 wrist
坐标系，`T_A_B` 表示把 B 系坐标映射到 A 系。初始化参考固定，随后 MPPI 仍可
调整全部 26 维 hand q。尺寸 IK 会先适应张开/收拢角度，不复用历史优化姿态。
默认 `pose_weight: 0`，没有把关节角拉回 1× 模板的代价。
该刚体匹配提供初值，不保证固定手指姿态已经达到最终的 80% 接触贴合要求。

全量初始化先用 `WujiContactKinematics` 将全部固定模板一次送入 CUDA FK 批处理，
在 GPU 上计算接触点和法向；matcher 复用该结果，不再调用 CPU FK。MPPI 的所有
候选 FK 使用 MuJoCo-Warp CUDA。CPU 负责静态资产准备及独立结果校验。
`template_initialization/manifest.json` 记录模板 FK 后端和批量大小，每例诊断中的
`fk_backend` 记录候选 FK 后端；CPU 参考路径仅用于测试和独立校验。

MPPI 外层仅采样二十六维 hand q。每个候选重算固定模板接触的 FK、物面投影、
投影法向、距离及法向误差，并区分手物软压入与手部自穿透。内层仅优化每个模板接触的非负
法向预载幅值，ComFree 计算摩擦响应，通过十二个正负轴向扰动得到力闭合代价。
初始接触距离不会关闭投影力闭合计算；贴合程度由独立的距离与法向代价引导。

`solver` 控制 MPPI 采样预算；`environment.template_mppi.objective` 控制几何与
ComFree 目标，默认两阶段。`object_mass` 为预测中的物体质量（千克）。
`task.templates` 接受编号或完整模板名，原有 `radius_scales`、`height_scales` 和
`task.options` 的 `primitive`、`radius`、`height` 可选择物体；也可设置
`task.object` 指定自定义网格。未提供尺寸列表时使用现有 taxonomy 尺寸策略。
自定义网格需闭合且法向朝外；顶点完全重合的接缝会在拓扑检查时合并，几何保持不变。

## 代价与预载求解

距离和法向联合评分，选取最好的 `ceil(0.8*n)` 个固定 slot，另保留 0.05 倍全点
引导。两个分项使用相同 slot。默认距离尺度 2 mm；原模板点覆盖率要求有符号距离在
`[-hand_object_allowance_m, 2 mm]` 内，且
手部外法向与物体内法向夹角 ≤30°。法向项使用
`(1-cos(theta))/(1-cos(30°))`，因此 30° 和 2 mm 的单点代价均为 1；旧的
`(1-cos(theta))²` 在 30° 时仅为 0.018，几乎不能纠正法向。
每个原生评测必需的 body 组另取组内最小联合误差，防止整个掌部组被 top-k 丢掉。
还累计各扰动下按实际响应力占比加权的联合误差，惩罚依赖远离物面的虚拟支撑点；
投影接触仍不按真实间隙开关。其余物理关节边界仍强制满足。

### 软接触与穿透限值

ComFree 已使用柔顺阻抗响应及带扭转的软指摩擦；原来的外层穿透代价和验收却接近
零穿透。现在使用 **2 mm 手物软压入阈值**，适用于全手与物体的碰撞，不仅是模板点：

- `hand_object_allowance_m: 0.002`：手物软压入阈值；范围内距离项视为贴合。
- `soft_penetration_weight: 0.1`：弱压入代价 `0.1*(手物压入/2 mm)^2`。
- 超过软阈值的部分使用 `penetration_weight: 1e4`、`penetration_scale_m: 0.001` 的平方惩罚；
  最终验收不按手物压入深度一票否决，因此 2.0019 mm 不会丢掉已有力闭合证据。
- 手部自穿透没有软层豁免，仍以 `1e4*(自穿透/1 mm)^2` 惩罚；最终只允许默认
  `penetration_tolerance_m: 1e-5` 的数值误差。零距离的自接触允许。
- `collision_clearance_m`、`collision_margin_m` 默认均为零，检测实际重叠，取消
  正间隙要求，也避开此前已复现的正 margin 下 GPU 虚假深自碰撞问题。

压入 0.5 mm 的手物项现在仅为 0.00625；2 mm 为 0.1；3 mm 超限后的代价超过
10000。相同 0.5 mm 手部自穿透仍产生 2500 的代价。设置 allowance 为零可恢复
手物零压入的强引导，但仍不是手物硬验收门槛。这是运动学搜索的软压入容差和 ComFree 投影接触响应；不会模拟
手指材料形变，也没有新增原生动态仿真。MuJoCo 的柔顺接触也允许非零穿透，
但具体压入限值是本项目可配置的搜索设置，见 [MuJoCo 约束模型](https://mujoco.readthedocs.io/en/stable/computation.html)。

内层对一组共享的 `alpha[n]` 求解，默认初值 2 N、上界 20 N，LM/GN 共 8 轮，
初始 damping 为 0.01；入口配置使用 2 轮快速预设，可改为 8 轮精化预载。
最终 q 保持不变，输出前另用至少 8 轮 GN 精化预载；已有的实际力闭合证据不会
因精化求解未收敛而丢弃。`optimization.json` 保留快速搜索代价，最终投影文件
记录精化代价、预载及其额外耗时。
每个投影点默认构造六条固定摩擦射线：四条 `inward ± mu*tangent` 滑动射线，
两条 `inward` 加正负法向自由扭矩的扭转射线。默认滑动系数 `mu=0.6`，
`torsional_friction_m=0.02`，满足 `abs(tau_normal) <= 0.02 * normal_force`，
扭矩单位 Nm。采用共享滑动/扭转预算的保守棱锥，不能在零法向力时产生自由扭矩；
设置 `torsional_friction_m: 0` 可复现无扭转的四射线模型。扭转系数以米为单位，
与 [MuJoCo 的接触摩擦定义](https://mujoco.readthedocs.io/en/stable/computation.html#contact) 一致。
预载按射线数 R 均分，通过 `gap = normal_J*z - alpha/(R*kappa)` 转成虚拟法向压缩，
射线有效质量与预载无关。静态响应由原有 Torch ComFree 的分段线性公式导出，
有限时长路径直接调用原有原语；手推前向灵敏度经过 autograd 和有限差分检查。
没有优化额外的三维力方向或实际物体状态。

响应场景为零外载加正负六维轴向外载。令 `w=(force, torque/radius)`，
测试幅值默认为 `max(object_mass*9.81, 1 N)`。默认 `response_mode: equilibrium`
固定投影及虚拟压缩，求解原生分段线性响应的静态平衡：
`lambda = relu(alpha/R - drag * A * u)`、`A.T * lambda + w = 0`。
六维 `u` 是由响应平衡导出的虚拟状态，不是额外的抓取优化变量；外层仍是 26 维 q，
内层仍只优化 n 个共享预载。每次响应最多 16 轮半光滑 Newton，带批量回溯，
每个场景保留求解途中残余最小的有限实际响应及其对应梯度，避免不可行布局的
无下界虚拟状态把残差任意放大。它仍是有限预算残差代理，不保证求得全局最小
不可行残差；退化布局不会被正则矩阵的可逆性伪装成闭合。
cost 为最坏场景平方残差加 0.25 倍平均残差，另有实际响应力的弱平方正则
（默认权重 `1e-4`）和预载正则。原先 `0.01` 的响应力正则会让可闭合的预载
比未闭合的预载更贵，已降低其优先级。
射线质量是 impedance 加权有效质量近似，输出元数据明确记录它与真实引擎
约束质量装配的区别。

旧版 `response_mode: finite_horizon` 保留用于复现：64 步、步长 0.01 s，末 16 步
残余 wrench 计分，另有弱虚拟状态正则。它衡量有限时长响应，不能单凭尚未平衡的
瞬态判定接触锥不闭合。默认静态模式不再把动态响应收敛速度当作力闭合能力。

闭合证据取实际非负射线响应力（旧有限时长模式取末段平均），检验其在同一组投影点上平衡十二种外载的能力。
默认最大归一化误差 ≤0.05；还记录保守六维 wrench 球下界
`F_test*(1/sqrt(6)-max_error)`。虚拟速度或位移不是闭合硬门，也不代表真实 hold。
开启扭转后，两点抓型也能抵抗绕接触连线的力矩，但仍需检查其他外载方向。
软指模型使用滑动/扭转系数 `[0.6, 0.02]` 的保守摩擦棱锥近似。

每例保存以下证据：

- `template_contacts.json`：固定接触、模板角度及来源；优化期间不会重选手表面点。
- `template_initialization/<模板>_<形状>/reference.json`：1× 参考尺寸、object↔wrist
  变换、选中物面点/法向（物体系及 wrist 系）、候选分项及模板身份哈希。
- 同目录 `scene_1x.xml`、`object_1x.obj`、`candidates.npz`：可回放的参考场景及所有
  1× 候选、投影点和内法向。
- 每例 `initialization.json` 和 `initialization_candidates.npz`：引用的同一 1× 参考、
  参考 SHA256、该例初始 q 和原始 1× 候选。并行 worker 直接读取参考，不重新拟合。
- `optimization.json`、`trajectory.npz`：MPPI 成本与实际优化轨迹。
- `projection_force_closure.json`：最终投影、法向、预载/GN 和扰动诊断。
- `grasp.npz`：最终抓姿，含 26 维 q 和兼容旧工具的 27 维四元数格式。
- `summary.json`：优化完成状态和投影几何成功状态。

模板 MPPI 已移除原生物理评测，不调用过滤、squeeze/pregrasp 或六方向物理测试，
也不生成 `evaluation/` 或 `native_candidate.*`。`evaluate_native` 仅接受 `false`。
`task_success` 现在使用 `success_criterion: projected_soft_contact_closure_v2`：
计算有效、投影摩擦锥闭合且手部自穿透不超过数值误差。
**手物压入深度和 80% 覆盖率继续作为引导和诊断，不再作为成功硬门槛**。
`soft_contact_success` 记录上述验收；`geometry_success` 另外要求 80% 覆盖率。
`hand_object_within_soft_allowance` 标记是否在软层内，但不影响成功判定。
距离、法向、必需接触组和力加权贴合仍参与优化。投影虚拟接触仍可能远离真实接触，
因此通过该判据不代表真实物理 hold；结果明确记录贴合率及实际间隙。
manifest 使用 `method: mppi` 和 `objective: template_comfree_soft_contact_v4`。
viewer 显示软接触验收、手物压入及限值、自穿透，历史结果沿用历史判据。
名义预载上界不限制 ComFree 扰动响应的实际法向力，二者分别导出；扭转上限绑定
实际响应法向力。`response_intrinsic_torques_nm` 仅记录接触自由力矩，
`response_torsional_torques_nm` 为沿物体内法向的标量分量，均不含力臂产生的 `r×F`。


此求解需要 CUDA。CPU 单元测试只验证参数、结果契约和边界情况，不作为 GPU 优化
或真实力闭合性能的证据。请使用新的 `output.name` 保留既有结果。

## 全尺寸 CUDA 并行批次

Type 18 与 Type 30 使用扩大的圆盘：1× 分别为直径 160 mm 和 100 mm，厚度均为 8 mm。
Type 18 固定直径 160 mm，扫描厚度 8–16 mm（9 档）；Type 30 固定
直径 100 mm，厚度扩展到 0.25–12 mm（10 档）。Type 22 的板面为 **150×105 mm**，
面积约为原 105×73.5 mm 的 2.04 倍，同样扫描 0.25–12 mm 厚度。
Type 24 保留标准直径 12 mm、长度 50 mm，扫描直径
**0.05、0.1、0.25、0.5、1、2、4、6、9、12、15、18 mm**，最小半径 0.025 mm。
Type 18 的尺寸策略版本为 6，Type 22、24、30 为 5，Type 10、12 为 4，其余为 3。
可用 `configs/grasp_template_mppi_tip_pinch.yaml` 重算 Type 24；其所有尺寸共享精确
1×（直径 12 mm）物体 pose 初始化。扭转摩擦默认对所有类型启用，包括 Type 24，
`torsional_friction_m: 0.02`；六射线模型与正负六维 wrench 平衡计算包含力矩响应，
原生动态物理评测仍关闭。
可用 `configs/grasp_template_mppi_broad_disks.yaml` 只重算 Type 18/30；
新的 1× 位姿分别重新求解，并在各自五个尺寸间复用。

```bash
.venv-contact/bin/python run.py --config configs/grasp_template_mppi_all.yaml
```

覆盖全部 33 个类型的选择：31 个有单物体模型的类型共 183 个尺寸实例，Type 19（剪刀）
与 21（成对筷子）缺少关节/多物体资产，写入 `skipped.json`。尺寸使用共享的物理尺寸策略，
每类使用主要形状族。显式给出 `radius_scales` / `height_scales` 会改成用户自定义网格。

每例 256 个候选一次进入 CUDA，2 阶段各 16 轮，合计 8192 个采样候选。
`equilibrium_backend: cuda_kernel` 将每个候选与 wrench 场景的全部 Newton、回溯和
预载前向灵敏度计算融合为 Warp 编译的 CUDA 内核；外层有界 GN、FK、投影、碰撞和
MPPI 张量运算均在 GPU 上。CPU 负责独立案例调度、静态资产准备和写文件，迭代依赖仍按序执行。
Torch 实现保留为 CPU/自动微分数值参考，显式选择 `equilibrium_backend: torch` 可对照。

`parallel.memory_fraction: 0.8` 按启动时**空闲显存**设置预算，计入 CUDA context、
Warp 和 Torch 分配。先按接触数分别实测整例峰值，再按各任务显存额度与安全系数动态接纳并行案例；
校准案例的结果保留，不重复求解，每例结束释放进程和 CUDA context。并发数取整且存在启动/收尾，因此实际使用量可能低于 80%，
不会分配无用张量填满显存。`resources.json` 记录预算、校准峰值、并发数和设备实测峰值；
`manifest.json` / `summary.json` 每完成一例即更新。所有结果写入新的批次目录。

## 尺寸适配与物体侧面接触

`environment.template_mppi.size_ik` 默认开启：4 个 CUDA 种子、32 轮有界 LM、damping
0.01。先拟合标准物体 pose，再完成一次 1× 抓姿求解，作为所有尺寸共享的手势与物体接触
参考；不能把 root matcher 尚未建立有效手势的投影冒充 1× 抓取对应。
`bootstrap_canonical: true` 默认启用该标准抓姿阶段；标准参考使用
`template_1x_solved_grasp_reference_v2`。`canonical_grasp_directory` 可读取经过独立审计的
既有 1× 求解：逐项验证模板签名、形状尺寸与来源哈希，不复用未经检查的历史手势。
新增姿态范围后，完整重算配置默认重新求解标准抓姿；不直接使用含旧反屈/交叉姿态的缓存。
参考保留 `root_match_initial_qpos`，实际共享种子为 `canonical_grasp_qpos`；尺寸初始化
标记 `solved_canonical_contacts_cuda_lm_v2`，保存 `source_initial_qpos`，明确区分模板
骨骼、标准抓姿与新尺寸的关节初值。
接触槽位和手部 local 点/法向不变；移动的是 hand q，目标来自标准物体接触点
在新尺寸上的物面投影。位置与法向共同拟合。每次 LM 更新都重新进行原生 CUDA
全手碰撞查询，冻结当前接触的 body-local 刚性附着点与接触法向，通过可微 GPU FK
计算碰撞距离导数；候选接受也重算真实碰撞，防止纯接触 IK 产生自穿透后退回原姿态。
还使用全手表面查询点的解析 primitive SDF 作手物碰撞引导。保留各 seed 的下降历史、
候选 q、碰撞筛选分项、实际选择结果与耗时，写入 `size_ik.json`。

`initialization.json` 中 `reference_initial_qpos` 是共享的标准姿态，`initial_qpos`
是该尺寸 IK 的实际结果，两者不要求相同。若全部 IK 候选更差，保留明确标注的
`canonical_fallback_selected`，不把退回结果冒充关节已适配。手部仍满足物理关节界限，
IK 与 MPPI 都不通过姿态先验约束到 1×。

Type 6 的五个手指接触槽位指定为圆柱侧面。Type 10 每根手指按标准抓姿的法向相容性
选定一个固定侧面槽位，在全部尺寸间复用；同一手指的其他指节仍可接触圆盘上下表面，
避免把中节、末节都强压到薄侧壁。初始化投影及每轮 MPPI 投影一致。闭合的完整物体
网格仍负责内外判定和全手穿透检查；指定侧面槽位靠近顶盖不能替代侧面接触。
`canonical_side_contact_slots` 与 `side_contact_slots` 记录这些固定身份，独立审计核对
跨尺寸一致性及投影法向的轴向分量为零。patch 指标另报告逐手指侧面贴合数。

Type 7 的四个固定接触也指定为圆柱侧面。Type 6、7 使用
`prismatic_non_reversed_flexion_v1` 抓姿范围：屈伸关节下限默认 0°，外展关节保留
物理范围；`prismatic_reverse_limit_deg` 可显式允许反屈。它不要求保持任何初始角，
没有 1× 姿态回拉代价。新的指腹模板使旧指尖 1× 缓存失效，必须重新求标准抓姿。
patch 显示与贴合诊断也限制在指腹中段，不把同一指节更靠近指尖的接触计为中段贴合。

Type 10 默认半径倍率扫描 **0.8–1.3**（步长 0.05），直径 **64–104 mm**，
厚度保持 15 mm，标准参考仍为精确 1×。显式自定义网格仍由配置决定。

后续修正的姿态类型使用 `grasp_articulation_and_lateral_order_v2`：非拇指 MCP
允许小幅伸展至 −5°，PIP/DIP 不反屈，PIP 默认上限 100°。空闲手指使用较宽的
张开/收拢区间，不锁定到初始角；参与接触的手指保持正常张合自由。拇指保留模板
所需的伸展包络及对掌能力。相邻食指至小指的 PIP、DIP、指尖在掌横轴上的顺序
受到约束，Type 29 使用 10 mm 中心线间隔，其余为 4 mm；完整原生网格自碰撞
检查仍单独执行。该顺序规则可阻止没有穿透却互换位置的交叉手指。
顺序残差通过 GPU FK 求导并进入尺寸 IK，MPPI 拒绝顺序反转的候选。

尺寸修正：Type 12 为 **0.75×–1×**（6 档），Type 18 为 **1×–2×**（9 档）。
Type 22、30 扫描厚度 **0.25、0.5、0.75、1、2、4、6、8、10、12 mm**，
板面或圆盘半径保持既定值；整体主要形状策略共 183 个实例。
`configs/grasp_template_mppi_articulation.yaml` 重算对应 15 类共 97 个尺寸实例。

Type 24 可启用 `antipodal_weight: 1`、`antipodal_scale_m: 0.001`。这一 q-only 几何项
减少两个固定投影点沿接触切平面的错开，避免细长圆柱两端各有接触却指间仍有间隙。
它不更改模板点、射线方向或预载优化变量；默认对其他模板关闭。

## 仅用于统计与显示的 patch 贴合率

`anatomical_patch_interior_v1` 根据固定 Wuji2 模板点所在 body、局部位置及法向，
在解剖坐标系中确定掌侧（front/palmar）、背侧、两侧或指尖 patch。同一 body 同一
patch 的多个模板点只计一次，不同侧面保持独立；原始 slot 的顺序、局部坐标、法向
及 ComFree 接触身份不变。patch 身份由实际 Wuji2 模板位置决定，避免把 Shadow
源标注的 tip 标签套到已映射到 Wuji2 侧面的点上。

每个 patch 使用该 body 暴露表面的查询池，允许整个语义侧面内部接触，不再要求
原模板点本身贴合。剔除距语义边界或超过 45° 的真实网格棱边不足 **1 mm** 的查询点；
平滑面的三角剖分对角线不视为边缘，物体边缘规则未额外收紧。内部至少一个查询点
同时满足有符号距离 `[-2 mm, 2 mm]`、外法向反向夹角不超过 30°，即算该 patch
贴合。分母为必需 patch 数；没有内部候选时明确记为未贴合，不退回整个 body。
结果保存边缘距离、查询身份、逐 patch 状态和接触证据。

此定义只修改输出 `coverage` 和显示，旧值保留为 `template_point_coverage`。
MPPI 的距离、法向、接触分组与力加权代价，以及力闭合、软接触验收均保持原定义。
新求解仅在最终结果导出时计算 patch 诊断；重统计已有结果不会调用 MPPI 或力求解。

```bash
.venv-contact/bin/python scripts/report_template_patch_coverage.py output/已独立审计批次 output/新统计批次
```

统计批次通过绝对路径引用原模型和轨迹，单独保存摘要与 `patch_contact_coverage.json`。
`patch_coverage_audit.json` 校验原始姿态/代价/响应力文件哈希不变，并使用 CPU MuJoCo
独立核对 GPU FK 的接触证据。可视化读取这些 sidecar，仅展示最终姿态。

## 验证命令

```bash
.venv-contact/bin/python -m pytest tests/primitive/test_template_mppi_inputs.py tests/primitive/test_template_mppi_continuation.py tests/grasp/test_projected_comfree.py
```

当前精简测试覆盖输入身份、原始接触来源、尺寸延续、root matcher/FK 接口以及投影力闭合；部分 CUDA 检查需 GPU。原先完整的 MPPI 更新、分块/padding 等历史测试可从提交 a24bc7b2 恢复。

CUDA 默认缓存完整内层求解的 CUDA Graph，相同 batch/接触数重放时不重复逐算子
Python 调度；`closure.cuda_graph: false` 可选择数值相同的 eager 参考实现。
单个 q 的诊断和 MPPI 均值检查只求解实际候选，不为补齐碰撞批量重复求解预载。
MPPI 的加权均值若比当前最佳样本更差，会回退到已评估的最佳样本；原有其他入口
默认不启用这一选项。

可用以下命令分开测量 FK/碰撞/投影、闭合和完整评分，首次建图与后续耗时分开记录：

```bash
.venv-contact/bin/python scripts/benchmark_template_mppi.py \
  --case output/grasp_template_mppi_validation_20261102_v1/004_11_Power_Sphere \
  --worlds 128 --gn-iterations 2 --torsional-friction-m 0.02 \
  --repeats 5 --output /tmp/template_mppi_benchmark.json
```

## 本机实测

早期 v1 于 2026-11-02 在 RTX 5060 Ti 上，Type 1、6、9、10、11 标准尺寸各运行 seed 0，
每例使用默认 8192 采样、64 步 ComFree 和 8 轮 GN。五例均完成优化、代价下降，
最终无手物或手手穿透；中位求解 71.16 秒。贴合覆盖率依次为 50%、40%、100%、
0%、20%，五例均未通过投影闭合，原生物理通过为 **0/5**。
Type 1、10 缺少必要掌部接触；Type 6、9、11 通过初始过滤后未通过六方向外力测试。
这些结果尚未证明新方法可以稳定抓取。

完整证据位于 `output/grasp_template_mppi_validation_20261102_v1/`，包含
`validation_report.md`、`validation_metrics.json`、逐例完整诊断和原生评测。
运行后源码哈希核验通过；相关 CPU/CUDA 回归 85 项通过。旧 ComFree AST 哈希测试
在未改动的 HEAD 源码上同样失败，本次没有修改原始 kernel 或其记录哈希。

修正后的软指 v3 使用相同 8192 个外层候选、worlds 128、GN2 快速预设、扭转系数
0.02 m；每例另外以 GN8/float64 审计最终投影，保留真实响应力而不修改 q。

| Type | 接触数 | 优化秒数 | 几何覆盖率 | 投影闭合 GN8 | 原生物理通过 |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 16 | 6.89 | 43.75% | 是 | 否 |
| 6 | 5 | 3.59 | 60% | 是 | 否 |
| 9 | 2 | 2.92 | 100% | 是 | 否 |
| 10 | 11 | 6.11 | 9.09% | 是 | 否 |
| 11 | 10 | 5.00 | 10% | 是 | 否 |

中位优化时间从 71.16 s 降至 5.00 s，包含首次内层建图，尚未达到 1 s。最终预载
精化另需约 0.17 s，不修改 q，并保留较好的完整代价或已有证书。GN2 是
近似预载精化，不能称为与 GN8 完全等价；原始默认 GN8 仍可显式选择。五例均无
手物或手手穿透，原生物理仍为 0/5，不能宣称整体抓取问题已解决。Type 1/10 缺少
必要接触，Type 6/9/11 未通过六方向物理测试。五类最终投影均有实际力闭合证据，
但只有 Type 9 达到 80% 几何覆盖；扭转不会使任意两点布局自动闭合。

分项测量证明旧版 70 s 的主因是内层：每 32 个 q 的几何约 3.20 ms、闭合约
267.96 ms，8192 个候选仅评分就约 69.93 s。最终 v3 在 Type 11、每批 128 个 q
时完整评分约 62.54 ms，其中闭合约 57.68 ms；8192 个候选温启动评分约 4.00 s，
另有均值检查、MPPI 更新和首次建图。根初始化及约 1.6–1.8 s 的独立原生评测
另行记录，未混入优化秒数。

最新完整证据位于 `output/grasp_template_mppi_softfinger_validation_20261102_final_weak_regularization/`，
相关 CPU/CUDA 回归共 151 项通过；运行后核验 158 个源码文件哈希均一致。

### Type 9 和相邻尺寸继续求解

Type 9 保留原始两处固定手部锚点。原始约 5 mm 的捏合间距不能通过纯刚体初始化直接匹配直径 36 mm 的圆柱：最近点会落入同一侧／底盖局部解。现在先采样圆柱轴向，以两处对向侧面接触作为 GPU IK 目标，允许手指张开；随后使用原有 MPPI。两锚点仍由固定 body、局部点和局部法向唯一确定。反屈、手指顺序和自穿透检查继续生效。对向引导同时惩罚物面法向不对向，避免重合投影得到虚假的零轴向误差。

`run.py --config configs/grasp_template_mppi_dense.yaml` 使用已审计的 1× 抓姿作为各类型锚点，将现有每段尺寸区间细分为四份，共 640 格（包括 Type 26 额外插入的精确 1×）。从 1× 向大、小两侧推进；每格以相邻已完成结果的物面投影点按实际半径／厚度缩放，重投影后使用原有 CUDA LM IK，再运行 MPPI。保留完整原始尺寸范围。

首次尝试固定相邻结果的物体／手腕相对位姿，关节可在相邻值 ±8° 内变化。贴合或闭合不足时分别允许 2 mm／2°／12°、3 mm／3°／16° 的相邻变化，保留各次结果并选择有效性、闭合和贴合较好的解。物体位移在**物体相对手腕的逆变换**中测量，旋转采用 SO(3) 距离；禁止以恒等物体世界位姿替代连续性检查。连续性残差和信赖范围均相对于相邻尺寸，不使用固定 1× 手指姿态先验。

调度器仅提交父尺寸已经求解完成的节点，依据实测的每种接触数量峰值分配 80% 可用显存预算。GPU 负责 FK、投影、IK、候选碰撞和 ComFree；CPU 仅调度依赖并导出文件。每格保存 `continuation.json`、父结果哈希、接触运输目标、各次 IK／MPPI 和连续性度量。`scripts/audit_template_mppi_results.py` 支持独立检查相邻依赖及源结果、CPU FK／碰撞／力平衡和姿态变化；`scripts/report_template_mppi_dense.py` 导出连续性报告与 CSV。不运行原生动态物理评测。


## 本地连续尺寸预览（2026-11-03）

Type 18 的 1× 已从模板重新拟合并重算其余 32 个高度；Type 26 的中指新增 ±5°
横向角范围、手指顺序与间距均衡目标，并重算全部 18 个尺寸，保留食指接触。
当前缓存为 `results/grasp_current_20261103`，默认可视化会自动创建本地入口。

全部 31 个 grid 的 640 个求解点现加密为 39,007 个姿态，数据位于
`results/grasp_current_20261103/dense`。物体尺寸、根平移和手指关节线性插值，根旋转用
SO(3) 最短弧插值；原始尺寸和手姿逐位保留。插值点没有重新求解或做物理评测。

Viser 的尺寸滑条复用网格，仅更新节点变换和物体缩放；相机保持不动，以掌心为
固定显示参考。运行 `python visualize.py --port 8089`，默认进入修正后的 Type 18 1×。
`--batch` 仍可查看其他批次；没有加密数据的历史批次保留离散尺寸控制。

数据现已打包为紧凑姿态缓存：仅一份场景 XML，共享几何，float64 NPZ 姿态数组，见 [grasp_pose_cache.md](grasp_pose_cache.md)。

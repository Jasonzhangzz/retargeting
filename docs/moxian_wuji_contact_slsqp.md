# Moxian glove → Wuji2 触觉 SLSQP

源模型是 Moxian 原生骨架和 Qnbot SKN。同一份已保存骨长、EMF 外参、蒙皮和材料绑定驱动源表面；源 IK 使用原生编译的 Studio 同类 QP，标定保持不变，新链路不加载 MANO。机器人使用仓库的原生 Wuji Hand2.2 mesh。

## EMF → Moxian 源骨架

实时和原始录制重建默认使用 `moxian_studio_native_qp_v1`，入口 `studio_emf_solver.py`，迭代核心 `native/studio_emf_qp.cpp`。它参考本机 Studio 2026.7.1 的逐指 Gauss–Newton、active-set / Cholesky 有界小 QP，并适配当前 Moxian 25／29 自由度；不是官方固定21自由度 SDK 二进制。保留全部 MCP twist、已有骨长、RX／TX 外参、目标权重、限位与整帧运动预算，上一原始解作为初值，所有迭代 FK／解析 Jacobian／小矩阵求解在 C++ 内执行。Python `emf_qp.py` 保留作离线数值参考，不再作为默认实时源。

全量40／36轮与重试预算保持不变；没有用减少迭代或低通来提速。当前同标定60帧数值对照最大关节差约5e-8 rad，原生源求解平均约0.47 ms。该数字为离线源求解测量，完整实时延迟另含设备、SKN、retarget、碰撞验收及传输。

首次构建需要 `g++`、Eigen3 headers 和 NumPy/SciPy，可先运行 `.venv-moxian-ik/bin/python scripts/setup_moxian_studio_qp.py`。构建按 C++ 源码 SHA256 缓存到 `output/runtime/moxian-studio-qp/`，policy 公布源码hash、solver_dof及 `official_sdk_binary=False`。不会修改已保存标定。

## 材料绑定与语义图

460 个 SID 复用设备的 face、barycentric、法向偏移和 patch 分区；位置与外法向在当帧 Moxian 原始三角面上求值。人工绑定继续生效。初始化为解剖映射，不宣称已完成实物传感器位置实测。

每个激活 sensor patch 生成一个加权表面代表点，保存 SID、材料坐标、法向和 semantic_link。同一手指同一节共用语义标签，六个掌心 sensor patch 都属于 palm。只在不同语义 link 之间连边，最多20点175边；同一 link 的多团接触保留各自代表点，不互连。法向与绝对位置不参与触觉 retargeting cost。

一次求解内机器人材料坐标固定；左右手镜像同时作用于点和物理法向。新映射缓存与旧 MANO 缓存通过 policy/version 分开。SLSQP 材料对应先最大化解剖局部坐标中法向夹角不超过60°的匹配数，再优化位置和法向软代价；无兼容候选的 SID 以 `normal_valid=False`、`normal_rejected_sids` 公开，仍可视化但不参与接触图。原始 `observed_contact_mask` 保留，不将其伪装成无触觉。代表点投影再次限制三角面法向与有效映射点的强度加权法向相差不超过60°。CEM 保留原有软法向映射及缓存版本。这是解剖材料对应，均不等同于实物传感器注册，也不保证运动后的法向相同。

## Cost 和求解

源适配、cost、solver、碰撞约束分别位于 wuji_contact_source.py、wuji_contact_cost.py、wuji_contact_slsqp.py 和 wuji_contact_collision.py。CEM 通过独立几何适配读取 Moxian，保留原 cost、solver 和预算。

参考公式与文件摘要见 third_party/emf_wuji_reference/SOURCE.json。只保留骨架图及 cost 数学；本地腕部按两边实际掌部骨架对齐，所有点、法向和 Jacobian 使用同一变换，不使用 SDK/MANO 坐标常量。

当前算法为 `moxian_patch_envelope_slsqp_v4`。SLSQP 仍保留为主求解器，碰撞修复仍使用独立的有界 SLSQP。接触 patch 不再压缩为一个 glove 点，也不再把 wrist 加入接触包络：每个 patch 使用自身的 sensor 点和局部 segment/掌部骨架点构造固定方向盒状凸包近似。其表面采样与 glove 包络做双向 soft Chamfer；退化到无法形成二维局部包络的 patch 只保留诊断，不进入优化图。这个固定方向包络替代逐帧 Qhull，避免在 SLSQP 的每次目标评估中重建三维凸包。

所有可形成稳定局部包络的 patch 都保留一个很低的背景 Chamfer 权重；触觉激活 patch 根据 response 提高权重，避免整手背景几何稀释对指目标。落入包络内部的 Wuji2 表面样本不会被过滤，而是作为穿透接触 witness 获得额外权重。接触 edge 只连接仍有有效包络的活动 patch，分别匹配包络中心的长度和方向。骨架图始终保留整只手：有触觉响应的 semantic link 提高局部骨段位置／方向先验和接触项权重，没有触觉的 link 仍以低权重留在图中；因此不会因触觉拓扑变化而删掉非接触骨骼代价。上一帧速度、上上帧加速度和原有关节平滑项共同用于指尖对搓连续性。

当前唯一的 SLSQP 目标不拟合凸包。它参考 CEM 的事件代价，把 2 mm 接触容差作为 soft hinge：容差内的 glove taxel 与 Wuji2 表面 witness 代价接近零，超出容差后平滑增长；同时保留 response 加权的双向 soft Chamfer、软法向和接触中心 witness。这样接触点可以因连杆形状不同而错开，但仍要求同一 link 的接触区域、法向和相对位置一致。穿透修复仍由独立的 signed-gap SLSQP 和稠密碰撞验收负责。

求解目标的 skeleton 节点始终是完整21点，material 节点包含所有通过局部退化检查的 patch；只有触觉激活 patch 的行权重和局部接触权重提高，未激活 patch 仍以背景权重保留。为兼容旧页面，`retarget_graph.source_nodes_m` 仍可只展示激活子图；完整图在 `all_source_nodes_m`、`all_robot_nodes_m`、`all_row_weights` 和 `envelope_graph` 中公开。`gesture_orientation` 和 `gesture_prior` 分项记录 wrist 坐标系下的骨段方向／位置先验，便于确认它们确实参与了每帧目标。

令 `a=lambda_lap=150`、`b=lambda_contact_edge=15000`，默认图邻接由三部分相加：参考骨架邻接 `I-L_ref`；材料点到所属骨架节点的双向连接，单边权重为 `1/所属节点数`；接触邻接 `sqrt(b/a) × A_contact/degree_contact`。从总邻接 W 构建同一个 `L=diag(W·1)-W`，源差分为 `D=L·V_source`，机器人与源使用相同 L。

唯一 Laplacian 项为 `a/2 × sum_i(row_weight_i × ||(L·V_robot-D)_i||²)`。骨架行保留参考 mask 与求和，材料行按有效材料点数取平均。接触边默认在图算子中放大10倍，对应平方误差系数100倍；混合节点残差包含骨架连接与接触边的交叉项，因此不能拆成两项独立 loss。接触增益之后不再按总度数归一化，防止大权重被抵消。a=0时仍支持仅接触系数的退化情况。基础骨段方向、关节平滑、拇指相对角和防反转继续复用参考实现。

| 项 | 初始系数 |
|---|---:|
| 骨架 Laplacian | 150 |
| 骨段方向：阶段1 / 阶段2 | 1 / 0.1 |
| 关节平滑 / 拇指相对角 / 防反转 | 0.01 / 0.5 / 20 |
| 统一图中接触边的平方误差系数 | 15000（算子增益10倍） |
| 接触位置 / 法向 tracking | 不加入目标 |

方向先验由两部分组成：`lambda_studio_direction=1.0` 固定复用 `emf_wuji_retarget` 的 wrist 坐标系单位骨段方向项，所有非拇指 finger-chain link 每个阶段都计算；`lambda_gesture_orientation=0.08` 是触觉 link 的局部附加项，激活 link 按 `1 + active_tactile_gain` 放大，未激活 link 保留基础权重。输出的 `studio_direction_cost` 可直接确认该项是否生效。直接点云 profile 的 CEM 接触容差为 `contact_tolerance_m=0.002`，中心 witness 相对权重为 `contact_witness_weight=0.35`。

位置单位米、角度弧度。单 patch、仅同语义 link、无有效跨 link 边或接触系数为0时，回到原参考骨架图；无有效触觉或时间差超过25 ms关闭触觉项。有跨 link 接触时，材料点相对骨架的偏移通过统一图连接参与残差；整张图共同平移不改变 Laplacian。没有额外的绝对位置或法向 tracking 项，法向仅用于可视化、诊断和防穿透几何。

NLopt LD_SLSQP 只求20个关节，固定源腕部；MuJoCo 原生运动学提供 float64 点/法向解析 Jacobian。首帧两阶段预算50/80，后续无接触1/10、有接触1/80，ftol_abs=1e-4。上一有限且限位内的原始候选作为下一帧初值，碰撞拒绝不清除 warm start，也不改变不可行标记；不追加低通或 SG。实时通过原始手套输入流的 epoch 检测超过250 ms断流，求解慢而跳过中间帧不会触发重置；离线没有流 epoch 时仍按源时间差检测。时间回退、模型/绑定或后端变化会重置。输出公开 initial_guess、reset_reason、initial_qpos、排队时间与整帧输出频率。

实时 `execution_profile=realtime_stream` 使用已做解析梯度及逐项对照的 NumPy 向量化 reference cost；源骨架／手套表面与对照页面以约60 Hz更新，CEM 保留独立30 Hz派发。每个 SLSQP 阶段另设6 ms时间预算，热启动帧目标预算25 ms、首帧50 ms；最终稠密验收始终执行，因此不宣称硬实时期限。以 `2π rad/s × min(源帧间隔,1/30 s)` 作为优化变量的单帧边界，避免接触图突变时跨分支大跳；这是优化内约束，不是输出滤波。实时在上一原始解处建立 signed-gap 约束后进入阶段2，每帧最多再做一轮穿透修复，下一帧继续迭代，失败仍明确不可行。离线默认 `offline_reference` 保留完整迭代／六轮修复预算，无上述单帧速度或时间限制；可通过构造参数 `realtime=True` 显式复现实时路径。

Moxian→worker 复用同帧打包 mesh，worker→页面使用带字段形状的 Float32 几何缓冲；避免逐顶点 JSON 浮点数反复展开、序列化。求解、qpos、cost 和解析 Jacobian 保持 float64，仅可视化坐标打包。普通 Python 快照接口和离线结果仍提供展开坐标。Wuji2 对照页直接显示求解结果内同帧源 mesh，不再重复请求当前源 mesh。

权重列表只缓存兼容性元数据，以文件 inode／大小／mtime 检测更新，预览时再读取完整模型并核对摘要；不再常驻历史训练帧和网格残差。只读串口进程在连接前将 SDK 各通道历史容量从默认1024缩为16帧；实时消费者只读取 latest，同帧 EMF／ADC 序号检查、独立原始录制与标定采集不变。SDK builder 没有容量参数，因此适配集中在 `protocol.bound_live_sdk_history`，校验 SDK buffer 类型且禁止连接后修改。两项一起减少长寿命嵌套数据的 GC 扫描；只缩 SDK 缓存尚不足以消除长停顿。修复前实测主进程 generation-2 回收暂停113–224 ms，不能把这段时间算作源 QP 计算。保持 Python 循环垃圾回收开启。

GPU 每轮修复的各连杆对最深穿透 witness 在设备端归约，仅回传各对的最深样本及其精确间距／点／法向；不再把约79 MB的全部候选样本矩阵拉回 CPU。多 witness 调试模式仍保留完整路径，稠密最终验收标准不变。

## 零间距与穿透修复

离线路径在触觉优化后，对穿透原生 mesh 建立有符号最近点约束 gap(q)≥0；考虑两侧刚体运动和法向旋转，以最小关节改动投影到零间距边界。每轮刷新各连杆对的最深 witness、必要时回溯步长，并稠密验收；最多6轮、每轮60次评估。已知物体也参与约束。激活本身不强制所有 patch 相互闭合。

实时若上一原始候选已经穿透，则暂停本帧姿态 tracking，优先使用独立 SLSQP 最小化负间距平方误差，且恢复阶段始终预留6 ms求解时间，不会在前向／验收用完帧预算后只评估初值。此阶段不强求一帧内达到零间距，可在原关节限位及单帧速度边界内逐帧恢复；数值中止时也保留已经找到的有限下降候选，再经过独立稠密验收。局部修复采用自适应运动范围及回溯，只有稠密穿透深度下降才接受。

局部修复连续停滞时，先在当前腕部与物体变换下重新验收上一可行姿态；没有历史可行姿态时尝试原生张手姿态。只有端点确认无穿透，才用 SLSQP 在原单帧运动边界内向该固定端点分步前进。它不是直接替换成历史输出；每一步均独立验收，尚有穿透始终标记不可行。退出局部极小时可能暂时加深穿透，诊断记录 `escape_guide.depth_increased`；不能将此路径宣称为全程无碰撞运动规划。端点没有通过当前验收时不使用该引导，也不承诺任意障碍布局都能恢复。达到零间距后继续正常 tracking。

保留原有自穿透0.5 mm、已知物体3 mm检查；新后端进一步要求剩余穿透≤0.01 mm数值容差才可行。修复失败或预算耗尽时显示当前有限候选、不可行状态与 separation_recovery 诊断，不放宽门槛。检查使用顶点、三角面中心和固定样本，连接圈排除与原生检查一致；不构成连续三角相交或帧间扫掠证明。

## 入口

在 .venv-contact 安装 requirements.txt。为避免 glove/Wuji2 连杆长度差异诱发翻折，distal skeleton 的位置和 wrist-frame 方向只作为低权重分支先验，接触几何负责最终 distal 定位；Wuji2 每指的中间、远端关节使用更高权重，并对偏离当前弯曲符号的翻折施加软惩罚，以消除指尖位置相同造成的多解跳变。网页保留五个入口：`cem`（CARD）、`slsqp_graph`（离散接触图选择后固定图求解 Laplacian，默认）、`wuji-retargeting`（已标定 Moxian 骨架 → 原版 Wuji2）、`geort` 和 `cart`（独立接触驱动网络）。切换会重建求解历史。图方法算法版本为 `moxian_discrete_contact_graph_slsqp_v8`。

运行 `scripts/start_moxian_web.sh --port 8890` 后打开 `http://127.0.0.1:8890/?view=wuji2&solver=slsqp_graph`，直接进入并选择实时 SLSQP 双视图。紫色显示材料点，黄色显示有效触觉，机器人端红色标出法向不兼容的材料对应；源端仍显示原始实测激活。橙色显示 patch 代表点、跨语义 link 接触边及材料点到骨架的连接。页面明确显示画面后端、算法版本、求解会话及源帧时间，并报告统一图 Laplacian RMS、仅作比较诊断的接触子图 RMS、边向量 RMS、分项 cost、输出频率、整帧耗时和零间距验收状态；不可行候选、穿透恢复模式、法向拒绝数及未核验SID分区明确标注，同时展示拇指最近跨指的源／机器人材料点距。同后端断流保留最后几何并关闭接触叠层；切换后端时清掉旧后端姿态，直到对应后端的新帧到达。切换求解器、重建映射和设置物体在串口采集线程之外等待 worker。

已保存 Moxian mesh 的录制：

    .venv-contact/bin/python scripts/retarget_moxian_wuji2.py --input source.jsonl --output slsqp_graph.jsonl --solver slsqp_graph

原始 EMF/触觉，或包含原始输入的历史录制：

    .venv-contact/bin/python scripts/retarget_moxian_wuji2.py --input raw.jsonl --source-model moxian-model.json --output slsqp.jsonl --device cuda:0

重建要求设备初始化指纹、手侧、SID分区与模型匹配；原始 EMF 重新经过 Moxian QP，再驱动 SKN。动作标签仅用于评估窗口分组，不进入 IK/retargeting。缺少原始数据的 MANO 录制明确拒绝。离线拒绝覆盖输入或已有文件；--pose-only关闭触觉项，--limit限制读取帧数。

Python API 为 `ContactSLSQPRetarget(topology, cache=None, device='cpu', settings=None, object_sdf=None, objective_profile='contact_graph')`。可选 `objective_profile='contact_graph'` 先固定离散接触图，再使用旧的统一 Laplacian、骨骼方向、非穿透和时间项；此构造器仅用于图方法。原 SLSQP·CARD 入口与专用代码已删除。原版机器人重定向后端独立使用 `create_retarget('wuji-retargeting', ...)`，详见 [接入说明](moxian_wuji_reference.md)。

## 运行期校验

保留两侧实际源 mesh、SID 随动、版本与手侧匹配、接触和穿透的运行期校验。独立测试与调试比较脚本不包含在当前源码发布中。

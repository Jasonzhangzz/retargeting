# CART：可微策略与在线碰撞距离模型

当前录制协议为 `cart_diff_v5`。训练直接使用手套观测和机器人原生 FK，不求解离线 Tube，不使用关节轨迹监督或 teacher forcing。预处理会运行原版 `wuji-retargeting` IK，只为随机训练窗口提供参考初始状态。模型输出下一帧的绝对关节角；短窗口内通过预测状态连续反传，窗口之间截断计算图。

## 网页与采集

重启 `moxian_web.server`、手套 worker 和 Wuji worker，并刷新网页。应用已有源手标定后，在 CART 面板分别录制训练会话和独立验证会话，点击 **准备观测、触点与 IK**，选择 **Attention** 或 **FSQ**，再点击 **训练可微策略**。只有通过真实几何验收的正式模型才能手动启用，失败、取消和 smoke 权重不会替换活动模型。

录制保存同帧原始五指 EMF 位置、旋转、有效性和时间戳，以及转换后的 wrist 位姿、源关节角、460 路触觉二值状态、响应、SID 绑定和标定指纹。触点由标定后的手套 FK 与材料位置重建；未知触觉不会变成“无接触”。原始 ADC 响应只用于当前 CARD 接触权重，不进入策略触觉 token。

旧 `cart_contact_v3`／`cart_contact_tube_v4` 文件保留在磁盘。旧权重不能作为 v5 策略启用；网页历史 Tube、精英残影及下载入口已删除，由缓存 IK 回放替代。缺少原始 EMF 的录制不能直接训练新模型。可以显式关联匹配的原始录制：

```bash
.venv-contact/bin/python -m moxian_web.cart_cli import-emf \
  --dataset RECORDING_ID --raw /path/to/raw-capture.jsonl
```

关联检查设备身份（serial 或已有设备目录）、手侧、标定、源骨架版本和 25 ms 内的一一时间匹配，写入独立 `cart_emf.jsonl`，不改写原录制；无法匹配时需补采。不会用手套 FK 的接收器预测冒充原始 EMF。

## 逐帧录制的 wrist 三维数据

新录制会在原始帧写盘的同时，启动独立几何保存进程。它按顺序读取全部原始帧，不使用丢帧预览邮箱；停止录制后继续处理积压，全部完成才发布 `cart_capture.jsonl`。界面显示原始帧与三维数据保存进度，完成后可下载 **Wrist 三维点完整记录**。训练前的“准备观测与触点”流式读取这份记录，也能补算旧录制的几何数据。训练内存保留原有 link 对接触摘要，全部穿透见证留在完整 wrist 文件中，不会将每帧数千个未使用的源见证全部常驻内存。

所有位置均以米为单位，使用未镜像、未做训练归一化的物理 `wrist_local` 坐标。训练时另外转换到策略的统一掌部坐标。

| 文件／字段 | 保存内容 |
| --- | --- |
| `frames.jsonl` | 不可变原始 EMF、手套关节角、触觉观测，以及显式 460 维 `tactile_state`（整数 0/1）和 `tactile_valid` |
| `cart_capture.jsonl` | 与原始帧逐一对应的完整 wrist 几何快照，`schema=cart_capture_wrist_v1` |
| `emf_observation` | 实测 EMF 原始位姿、wrist 位姿、时间戳和有效性 |
| `tactile_positions_m`／`tactile_normals` | 全部 460 个税元的位置和法向，包含当前没有信号的点 |
| `tactile_state`／`tactile_valid`／`tactile_position_valid` | 原始 0/1、同步信号有效性、材料绑定有效性；过期位不作为当前标签 |
| `tactile_signal_sids`／`tactile_signal_positions_m` | 当前有效激活 SID 与其三维位置，包含有绑定的掌面税元 |
| `contacts.object_contacts` | 全部触觉接触点、法向、SID、响应和材料来源，不只保存分区中心 |
| `contacts.self_pairs` | 按方向退出深穿透后选取的 link 对见证摘要，保留原始材料来源 |
| `contacts.self_penetrations` | 现有非相邻材料 link 检测范围内，所有正深度采样见证；双向保存，不做 top-k 或去重 |

`self_penetrations` 以列保存 `positions_m`、`surface_positions_m`、双方法向、`depths_m` 和 link 名称，并保留源采样索引、三角面／重心坐标，能重建每个见证点。每个原始三角面的顶点及面心参与采样；“全部”指该检测范围内的全部离散见证，不是连续曲面的无限点集。

接触见证使用 `directional_depenetration_then_contact_v2`。内部深度超过 0.5 mm，或最近面的法向朝向错误时，先沿可信方向将查询点虚拟排斥至封闭 link 外，再查询原生接触表面。自接触采用源材料外法向的反方向，且深穿透样本必须朝向另一 link；触觉到机器人接触采用实测材料外法向。退出到封口或其他语义表面、缺失方向及中心重合无法消歧时，不回退到背面猜测接触。`contact_resolution` 记录重选和未解决采样数，`self_pairs.witness_resolution` 区分普通查询与方向退出。

这一步只修正接触面的选择，不修改原始关节角、实测触点、EMF 或 mesh；完整 `self_penetrations` 仍保存真实最近距离，`self_pairs.separation_m` 另存沿正确侧选面后的间隔。上述离散选面用于采集、真实 mesh 标签与独立验收。策略训练改为方向条件化的连续接触场，不在策略梯度路径选三角面或固定见证；无法消歧的原生标签不加入接触场回放，验收中的未解决见证直接失败。接触场实现变化会拒绝旧模型启用，不要求为这一训练改动重算已完成的原始观测。

`cart_capture_status.json` 显示进度或错误，`cart_capture_meta.json` 保存完成帧数及输入／输出校验和。中断时保留 `cart_capture.pending.jsonl`；重启可验证前缀并恢复，损坏的半行会截断，不改动原始录制。也可单独补算：

```bash
.venv-contact/bin/python -m moxian_web.cart_cli capture --dataset RECORDING_ID
.venv-contact/bin/python scripts/check_cart_capture.py
```

## 输入、输出与实时延迟

每步使用当前 Wuji2 `q_t[20]` 与 `t+1…t+10` 的实测观测，预测 `q_{t+1}`。输入 q 以弧度传入，网络内部按当前 Wuji2 关节域映射到 `[-1,1]`；网络和 Predictor 默认输出归一化绝对关节角，FK、loss 与运行时状态使用弧度。

EMF 表示为五指接收器 wrist 位置和旋转 6D，共 `[10,5,9]`，位置均值与尺度仅由训练会话拟合。左右手、EMF、触点和机器人 FK 使用同一坐标适配；反射时对旋转两侧换基，保持 SO(3)。时间输入保留实际间隔。训练窗口直接读取录制轨迹中当前帧之后 `t+1…t+10` 的 EMF、逐帧触觉三维位置、0/1 与有效性，目标是 `q_{t+1}`；下一步将读取范围整体后移一帧。轨迹尾部不足十帧时不补齐，跨断流、标定或输入 epoch 的窗口不使用。

- **Attention**：两层宽度 128、四头 cross attention，以 EMF 查询触觉 token，再由当前 q token 汇聚。触觉 token 包含时间、SID、FK 三维位置和 0/1；460 个单元均保留，有效的 0 与未知通过独立 mask 区分。全未知时使用 null token，避免 attention NaN。
- **FSQ**：当前 q、未来 EMF、有效性和时间输入 `256→256→8` encoder，经 FSQ 后进入 `256→256→20` decoder。levels 沿用 `[8,8,8,8,5,5,5,5]`，无连续旁路。该分支不读取触觉，但使用相同触觉损失。量化使用直通梯度估计，并非精确解析导数。

```python
from moxian_web.cart_training import Predictor

predictor = Predictor('output/moxian/cart/models/MODEL_ID')
q_normalized = predictor.predict(
    q_current, future_emf, emf_valid, future_dt,
    tactile_positions, tactile_state, tactile_valid,
)
# 单样本：q[20], emf[10,5,9], emf_valid[10,5], dt[10]（秒、相对 q_t 时间）
# tactile_positions[10,460,3]（米），state/valid[10,460]
# normalized=False 返回弧度。FSQ 可省略 tactile 参数。
```

实时路径缓存 11 个真实观测（当前状态时刻和 10 个未来时刻），输出第一个未来时刻对应的机器人与手套画面，显示其时间戳和实际缓冲延迟。首段等待；超过 250 ms 的断流、时间回退、seek、输入 epoch 或模型变化均重置缓冲与机器人状态。离线回放直接读取保存的未来帧，暂停或 seek 后可立即查看有完整窗口的结果；结束时不补造短尾。该模型需要真实前瞻，不能作为零延迟策略使用。

## 在线距离场、接触场与损失

碰撞模型为 link-conditioned SDF：21 个 link 的 16 维 embedding，加三层 128 宽 Softplus MLP，输出米制有符号距离（外正内负）。对外接受 q、wrist 查询点及目标 link，通过精确 FK 将点变换到 link 局部坐标。关节和查询点梯度均保留，不学习 FK。

另加方向条件化接触头：输入 link 局部查询位置、可信退出方向、固定的指腹／全 mesh 语义标记和共享 link embedding，使用 `23→128→128→128→6` Softplus MLP，连续输出表面位移及外法向。深穿透标签沿正确方向退出后生成，包含从错误外侧穿过 link 至正确退出面的样本。网络学习对应表面，不输出三角面编号。

默认热启动 2000 次几何模型更新。SDF 使用原生 mesh 的符号距离与法向监督，包含距离、符号、法向和 Eikonal 项；接触头使用原生表面位置与法向监督。每个真实并行 batch：

1. 当前策略只展开一次，保留完整 BPTT 图；将预测 q 的分离副本合批转换为自碰撞、有效触点及当前接触对应查询。
2. 用真实 mesh 标注查询及附近扰动，与全域／近表面随机点混合；距离与接触各保留最多 20 万条回放。
3. 从混合回放联合更新距离头、接触头与共享 embedding 两步。
4. 固定整个几何网络的参数，直接在保留的策略状态图上构建接触损失并反传；几何网络只在此时进入策略图，保留对输入的梯度，不重跑策略。

设置 `--sdf-warmup-steps 0` 可从 scratch 联合训练。距离与接触两个头在整个策略训练中持续更新，不会热启动后冻结。原生 mesh 查询不参与反传，部署使用对应 checkpoint 中冻结的几何网络。

自碰撞保留现有非相邻 link 碰撞对与 MCP collar 排除规则。策略阶段每个 link 取 8 个固定材料采样点；验证使用原有密集顶点与面心检查。有效触点穿透查询全部 21 个 link，包括掌面，不仅检查语义对应 link。两者仍是采样曲面检查，不声称连续三角形无相交证明。

策略默认代价为：

- 单位骨骼方向：沿用全手骨链及原方向权重。
- 相邻动作平滑：`.5 × .01 × ||q_next-q||²`。
- 当前接触位置与法向：沿用 CARD 语义曲面和响应权重，由连续接触场预测对应位置／法向，通过双方 FK 反传；自接触将双向固定材料候选进行带梯度的 softmax 汇聚。权重分别 20 和 5。
- 自穿透与触点穿透：分别按 0.5 mm 归一化后平方，权重 100。
- link 接触保持：权重 1，按连续有效触觉中同一语义 link 的当前触点集合计算平滑接触奖励；允许更换 SID、滑动，不锁定历史材料点。释放、未知和断流立即撤销。持续存在的源自接触按 link 对保持。

关闭轨迹监督、Laplacian、拇指附加代价、反折附加代价及加速度代价。速度和加速度按真实时间戳报告；单位骨骼方向、平滑、位置、法向及穿透初始权重沿用旧配置。

默认 AdamW、学习率 `1e-4`、200 轮、16 步 BPTT，**真实并行窗口数为 0（按当前空闲显存自动选择）**。一次 optimizer 更新只接收一个可同时驻留的 batch；其全部窗口在 batch 维度合批前向，合批损失只反传一次，随后立即更新参数。已取消逻辑 batch 4096 及跨显存分块累积梯度；请求数大于当前估算容量时直接降低实际 batch，不累积到请求数。最后不足一个 batch 的窗口直接更新，不复制数据。旧 `--micro-batch-size` 仅保留为并行数上限兼容参数，网页已删除分块选项。16 个窗口 × 16 帧表示 256 个损失帧，每帧计算同一组损失，不含随机候选动作搜索。窗口内时间依赖仍按 BPTT 顺序反传；不同 optimizer 更新使用前一次更新后的参数。

观测按会话转成常驻设备张量，精确 FK 将五指的同层关节合批，完整变换与关节梯度对照原生逐节点实现验证。策略的 q 反馈仍逐时间步展开；预测完成后，将整个窗口的状态合并成 `T×B`，一次计算精确 FK 和按语义 link／link 对合批的连续接触损失。各时间步沿用自己的语义分母，首帧不计加速度监测，未在窗口内截断梯度。独立真实 mesh 密集验收保持逐时间步执行，避免其临时显存随 T 放大。采样 rollout 仅计算 FK、动作反馈和查询点，不再计算一次完整接触 loss。SDF 查询合批；无效输入在张量准备阶段检查，公开 Predictor 仍逐次检查。训练启动会停止当前接触预览，减少 GPU 争用。

GPU 标签查询使用 Warp 并行处理全部查询点，触点 FK 变换、接触对应查询与穿透 softmax 按 `T×B` 合批。保留一次策略展开图后，标签与两个几何更新使用分离的查询点；随后才构建几何损失图，避免策略反传需要的几何参数被原位更新。网页显示真实并行窗口数、每轮 optimizer 更新次数、真实 mesh 验证进度及损失帧吞吐，不再展示“4096 逻辑 batch”。阶段 wall time 使用 CUDA 同步边界测量。

CUDA Attention 默认在支持 BF16 的设备上仅对 attention 算子使用 BF16 autocast；q decoder、FK、距离／接触场和损失保持 FP32，可用 `--attention-precision float32` 对照。传感器编码器和固定几何网络采用非重入 activation checkpoint，在反传时重算中间激活以扩大真实 batch；没有截断输入梯度，几何网络学习阶段仍使用原有完整梯度和 Eikonal 高阶导数。与 q 无关的 EMF／触觉编码按 `T×B` 计算，q 汇聚和 decoder 保留因果顺序。编码与 SDF 查询的临时张量有分组上限，但它们属于同一损失图，只执行一次策略 backward 和 optimizer 更新。

“12 次更新”曾表示 1326 个窗口按真实并行数 112 分成 12 个 batch。显存足够时可一次放入全部窗口，计算联合梯度后更新一次；它不等价于同时完成原有 12 次 SGD 更新，因为第 2 次及后续更新依赖前一次更新后的参数。显存自动选择不再设 256 窗口上限。全可微也不能消除 `q[t+1]=policy(q[t], observation[t+1:t+10])` 的时序依赖；同一窗口的 16 个状态不能直接当作独立样本并行求解。

随机窗口在当前帧 t 使用离线 `q_IK[t]` 初始化，默认添加 `U(-0.1,0.1) rad` 并投影到现有关节域；CLI `--initial-noise-rad 0` 或网页噪声 0 可禁用扰动。采样 rollout 与策略更新复用同一次初始扰动，窗口内随后 16 步持续反馈策略输出，不再从中性姿态进行策略预热。短窗口验证使用无噪声 IK 初始状态；独立完整会话仍从中性姿态闭环验证启动与长期反馈能力。窗口不跨断流或输入 epoch。保存恢复以每次完成的真实 batch 更新为边界，记录实际并行窗口数；精确恢复保持相同 batch。

## 离线 IK 参考初始状态

`cart_preprocess` 和训练前准备会生成 `cart_ik.npz`、`cart_ik.json` 与进度文件 `cart_ik_status.json`。使用仓库中原版 `retarget_reference.RetargetSession.for_hand("WujiHand2", "right")`，保留其两阶段求解、float32 状态和 alpha=0.2 低通。输入是已经统一到右手规范坐标的实测手套骨架；左右手先经过现有 SourceAdapter，避免把左手关节角直接当右手机器人角度。检查 20 个关节名称顺序，并保存未投影的原版滤波输出 `raw` 和符合 CART 关节域的 `q`。

每帧角度按原始时间戳对齐；断流、时间回退和输入 epoch 变化重置求解器，失败帧不能提供窗口初始状态。缓存有输入、原版实现、URDF、关节域及输出校验和；中断保存求解器 warm/filter 状态，恢复继续。它是独立侧车文件，不重写原始录制或 wrist 几何，不生成可行精英或 Tube。

在网页选择一段录制，点击 **查看／刷新 IK 轨迹**，可同步播放缓存的 Wuji2 与原始手套。支持播放、暂停、倍速、拖动和跳到下一个有效 IK 帧；显示原始时间戳、求解器回退及关节域限制。回放只做缓存角度的 FK 与手套重建，不调用 IK 求解，不依赖训练模型。录制列表可下载 `cart_ik.npz`。

IK 只用于窗口起始 `q_t`，不作为 `q_{t+1}` 的监督标签，不逐帧重置动作。窗口内部的后续状态来自策略输出，梯度连续穿过完整 BPTT。部署输入仍是当前机器人状态和未来十帧传感器观测，无需在线调用 IK。

## 与 xMimic 的实现差异及速度测量

对照版本为 [xMimic 0385122](https://github.com/Open-X-Humanoid/xMimic/tree/0385122a232279210e941c7bb28da6a0bbb42356)。它使用 Isaac Lab 批量物理环境和 PPO；[环境配置](https://github.com/Open-X-Humanoid/xMimic/blob/0385122a232279210e941c7bb28da6a0bbb42356/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py)默认 4096 环境，[PPO 配置](https://github.com/Open-X-Humanoid/xMimic/blob/0385122a232279210e941c7bb28da6a0bbb42356/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/tiangong3/agents/rsl_rl_ppo_cfg.py)每环境采样 24 步、分为 4 个 minibatch、优化 5 轮。因此每次采样是 98,304 条 transition，每个 PPO minibatch 是 24,576 条；`num_mini_batches=4` 并不是 batch size 4。PPO 不通过接触物理反传，奖励的 Torch 表达式也不意味着仿真可微。其 [MotionCommand](https://github.com/Open-X-Humanoid/xMimic/blob/0385122a232279210e941c7bb28da6a0bbb42356/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L176) 在 reset 随机抽取预先重定向轨迹的一帧，以参考 q 加均匀噪声写入模拟器；任务配置将幅度设为 ±0.1 rad。后续 policy 的当前关节状态来自模拟器，参考 q/速度另外作为 command 输入；没有中性姿态策略预热。CART 对齐其参考状态初始化方式，目标输入继续使用原有未来 EMF／触觉，不添加 IK 轨迹监督。

CART 保留用户确认的 16 步 BPTT、当前观测和在线 SDF，没有 PPO critic、优势估计、ratio clip 或 Isaac 仿真；状态转移直接为策略输出的绝对 q，梯度沿精确 FK 和碰撞代理输入返回。曾将 4096 环境数映射为逻辑窗口 batch，并通过显存分块累积；这种映射不能复现 xMimic 的并行执行，现已删除，改用显存决定的真实并行窗口数。两者算法、batch 单位和每轮更新次数不同，不能直接比较 iteration 的秒数。

2026-11-18 在本地 RTX 5060 Ti 16 GiB、同一真实训练会话（10628 帧、1326 个 16 步窗口）完成整轮验证：旧实现 83 个显存分块的训练计算约 167.9 s；当前版本全部 1326 窗口组成一个 batch，一次完整策略 backward／更新的训练计算为 **52.54 s**，其中策略 rollout 与标签收集 4.76 s、原生标签及两次几何更新 3.19 s、损失及策略反传 44.60 s。固定 32 窗口的原生选模检查另需 **3.86 s**，合计 **56.41 s**；首次数据加载及最后完整会话验收另计。此前 112 窗口、12 次更新的优化版本整轮训练约 51.93 s，说明合成大 batch 主要改变参数更新方式，并不会令总运算量缩小 12 倍。该计时仅验证运行吞吐；单轮模型没有通过几何质量验收，未启用，也没有证明分钟级收敛。机器可读记录为 `output/moxian/cart/performance/local_full_parallel_epoch_20261118T153108.json`。

替换连续接触场之前的合批优化基准（不是当前接触场版本）：2026-11-18，RTX 5060 Ti 16 GiB、Attention、16 步、实际录制的四段固定窗口、相同 checkpoint 的局部测试（排除首次编译；分块 16 测试重复这四段窗口以隔离 batch 吞吐）：

| 实现 | 窗口数 | 每次更新耗时 | 窗口／秒 |
| --- | ---: | ---: | ---: |
| 原实现 | 4 | 1.80 s | 2.23 |
| 合批与张量缓存 | 4 | 0.79–0.80 s | 4.98–5.04 |
| 合批与张量缓存 | 16 | 1.43–1.44 s | 11.15–11.21 |

基准包含一次采样 rollout、两次 SDF 更新和一次完整 BPTT 参数更新；原实现还测得每次 checkpoint 写盘约 0.01 s，占比很小。约 2.3 倍的同 batch 提升及约 5 倍的样本吞吐提升仅是局部计算测量，不是任务质量或收敛曲线结论。分块 16 的峰值 PyTorch 分配约 5.1 GiB。可用当前实现对独立分布的真实窗口复测，脚本不写模型、不更改活动选择：

替换选面前另取同一训练录制中均匀分布的 **16 个不同窗口**，实测稳定 2.05–2.09 s／批、7.73 窗口／秒（约 124 个损失帧／秒），峰值分配约 5.22 GiB。窗口内接触数量不同会改变耗时。

**引入 IK 初始化之前的连续接触场基准**使用同一组 16 个不同窗口、同一旧策略及距离头初始化，新接触头随机初始化，合批 FK 后稳定测得 **2.12–2.18 s／批，约 7.45 窗口／秒、119 个损失帧／秒**，峰值约 5.31 GiB。包含在线距离／接触标签、两个联合几何更新和一次 BPTT 更新，不包含模型发布。新旧目标不同，不能从这组局部吞吐测试推断收敛快慢，更不能把 0.8 s／小批等同于几分钟完成训练；需要完整计时与真实验证收敛曲线。

IK 初始化改动前，单窗口另测训练录制中第 600 个窗口，16 帧全部含接触：一次 16 步更新 **307 ms**，约 60 ms 用于采样 rollout、32 ms 用于原生标签和两个几何更新、211 ms 用于策略前向与反传。这个窗口与上面的 16 个不同窗口负载不同，不能直接相乘推算整个训练。独立 2048 点查询，包括 FK 和对 q／查询点的梯度，距离场约 **1.74 ms**，连续接触场约 **2.12 ms**。几何网络查询达到毫秒级不代表完整训练更新也只有几毫秒。

改为离线 IK 窗口初始化后，同一真实第 600 个窗口、16 步、±0.1 rad 初始扰动，稳定完整更新为 **261–270 ms**：采样 rollout 41–43 ms、真实标签及两个几何更新 31–32 ms、策略前向与反传 185–195 ms。首轮建立原生接触面约 3 s，未计入稳定吞吐；不代表任务已经收敛。同样的 16 个不同窗口使用 IK 初始化与 ±0.1 rad 扰动，稳定完整更新 **1.89–1.90 s／批**，约 **135 个损失帧／秒**；其中 240/256 个损失帧含接触。初次 mesh 建立／编译另计；没有验证“几分钟收敛”。

将窗口内独立的策略损失按时间合批后，在同一显卡、相同 16 个真实窗口、16 步、同初始化、±0.1 rad 扰动及两个几何更新条件下重新对照：逐时间步损失实现为 **1.966–1.968 s／完整更新**，时间合批实现为 **1.176–1.192 s／完整更新**，约 **1.66 倍吞吐**（130→215–218 个损失帧／秒）。策略前向与反传阶段约 1.60→0.84–0.85 s，collect 仍约 0.29–0.31 s；峰值 PyTorch 分配约 5.31→5.30 GiB。测试时暂停原训练进程并保留其显存，结束后继续原任务，排除并发训练干扰和首次 mesh 建立。固定几何参数时，两种执行方式的 loss 和 BPTT 梯度有数值对照；合批 FK 的浮点差异可能改变边界采样与后续 mesh 标签，完整交替训练不要求逐位相同。此处是更新吞吐测量，未包含每轮独立密集 mesh 验证，也没有证明分钟级收敛。基准加 `--serial-losses` 可运行逐时间步参考版本。

```bash
.venv-contact/bin/python scripts/benchmark_cart_training.py \
  --dataset TRAIN_ID --model CHECKPOINT_ID --batch-size 16 \
  --micro-batch-size 0 --length 16 --repeats 3

# 单窗口真实接触负载的完整更新时间；不含动作探索
.venv-contact/bin/python scripts/benchmark_cart_training.py \
  --dataset TRAIN_ID --model CHECKPOINT_ID --batch-size 1 \
  --window-start 600 --length 16 --repeats 3
```

## 碰撞代理与 loss 的精确定义

几何代理共享 16 维 link embedding，包含距离头和连续接触头，不直接学习 FK。距离分支约 3.6 万参数。对 wrist 点 `x` 与 link `i`：`x_local=R_i(q)^T(x-t_i(q))`，将 `x_local/0.05` 与 16 维 link embedding 拼接，经过 `19→128→128→128→1` MLP，三层激活为 `Softplus(beta=10)`，最终乘 `0.05` 得米制距离 `D_i(q,x)`。`D<0` 判内部，接近零用于接触保持。模型参数冻结时仍对 q 与查询点求导。

SDF 训练令真实 mesh 标签为 d，输入梯度为 g，真实法向为 n：

```text
L_sdf = mean[w(d) · SmoothL1(D/0.001, d/0.001)]
      + 0.1 · mean[softplus(-sign(d) · D/0.001)]     (|d|>0.0001)
      + 0.1 · mean[1-cos(g,n)]                      (|d|<0.005)
      + 0.1 · mean[(||g||-1)^2]
w(d) = 1/(1+(d/0.01)^2)
```

策略按窗口帧平均；接触位置／法向按语义组平衡，默认为：

```text
L_policy = 0.05 · sum_edges ||u_robot-u_glove||²
         + 0.005 · ||q_next-q||²
         + 20 · L_contact + 5 · L_normal
         + 100 · [(p_self/0.0005)² + (p_taxel/0.0005)²]
         + L_keep
```

训练中先以 `z=0.00005·softplus(-D/0.00005)` 平滑负距离，再以 `softmax(z/0.00005)` 加权汇聚，得到 `p_self` 与 `p_taxel`。MCP collar 的随姿态变化边界用 0.1 mm 尺度 sigmoid 平滑，固定机械排除保持不变；真实 mesh 验收仍使用硬阈值和最大穿透。触觉查询全部 21 个机器人 link。`L_contact` 对触觉目标使用 `rho²||x_robot-x_glove||²/0.001²`；自接触使用双方间距平方加 `0.02` 倍接触中点跟踪平方，再除 `0.001²`。`rho=1+0.5·force/force_max`（无力时为 1）。法向项分别是 `rho²(1-n_robot·n_glove)/(1-cos30°)`、自接触的 `(1+n_a·n_b)/(1-cos30°)`。

连续接触头令 `h=x_local+0.05·MLP[:3]`，法向为退出方向加 `MLP[3:]` 后的平滑归一化。其训练项为 `SmoothL1(h/0.001, h_mesh/0.001)+0.1·mean(1-cos(n_pred,n_mesh))`，加入上述 `L_sdf`，仅接受方向与语义均有效的真实 mesh 标签。

自接触对两个 link 的双向固定材料候选计算接触场对应点，使用 `alpha=softmax(-(distance²/0.001²+(1+n_a·n_b)/(1-cos30°)))` 汇聚位置、法向及中点跟踪代价；不执行 argmin，不 detach 权重。固定材料采样不是每步选面，其位置仍由双方 FK 驱动。

对持续激活的同一 link，以当前触点集合的 `a=(D/0.001)²`、`alpha=softmax(-a)`，计算 `1-sum(alpha·exp(-a/2))`，再按保持项平均。自接触使用双向当前表面采样的平均奖励。释放、未知、断流即撤销，允许同一 link 换点。

策略更新不调用原生最近三角面查询或 argmin；连续接触位置、法向及候选权重均保留输入梯度。原生 mesh 的离散查询只生成无梯度标签、采集记录和独立验收。有效性、语义 link 及固定机械拓扑仍由观测／机器人定义；FSQ 保留 STE，不将量化宣称为精确可微。学习的对应关系只是代理估计，其精度必须由真实 mesh 检验。

## 检查点、验证与命令

用于选模的真实 mesh 检查默认在第 1 轮、每 10 轮和最后一轮运行，对独立验证会话均匀选取固定 32 个窗口。`--validate-every` 调整间隔，`--validation-windows 0` 检查全部窗口。训练结束仍对完整独立会话进行真实 mesh 闭环验收，保留原几何阈值；失败不启用。此前每轮约 40 s 的步骤是 259 个验证窗口 × 16 帧的原生密集查询，不是 backward，也不是策略训练必需的标签生成。减少选模检查频率和窗口数不会以代理网络自身评分替代最终验收。

`last.pth` 保存真实 mesh 验证组合代价最低的策略及配套 SDF；`resume.pth` 保存每个真实并行 batch 更新后的完整训练状态，包括两个 optimizer、回放、窗口顺序、位置及随机状态。CPU 恢复可逐值复现；CUDA attention 的浮点归约允许数值容差。`warmup.pth` 保存热启动检查点。恢复要求配置、数据及实现指纹相同；恢复失败不会覆盖已有成功模型状态。

CUDA 合成端到端检查会启用 Torch 确定性算法，排除快速 BF16 attention 反传的非确定归约，分别检查 Attention／FSQ 的中断恢复、未来缓存、seek、尾部与 IK 回放。当前两种结构在该受控检查中的恢复权重最大差均为 0。普通训练默认使用快速 CUDA 算子，保存随机状态不等于两次独立运行逐位相同。

```bash
.venv-contact/bin/python -m moxian_web.cart_cli train \
  --datasets TRAIN_ID VALIDATION_ID --architecture attention \
  --batch-size 0

.venv-contact/bin/python -m moxian_web.cart_cli train \
  --datasets TRAIN_ID VALIDATION_ID --architecture fsq \
  --resume-run MODEL_ID
```

恢复时保持原 epochs、batch、length、seed、IK 初始噪声幅度和热启动配置，以及 checkpoint 中的实际并行数。训练不写活动模型选择文件。

改变 batch 或执行版本时不能冒充精确恢复。可在新 run 显式使用 `--initialize-from OLD_RUN_ID`，检查输入、关节域、架构、标定、统计量和数据指纹一致后初始化新 run，不复用旧 iteration 光标。同接触场版本可导入两个 optimizer、replay 并计入已有热启动。没有接触头的旧 run 仅显式迁移策略、策略 optimizer、距离头及 embedding，新接触头、几何 optimizer 与回放重新热启动完整预算。旧 run 和 checkpoint 保留。

验证同时执行短窗口和完整会话闭环，使用真实 mesh 的距离、符号及密集自穿透，报告接触、法向、骨轴、保持奖励、速度、加速度、SDF 距离误差及穿透漏检。正式启用要求全部被评估帧通过现有接触间隙 0.5 mm、穿透 0.5 mm 和法向 30° 阈值，验证中有活动接触，另要求 SDF 距离 p95≤0.5 mm、深度超过 0.5 mm 的至少 20 个穿透样本漏检率≤1%，以及至少 20 个有效接触场验证样本的位置 p95≤0.5 mm、法向 p95≤30°。未达标记为 `rejected`；smoke 始终不可启用。

```bash
.venv-contact/bin/python -m pytest -q tests/test_cart.py tests/test_cart_differentiable.py
node tests/cart_frontend.cjs
.venv-contact/bin/python scripts/check_cart_pipeline.py
```

端到端脚本从已有动作构造显式合成 EMF fixture，检查两种架构的预处理、训练、恢复、真实几何验证和缓冲回放。fixture 并非真实传感器监督或独立任务泛化证据；测试模型不会启用。实际收敛速度与接触质量需新增原始 EMF／触觉齐全的独立会话验证。

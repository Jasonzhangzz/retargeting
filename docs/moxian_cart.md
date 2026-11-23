# CART：未来 EMF、触觉与可微重定向

仓库只提供两种训练目标，使用同一套未来十帧 EMF attention 编码器和五个独立手指 decoder：

| 版本 | objective | 策略状态与输出 |
|---|---|---|
| v9，固定腕部 | `card_raw_neural_self_contact_v4_actuated` | 当前 20 关节角 → 20 个关节增量 |
| 腕部 pose | `card_raw_neural_self_contact_v5_wrist` | 当前 20 关节角＋6 维相对腕部位姿 → 对应 26 维增量 |

输入为未来十帧实测 EMF（每帧五个接收器的位置和六维朝向）、触觉点位置、二值接触状态、有效标记及实际时间间隔。IK 仅初始化窗口当前关节状态，不向网络提供未来关节角或骨骼。五个手指各输出四个关节增量，腕部版另有一个六维 decoder。窗口内保留 BPTT；昂贵的观测编码按时间和窗口批处理，状态递推保留逐帧因果关系。

## 目标与梯度

两版保留 CARD 的骨骼方向、物理指尖、末节法向、触觉接触位置/方向及材料保持项，并加入接触 link 连续性、源手自接触闭合和 5 mm 穿透超限软惩罚。预期接触手指组使用穿透势能，其他组保留 6 mm 自碰撞裕量。触觉点对全部 21 个机器人 link 查询 SDF，容许 2 mm 后开始穿透惩罚。约束属于软损失，实际效果由独立真实 mesh 验证报告。

完整 raw cost 在 batch/窗口帧求均值后归一化为 `C/(C+1e8)`，范围为 `[0,1)`。W&B 的 `train/total` 是归一化值；`train/raw_cost` 与各 CARD 项保留物理代价。策略及几何网络支持 Muon 矩阵更新与向量/embedding 的 momentum SGD。

v9 的腕部固定，掌部触觉位置和法向仅作物理诊断；手指与掌部之间的接触/自碰撞仍计算。腕部版恢复掌部接触目标，并加入按实际时间计算的腕部平移、旋转速度和加速度平滑。两版 loss 的数值不能直接比较。

几何通路包含 `(q20, x, link) → 可微逆 FK → link 局部神经 signed distance`、指定语义 pad 的神经接触点/法向预测，以及 `q20 → 自碰撞势能/深度`。策略反传不搜索三角面或最近 link；原生 mesh 查询只用于监督和独立验证。自碰撞网络显式学习值、Jacobian 和梯度方向。参见 [碰撞场](cart_q_collision.md)、[v9](cart_actuated_contacts.md) 和 [腕部版](cart_floating_wrist.md)。

## 训练

```bash
.venv-contact/bin/python -m moxian_web.cart_cli --root output/moxian/cart train \
  --datasets TRAIN_ID VALIDATION_ID --epochs 1000 --length 16 \
  --architecture attention --optimizer muon_sgd --policy-loss-scale 100000000 \
  --policy-objective card_raw_neural_self_contact_v5_wrist
```

选择 v9 时将 objective 换成 `card_raw_neural_self_contact_v4_actuated`。不指定初始化选项即从零初始化策略、几何网络和优化器，并完成 mesh 监督预热。当前 raw 录制、触觉材料绑定、标定、观测缓存与离线 IK 均保留。录制协议中的旧标识仅用于读取既有数据，不提供旧策略或旧 loss 实现。

正式训练先独立训练并验证几何网络，再冻结其参数进行策略 BPTT；冻结参数仍保留对查询点和关节输入的导数。`last.pth` 保存完整验证会话闭环真实 mesh cost 最小的策略和几何网络，`resume.pth` 保存最新训练状态；写出续训文件不会使一次新训练自动加载旧 checkpoint。显式续训仍核对实现与数据指纹。

## 本地网页部署

本机 CART 使用 `remote_card_actuated_20261123_muon_v9` 的最佳 **epoch 70** 权重，验证 loss 为 **0.0928079**。参数保存在本地 `output/moxian/cart/models/`，不提交 Git。原训练状态为 cancelled，checkpoint 通过显式实验导入记录部署，仍保留原始配置、状态和校验和，不将中止任务改写为验收通过。

导入检查严格加载策略和碰撞参数、验证输入输出/关节域，再通过 `deployment.json` 固定文件及当前推理实现的指纹。CART 面板可选择模型并回放配套录制；实时启用仍检查设备、标定与触觉绑定，不覆盖当前实时标定。

实时输入在线程内保留滑动未来帧窗口，网络输入准备不做源 mesh 接触搜索。真实 mesh 诊断在后台执行，并标明诊断帧时间，避免阻塞实时推理。腕部训练权重包含 26 维状态；指关节网页运行时拒绝直接加载，部署腕部版须由可执行腕部 pose 的接口接收额外输出。

## v9 已知质量问题

上述 epoch 70 仅表示该任务中验证 objective 最低，不表示轨迹或接触达到要求。该轮真实 mesh 验证的指部接触间隙 p95 为 33.46 mm，骨骼方向平均误差为 25.64°，最大穿透为 8.42 mm，活动接触达标率为 0。

对部署的 SDF 复查 4096 个固定随机验证点：609 个真实内部点全部漏检；全部预测距离均为正，范围约 1.46–2.63 mm，空间输入梯度范数均值约 0.000409。该场近似塌缩为常数，触觉穿透惩罚可能为零而实际 mesh 仍然穿透。q-only 自碰撞场的梯度余弦均值约 0.517，约 15.6% 的有效梯度方向相反。这些实测限制是实验部署标记的原因。

验证 loss 0.0928079 对应 raw cost 10230238，经 `C/(C+1e8)` 得到；不能以其数值小推断物理误差小。修复方向是独立验证距离/符号/法向及关节梯度，防止未学好的几何场驱动策略，并检查完整闭环轨迹；继续降低相同的代理 loss 或仅增加腕部自由度不能证明穿透得到改善。


## v10 几何修复与训练门槛

原 v10 也出现了距离场近似常正值的问题，已停止。新训练的距离场学习刚性 link 的局部表面，不再让随机初始化的网络同时拟合关节运动学。世界/人手 wrist 坐标查询先经过精确可微的逆 link 变换，梯度沿该变换传回手指和相对腕部 pose；自碰撞势能仍只输入 20 维 q，不输入绝对腕部 pose。

距离场与接触投影从零使用 Muon 训练，矩阵初始学习率为 1e-3，向量/embedding 使用较小的 momentum SGD 步长，避免沿用预训练模型 1e-5 的微调学习率。接触投影同时拟合位置、法向和局部投影 Jacobian（仅监督阶段查询原生特征）。q 势能除总势能梯度外，增加随机分组投影的 Jacobian 监督，以约束不同碰撞组的导数。原有 CARD 项、触觉点 loss、接触闭合及 5 mm 穿透软上限保持原定义。

正式任务的策略更新前必须通过 `geometry_acceptance.json`：SDF 距离 p95 ≤0.5 mm、内部点漏检 ≤1%、空间梯度均值范数 0.8–1.2、方向余弦 ≥0.9；接触位置 p95 ≤0.5 mm、法向 p95 ≤30°、投影方向导数误差均值 ≤0.25，并单独验证 25 mm 扰动的远距离投影（位置 p95 ≤1 mm、方向导数误差 ≤0.25）；q 碰撞势能与 5 mm guard 的梯度余弦 ≥0.8，真实 mesh 总代价及超限穿透势能的退出下降率 ≥90%，并满足穿透验证覆盖与漏检要求。验证不过只继续几何训练，达到预算则终止，不会启动策略。通过后写出 `geometry.pth`，冻结几何网络并保留输入梯度。

W&B 的 `geometry/validation_*` 显示距离、漏检、空间梯度、投影 Jacobian 和真实 mesh 退出指标；`geometry/accepted` 表示是否允许策略开始。每次策略验证增加完整录制的连续闭环检查，结果保存到 `closed_loop_validation.json`、`validation_motion_latest.json` 和 `closed_loop/*` 曲线。低归一化 loss 仍不代表通过物理验收。

# CART：未来 EMF 输入与五指独立 decoder

CART 使用当前机器人关节角和未来十帧实测 EMF／触觉观测，预测下一步关节增量。仅保留五指独立 decoder；共享 decoder、历史关节网络、未来 IK/q 输入网络及其迁移入口已移除。原版 wuji-retargeting 作为独立 baseline 和窗口初始状态求解器保留。

## 输入和网络

| 输入 | 单样本形状与单位 |
| --- | --- |
| 当前 q | `[20]`，弧度 |
| 未来 EMF | `[10,5,9]`，五指接收器位置（米）与旋转矩阵前两列 |
| EMF 有效性 | `[10,5]` |
| 时间差 | `[10]`，相对当前 q 的秒数 |
| 未来触觉位置 | `[10,460,3]`，米 |
| 触觉状态／有效性 | `[10,460]`，0/1 与独立有效性 mask |

`cart_observations.py` 保存实测数据并完成坐标转换；`cart_emf_observations.py` 统一训练和推理的 `t+1…t+10` 打包。左右手的位置与旋转换到同一掌部坐标，反射通过旋转两侧换基保持 SO(3)。缺失 EMF、时间回退、超过 250 ms 的断流和不足十帧的尾部不补造观测。

Attention 使用两层宽度 128、四头 cross attention：EMF 查询触觉 token，当前 q 汇聚共享 context。FSQ 分支将当前 q、未来 EMF、mask 和时间输入 `530→256→256→8` encoder，再经有限标量量化；它不读取触觉 token。两个分支均使用五个独立 MLP：

```text
context → 256 → LayerNorm → GELU → 256 → GELU → 4 → Tanh
```

按 thumb、index、middle、ring、pinky 拼接成 20 维输出。参数互不共享，观察 context 共享，允许指间协调。策略输入没有未来骨架或未来 IK，当前帧 IK 只在窗口开始时提供 q；之后一直反馈预测状态。

```text
delta_norm = 2 * delta_rad / (upper - lower)
q_next_norm = clamp(normalize(q_current) + delta_norm, -1, 1)
```

输出层以零初始化，使新策略先保持当前状态。状态反馈保留完整窗口 BPTT；关节域投影在区间内部保留梯度，窗口之间截断。FSQ 使用直通估计。

## 数据和训练

网页 CART 面板分别录制训练与独立验证会话，再准备观测、触点和初始 IK。原始 `frames.jsonl`、完整 wrist 几何 `cart_capture.jsonl`、标定和时间戳保持不变；预处理与 CARD 目标使用独立缓存。旧录制格式可读取，不代表支持旧策略网络。缺少实测 EMF 的数据可通过 `cart_cli import-emf` 显式关联同会话原始录制，不能用 FK 预测代替传感器读数。

当前训练路径使用 `card_raw_neural_surface_v2`：保留 CARD 的骨骼方向、物理指尖、末节法向、接触位置／方向和材料保持表达式，加入 link 接触连续性及 5 mm 穿透软惩罚。完整 raw cost 对 batch 和窗口帧求均值后再计算：

```text
L = C / (C + 1_000_000)
```

`train/total` 和 `validation/loss` 是归一化值，`train/raw_cost`、`validation/raw_cost` 及分项保留原始尺度。指尖分项为 `3² × sum_5tips(||error_mm||²)`，五指合并 RMSE 可由 `sqrt(mean(style_tip)/45)` 还原。当前关节平滑权重仍为 `0.1²`，加速度只监控，不进入 loss。

接触位置／法向由连续 nearest-pad 网络预测；触觉点经 wrist 坐标适配后交给 q/point/link 条件化 Neural SDF。自碰撞使用直接输入 q 的碰撞势能网络，对值、关节 Jacobian 和退出方向进行监督，详见 [q 自碰撞模型](cart_q_collision.md)。策略求导时冻结几何参数，但保留对输入 q 和查询点的梯度；原生三角面查询仅用于独立标签与验收。

每个真实并行 batch 展开一次策略，保留状态图；用分离后的预测状态采集原生标签，更新几何网络两步，再通过冻结的几何网络计算策略 loss 和 BPTT。不将请求 batch 拆成多次梯度累积。

```bash
.venv-contact/bin/python -m moxian_web.cart_cli train \
  --datasets TRAIN_ID VALIDATION_ID --job-id RUN_ID \
  --architecture attention --epochs 1000 --length 16 \
  --batch-size 4096 --window-stride 2 \
  --policy-objective card_raw_neural_surface_v2 \
  --policy-loss-scale 1000000 --optimizer muon_sgd
```

Muon 更新矩阵参数，momentum SGD 更新 embedding、偏置和归一化参数。支持显存预算、BF16 attention、encoder 重算、cosine 学习率及断点恢复。`--initialize-from` 只接受相同五指 EMF 架构；几何迁移使用独立选项。W&B 为可选标量跟踪，不上传录制或权重。

验证在独立固定窗口上周期执行，完整会话验收在训练结束执行。`last.pth` 保存验证总 loss 最低的 policy 和完整几何权重；`resume.pth` 保存最新优化器、回放、随机状态和训练位置。两者用途不同。

## 推理和验收

```python
from moxian_web.cart_training import Predictor

predictor = Predictor('output/moxian/cart/models/ACCEPTED_MODEL')
delta = predictor.predict(q, emf, emf_valid, dt, taxel_positions, state, valid)
q_next = predictor.predict_next(
    q, emf, emf_valid, dt, taxel_positions, state, valid, normalized=False,
)
```

默认输出归一化增量；`normalized=False` 返回弧度。实时路径缓冲真实前瞻窗口，模型不是零延迟策略。断流、seek、输入 epoch 或模型变化重置状态。只有通过原生 mesh、SDF 和接触场验收的正式模型才能启用；5 mm 是训练软惩罚，不能替代真实几何检查。

本机候选权重放在被 Git 忽略的 `data/checkpoints/cart/`，附原始配置和验收状态。1000 轮实验 `remote_card_emf_20261121_muon_v6` 的最优 epoch 为 260，验证 loss 为 0.706505；完整验收未通过，候选权重未作为正式模型启用。

## 导入的实验模型

经用户指定、严格加载和推理验证的候选可以部署到 `output/moxian/cart/models/`，通过 `cart_deployment.register_import(..., experimental=True)` 显式注册。`deployment.json` 绑定 checkpoint、原始配置、验收状态、源标定与本机推理实现的校验和；没有导入记录的 rejected 权重仍然拒绝加载。原始 `status.json` 和 `config.json` 不改写，网页持续显示“实验模型 · 验收未通过”。

当前已部署 epoch 260。网页 CART 面板选择该模型和验证录制 `20261118T135558_05a2628d`，点击“回放所选一段”即可使用，不需要先替换当前实时标定或启用实时 CART。结束回放恢复先前求解器，GeoRT 与 CART 分别保存自己的模型路径。

实时启用仍检查设备、手侧、源标定、材料绑定和关节域。模型训练标定与当前实时标定不一致时，显示明确错误；不会自动覆盖当前标定或跳过身份检查。

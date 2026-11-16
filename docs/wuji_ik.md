> 历史实验记录：文中部分脚本已于2026-11-16清理；当前基线入口见 [baseline说明](../retarget/baseline/README.md)。

# EMF → Wuji2 并行 IK 初值网络

代码位于 `retarget/learning/wuji_ik`，入口为 `python -m retarget.learning.wuji_ik`。
本模块蒸馏缓存中的官方 SDK 关节角；预测过程不调用 SDK、不输入上一帧关节角。
训练、评估、推理都支持完整连续段，不会在显存不足时静默缩短历史。

| variant | 信息范围 | 目标 |
|---|---|---|
| temporal_causal | 当前帧及全部过去 EMF | 归一化 q 的 MSE |
| temporal_bidirectional | 完整段，包含未来 EMF | 归一化 q 的 MSE |
| frame_regression | 每帧独立 | 归一化 q 的 MSE |
| frame_flow | 每帧独立，加带噪 q 和 flow 时间 | 条件速度场 MSE |

两个时序实现分别在 `temporal_causal.py` 和 `temporal_bidirectional.py`，共享 backbone。
四组均为 **4 层、宽度 1024、16 heads、FFN 4096、GELU、dropout 0**。
每帧五个 EMF token 加一个关节读出 token；读出 token 输出 20 维关节角。
使用 finger/side embedding、帧位置 RoPE、输入 LayerNorm、adaLN-Zero blocks。
同帧所有 token 互相可见；因果遮罩只屏蔽未来帧。回归时间条件固定为零。
生产参数量：回归模型 76,915,732；flow 模型 76,937,236。

## 数据准备

四组共用 `WUJI2_mano_sdk_v1` 的时序数据：训练 758,942、验证 97,221、测试 93,737 帧。
按记录、缺帧及 SDK reset 切分为完整连续段；原 train/val/test 和 group ID 保留。
静态姿态不进入本次对比。预处理会检查全部来源分片的 SHA256，并验证 q 的命名顺序和物理范围。

源 EMF 是 **TX 系的合成传感器位姿**。读取时按 side 对应的固定 glove TF 转到 wrist 系：
`wrist_T_RX = wrist_T_TX @ TX_T_RX`。不再乘 tip→RX 偏移，输入仍是 EMF 传感器位姿。
特征为位置除以 0.1 m，以及旋转矩阵前两列按列展开的六维表示。投影至 1024 后使用 LayerNorm。

每侧根据固定 Hand2 URDF 和缓存关节名称确定上下限，保存到准备缓存及 checkpoint：
`q_normalized = 2*(q-q_min)/(q_max-q_min)-1`。标签异常超限报错，不静默裁剪。
回归输出使用 tanh；flow 速度不限制范围，仅采样终点裁剪到 [-1,1]。
归一化与反归一化都是逐关节可逆的固定映射，不对输出目标做逐帧 LayerNorm。

在远端仓库根目录执行：

```bash
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/bin/python
SDK_CACHE=/mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/WUJI2_mano_sdk_v1
IK_DATA=/mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/WUJI2_ik_segments_v1
HAND_URDF=/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget/thirdparty/wuji-retargeting/wuji_retargeting/wuji-description/hand2/hand2_beta1/body/urdf

$PYTHON -m retarget.learning.wuji_ik prepare \
  --source "$SDK_CACHE" --output "$IK_DATA" \
  --right-urdf "$HAND_URDF/right.urdf" --left-urdf "$HAND_URDF/left.urdf" \
  --scratch-dir /dev/shm/jasonzhang_wuji_ik_prepare
```

输出 `features.npy`、`q_normalized.npy`、`frame_id.npy` 为只读 mmap 数组，`segments.json` 保存连续段及逐行来源映射。
`manifest.json` 的签名包含原缓存签名、源 manifest 摘要、URDF 摘要、关节限位与坐标转换。
已有完整准备缓存会检查摘要后复用；不覆盖不兼容输出。每个 DataLoader worker 独立打开 mmap。

## 训练与恢复

PyTorch 2.10+ 内置 Muon、Flash SDPA 和 FlexAttention；不要求第三方 `flash_attn`。
隐藏 block 的二维矩阵使用 Muon，其余参数使用 AdamW；embedding、LayerNorm、偏置不衰减。
默认 Muon lr=0.02、momentum=0.95；AdamW lr=0.0003；weight decay=0.01。
默认 20 epochs，2% warmup 后 cosine decay，梯度裁剪 1.0，seed=42。

```bash
# GPU 选择通过 CUDA_VISIBLE_DEVICES 控制；按当前资源情况设置进程数。
$PYTHON -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m retarget.learning.wuji_ik train \
  --prepared "$IK_DATA" --output output/reports/wuji_ik/temporal_causal \
  --variant temporal_causal --batch-size 1 --accumulation 8 --checkpoint-blocks
```

分别将 variant 改为另外三种，并使用独立输出目录。四组的数据顺序与 split 一致。
默认每 GPU 每微批一条完整连续段，梯度累积八次，CUDA BF16。
DDP 按全局有效关节元素数归一化梯度，不平均各 rank 各自的局部均值。
训练 sampler 每 epoch 最多舍弃 `world_size-1` 条段以保持各 rank 步数一致，配置记录舍弃数量。
评估按 rank 无重复分片，不补齐重复样本。

`--backend auto` 在无 padding 的双向/当前帧注意力上使用 Flash SDPA，在 CUDA 帧级因果/有 padding 场景使用 FlexAttention。
`sdpa` 为明确的数学参考实现，长段会占用较多显存；`flash` 强制 Flash，遇不支持的遮罩或 dtype 会报错。
日志中的 `attention_backends` 报告实际执行后端。Flex 首次编译可能较慢；短段 auto 模式允许有说明的数学回退。

`latest.pt` 保存模型、左右手归一化、两套优化器及 scheduler、每 rank RNG、epoch/批次位置、数据签名和模型配置。
`best.pt` 按验证集归一化 q MSE 选择；flow 使用固定种子、8 步生成后的 q 评估。
断点恢复要求相同模型、world size、采样及训练总步数；workers、日志/保存间隔和停止调试点可调整。
显式指定 `--resume ... --extend-epochs --epochs 200` 可延长 epoch 预算；保留模型、两套优化器、RNG 和采样位置。
学习率从 checkpoint 的当前值连续进行 cosine decay 到新终点，不重新 warmup。
延长后的调度参数写入 checkpoint，普通恢复仍可精确续训。验证历史追加到 `validation.jsonl`。

```bash
$PYTHON -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m retarget.learning.wuji_ik train \
  --prepared "$IK_DATA" --output output/reports/wuji_ik/temporal_causal \
  --variant temporal_causal --checkpoint-blocks \
  --resume output/reports/wuji_ik/temporal_causal/latest.pt
```

`--max-steps` 限制完整训练预算；`--stop-after-steps` 用于提前保存退出的冒烟测试，保留原 scheduler 总预算以便恢复。
正常关闭或取消进程时使用最后已提交的 checkpoint；没有实现逐微批的异步抢救保存。

## Flow matching

采样 `z~N(0,I)`、`t~U(0,1)`，使用 `x_t=(1-t)z+t*q_normalized`，回归速度 `q_normalized-z`。
带噪 q 只进入 flow 读出 token，真实目标不会进入任何回归模型的输入。
默认 Euler 8 步，从噪声积分到 q。支持 1/4/8/16 步评估；没有蒸馏成一步采样或加入 CFG。
默认每段共享一个 20D 初始噪声，使帧间差异不再来自独立重采样噪声。
这不保证平滑；三个回归模型均不加额外平滑损失。
评估使用 seed 和 segment ID 确定噪声，结果不依赖评估 batch/rank 分片方式。

## 评估和并行预测

```bash
$PYTHON -m retarget.learning.wuji_ik evaluate \
  --prepared "$IK_DATA" --checkpoint output/reports/wuji_ik/frame_flow/best.pt \
  --split test --flow-steps 1 4 8 16 --output output/reports/wuji_ik/frame_flow/test.json
```

也可通过 torchrun 分片评估。报告弧度 MAE/RMSE、每关节误差、左右手/数据集误差、相邻帧增量误差、吞吐量、延迟、峰值分配显存及实际 attention 后端。
计时同步 GPU、排除磁盘读取，包含首次编译，并在报告中明确注明；双向使用未来信息，必须与因果结果分别解释。
吞吐量以总帧数除以各 rank 中最大的模型推理耗时计算，不包含训练/数据加载通信。

预测 NPZ 字段必须为 `position_wrist`、`orientation_xyzw_wrist`，形状分别为 `[T,5,3/4]` 或 `[B,T,5,3/4]`。
五指顺序为 thumb/index/middle/ring/pinky，米和 xyzw。可附 `lengths[B]`；每条输入必须是单一连续段。
不要直接把原缓存的 TX 系 `emf_position` 当作 wrist 输入。

```bash
$PYTHON -m retarget.learning.wuji_ik predict \
  --checkpoint output/reports/wuji_ik/temporal_bidirectional/best.pt \
  --input wrist_emf.npz --side right --output initial_q.npz
```

返回 `q_normalized` 和 `q_rad`，不需要关节角反馈。有效范围外的 padding 输出为零，使用 lengths 判断有效帧。
Python API `predict_wrist` 允许 batch 内不同 side；CLI 一次选择一侧。

```python
from retarget.learning.wuji_ik.training import load_inference
from retarget.learning.wuji_ik.inference import predict_wrist
model, normalizer, state = load_inference('best.pt', 'cuda:0')
result = predict_wrist(model, normalizer, state['variant'], position, orientation_xyzw,
                       side='right', lengths=None)
```

网络拟合的是合成 EMF 对应的 SDK 骨架重定向标签，并不证明实测手套域上的效果。
原 SDK 标签还包含其有状态求解与滤波行为；全局模型是近似蒸馏，并不保证逐位复现串行 SDK。
本模块不修改 SDK 或接入实体手。下游 IK 收敛时间和成功率需要在允许设置初值的优化器中另行评估。

## 验证

```bash
python -m pytest tests/learning/test_wuji_ik.py -q
```

覆盖 TF/6D/归一化、来源校验、段划分、因果/未来信息/跨帧隔离、四组过拟合、Muon 参数分组、flow 目标和积分、精确 RNG 恢复、两 rank CPU torchrun，以及 CUDA Flash/Flex 与数学参考的前后向对比。
测试中的小模型仅为低成本正确性检查；生产 CLI 固定使用上述 4×1024 结构。

正式宽度的独立冒烟入口，每组只更新一次，并保存验证报告：

```bash
$PYTHON scripts/smoke_wuji_ik.py --prepared "$IK_DATA" --device cuda:0 \
  --max-frames 8 --memory-limit-gib 4 --output output/reports/wuji_ik/production_smoke.json
```

该脚本明确取一条真实训练段的短前缀用于冒烟；正式 train/evaluate/predict 不截断完整段。
加 `--inference-only --longest-segment --variants temporal_causal temporal_bidirectional` 可验证最长训练段的整段推理，优化器更新数为零。

## 四卡串行实验：每卡有效 8192 帧，5% 分层验证

`resplit` 保留原测试集，将原 train+val 合并为候选池，在 group ID 层面重新划分。
按数据集×左右手的帧数量匹配约 5%，同时匹配 20 个关节的八箱直方图。
同一 group 的左右手、多相机和所有连续段统一归属，避免邻帧/相机泄漏。
实际训练 813,432 帧、验证 42,731 帧、测试 93,737 帧；候选池验证占比 4.991%。

```bash
$PYTHON -m retarget.learning.wuji_ik resplit --prepared "$IK_DATA" \
  --output /mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/WUJI2_ik_segments_val5_v1 --fraction .05 --seed 42

$PYTHON scripts/launch_wuji_ik_queue.py \
  --prepared /mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/WUJI2_ik_segments_val5_v1 \
  --output output/reports/wuji_ik_4gpu_bs8192_val5_20261110 \
  --frames-per-gpu 8192 --epochs 200 --memory-limit-gib 12
```

队列冻结代码快照，依次启动 causal、bidirectional、frame regression、frame flow；
只有当前任务成功完成后才启动下一个。每个任务使用四个 DDP rank，各自设置 12 GiB 分配上限。
GPU 空闲显存不足时等待；任务失败会停止队列，后续任务保持 queued，不自动重试或并发启动。
`STATUS.json`、各任务 `.log` 和 `train.jsonl` 提供进度；创建队列目录中的 `STOP` 文件可终止当前任务并停止后续任务。
停止旧 controller 并确认其 torchrun 子进程退出后，使用相同命令加 `--continue-existing` 续接队列。
允许增加 `--epochs`；原配置和代码快照存入 `revisions`，当前任务从最近完整 checkpoint 继续，后续任务按新预算启动。

## W&B 曲线同步

```bash
$PYTHON -m wandb login --host https://wandb.plat.x2robot.com
$PYTHON scripts/sync_wuji_ik_wandb.py \
  --campaign output/reports/wuji_ik_4gpu_bs8192_val5_20261110 \
  --entity x2robot --project 'ik policy' --name-prefix 'ik policy' --follow
```

同步器不运行模型或占用 GPU；读取已提交 checkpoint 对应的日志，合并断点续训前后重复步数，过滤尚未保存的更新。
四种模型分别使用稳定的 run ID，按 global step 展示训练 loss、学习率、梯度范数，以及验证集整体、逐关节、左右手和数据集误差、延迟、吞吐与显存。
只同步曲线及训练配置，不上传模型、原始 EMF/q 数组或仓库代码；W&B 凭据由 SDK 的登录配置或环境变量提供。
重复执行会继续同一个 run；`--name-prefix 'ik policy'` 将四组命名为 `ik policy / variant`。
加 `--follow` 可在后台每 30 秒导入后续 checkpoint 的新曲线，训练期间保持当前 run 在线，暂停或完成后关闭该会话。
`training_state=resuming` 表示已启动续训但还未写入新训练步；不据此宣称已产生新的参数更新。
`wandb_sync.json` 保存链接和同步进度。同步命令不会启动或恢复训练。

训练期间可将验证最优 checkpoint 固定为独立硬链接或副本，再执行
`scripts/evaluate_wuji_ik_snapshot.py --checkpoint ... --prepared ... --output ...`。
该入口在一张卡上限制显存，评估整个测试集，并记录确切 checkpoint 步数与数据签名。
把报告保存为对应模型目录下的 `test_snapshot.json` 后，同步器将结果写入 W&B 的
`test_snapshot/*` summary；不会占用训练曲线的后续 global step，也不会用测试误差选择权重。

`--effective-frames-per-gpu 8192` 使用完整轨迹微批，通过梯度累积精确计入 8192 个有效损失帧。
若轨迹跨越更新边界，会在边界两侧都保留完整注意力上下文，仅使用互不重叠的 loss range。
因此每个 epoch 每个训练帧恰好计入一次，不为凑批而裁掉轨迹上下文或重复损失。
全局常规 batch 为 32,768 帧；epoch 最后一步使用剩余帧，不补重复样本，逐 rank 实际帧数写入日志。
该模式下 `batch-size=1` 表示物理微批是一条完整轨迹，固定 `accumulation` 参数不再控制更新边界。
checkpoint 保存每个 rank 各自的微批位置，以支持不同累积次数的精确恢复。

## 时序平滑训练

`train --smoothness-weight 0.1` 为 `temporal_causal` 和 `temporal_bidirectional` 加入归一化 q 的二阶差分均方惩罚：
`L = mean((q_pred-q_SDK)^2) + 0.1 * mean((q_pred[t]-2*q_pred[t-1]+q_pred[t-2])^2)`。
它直接约束预测加速度；匀速运动不受惩罚，但真实快速动作也可能受到抑制，需同时比较回归误差与跳变指标。
只在连续段内部计算，屏蔽 padding/缺帧；跨有效帧 budget 时由三帧的末帧拥有损失，保留之前的上下文，既不漏算也不重复。
回归项与平滑项按各 rank 的全局有效关节元素数分别归一化（平滑项使用有效三帧数）。
当前帧模型不允许启用该项。W&B 分开记录 `train/regression_loss`、`train/smoothness_loss`、`train/weighted_smoothness_loss` 与总 `train/loss`。
验证/最优 checkpoint 仍按原有 normalized MSE 选择，以保留回归精度的可比性。

若训练目标发生变化，使用新的输出目录和 `--initialize-from OLD/latest.pt` 只继承模型权重，重新初始化优化器、scheduler 和步数；不是精确续训。
checkpoint 和配置均记录初始化来源及来源 step。相同目标下可正常 `--resume`，更改平滑权重的精确 resume 会被拒绝。
串行队列的 `--temporal-smoothness-weight 0.1 --initialize-causal-from OLD/latest.pt` 会先训练平滑 causal，再训练平滑 bidirectional，最后训练原有的两个当前帧模型。

微调已经收敛的 causal 模型时，队列支持独立的 `--causal-muon-lr 0.003 --causal-adamw-lr 0.000045 --causal-warmup-fraction 0.005`。
在 200 epochs / 5000 steps 下 warmup 为25步，之后 cosine 衰减，学习率始终不超过这些显式上限；其它从头训练的模型仍用其原有默认学习率。
`--initialize-causal-optimizer` 配合 `--initialize-causal-from` 可保留相同训练目标 checkpoint 的 Muon/AdamW 优化状态，同时启动新的低学习率调度。
加载优化器状态后会显式覆盖源 checkpoint 的 lr、initial_lr 和 weight decay，避免旧 param_groups 把高学习率带回新任务。
这仍是独立微调任务：训练步数、数据位置及 scheduler 从头开始，初始化来源和策略保存在 checkpoint 与配置中，之后可精确 resume。

# Wuji2 RVQ 重建与分类联合训练

当前默认任务为 `--task joint`，同一模型同时重建20维 hand pose并预测31类 grasp type。结构为 `pose → encoder → RVQ → decoder共享主干 → pose头 / 类别头`，不需要物体、接触标签或腕部位姿作为输入。两个输出头共同更新decoder主干与encoder。

## 数据与划分

```bash
.venv-contact/bin/python -m retarget.learning.wuji2_rvq prepare
```

通过 `ssh -p 1016 8.147.67.185` 只读提取远端 `results/wuji2_multi_20261107_x8iizrs9` 的 Wuji2 结果，不使用51维 MANO cache。由源模型关节名称核对26维 qpos 的后20维，采用原生关节范围（包括负角度）和弧度单位。

每类5000条，共155000条；每类训练4000、验证500、测试500。同一源物体，以及按1e-5 rad量化重复的姿态组成不可跨分区的连通组。去除重复源记录，数量不足时报错，不复制补齐。1247条 primitive 模板作为额外训练锚点，移除与验证/测试重复的姿态，约占训练采样的10%。

数据存放于 `data/datasets/wuji2_rvq_v1/`。`manifest.json` 记录源文件指纹、原生范围、关节顺序和分区计数，加载时验证 `poses.npz` 哈希。入选要求 valid、有限数值和原生关节范围；contact/stable仅作为分组评估标记。手势分类准确不等于物理抓取成功。

## 训练目标与模型选择

Encoder 为 `20→256→256→64`，隐藏层 LayerNorm + GELU。四层独立EMA残差码本，每层256个64维向量，EMA衰减0.99，距离与统计使用float32。Decoder的共享主干为 `64→256→256`，每层LayerNorm + GELU；从相同的256维隐藏表示连接两个线性输出头：

- pose头：`256→20→tanh`，映射回原生关节范围内的弧度值。
- 类别头：`256→31` logits，推理时通过softmax/argmax输出概率与类别标签。

联合损失为 **归一化关节MSE + 0.05 × 分类交叉熵 + 0.25 × commitment**。两个任务权重通过 `--reconstruction-weight 1 --classification-weight 0.05` 配置，必须均为有限正数；0.05用于平衡交叉熵与归一化关节MSE的量级，不代表已经调参得到最优权重。

前20轮为连续联合预训练，两个头同时训练；随后仅从训练集latent进行逐层残差K-means初始化，再训练量化联合模型最多300轮。两个头始终使用同一份连续/量化表示，没有绕过RVQ的分类分支。按**验证集加权重建MSE + 分类交叉熵最小**保存模型，量化阶段连续30轮未改善则早停；分别记录MAE/RMSE/P95、accuracy、macro F1与两项损失。测试集不参与选模。

Muon用于encoder及decoder主干的矩阵，pose和类别输出头、偏置及LayerNorm使用AdamW；RVQ码本由EMA更新。Muon momentum=0.95、Nesterov、5步Newton–Schulz、`adjust_lr_fn="match_rms_adamw"`，基础学习率3e-4、权重衰减1e-4，余弦衰减至3e-6。依据 [PyTorch 2.10 Muon文档](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Muon.html)。

## 远端四卡

服务器 `ssh -p 1016 8.147.67.185`，仓库目录 `/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget`，使用已有 `miniconda3/envs/pi0` 的PyTorch 2.10.0+cu128。

```bash
cd /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget
bash retarget/learning/wuji2_rvq/launch_4gpu.sh
# 恢复同一作业：
bash retarget/learning/wuji2_rvq/launch_4gpu.sh --resume
```

每卡实际batch为8192，四张A800全局batch32768。8192不能被31整除，每批各类264～265条，额外8条所属类别由已保存状态的随机采样器轮换选择。每轮250次更新，约819万次采样，包含重复抽样与约10% primitive。通过 `--batch-size 8192` 指定实际每卡batch；未指定时沿用 `31 × --samples-per-type`。DDP同步encoder及共享decoder和两个输出头的梯度，各层EMA统计跨卡求和。相同数据、GPU数、batch及阶段配置下可以恢复；checkpoint保存Muon/AdamW、调度器、各rank随机状态、采样器状态和阶段进度。

```bash
cat output/wuji2_rvq_joint_4gpu_muon_v1/status.json
tail -f output/wuji2_rvq_joint_4gpu_muon_v1/training.log
```

单卡命令：

```bash
.venv-contact/bin/python -m retarget.learning.wuji2_rvq train \
  --task joint --reconstruction-weight 1 --classification-weight 0.05 --optimizer muon \
  --output output/wuji2_rvq_joint_v1
```

## 推理、编码与评估

输入 `.npy` 为 `[B,20]` 弧度数组，或 `.npz` 的 `qpos` 字段。顺序必须与模型 `joint_names` 一致。

```bash
PY=/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/bin/python
RUN=output/wuji2_rvq_joint_4gpu_muon_v1
$PY -m retarget.learning.wuji2_rvq predict \
  --checkpoint "$RUN/best.pt" --input poses.npz --output predictions.npz
$PY -m retarget.learning.wuji2_rvq encode \
  --checkpoint "$RUN/best.pt" --input poses.npz --output tokens.npz
$PY -m retarget.learning.wuji2_rvq decode \
  --checkpoint "$RUN/export/decoder.pt" --input tokens.npz --output decoded.npz
$PY -m retarget.learning.wuji2_rvq evaluate \
  --checkpoint "$RUN/best.pt" --split test --output "$RUN/independent_test.json"
$PY -m retarget.learning.wuji2_rvq export \
  --checkpoint "$RUN/best.pt" --output "$RUN/export_copy"
```

`predict` 输出 reconstructed_qpos[B,20]、probabilities[B,31]、零起始 class_index[B]、grasp_type、class_names及四个uint8 token。`decode` 使用同一组token通过两个输出头返回qpos及类别概率/标签；可使用完整模型或独立双头decoder包。

```python
from retarget.learning.wuji2_rvq.train import load_model
from retarget.learning.wuji2_rvq.joint import JointTokenDecoder
model, _ = load_model("output/wuji2_rvq_joint_4gpu_muon_v1/best.pt")
tokens = model.encode(qpos)  # qpos: float32 Tensor[B,20], radians
outputs = model.decode_outputs(tokens)
reconstructed = outputs['qpos']
probabilities = outputs['logits'].softmax(-1)
# JointTokenDecoder(config)加载export/decoder.pt中的decoder状态，
# 不需要encoder；直接接受tokens，返回qpos、logits和probabilities。
```

输出包含 `warmup/`、`rvq/` 的最佳与恢复检查点、history及optimizer分组；`best.pt`、`export/joint.pt` 为完整模型，`export/decoder.pt`为独立码本和双头decoder，不含encoder。`interface.json`记录关节顺序、原生范围和31类名称。

`report.json`、`REPORT.md`、训练曲线与混淆矩阵同时报告分类accuracy、balanced accuracy、macro F1、top-3、逐类precision/recall/F1，以及总体/逐类/逐关节重建MAE/RMSE/P95、接触/稳定性分组、指尖FK误差和码本利用率/perplexity。`continuous_test.json`为20轮连续联合预训练模型的参考结果；`test_predictions.npz`保存源行索引、目标/重建关节角、标签、概率和token。测试MAE≤1°且P95≤5°为重建目标，未达到时状态为 `completed_targets_unmet`；分类指标独立如实报告。

临时检查验证两个头分别都能向共享decoder主干和encoder传回梯度、独立双头decoder与完整模型按相同token解码逐值一致、联合训练断点恢复逐值一致。四卡短流程检查各rank模型和EMA码本一致，不创建持久测试目录。

## 保留的单任务实验

仅分类实验可以显式指定 `--task classification --output output/<新目录>`，使用交叉熵加commitment、不包含decoder。

原重建任务仍可显式指定 `--task reconstruction --output output/<新目录>`：20轮AE预训练，随后 `MSE + 0.25 commitment + λ SupCon`，比较λ=0、0.01、0.05，必要时0.005，按重建与检索指标选择；支持原 `encode/decode/export` 及独立decoder包。

旧远端AdamW重建结果保留在 `output/wuji2_rvq_4gpu_v1/`，Muon重建结果保留在 `output/wuji2_rvq_4gpu_muon_v1/`。旧仅分类结果保留在 `output/wuji2_rvq_classifier_4gpu_muon_v1/`。当前联合任务从头进行联合预训练，使用独立输出目录，不加载旧单任务优化器状态。数据与输出均被Git忽略，不修改网页、手套标定或已部署模型。

## W&B 实时曲线与补传

独立同步进程读取已有 `warmup/history.json` 与 `rvq/history.json`，补传并跟踪每轮指标，不需要重启GPU训练。目标为 `https://wandb.plat.x2robot.com`，默认project为 `wuji2-rvq`，entity使用已登录账号的默认空间，也可追加 `--entity <team>`。

```bash
# 在服务器终端交互登录；不要将API Key写进代码或聊天。
/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/bin/wandb login \
  --host https://wandb.plat.x2robot.com
# 从仓库根目录启动；若尚未登录，会等待认证。
bash retarget/learning/wuji2_rvq/launch_wandb_sync.sh
```

W&B记录训练总损失、重建MSE、分类交叉熵、commitment、学习率，验证accuracy/macro F1、关节MAE/RMSE/P95与联合损失。横轴为跨预训练/RVQ的连续epoch，并保存每阶段epoch和累计optimizer_step。结束后上传最终指标摘要、评估JSON、报告及曲线/混淆矩阵图片，不上传数据样本或模型权重。

`wandb_run.json`保存run链接与ID，`wandb_sync_status.json`记录同步进度。重复启动会恢复同一个run；目录锁防止同时出现两个写入进程。W&B连接不进入GPU训练进程，认证或同步问题不改变训练与检查点。

## 接入 Moxian Glove 网页

训练后的联合权重已下载到本地 `data/models/wuji2_rvq_joint_v1/`：`joint.pt`为完整encoder、RVQ及双头decoder，`decoder.pt`为独立双头解码包；`download.json`记录远端来源与逐文件SHA256。

按原方式启动 Moxian 网页并应用个人六动作标定，选择 **Moxian／Wuji2 对照**。先通过当前选定的重定向后端得到Wuji2原生20维关节角，再进行RVQ推理。该模型不直接接收Moxian的25＋4维原生角或EMF信号。

新增“RVQ · 手势识别与重建”面板显示类别、模型概率、Top-3、四个token及相对同帧重定向输入的关节重建均差。右侧灰色机器人保留原重定向结果，蓝色线框显示RVQ重建；可取消“叠加重建手型”。重建的同帧手腕变换来自实际机器人根link，保留现有左右手镜像约定。

推理默认在Wuji2独立worker中使用CPU，模型只读、EMA不更新；它不替换重定向结果，也不发送机器人或手套控制命令。缺少权重、关节名称/范围不匹配或输入无效时，RVQ单独报告错误，原重定向仍可使用。新增结果位于 `/api/state?view=wuji2` 的 `wuji2.rvq`，含概率、token、重建关节角与link变换；模型来源和类别列表位于 `/api/wuji2/model` 的 `rvq`。

环境变量（在启动网页前设置）：

```bash
# 可指定另一个联合模型；目录默认读取其中的joint.pt。
MOXIAN_RVQ_MODEL=/absolute/path/to/joint.pt bash scripts/start_moxian_web.sh
# 关闭附加推理：
MOXIAN_RVQ_ENABLED=0 bash scripts/start_moxian_web.sh
```

权重更新后可点击“重启求解”；不需要重新标定。GeoRT或CART若提示设备/标定身份不匹配，需要选择兼容模型或其他可用重定向后端，RVQ不能绕过上游标定校验。

原版 `wuji-retargeting` 的右手 URDF 使用自身 FK 绘制重建，其他原生 Wuji2 后端使用 MuJoCo FK。原版左手 URDF 的角度约定尚未映射到该右手训练模型，使用该组合时 RVQ 会单独报错；已输出标准右手 Wuji2 角度的镜像后端不受此限制。

对照画面右上角的小窗显示当前预测的 grasp type、概率及其 **primitive 静态示意图**。31类插图取自原始MANO primitive（选择有效样本中 `size_scale` 最接近1的固定行），类别不变时图片不变，不受实时关节姿态影响。它与蓝色RVQ重建叠加是两个独立视图。插图和来源指纹在 `moxian_web/rvq_primitives/`；可用 `.venv-contact/bin/python scripts/render_rvq_primitives.py` 重新生成，目录通过模型拓扑一次加载，不随每帧重复传输。

# 两阶段姿态编码与条件 Flow Matching

新 `train --model diffusion` 默认训练 **VQ-VAE + pose flow**；`--latent-kind vae` 使用连续 VAE，`both` 分别训练两套模型。原 `results/grasp_manifold_20261103` 保留旧 DDPM/DDIM 权重，加载和预测兼容。本次只进行合成数据 smoke test，没有运行真实数据完整训练或产生性能排名。

## 网络与损失

所有网络处理按关节限位归一化的 20 维手指角，手腕前六维由调用方原样保留。新路径直接重建原始标准抓姿，不经过 PCA 投影，也不裁剪输出。

Encoder 为 `20 → 256 → 256 → latent`，隐藏层使用 SiLU，latent 默认 16 维。VQ-VAE 使用 128 个 16 维码字及 straight-through 梯度；连续 VAE 输出均值／对数方差并通过重参数采样训练。Decoder 拼接 latent、8 维 primitive 嵌入和归一化对数尺寸，经两层 256 单元 MLP 输出标准 pose。Encoder 只读取 pose，decoder 不读取 grasp type。

训练 batch 包含同抓型不同尺寸的正对及不同抓型的负对。自身重建平均干净输入与带噪输入的重建误差；跨尺寸重建使用编码与配对样本的尺寸重建该尺寸标准 pose。监督对比损失使用 grasp type 标签。

默认权重：自身重建 `1`、跨尺寸重建 `1`、干净／带噪特征一致性 `.1`、监督对比 `.1`，对比温度 `.1`；连续 VAE 的 KL 为 `1e-4`；VQ 码本为 `1`、commitment 为 `.25`。均可通过 CLI 配置。

统一加噪先均匀选择 0–5 根手指，再为每根选中手指独立采样 1–5 轮，每轮对其四个关节加入独立高斯增量。单轮 sigma 按样本从 `[.01,.35]` 均匀采样，未选手指不变；原上限 `.15` 已提高至 `.35`。各指累计理论标准差为 `sqrt(rounds_finger) * sigma_step`。验证同时检查单轮 sigma `.03/.1/配置上限`，避免只用小噪声选择大噪声训练的权重。验证／评估 NPZ 保存实际掩码、轮数和累计噪声，模型推理不读取这些真值。

第二阶段恢复第一阶段验证集最佳权重，冻结整个 autoencoder（含码本及 decoder），只优化速度场。连续 VAE 固定使用均值，VQ-VAE 固定使用量化 latent：

```text
y = 带噪标准姿态
z = frozen_encoder(y)
t ~ Uniform(0, 1)
x_t = (1 - t) * y + t * x
目标速度 = x - y
loss = MSE(v(x_t, t, z, grasp_type, primitive, size), x - y)
```

速度场拼接当前 pose、latent、32 维 grasp type 嵌入、8 维 primitive 嵌入、归一化对数尺寸及 32 维正余弦时间特征，通过三层 256 单元 SiLU MLP 输出速度。推理从观测开始，用固定 latent 做默认 32 步 Heun 积分，时间从 0 到 1；不额外采样噪声。零噪声输入也经过该网络，不返回 PCA 替代结果。

## 使用

以下完整训练命令供后续使用，本次未执行。使用独立目录及 prepare 默认种子 7 可复用原实验的 424 训练／108 验证／108 测试尺寸节点划分：

```bash
.venv-contact/bin/python scripts/grasp_manifold.py prepare \
  --source results/grasp_current_20261103 --output output/grasp_pose_flow

.venv-contact/bin/python scripts/grasp_manifold.py train \
  --output output/grasp_pose_flow --model diffusion --latent-kind both \
  --seeds 7,17,27 --stage all
```

默认每阶段上限 10,000 步，batch 256，AdamW、学习率 `.001`。每 250 步按抓型宏平均验证；至少 2,500 步后，连续 12 次验证未改善可提前停止。第一阶段按 decoder 重建误差、第二阶段按完整积分去噪误差选权重，测试节点不参与选择。

分开执行时，第二阶段必须明确提供第一阶段 checkpoint；latent 维数、隐藏宽度、码本大小和数据指纹须兼容：

```bash
.venv-contact/bin/python scripts/grasp_manifold.py train \
  --output output/grasp_pose_flow --model diffusion --latent-kind vq \
  --seed 7 --stage autoencoder

.venv-contact/bin/python scripts/grasp_manifold.py train \
  --output output/grasp_pose_flow --model diffusion --latent-kind vq \
  --seed 7 --stage flow \
  --autoencoder-checkpoint output/grasp_pose_flow/models/diffusion_vq_seed7/autoencoder.pt
```

已存在的阶段权重拒绝覆盖。两种模型分别保存在 `models/diffusion_vq_seed*/` 和 `models/diffusion_vae_seed*/`，包含 `autoencoder.pt`、`best.pt`、阶段历史、配置和验证噪声。Checkpoint 记录架构版本、阶段、条件标签、数据指纹与第一阶段权重指纹；最终 flow checkpoint 自包含冻结的编码器。

`models.json` 保存别名及选定种子，不修改参与指纹校验的 `dataset.json`。有 VQ 时 `diffusion` 默认指向 VQ；仅训练 VAE 的实验指向 VAE。旧实验没有此注册文件，继续加载旧 DDPM。明确请求的新权重缺失时直接报错。

```python
from retarget.env.grasp.manifold import ExperimentData, Predictor

data = ExperimentData('output/grasp_pose_flow')
predictor = Predictor(data, 'diffusion_vq')  # 或 diffusion_vae
clean = predictor.predict(observed_normalized, groups, sizes)
```

输入分别为 `(N,20)`、`(N,)`、`(N,)`；旧 `sigma` 参数仍保留，新 flow 不使用其值。新模型的尺寸曲线从模型对训练节点的预测拟合，无 PCA 输出约束。

## 评估和检查页

```bash
.venv-contact/bin/python scripts/grasp_manifold.py evaluate \
  --output output/grasp_pose_flow --seeds 7,17,27 \
  --baseline results/grasp_manifold_20261103

.venv-contact/bin/python scripts/inspect_grasp_denoising.py \
  --experiment output/grasp_pose_flow --method diffusion_vq \
  --noise-mode fingers --output output/grasp_pose_flow_inspector
```

比较要求基线与新模型使用相同的原始节点、关节范围、标签及划分。两套 flow、PCA、尺寸插值和旧 diffusion 接收同一组带噪观测：无噪声及单轮 sigma `.03/.1/已训练模型上限` 的随机手指／轮次扰动；`--noise` 可覆盖默认级别。旧 DDPM 需要标量 sigma，因此使用各指标准差的 RMS 作为各向同性近似，这一限制记录在结果中。Flow 不使用此近似。

`evaluation.json` 包含各模型／种子、各噪声强度和手指数的关节 RMSE、latent 同型跨尺寸／异型距离、VQ 码本利用率及预测碰撞／限位诊断；`evaluation_noise.npz` 保存共享扰动，`comparison.md` 提供汇总。只根据验证集选择种子和 flow 后端，单独报告总体最佳比较方法；结果代表已知抓型内的尺寸插值，不代表动态抓握稳定性。

检查页通过 `--method` 选择一套 flow 与 PCA 并排显示，Flow 显示积分时间和步数，旧 DDPM 显示噪声时刻；均显示实际各指 sigma 和轮数。检查页的收紧演示仍使用两方法共享的训练节点 PCA 尺寸曲线，用于控制演示变量，不构成新模型的去噪或输出约束。`--noise-mode all` 可复现旧的全关节加噪规则。

## Smoke test

```bash
.venv-contact/bin/python -m pytest -q \
  tests/test_grasp_pose_flow.py tests/test_grasp_manifold.py \
  tests/test_grasp_denoising_inspector.py
node --check retarget/env/grasp/denoising_inspector.js
```

测试使用合成小数据、窄网络和每阶段两步更新，覆盖噪声统计、VAE／VQ 梯度、冻结、积分方向及收敛、保存／加载、错误权重拒绝、旧权重兼容和检查页／比较导出。CUDA 可用时增加两种后端的前后向与积分检查。本次没有运行完整训练，也没有验收新页面的实际 WebGL 渲染。

## 输入采样分布检查

```bash
.venv-contact/bin/python scripts/inspect_pose_flow_sampling.py
```

脚本直接复用第一阶段的配对采样器、第二阶段的节点采样规则和统一加噪器，不训练网络。默认每阶段采样 102,400 个输入，在 `output/grasp_pose_flow_sampling_broad_20261103/` 导出交互 HTML、12 面板 PNG／PDF、采样 NPZ 与统计 JSON。可按抓型、阶段、原始关节维度查看散点，并与固定干净训练 PCA 和数学上的均匀关节盒对照。均匀关节盒不代表有效抓姿。

原范围 `[.01,.15]` 的报告保留在 `output/grasp_pose_flow_sampling_20261103/`：覆盖全部 31 个抓型、424 个训练节点和 32 种手指组合；第二阶段约 65.2% 输入距最近干净节点的归一化关节 RMSE 不超过 0.1，约 28.7% 输入有至少一个关节越限。

新范围 `[.01,.35]` 使用相同采样种子：噪声 RMSE 中位数从 `.069` 增至 `.151`，99% 分位数从 `.280` 增至 `.649`；距干净节点 RMSE 不超过 `.1` 的比例降至约 39.3%，有越限关节的输入增至约 46.1%。这扩大了连续扰动范围，但没有提供新的非抓取语义，也不保证覆盖放松姿态。两组数据未裁剪、未做碰撞筛选。

## 放松与非抓取状态：后续数据／模型建议

用户期望放松输入保持或重建放松，而不是被吸引到某个抓型。当前数据只有抓取参考，没有带标签的放松／非抓取样本；以下状态分支和拒绝机制尚未实现，本次仅修改噪声范围与对应验证覆盖。

- 将自然放松、张开、轻微屈曲和抓取／松开过程单独记录。已知放松是正常数据状态，不能只作为低置信度异常处理。随机越限或随机均匀关节角不能替代真实放松标签。
- 为样本增加 `state`（grasp/rest）、干净重建目标与条件有效掩码。Rest 不应伪造 primitive、尺寸或某个抓型，也不应复制到多个尺寸节点制造数据量；数据划分按原始姿态／轨迹隔离。
- 第一阶段同时重建抓取与放松，分别做相应加噪；第二阶段对 rest 学习 `v*=clean_rest-observed_rest`，而不是映射到抓取目标。多样非抓取动作不宜全部被对比损失强拉成单一 latent。
- 在原始观测及量化前编码特征上增加状态／抓型预测分支，在 flow 积分前路由。状态置信度与重建成功置信度分开：已知放松可以很有把握，未知或严重歧义输入才应拒绝强制投影、保留原姿态。
- 预测成功评分需要留出的已知目标误差或显式负例监督；用留出放松／抓取／过渡姿态校准阈值，重点报告“放松被误变抓取”的比例、抓取召回率和接受样本的重建误差。生成密度或重建残差可作辅助，不能直接当作可靠 OOD 概率。
- 大噪声抓取观测可能与真实放松重叠。若同一静态 pose 要对应不同目标，就需要任务模式或短时序上下文；继续增加 sigma 不能消除这一歧义。保持低噪声／干净样本覆盖，对重叠输入降低抓取恢复强度。

参考：[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)、[Outlier Exposure](https://arxiv.org/abs/1812.04606)、[生成模型 OOD 检测的局限](https://proceedings.mlr.press/v139/zhang21g.html)。这些工作支持拒绝机制与异常监督的一般方向，不是本项目的识别性能证据。

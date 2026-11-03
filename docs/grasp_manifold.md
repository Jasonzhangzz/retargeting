# 抓取流形去噪、尺寸收紧与残差叠加

新的训练入口已改为[两阶段 VAE／VQ-VAE + 条件 Flow Matching](grasp_pose_flow.md)，支持按手指独立轮次加噪。下文已发布的指标及 checkpoint 仍属于旧 DDPM／DDIM 实验；新架构本次只做 smoke test，未运行完整训练。

新增的[多轮高斯加噪检查页](grasp_denoising_inspector.md)逐类使用标准 1× pose，并排比较 PCA 与 diffusion，默认在收紧后加回**完整原始高斯噪声**。下文旧演示中的“叠加”主要指 PCA 正交残差，两者有明确区别。

这是独立的研究实验，不改写原始 pose、模板或现有抓取求解器。实现四种方法：每类 PCA、尺寸条件插值、128 原子去噪字典、条件 DDPM / DDIM。

## 已生成的结果

最终精简结果位于 `results/grasp_manifold_20261103`。每类 PCA 已拟合；字典与 diffusion 均完成三个初始化种子（7、17、27）的训练，每次 3,000 步。现仅保留按验证集宏平均 RMSE 选出的字典 seed 17、diffusion seed 27 两个 checkpoint、干净姿态、PCA 参数及精简评估摘要。

加噪／收紧演示、HTML、图片、逐样本评估、训练历史及其他种子的权重已清理。原 `output/grasp_manifold_20261103` 只是指向最终结果的本地链接，不重复存储。`dataset.json` 使用相邻源快照的相对路径，可随 GitHub 仓库一起克隆使用。

```bash
# 仅在需要展示时重建演示；自动使用保留的最佳 checkpoint。
.venv-contact/bin/python scripts/grasp_manifold.py demo --model selected

# 查看重建的离线报告。
.venv-contact/bin/python -m http.server 8093 --bind 127.0.0.1 \
  --directory results/grasp_manifold_20261103
# http://127.0.0.1:8093/demo/index.html
```

重建的报告包含真实 WUJI2 手部网格，支持抓型、方法、已知偏置、噪声、收紧比例、裁剪状态切换和四视图相机联动，并生成静态科学图。此前已检查 31 类共 3,348 组演示；浏览器 FK 与 MuJoCo 的误差低于 `1e-10 m`。当前环境没有可连接浏览器，因此没有进行真实浏览器截图验收；应用 JavaScript 已经过语法检查及带模拟 DOM / Plotly 的控件运行检查。

此次运行的测试集结果如下。误差为按抓型宏平均的归一化关节 RMSE，统计噪声水平为 `0, 0.03, 0.1, 0.3`，噪声种子为 `11, 23, 37`。

| 方法 | 验证集 RMSE | 测试集 RMSE |
| --- | ---: | ---: |
| PCA | 0.049807 | 0.049617 |
| 尺寸插值 | **0.026685** | **0.026498** |
| 128 原子字典 | 0.028749 | 0.028383 |
| 条件 diffusion | 0.035000 | 0.034609 |

仅根据验证集选择的默认方法是尺寸插值。这是已知抓型和源尺寸、每类参考姿态相对稀少时的结果，不代表 diffusion 在更丰富数据上必然更差。

在每类随机选取的一个测试姿态上，无额外偏置和噪声、期望收紧 10% 时，插值方案使 29/31 类的接触包络 RMS 减小，平均变化约为 `-1.53 mm`。同时部分结果有明显穿透，最大手物穿透约为 `9.82 mm`。这支持将方法作为候选姿态生成器，并不能证明生成了物理可执行或更稳定的抓取。

## 从头复现

使用已有抓取环境，补充报告依赖：

```bash
python -m pip --python .venv-contact/bin/python install -r requirements-manifold.txt
```

选择新的输出目录，避免覆盖训练结果：

```bash
.venv-contact/bin/python scripts/grasp_manifold.py prepare \
  --source results/grasp_current_20261103 --output output/manifold_reproduction

.venv-contact/bin/python scripts/grasp_manifold.py train \
  --output output/manifold_reproduction --seeds 7,17,27 --steps 3000

.venv-contact/bin/python scripts/grasp_manifold.py evaluate \
  --output output/manifold_reproduction --seeds 7,17,27

.venv-contact/bin/python scripts/grasp_manifold.py demo \
  --output output/manifold_reproduction --seed 7
```

默认自动使用 CUDA，无 CUDA 时使用 CPU。`train` 和 `demo` 的 `--seed` 指模型初始化种子；`prepare --seed` 控制数据划分，`demo --noise-seed` 控制已知偏置及高斯扰动。数据划分保存后不会随训练种子改变。`demo` 和 `evaluate` 省略模型种子时，从结果元数据选择保留的最佳种子；新实验没有该字段时使用 seed 7。

```bash
# 仅显示一个抓型和一种方法，输出到该实验的 demo/；会替换该目录下既有报告。
.venv-contact/bin/python scripts/grasp_manifold.py demo \
  --output output/manifold_reproduction --grasp-type 24 \
  --model dictionary --noise 0,0.03,0.1 --tightening 0,0.1,0.2

# --model selected 读取 evaluation.json 中验证集选定的方法。
# 重新评估单一已训练种子：
.venv-contact/bin/python scripts/grasp_manifold.py evaluate \
  --output output/manifold_reproduction --seed 7 --noise-seeds 11,23,37

.venv-contact/bin/python -m pytest -q tests/manifold/test_grasp_manifold.py
```

`prepare` 已存在数据集、`train` 已存在相同方法和种子的最佳权重时会拒绝覆盖。`evaluate` 会更新评估文件，`demo` 会更新演示文件。`--device cpu` 可用于 CPU 检查。所有 CLI 子命令支持 `--help`。

## 表示和数据划分

输入为仓库压缩抓姿快照。640 个原始已求解 pose 全部通过本次有限值、手指限位和原生自碰撞检查，划分为 424 训练、108 验证、108 测试姿态。插值出的 39,007 个 pose 不进入学习与评估。每类独立按唯一尺寸节点划分；相同尺寸节点始终属于同一集合，端点保留用于训练。因此评估为已见抓型内部的留出尺寸插值，不是跨物体、跨抓型泛化测试。

手腕前六个坐标在演示中固定。末 20 个关节角按各自限位中心和半范围归一化：

```text
x = (finger_angles - joint_midpoint) / joint_half_range
```

PCA 均值与基只从训练集估计，每类保留 95% 方差，上限为 `min(19, n_train - 1)`。本次实际保留 1–6 个方向。特征值、数值秩、保留方差比例和截断 RMSE 保存在 `dataset.json`。

共享字典为 `D.shape == (20,128)`，不是 128 个独立主成分。编码器使用抓型嵌入、归一化对数尺寸和时间嵌入，通过两层 256 单元 SiLU MLP 预测稀疏编码。训练包含重建、`1e-3` 稀疏项、`1e-2` 同姿态干净／加噪编码一致性，以及干净样本重建项。字典列归一化，输出投影到对应类 PCA 子空间。不同尺寸的编码允许变化，不要求全类塌缩为同一个向量。

Diffusion 使用相同条件和干净子空间目标，预测高斯噪声；1,000 步 DDPM 线性 beta 为 `1e-4 → 0.02`，推理默认最多 50 步确定性 DDIM。对外噪声参数 `sigma` 表示归一化关节空间的加性噪声标准差。程序寻找最近的训练噪声时刻并报告实际 sigma：

```text
observed = clean + sqrt((1-alpha_bar[t])/alpha_bar[t]) * epsilon
x_t = sqrt(alpha_bar[t]) * observed
```

推理从该 `x_t` 开始，不重新添加随机噪声；`t=0` 是精确干净时刻。用户输入任意 pose 时必须明确其噪声强度，程序不会从 pose 自动识别噪声。

所有学习式模型都条件于已知源尺寸。插值基线使用同样的尺寸信息；PCA 基线仅做正交投影。验证集选择最佳 checkpoint，之后按各类平均的验证误差选择默认方法；测试集不参与选择。

## 投影、收紧和残差的含义

```text
principal = mean_g + (x - mean_g) @ U_g @ U_g.T
residual  = x - principal
result    = clean_estimate + curve(target_size) - curve(source_size) + residual
```

尺寸曲线只使用训练节点，并在 **PCA 系数空间**做 PCHIP，再映射回 20 维空间，确保曲线差值仍在类子空间中。直接对 20 个坐标分别 PCHIP 通常不能保证这一性质。目标尺寸限制在训练范围内；若源尺寸已在最小端点，收紧可能被截断为零，报告会给出实际目标尺寸。

每种方法都导出自己的尺寸曲线。Diffusion 在无噪声时返回正交投影，因此当前 diffusion 曲线与 PCA 曲线相同。字典曲线由训练节点的无噪声编码重建形成。纯 PCA 路径使用精确投影，零收紧时应还原原输入；学习式路径可能同时修正子空间内扰动，不要求零收紧等于输入。

对 `observed = clean_reference + bias + noise`，有：

```text
extracted_residual = reference_truncation_residual
                   + perpendicular_part(bias)
                   + perpendicular_part(noise)
```

因此原样加回残差也会加回其中的噪声，而一般方向偏置的平行部分不会被该残差保留。本实验同时展示无偏置、已知正交偏置、已知一般方向偏置，保存真实噪声和偏置以便核验这一点。

原始候选与 `[-1,1]` 限位裁剪候选分别保留；裁剪可能改变正交残差，其变化量也单独记录。不做接触优化、物体缩放或隐式修复。

## 文件与 Python 接口

核心代码：`retarget/env/grasp/manifold.py`；离线报告：`retarget/env/grasp/manifold_report.py`；CLI：`scripts/grasp_manifold.py`。

```python
import numpy as np
from retarget.env.grasp.manifold import ExperimentData, Predictor, PoseChecker

data = ExperimentData('results/grasp_manifold_20261103')
index = int(np.flatnonzero(data.mask('test'))[0])
group = int(data.group[index])
size = float(data.size[index])
observed = data.x[index]  # 可替换为归一化后的用户手指姿态

principal, residual = data.subspaces[group].decompose(observed)
predictor = Predictor(data, 'dictionary')  # 自动选择保留的 seed 17
clean = predictor.predict(observed[None], [group], [size], sigma=0.)[0]
tighter, target_size, delta = predictor.curves[group].tighten(
    clean, residual, source=size, fraction=.1,
)
q = data.qpos[index].copy()  # 固定源姿态的手腕
q[6:] = data.denormalize(tighter)
diagnostics, frame = PoseChecker(data, index).inspect(q)
```

主要输出：

- 发布的精简结果包含 `clean_poses.npz` 和 `pca.npz`：前者保存 640 个参考姿态和四种方法对无额外噪声输入的估计，后者提供可直接用 NumPy 读取的每类 PCA。Diffusion 的无噪声路径直接返回 PCA 投影；非零噪声的学习式去噪由保留的 checkpoint 提供。
- 发布目录的 `evaluation.json` 保留原三种子总体指标、六次训练完成记录与最佳种子选择依据；`checksums.json` 校验所有最终文件。
- `dataset.npz / dataset.json`：原始样本索引、尺寸节点划分、归一化参数、PCA、质量标记与源文件哈希。
- 发布的模型目录仅保留 `best.pt`、`config.json`，字典另保留 `dictionary_codes.npz`。重新训练会生成 `history.json` 和 `reference_codes.json`，这些中间文件未发布。
- 重新运行 `evaluate` 会替换精简摘要，生成逐样本误差、计时、编码一致性、预测方差、bootstrap 子空间稳定性与收紧几何诊断。
- 重新运行 `demo` 会生成 `demo/index.html / comparisons.json / poses.npz / preview.png`；这些可重建文件未发布。HTML 内的 pose 舍入至 10 位小数，完整精度以 NPZ 为准。

几何检查使用原生 MuJoCo 自碰撞／手物穿透与精确三角形最近点。接触覆盖率是固定模板锚点的距离／法向指标，区别于原缓存的 patch 内部覆盖率；网格边缘处有符号间隙使用最近面法向作为局部近似。接触包络 RMS 与活跃指尖平均间距是运动学开口指标，不是力学抓握尺寸的唯一度量。

## 参考与结论边界

- [EigenDEXplore §3.1–3.2](https://arxiv.org/pdf/2610.07681)：从重定向关节数据提取协调 PCA，并用于相关探索。该论文没有证明本实验的残差分离假设。
- [DDPM](https://arxiv.org/abs/2006.11239)、[DDIM](https://arxiv.org/abs/2010.02502)：加噪训练与确定性反向推理依据。

要迁移到真实物体，需要额外的同尺寸 primitive／真实物体成对抓姿来验证几何残差，并对输出候选施加接触、碰撞与必要的柔顺控制约束。当前代码没有真实物体几何监督，也不把关节收紧量解释为夹持力。

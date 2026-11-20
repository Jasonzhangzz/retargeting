# Wuji2：关节角条件化 Neural SDF

本模块直接学习 `s_theta(q, x, link) → d`：`q` 为 Wuji2 的 20 个关节角，`x` 为固定腕部的原生模型坐标中的三维点，`link` 为 0–20。距离单位米，外正内负。同一 `x`、同一 link 换一组关节角，SDF 会变化。空间点和关节角均可微。

与 `moxian_web/cart_collision.py::LinkSDF` 的区别：该模块的 MLP **直接看到 q 和腕部坐标点，推理不使用 FK，也不将点转换到 link 局部坐标**。真实 FK 仅用于离线标签和独立验收。它没有修改现有 CART / GeoRT 模型或活动模型选择。

## 采样与碰撞标签

复用 GeoRT 的 `nonhyperextended_flexion_uniform_v2`：在 Wuji2 原生限位与无反伸屈伸范围的交集中均匀随机采样完整 20 维 q，不通过裁剪制造姿态。CMC / 外展轴保留原生有符号范围。这是独立关节均匀分布，不代表自然动作频率或人体关节耦合。

默认提议 **100,000 组姿态，每姿态 21 × 8 = 168 个查询，合计 16,800,000 个距离标签**。默认保留全部姿态，逐姿态保存密集碰撞查询和最大穿透深度。需要排除穿透姿态时，用 `--pose-filter free`；此时 `--poses` 仍表示提议预算，最终保留数可能更少。

每个 link / 姿态独立按三角形面积抽取新的曲面点，默认 8 点组成：

- 4 个近表面点：1 mm / 6 mm 高斯扰动；
- 2 个 link 局部包围盒扩展 15 mm 内的点；
- 1 个覆盖整手运动范围的固定全局包围盒点；
- 1 个来自其他非连接 link 的运动曲面点，补充穿透和接近区域。

先做原生精确 FK，再对指定 link 的真实三角形做最近点距离与 winding-number 符号查询。近表面点可能在内也可能在外。默认 0.5 mm 仅是姿态碰撞分类阈值，训练距离不截断，也不将阈值以内的负距离改为零。

碰撞检查复用 `WujiMeshCollision`：所有顶点、面心和固定曲面样本双向查询，覆盖跨指、同指非相邻 link 及非连接掌面，并保留机械连接 / MCP collar 排除。它是密集离散表面检查，不是连续三角形相交证明。部分原生 link 网格不封闭；不修补或替换资产，符号采用现有 winding-number 约定，`manifest.spec.geometry.watertight` 记录每个 link 的状态。数据中的“无碰撞”应理解为上述查询范围和阈值下通过。

## 执行

复用仓库的 `.venv-contact` 环境，无新增依赖：

```bash
# 一次执行大规模采样和 30 轮完整训练；相同配置可重复执行以恢复。
bash scripts/train_wuji2_sdf.sh

# 单独采样；输出目录必须新建或显式 --resume。
.venv-contact/bin/python -m retarget.learning.wuji2_sdf sample \
  --output output/wuji2_sdf/random100k_v1 \
  --poses 100000 --per-link 8 --shard-poses 1024 --pose-batch 32 \
  --device cuda:0 --resume

.venv-contact/bin/python -m retarget.learning.wuji2_sdf train \
  --data output/wuji2_sdf/random100k_v1 \
  --output output/wuji2_sdf/mlp100k_v1 \
  --epochs 30 --batch-size 4096 --device cuda:0 --resume

# 遍历完整独立测试集，不限查询数。
.venv-contact/bin/python -m retarget.learning.wuji2_sdf evaluate \
  --data output/wuji2_sdf/random100k_v1 \
  --model output/wuji2_sdf/mlp100k_v1/best.pt \
  --output output/wuji2_sdf/mlp100k_v1/test_all.json --device cuda:0

# 导出三组姿态的真实 / 学习 SDF 截面与训练曲线。
.venv-contact/bin/python -m retarget.learning.wuji2_sdf plot \
  --model output/wuji2_sdf/mlp100k_v1/best.pt \
  --output output/wuji2_sdf/mlp100k_v1/figures --device cuda:0
```

脚本支持环境变量 `WUJI_SDF_PYTHON / DATA / MODEL / DEVICE / POSES / EPOCHS / BATCH`（共同前缀为 `WUJI_SDF_`）。例如百万姿态使用新的 DATA / MODEL 路径与 `WUJI_SDF_POSES=1000000`，不要把已完成十万姿态目录的采样预算改大。默认按分片工作，数据量增加不会把所有点常驻显存。`--pose-batch` 限制密集几何临时显存，`--batch-size` 限制 MLP 训练批量。

CPU 可用 `--device cpu --threads 4`。功能测试可以设置 `--poses 64`，训练时 `--epochs 2 --steps-per-epoch 3 --width 64 --depth 3`；`steps-per-epoch=0` 才是完整遍历训练集。缩小步数的结果不作为正式精度结论。

## 模型与训练

共享 21-link embedding（16 维）+ 四层 256 宽 Softplus MLP；输入为归一化空间点、归一化 q、sin(q)、cos(q) 和 link embedding。从原生模型的父链自动生成每个 link 的关节依赖 mask，只保留真实祖先关节。这保证其他手指的关节不会改变该 link 的几何，固定掌部不随 q 漂移；不手写运动变换。

AdamW（初始 lr 5e−4，余弦衰减），回归**不截断**的米制距离，10 mm 内权重 4；增加符号损失和小批量 Eikonal 正则。默认完整训练 30 轮。Eikonal 对 x 求梯度，保持欧氏距离斜率；对 q 的可微路径也保留，训练过程中不会 detach q 后替换成静态场。

按完整姿态独立 80% / 10% / 10% 分割，属于一个 q 的所有点只进入同一 split，避免同姿态查询点泄漏。验证集挑选 `near_mae_mm + 0.1 × mae_mm` 最小的权重，测试集不参与挑选。每次训练内的验证 / 测试默认最多 200,000 个查询，独立 evaluate 默认遍历全部查询。

## 文件与恢复

采样 `manifest.json` 保存资产与代码 SHA、采样域、坐标、单位、符号方法、每 link 网格封闭状态、查询分布、提议 / 保留 / 碰撞姿态数、split 数量及各分片 SHA。每片 NPZ 字段：

| 字段 | 形状 / 含义 |
|---|---|
| qpos | `[P,20]`，弧度；不在磁盘重复保存每个查询的 q |
| pose_id / split | `[P]`；提议的全局 ID；0=train、1=val、2=test |
| points | `[P,21*K,3]`，固定腕部模型坐标，米 |
| links | `[P,21*K]`，原生 owner 编号，不是 MuJoCo body ID |
| sdf | `[P,21*K]`，米，内负外正 |
| max_penetration_mm / collision | `[P]`，原生密集检测的最大穿透与分类 |

分片与清单原子写盘；中断后最多重算未提交的一个分片。种子按分片固定；恢复校验配置、几何、代码和已提交分片 SHA，不静默复用旧域 / 旧几何。训练目录有排他锁；数据未完成时拒绝训练。

训练输出 `configuration.json`、`status.json`、`history.json`、`best.pt`、`last.pt`、`report.json`。`last.pt` 包含 optimizer / scheduler / RNG / 当前 epoch 与批次位置，默认每 500 步、每轮及收到 SIGINT / SIGTERM 时保存，可用原命令加 `--resume` 恢复。更改数据、代码或关键训练参数需要新输出目录。`best.pt` 为验证集选出的可直接加载模型，包含模型结构和几何元数据。

报告包括距离 MAE / RMSE / p95、10 mm 内误差、逐 link MAE、符号准确率、真实负距离超过 0.5 mm 的漏检率，以及来自碰撞 / 非碰撞姿态的查询误差。`configuration_response` 固定同一个 x 与 link，切换两组独立测试 q，比较真实距离变化与预测距离变化；同时报告完全忽略 q 的零变化基线误差，防止把静态 SDF 的普通点误差误认为学会运动。

## 使用与梯度

```python
import torch
from retarget.learning.wuji2_sdf.model import load_model

model, info = load_model('output/wuji2_sdf/mlp100k_v1/best.pt', 'cuda:0')
# 冻结参数不会切断对输入 q 和 x 的梯度。
model.requires_grad_(False)
q = torch.zeros(1, 20, device='cuda:0', requires_grad=True)
x = torch.tensor([[[0.02, 0.02, 0.10]]], device='cuda:0', requires_grad=True)
links = torch.tensor([[8]], device='cuda:0')  # index distal，顺序见 checkpoint geometry
d = model(q[:, None, :], x, links)          # [B,N]，米
dq, dx = torch.autograd.grad(d.sum(), (q, x))
all_links = model.per_link(q, x)            # [B,N,21]
hand_field = all_links.amin(-1)             # [B,N]
```

`min(link SDF)` 在手外给出最近距离，零水平集和符号表示组成 link 的并集；多个 link 内部重叠时不保证等于并集边界的严格欧氏内距离。逐 link 输出仍可用于 link 对碰撞 / 接触损失。自碰撞需要将源 link 的运动曲面点查询到其他允许目标 link，不能把点对自己的 link 的零距离当作无碰撞。未知物体还需要物体几何。模型精度以实际独立验收为准，不将神经预测当成严格碰撞证书。

## 本机已执行结果（2026-11-18）

已完成 `output/wuji2_sdf/random100k_v1`：100,000 组姿态、16,800,000 个标签、98 个分片，压缩约 246.7 MiB。训练 / 验证 / 测试姿态分别为 79,815 / 10,093 / 10,092；76,419 组提议姿态在现有原生密集检查下穿透超过 0.5 mm。所有姿态均保留。

`output/wuji2_sdf/mlp100k_v1` 完成全部 30 轮、99,540 次 optimizer 更新，最优权重为 `best.pt`（第 30 轮）。以下来自完整独立测试集的 **1,695,456 个查询**，报告保存在 `test_all.json`；`report.json` 是训练结束时默认上限 200,000 查询的测试结果，两者不要混用。

| 完整独立测试指标 | 结果 |
|---|---:|
| 全部查询距离 MAE | 0.253 mm |
| 表面 10 mm 内距离 MAE | 0.200 mm |
| 绝对距离误差 p95 | 0.778 mm |
| 距表面超过 0.1 mm 的符号准确率 | 98.06% |
| 真实负距离深于 0.5 mm 的查询点漏检率 | 0.823%（179,137 个内部查询） |

固定 x / link、切换两组独立测试 q 的 20,480 个查询中，真实距离变化平均幅度为 18.373 mm；模型预测**距离变化**的 MAE 为 0.398 mm，而完全忽略 q 的零变化基线误差为 18.373 mm。真实变化超过 5 mm 的 12,822 个查询的变化方向全部正确。此项证明所学场随姿态变化，不代表所有未采样配置的严格精度保证。

`figures/configuration_slices.png` 展示固定截面、三种关节姿态下真实网格与 MLP 的距离场，`figures/training_curves.png` 保存训练曲线。`inference_check.json` 记录冻结模型后的输入梯度检查：q / x 梯度有效，无关手指梯度严格为零，关节梯度与中心有限差分吻合。自动测试 `tests/test_wuji2_sdf.py` 的 3 项测试通过，覆盖 FK / 标签独立核对、动态 q 标签与梯度、分片隔离 / 完整性及中途中断后的逐值恢复。

上述误差针对记录的随机查询分布；内部点漏检率是点级指标，不是整姿态自碰撞漏检率。模型已接入独立网页碰撞诊断，不替换当前 CART / GeoRT 活动模型。数据与权重位于 Git 忽略的 `output/` 下，需要单独保留。

## 重新采样的外部测试集

另用种子 `2026111801` 新生成 10,000 组姿态，每组 21 × 16 个查询，共 3,360,000 个标签，保存到 `output/wuji2_sdf/fresh_test10k_seed2026111801`。检查与原来全部 100,000 组姿态的精确 q 重合数为 **0**；新数据的所有分片、所有 split 均只用于测试，模型冻结且没有再训练。关节域、几何与查询分布族与训练一致，因此是新样本泛化测试，不是域外测试。

完整报告 `benchmark.json`：MAE **0.252825 mm**、RMSE **0.390019 mm**、p95 **0.778951 mm**、p99 **1.427853 mm**；表面 10 mm 内 MAE **0.200556 mm**。符号准确率 98.051%（排除距表面 0.1 mm 内查询）；真实深于 0.5 mm 的 355,216 个内部点中漏检 3,060 个，点级漏检率 **0.8614%**。

最大误差 **12.728 mm**，对应拇指末节的远场查询：真实距离 188.166 mm，预测 200.894 mm。误差超过 5 mm 共 160 / 3,360,000 个点。`error_examples.json` 保存极端样本的 q、空间位置、真实 / 预测值及近表面分区统计；`predictions/` 保存每个分片的逐点预测和误差，可与源分片直接对齐。

复现：

```bash
.venv-contact/bin/python -m retarget.learning.wuji2_sdf sample \
  --output output/wuji2_sdf/fresh_test10k_seed2026111801 \
  --poses 10000 --per-link 16 --shard-poses 1000 --pose-batch 32 \
  --seed 2026111801 --device cuda:0 --resume
.venv-contact/bin/python -m retarget.learning.wuji2_sdf benchmark \
  --data output/wuji2_sdf/fresh_test10k_seed2026111801 \
  --reference-data output/wuji2_sdf/random100k_v1 \
  --model output/wuji2_sdf/mlp100k_v1/best.pt \
  --output output/wuji2_sdf/fresh_test10k_seed2026111801/benchmark.json --device cuda:0
```

`benchmark` 会校验标签几何 / 坐标 / 关节域与模型相同、训练参考集指纹正确，且新姿态与整个原始数据集无重合。评估过程中不更新模型参数。新增误差统计与 q 重合检测单元测试后，`tests/test_wuji2_sdf.py` 共 4 项通过。

## Moxian 网页实时碰撞对照

刷新 `http://127.0.0.1:8890`，选择 **Moxian／Wuji2 对照**，查看 **Wuji2 · 真实网格与 Neural SDF 碰撞对照**。默认开启，按钮可停止 / 重新启动。使用当前已发布的 Wuji2 q，支持 SLSQP、CEM、wuji-retargeting、GeoRT、CART 的原生 20 关节顺序以及录制回放。它不依赖实时 Glove 曲面重新标注，不改变 q 或现有控制 / 学习模型。

每次约 250 ms 抽样一个已经发布的完整姿态，三种方法共用这组 q，忽略不影响自碰撞的整体腕部刚体位姿：

| 网页列 | 查询方式 |
|---|---|
| 真实网格 · 密集 | 原生顶点、面心与固定曲面样本，复用 `WujiMeshCollision` |
| 真实网格 · 同点 | 每 link 固定 128 个曲面点，对其他允许 link 的真实网格查询距离 |
| Neural SDF · 同点 | 与上一行完全相同的 x / q / target link，用 `mlp100k_v1/best.pt` 预测距离 |

三种方法均以最大穿透 **>0.5 mm** 为碰撞，共用相同连接 / MCP collar 排除。对照样本包含原生粗采样的 54 点，再补充面积均匀曲面点；源点属于本 link 时不查询它自身。每帧约 4.8 万个允许的有向查询，按 link 对聚合双向最大深度。

显示全手碰撞状态、最大穿透、碰撞 link 对数、各项耗时、同点距离 MAE、同点与密集最大深度差及累计判定不一致帧数。表格列出任一方法深度超过 0.1 mm 的 link 对。密集 / 同点网格之间的差异属于离散覆盖差异，**同点网格 / MLP** 才是相同见证点上的模型误差比较。

API：`GET /api/state?view=wuji2` 的 `wuji_collision`；开关使用 `POST /api/command`：

```json
{"command":"wuji2_collision_set","enabled":true}
```

检测在独立子进程执行，Web / SDK 线程不加载 PyTorch，最多一个请求在途。保存该诊断帧的源时间戳、q、求解器 session、模型名称与 checkpoint SHA。断流、过期、切换求解器 / 模型 / 回放会清理旧结论；暂停回放保留该帧；若 q 超出训练域，明确标记域外预测且不夹角。权重缺失或不兼容时保留真实网格结果，并显示 MLP 不可用，不把错误当作无碰撞。

默认权重路径可通过 `MOXIAN_WUJI_SDF_MODEL` 覆盖，检测解释器沿用 `MOXIAN_GEORT_PYTHON`。子进程日志：`output/moxian/wuji2_collision/collision-comparison.log`。GPU 验证预热后每帧总计算约 11–12 ms（含密集网格与同点网格 / MLP），初次加载和编译更慢。频率仍限制为约 4 Hz，以减少与其他训练 / 求解任务争用。

2026-11-19 已用兼容 GeoRT 真实录制验证网页 API 的同帧 q / 时间戳、暂停、定位更新及停止回放后清除结果；验证后返回实时输入。检测进程另验证了无碰撞和随机碰撞两种姿态。`tests/test_wuji_sdf_comparison.py` 覆盖同点 oracle 一致性、无效 / 过期 / 切换 session、域外角度与模型缺失；`node tests/wuji_collision_frontend.cjs` 覆盖控制按钮、差异显示与清除过期结果。

### 双三维可视化

打开 `http://127.0.0.1:8890/?view=wuji2&collision=test#wujiCollisionPanel`，自动展示独立测试姿态中的一个误报案例。点击“返回实时”显示当前求解器姿态。测试姿态只切换诊断视图，不改变实时求解器或发送手部控制。

左右两侧使用同一帧 q、原生碰撞网格、尺度和相机，分别根据真实网格与 MLP 的碰撞结果着色。默认以“同点网格”为参照：红色表示两者均检出，蓝色表示网格检出但 MLP 漏检，橙色表示 MLP 误报，灰色表示未检出碰撞。可同步旋转、缩放，选择 link 对淡化其他部位，打开线框或穿透查询点，悬停查看 link 名称及有符号距离。切换“密集网格”参照时，差异也包含曲面采样覆盖差异，页面会相应标注。

这里显示的是**原生网格上的碰撞判定对照**，并非重建出的 MLP 零等值面。查询点来自实际参与本帧检测的曲面点，按 link 对和检出类别保留最深点，最多显示 768 个；全量查询仍用于数值统计。双方几何变换随 q 更新，断流或上下文切换会清空旧图形。

6 个示例来自独立 10k 测试集，包括无碰撞、接近阈值、深穿透与误报姿态。可重新导出：

```bash
.venv-contact/bin/python -m moxian_web.wuji_sdf_comparison \
  --export-visuals output/moxian/wuji2_collision \
  --example-data output/wuji2_sdf/fresh_test10k_seed2026111801
```

静态资产通过 `/api/wuji_collision/model`、`/examples` 和 `/example?index=N` 提供。2026-11-19 已验证这三个 API 与全部前端模块正常响应，以及当前 `wuji-retargeting` 实时帧的 21 个 link 变换、48,255 次查询和可视化输出（该帧总计算约 14.5 ms）。Python 两个测试文件共 9 项通过；三个碰撞前端测试分别覆盖测试姿态切换、差异颜色、真实 Three.js 矩阵变换与失效清理。静态示例图：`output/wuji2_sdf/collision_visual_comparison.png`。

# ADMM 后接 MPPI 的配对实验

## 七类手势审阅策略

最新开发策略见[七类逐阶段验收](grasp_gesture_review.md)。新增可选 `gesture_reference`，共享独立于本次 ADMM 输出的整手形状、接触面及软朝向参考；没有该字段时维持旧代价。新 type 9 不固定旋转通道，旧 `reference_orientation_wxyz` 仅为历史策略兼容。
初始化失败或 ADMM 导出不完整的样本仍进入 MPPI 报告分母，明确跳过优化，不从旧结果补齐。
手势确认前不将审阅批次接入正式 viewer，也不启动 410 例批跑。

## Type 6–9 修正批次

已完成 pi0 CUDA 环境的 200 例软偏好重算，数值、限制和 viewer 接入说明见
[实测报告](grasp_soft_center_results.md)。原生物理通过为 ADMM 13/200、MPPI 直接
输出 12/200；单列验证后回退策略为 17/200。

下面的历史数值尚未包含新的接触区域/朝向约束。新接口读取 ADMM
`native_candidate.json` 中可选的 `contact_regions`、`reference_orientation_wxyz`：
6–8 的必要接触和对应 wrench 候选覆盖整个暴露的 distal，用有限软代价鼓励
掌侧中心接触，允许几何收益更好的区域外接触；9 的旋转
通道固定在 1.5× 初始抓姿经掌面内 +90° 旋转后的朝向。没有这些字段时保留旧行为。

ADMM 保存 `contact_region_diagnostics.json` 和逐轮局部锚点；MPPI 在
`refinement.json` 保存优化锚点诊断、参考朝向及轨迹误差；公共原生评测在
`evaluation.json` 独立记录实际接触区域。`semantic_success` 单独统计，不替代
`graspadmm_native_v1` 的物理成功判据。
软偏好不要求每个优化锚点或实际接触都处于 0.3–0.7 诊断区；审计检查候选保留
区域外点、软代价有限，并单独报告实际中心接触比例。

本机通过 `pi0` 派生的两个项目虚拟环境运行，保留其 Torch 2.10.0+cu128 以支持
RTX 5060 Ti。原生 ADMM/评测使用 MuJoCo 3.3.3、Warp 1.8.0；MPPI 使用
MuJoCo 3.10.0、MJWarp 3.10.0.3、Warp 1.16.0；未改动 `pi0` 本身。
GPU 需要在允许设备访问的执行环境运行；沙箱内 `cuda.is_available()` 为 false
不能据此判断宿主机无 CUDA。实际版本和 GPU 信息随批次保存；本次性能不与历史
RTX 4090/Torch 2.5.1 批次作直接速度比较。

在依赖齐全的 CUDA 环境先运行 20 例 smoke，再运行 200 例完整修正批次：

```bash
.venv-graspadmm/bin/python scripts/run_graspadmm_batch.py --types 6 7 8 9 --seeds 0 --output graspadmm_region_pose_smoke
.venv-contact/bin/python scripts/refine_graspadmm_mppi.py --source output/graspadmm_region_pose_smoke --output output/graspadmm_region_pose_mppi_smoke
.venv-contact/bin/python scripts/audit_graspadmm_mppi.py --batch output/graspadmm_region_pose_mppi_smoke

.venv-graspadmm/bin/python scripts/run_graspadmm_batch.py --types 6 7 8 9 --output graspadmm_region_pose_200
.venv-contact/bin/python scripts/refine_graspadmm_mppi.py --source output/graspadmm_region_pose_200 --output output/graspadmm_region_pose_mppi_200
.venv-contact/bin/python scripts/audit_graspadmm_mppi.py --batch output/graspadmm_region_pose_mppi_200
```

每个 type 9 随机 seed 只有一次原生 1.5× 初始化，不复用优化后的抓姿，不按结果
重试。smoke 和完整批次使用独立目录。审计通过且检查实际接触诊断后，将 viewer 的
`visualization.batch` 指向新批次即可回放；未完成重算前继续展示已有结果。

## 历史完整批次

1610 例已全部实际完成，均使用原版过滤、squeeze/pregrasp 和六方向外力测试。

| 方案 | 成功数 | 成功率 |
|---|---:|---:|
| 原 GraspADMM | 524 / 1610 | 32.55% |
| ADMM → 局部 MPPI，直接输出 | 573 / 1610 | 35.59% |
| MPPI 验证失败时保留原 ADMM，单列回退策略 | 647 / 1610 | 40.19% |

直接微调净增 **49 例、3.04 个百分点**：修复原失败 123 例，同时损失原成功
74 例。修复中 99 例原先止于 grasp filter、24 例原先六方向测试失败；损失中
31 例止于 grasp filter、43 例六方向测试失败。正式批次没有运行故障或超时。
全部 161 个尺寸组合均保留 10 个种子，来源姿态与网格的 8050 项哈希核验通过。
所有 573 个成功案例均验证了完整六方向、每方向十个检查点及严格阈值。

收益随抓型变化：type 29 为 9→26/50，type 26 为 30→40/50，type 28 为
42→50/50，type 10 为 16→29/110；type 5 则从 47 降为 30/50，type 4 从
50 降为 42/50。对这组数据，验证后回退能保留所有原有成功并增加 123 例成功。

新增 MPPI 优化中位 **.337 s/例**，P95 .347 s；1610 例纯优化共 543.66 s，
原生评测共 450.05 s，含几何准备、I/O 和逐例统计的串行总墙钟为 **1206.12 s
（20.10 分钟）**。不包含已有 ADMM 生成时间。此前同输入、同预算的原版 ADMM
核心中位 1.162 s、迁移封装 1.347 s，比例 1.16×，数量级一致；新 MPPI 使用
GPU 几何评分，不能将这两个时长解释为等价工作量下的优化器加速比。

完整结果位于 `output/graspadmm_mppi_wuji2_results`：`report.json`、`by_type.csv`、
`by_size.csv`、`trials.csv`、`timings.json`；图为 `type_comparison.png` 和
`size_comparison.png`。`audit.json` 与 `source_audit.json` 均通过，四个独立进程
重放案例逐位一致，记录在 `deterministic_replay_audit.json`。

推送到 Git 的自包含副本位于 `results/graspadmm_mppi_best`，包含全部 1610 例、
ADMM/MPPI/所选抓姿、647 个成功的真实原生回放、失败记录、相对路径的网格与模型，
以及测量时的代码快照和完整报告。可用 `scripts/publish_grasp_results.py --verify
--activate` 校验并接入回放。原始 `output` 不改写；副本的 evidence 中保留的绝对
路径只是来源记录，实际回放只使用包内相对路径。

该入口从已有 GraspADMM **最终原始抓姿**出发，固定实际物体网格、接触 body 分组和
每例的种子，做静态局部 MPPI 微调。采样器复用 `retarget.solver.spider_mppi.optimize`；
搜索变量是手根 xyz、XYZ 欧拉角和 20 个 WUJI2 手指关节。它使用新的局部几何代价，
包含接触间隙、法向、真实碰撞、近似 wrench support 和相对 ADMM 姿态的位移惩罚。
这是新增的组合方法，不能将它的代价称为原版 ADMM 或原版 SPIDER 的代价。

统一通过条件仍是固定版本的 `graspadmm_native_v1`：原版 grasp filter、QP、
squeeze/pregrasp 以及六方向外力测试。优化过程没有调用物理评测来选采样或重试。
每例固定两阶段，每阶段 8 轮、每轮 256 个候选，共 4096 个采样。第一阶段噪声为
1.5 mm / .015 rad / .04 rad，第二阶段缩至四分之一；保留几何代价最好的实际候选。
完整配置见 `configs/graspadmm_mppi_refine.yaml`。

物面接触目标间隙为 .7 mm；优化时使用 .1 mm 碰撞检测 margin 和 .05 mm 的小幅
间隙惩罚，减小平方穿透代价允许数微米穿透而被原版零穿透过滤拒绝的风险。MJWarp 的正
margin 不支持 MULTICCD，因此仅优化模型使用 single CCD；公共评测会重新构建原版
模型，碰撞设置和通过阈值不变。手指关节边界内缩 1 μrad，避免 float32 导出越界。

这类局部代价不能保证物理成功：新版 GPU single CCD 与原版 MuJoCo 的接触生成
存在差异，近似 wrench support 也不能替代原版 QP 和六方向测试。正式批次保留
所有因此产生的失败和退步，不能用代价下降或正间隙惩罚为零冒充通过。

预试验使用 8 个代表抓型、三个尺寸、seed 0，共 24 例。最初零 margin 的探索版本
仅 4/24 通过，原 ADMM 为 8/24；加入上述间隙后为 11/24，修复 4 例、损失 1 例。
两次预试验都保留在 `output/graspadmm_mppi_pilot` 和
`output/graspadmm_mppi_clearance_pilot`。正式运行固定参数，换用新的 MPPI 随机种子
偏移 200000；覆盖全部 1610 例，没有按抓型调参、追加候选或挑选中间帧。

统计明确区分三个结果：原 ADMM；MPPI 直接输出；单独的“MPPI 验证失败则保留原
ADMM”策略。最后一种包含额外的验证与回退，不作为 MPPI 直接成功率报告。运行
失败留在分母。报告同时给出 retained / lost / recovered / neither 配对计数。

## 环境与运行

规划与原版验证使用不同环境，避免新版 MJWarp 修改原版 MuJoCo 依赖：

```bash
bash scripts/setup_graspadmm.sh
python3.12 -m venv .venv-contact
.venv-contact/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv-contact/bin/pip install -r requirements-graspadmm-mppi.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-contact/bin/python scripts/refine_graspadmm_mppi.py \
  --source output/graspadmm_wuji2_results --output output/graspadmm_mppi_wuji2_results
```

已有 `.venv-contact` 可直接使用。完整实验使用 RTX 4090，Torch 2.5.1+cu121、
MuJoCo 3.10.0、MJWarp 3.10.0.3、Warp 1.16.0；原版评测另用
`.venv-graspadmm` 的 MuJoCo 3.3.3。运行中保存实际版本、源码副本与哈希、每个
原始抓姿及真实网格的哈希。重复相同命令即可跳过已完成项；配置、源码或输入
改变时拒绝混入同一目录。默认串行执行，一个 GPU 规划器和一个持久 CPU 评测器。

小规模运行可附加 `--types 4 9 11 --seeds 0 --size-indices 0 2 4`，并使用新输出
目录。中断不会删除已有结果；已记录失败属于完成的试验，不自动追加重试。

```bash
.venv-contact/bin/python scripts/plot_graspadmm_mppi.py --batch output/graspadmm_mppi_wuji2_results
.venv-contact/bin/python scripts/audit_graspadmm_mppi.py --batch output/graspadmm_mppi_wuji2_results
.venv-contact/bin/python run.py --config configs/web_graspadmm_mppi.yaml
```

输出含 `report.json`、`trials.csv`、`by_type.csv`、`by_size.csv`、耗时和配对图。
每例保留 raw MPPI 抓姿、阶段轨迹、代价分解、原生评测轨迹与回退选择。汇总命令
为 manifest 补齐展示阶段：`refinement` 对应“接触建立”，`native_test` 对应
“GraspADMM评测”；前者默认展示 MPPI 最终姿态，播放的是优化记录。通过状态
只读取公共原生评测。

## 历史 MPPI 重评

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-contact/bin/python scripts/reevaluate_grasps.py \
  --source output --output output/mppi_native_reevaluation --methods mppi
```

本次重新实际评测历史 237 例 MPPI 输出，0/237 通过，236 例止于 grasp filter，
1 例缺少有效最终输出；耗时 23.84 s。第一项失败原因依原版检查顺序统计：手物
穿透 167、手手穿透 27、关节越界 15、无接触 26、required group 缺失 1，另有
1 例无有效输出。文件保留在 `output/mppi_native_reevaluation`，来源没有被改写。
这些历史对象、初始化和预算与新 1610 例不同，不构成相同输入上的 MPPI/ADMM
优化器对照。它们包含历史静态 MPPI 和 SPIDER/跟踪批次，来源明细见
`failure_breakdown.json`。

本仓库通过一个 YAML 接口运行 grasp、dex 和 humanoid，支持 GPU CEM、MPPI、旧 SCSP，以及搬箱与 WUJI2 抓取的 DexForge Force-Aware 优化。

新增 [WUJI2 GraspADMM 与公共评测](docs/graspadmm.md)：`solver.type: graspadmm` 直接使用固定版本的原版 ADMM、QP、MuJoCo、filter 和 pregrasp 流程。运行 `bash scripts/setup_graspadmm.sh` 后使用 `.venv-graspadmm/bin/python run.py --config configs/graspadmm.yaml`。新 grasp 结果统一以原版六方向外力测试决定 `task_success`，旧载荷/接触指标保留为诊断。共享 primitive 尺寸策略覆盖 31 个抓型、161 个尺寸组合；十种子批测和历史重评命令见文档，完成情况以各输出目录的 `status.json` 为准。

[完整实测](docs/graspadmm_results.md) 已完成：524/1610（32.55%），type 9 的侧面初值修正后全50例重测并完整替换；515例历史结果已独立重评。同输入、同50×10预算下，原版核心与迁移记录包装为1.1619/1.3473秒，约1.16倍，最终抓姿逐位一致。

```bash
pip install -r requirements.txt
python run.py --config configs/grasp.yaml
python run.py --config configs/dex_sugar_box.yaml
python run.py --config configs/humanoid_sprint.yaml
python run.py --config configs/humanoid_move_largebox_contact_mppi.yaml
python run.py --config configs/web.yaml
```

CUDA 是求解所需条件。当前验证环境为 NVIDIA RTX 5060 Ti、MuJoCo 3.10.0、MJWarp 3.10.0.3、Warp 1.16.0、Torch 2.13.0。requirements 中的本机引擎版本用于重现本次验证，不代表 SPIDER 上游锁定环境。

根目录 `task.yaml` 现在选择 [DexForge Force-Aware 搬箱](docs/humanoid_forceaware.md)，使用隔离的 Python 3.10 环境：

```bash
bash scripts/setup_forceaware.sh
.venv-forceaware/bin/python run.py --config configs/humanoid_move_largebox_forceaware.yaml
```

该分支直接使用数据中的关节角，不运行 IK、Contact-Aware、ADMM、接触点 CEM 或 IPOPT。规划和执行使用固定提交的原版 FA native dynamics/adjoint；完整运行先通过真实动力学梯度检查。以下 SPIDER/ComFree 响应及 SCSP 描述针对各自的显式旧模式。

统一配置示例：

```yaml
mode: solve                    # solve / replay / web
task:
  type: humanoid               # grasp / dex / humanoid
  name: move_largebox          # humanoid 另支持 sprint
  data: data/humanoid/move_largebox
solver:
  type: cem                    # cem / mppi
  population: 1024
  iterations: 16
  worlds: 128
  elite_fraction: 0.05         # CEM
  temperature: 0.1
  final_noise: 0.2
environment:
  device: cuda:0
  # frames: 1                  # 仅用于短测；省略时执行完整参考
seed: 0
output:
  name: humanoid_move_largebox_cem
visualization:
  live: false
  save_video: true
```

输入路径相对于 YAML 所在目录解析，示例配置位于 configs，因此使用 `../data/...`。抓取 `task.options` 中显式指定的资源路径也相对于 YAML 解析；内置默认资源相对于 `task.data`。在 CEM 与 MPPI 间切换只需修改 `solver.type`；省略 temperature/final_noise 时，CEM 默认 0.1/0.2，MPPI 默认 1.0/0.1。跑步示例显式采用上游 temperature=0.6。

输出固定为仓库内 `output/<name>`。配置和输入校验通过后会替换同名目录，其他结果目录保留；名称禁止包含路径或跳转，同名符号链接也会拒绝。结果包含解析后的 config.yaml、status.json、manifest.json、summary.json、轨迹、逐轮日志和来源记录。抓取批量任务使用 case 子目录。失败初始化或无硬约束可行候选会保留失败诊断，不能作为成功抓取导出。

目录职责：

| 目录 | 内容 |
| --- | --- |
| `retarget/solver` | CEM/MPPI、旧 SCSP 两层优化、原版 DexForge FA 的目标函数适配 |
| `retarget/task` | 各任务 cost、参考、搜索变量和验收指标 |
| `retarget/env` | 模型、接触查询、控制解码、rollout、执行与记录 |
| `retarget/dynamics` | ComFree 接触响应、独立的轻量显式接触预测 |
| `retarget/initialization` | 输入 IK、保留的 Dexonomy 初始化数值实现 |
| `retarget/config`、`common`、`viewer` | 配置、结果与来源记录、回放和网页查看 |
| `data/dexycb` | 两批共十条原始打包序列、原始资产/标定和 prepared 示例 |
| `data/humanoid` | move_largebox、sprint 与共享 G1 场景资产 |
| `data/grasp` | 抓取模板、两代手资产、冻结手腕参考 |

CEM 使用 CARD 精英成本标准化后的 softmax 均值更新，精英数量向上取整，不拟合精英方差；噪声按 `sigma * final_noise ** (i / iterations)` 退火，每轮包含历史最佳与均值。MPPI 使用前 10% 候选加权、同形式退火和可配置提前停止。两者分离 population 与 worlds，并通过 Torch/Warp 共享数组分批评分。日志记录实际迭代数、采样数与额外评分数。

Humanoid 使用原始 SPIDER reward、terminal reward、参考插值和 post-step 时域聚合；搬箱保留加权非平方 L2，权重为根位置/根旋转/关节/箱体位置/箱体旋转 `1/1/1/1/0.3`，无附加平衡或接触目标。48 步预测、6 步执行、8 个控制 knots、CPU 初始化 pre-step、DR 与执行前参数恢复均保留。只优化关节控制增量，未添加控制裁剪。版本和函数 AST 哈希在任务包 source.json，迁移前冻结评分样本在 tests/fixtures。独立的 3 秒初始化预检会记录失败并继续，实际初态不受其影响。

可选的 [固定接触模型与梯度 MPPI](docs/humanoid_cost_contact.md) 使用 `cost_contact.py`：每个 MPC 窗口从 Warp 碰撞和动力学建立一次缓存，以 ComFree 可微递推对整个控制序列评分。`solver.gradient_steps` 开启同一 MPPI 内的梯度更新与回溯，默认关闭；原 `cost.py` 保持不变。示例配置执行完整搬箱参考，输出实际 MJWarp 执行轨迹及独立验收指标。

[SCSP 两层优化](docs/humanoid_scsp.md) 使用 `cost_contact1.py` 和 `solver/scsp.py`：机器人接触点为双手手心和双脚脚心，单组手心投影生成每手 11 个候选，GPU CEM 在 121 种组合内搜索；接触力使用梯度优化。IPOPT 根据当前轨迹的真实碰撞、Jacobian 和 ComFree 梯度更新机器人 action，ADMM 协调手部接触位置与反力。额外安装 `requirements-scsp.txt` 后运行 `python run.py --config configs/humanoid_move_largebox_scsp.yaml`。该模式需同时设置 `solver.type: scsp` 和 `environment.dynamics: scsp`。

Sprint 保留原 qpos/qvel/ctrl 数组，原始文件单独保存在 reference_raw.npz。实际 reference.npz 中两脚接触标签由参考 FK 的足部碰撞几何有符号地面间隙生成，阈值 35 mm；目标为 left_foot/right_foot site 的世界坐标。原始接触字段中的占位数据被替换，生成来源和阈值见 contact_provenance.json。这是推导标注，未冒称 SPIDER 官方标注。

Grasp 的两种求解器直接评分完整几何项、手腕参考、显式物体 pose cost 和硬约束，搜索手腕 6 维、手指 20 维与独立伺服偏移 20 维。显式预测对每个候选分批执行，无几何优化后的梯度细化。保留原输入初始化、接触区域、指腹中心策略和 WUJI1/WUJI2。预测采用轻量准静态接触模型，summary 中明确区分预测稳定与实际物理测试。更换手型使用 `task.options.hand_model: wuji1` 或 `wuji2`；更换抓型使用 task.templates，尺寸使用 radius_scales/height_scales。冻结手腕参考缺失时会记录输入失败，不会自动搜索历史输出。当前冻结库是 WUJI2 的已有参考；WUJI1 可提供匹配的冻结参考，或显式设置 `task.options.wrist_pose_prior: false`。

新增 [WUJI2 primitive tracking cost](docs/grasp_tracking.md)：`configs/grasp_tracking_mppi.yaml` 保留静态 46 维抓姿搜索，沿预测轨迹评分穿透、模板接触点接近、接触保持和静止物体 tracking 四项。公式复用本仓库 ForceAware/DexYCB dex cost，不宣称官方 SPIDER 灵巧手实现。独立配置不依赖缺失的 DexYCB 物理参考或冻结手腕参考；资产可由 `scripts/setup_grasp_assets.py` 准备。

[ContactAware 参考生成](docs/grasp_contactaware.md) 使用新增的原版 DexForge CA SQP/Clarabel 求解器，先联合优化物面锚点与 WUJI2 IK，再将所得手姿作为 MPPI 的第五项 `hand_tracking` 参考。运行 `configs/grasp_contactaware_mppi.yaml`；额外依赖见 `requirements-grasp.txt`。模板只提供语义 patch 和初值，失败的 CA 参考不进入 MPPI；后续接触预测按数值残差验收。

[WUJI2 ForceAware](docs/grasp_forceaware.md) 将 CA 结果交给固定版本的原版 FA 优化器、损失和滚动执行。[修复记录](docs/grasp_forceaware_repair.md) 说明 WUJI2 的显式 PD 积分适配、完整手部球云约束、active mask、棱边法向、静态预载和逐步接触力审计；上游源码哈希与本地适配边界分别记录。运行 `.venv-forceaware/bin/python run.py --config configs/grasp_contactaware_forceaware_fine_samples.yaml` 可复现三尺寸 2 ms 执行验证。求解完成、物体保持、native 五指载荷和独立 MuJoCo 五指载荷分别验收，未全部通过时不会报告抓取成功。

[ForceAware 性能对照](docs/grasp_forceaware_performance.md) 提供 `configs/grasp_contactaware_forceaware_fast.yaml`：恢复原 40 次预算和 64 starts，同机三尺寸平均纯优化 37.27 s、含回放审计 47.70 s；每窗 2.66 s，与原版资产的有接触代理基线 2.85 s 相近。当前快速配置三例均未完整通过五指保持；200 次质量验证结果单独保留，不能混用成功结论。

[WUJI2 SPIDER MPPI](docs/grasp_spider_mppi.md) 保持已验收的 ContactAware 参考不变，以 SPIDER 灵巧手的配置/物体轨迹/接触点 tracking 和 native MuJoCo Warp 替换 FA 优化。原 tracking 对照保留在 `grasp_contactaware_spider_tracking.yaml`，原始权重对照另见 `grasp_contactaware_spider_mppi.yaml`。加入接触保持项前，三尺寸 0.4 s 测试平均纯优化 12.47 s，物体保持 3/3，五指连续载荷 0/3。这是独立的时序控制 MPPI 入口，记录每个 2 ms 执行步的真实五指接触力，并独立 CPU 回放验收。预算、效果和与旧 FA 的比较边界见文档。

新增 [CA 手指接触保持惩罚](docs/grasp_spider_mppi.md#ca-手指脱离接触惩罚)：推荐运行 `.venv-contact/bin/python run.py --config configs/grasp_contactaware_spider_contact_keep.yaml`。在相同 CA、tracking 权重和 MPPI 预算上，对指定手指的实际法向载荷低于 0.05 N 加罚；GPU 全程逐步测量，未调用 FA 优化器。三尺寸五指同时承载比例从 13%/52.5%/35% 提升至 69%/81.5%/86%，平均纯优化 15.97 s，物体保持仍为 3/3；每一步五指均承载的严格验收仍是 0/3。

[全抓型五尺寸后台批次](docs/grasp_spider_batch.md) 使用 `scripts/run_spider_grasp_batch.py --launch --workers 2`：每个抓型固定一个主 primitive，重新采样半径倍率 0.5/0.75/1/1.25/1.5，type 1–6 的 cylinder 长度为 0.18 m，共 155 个可运行组合；19/21 两类单独记录缺少模型。历史 190 例批次保留原始结果，Viser 仅展示其中 155 个主 primitive 案例。每例先求 CA，再跑上述 MPPI 和 CPU 回放，逐例失败和保持验收分开记录；后台状态及代码快照保存在新批次目录。

[GraspADMM 与统一评测](docs/graspadmm.md) 提供独立原版 ADMM 后端和修正后的 161 个 primitive 尺寸组合。新增 [ADMM→MPPI 配对微调](docs/graspadmm_mppi.md) 从保存的最终 ADMM 抓姿出发，固定预算跑局部 MPPI，再使用同一原生物理判定；分别统计直接输出、修复/损失及验证回退策略，支持断点续跑和网页回放。

仓库包含[最高成功率的完整可移植结果](results/graspadmm_mppi_best/README.md)：验证回退策略 **647/1610（40.19%）**，直接 MPPI 为 573/1610（35.59%），原 ADMM 为 524/1610（32.55%）。保留全部失败、两种原始候选、真实几何、原生评测与回放，不依赖原机器的 output 目录。运行 `.venv-contact/bin/python scripts/publish_grasp_results.py --verify --activate` 后，以 `configs/web_grasp_best.yaml` 打开网页；基础 WUJI2 几何、查询点、模板及许可证也随仓库提供。

旧模式的 ComFree 只实现接触响应：`retarget/dynamics/comfree/{kernels,response}.py`。三个 kernel 的函数体来自 [上游 ComFree Warp](https://github.com/asu-iris/comfree_warp/tree/ba8b9969f5f7859d51b1f738f458f8d2f49be080)，固定 commit 和 AST 哈希见同目录 source.json。原始响应为预测速度、预测侵入、非负行接触力与 `J.T @ force`；没有 softplus 或多轮线性化。`retarget/env/mjwarp.py` 调用已安装的 MJWarp 完成碰撞检测、约束行装配、FK、质量矩阵、平滑动力学和积分，仅替换接触求解阶段。模型 YAML 的 `contact_physics.stiffness/damping` 默认为 `0.2/0.001`，行阻抗宽度按上游固定为 `0.01`。不支持的双侧等式和 frictionloss 约束会明确报错。

旧模式的物理输入使用标准 MuJoCo mesh。MJWarp 的 mesh 碰撞采用凸碰撞表示，不能宣称与 SDF 接触几何的动力学结果一致。手部 NPZ 采样点和 `retarget/env/surface.py` 的 Warp 内置 mesh 距离查询仅为任务 cost 提供表面测量，不生成物理接触。新 DexForge 分支独立使用固定版本的 source-cloud/mesh 接触与 native adjoint。Grasp 的 `explicit.py` 仍是独立的轻量准静态预测。

Dex 默认使用 prepared/sugar_box，保留原主目标公式、参考和控制插值，动力学采用上述 MJWarp + ComFree 响应。初态固定，搜索动作 knots，MANO 接触指导固定为参考；没有 consensus/dual、接触点交替优化或物体虚拟控制。原始序列尚未包含完整 prepared 输入时会明确报错；本次不实现 MANO 自动转换。仓库外 MANO 模型不受本次整理影响。

回放和网页都读取 manifest.json：

```bash
python run.py --config configs/replay.yaml
python run.py --config configs/web.yaml  # http://localhost:8081
```

回放配置的 output.name 指向已有结果；不会清空该目录。`live: false, save_video: true` 可用于无显示器的视频导出。网页自动扫描 output，支持任务、抓型、手型及 CEM/MPPI 筛选。统一入口没有旧脚本或命令行选项的转发兼容层。

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

验证仅使用短测，没有重新完成整套任务求解。本次对受影响的 grasp、dex 分别完成 CEM/MPPI 短测，固定 SPIDER reward 样本逐元素一致，新增接触响应及 MJWarp 委托验证；旧的模型数组一致性结论不适用于这次碰撞几何替换。具体预算、结果、耗时与验证限制见 [验证记录](docs/validation.md)。这些结果证明新流程可运行，不构成任务成功率提升或完整搬箱成功的结论。来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

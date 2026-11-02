# WUJI2 GraspADMM 与公共抓取评测

`solver.type: graspadmm` 是独立的 Dexonomy GraspADMM 后端。固定源码提交为
`d561ff9df8545deef7a3f9e7cfb4045ef66f6913`；原文件与 SHA-256 清单位于
`third_party/Dexonomy/GRASPADMM_UPSTREAM.json`。WUJI2 的 free-root 场景、模板映射、
固定指尖 pad 归属和输出格式是本地适配；不使用 MPPI、ContactAware 或 ForceAware
作为该后端的优化器。

```bash
bash scripts/setup_graspadmm.sh
.venv-graspadmm/bin/python run.py --config configs/graspadmm.yaml
```

初始化在 `environment.device` 指定的 CUDA 上运行。ADMM 保留上游默认 CPU 路径：
物面切向梯度更新、Clarabel contact QP、MuJoCo 弹簧力推进和对偶更新。初始化批量
QP 使用上游 RELUQP。默认 50 轮、10 仿真子步、rho=1000、学习率=0.0002，保留
原版 QP 失败 fallback、grasp filter、squeeze 和 pregrasp；数值参数位于
`solver.graspadmm`。上游源码完整性在载入时验证。

每个试验采用 `epoch=1, n_final=1`，固定模板，不回灌先前成功结果，不因失败重采样。
Python、NumPy、Torch 和物面采样都使用显式种子。模板的 Shadow→WUJI2 迁移属于
机器人输入适配，不改变 ADMM 的 cost 或增加本地通过条件。

`task.options.primitive_seed_policy` 默认 `taxonomy`：type 9 的原始物体目标若落在
端盖，增加解析手根/物体侧面目标的几何输入桥接；原本已在侧面的初值和其他抓型
严格不变，20 个指关节角全部保留。不添加 IK、优化器、cost 或过滤条件。
`upstream` 可复现纯原版初始化。原始与修正初值、触发条件及初始穿透均分别保存。

## 尺寸批次

```bash
.venv-graspadmm/bin/python scripts/run_graspadmm_batch.py
.venv-graspadmm/bin/python scripts/run_graspadmm_batch.py --resume output/graspadmm_all_types_10seeds
```

默认单 worker，31 个可运行抓型、161 个主 primitive 尺寸组合、seed 0–9，总计
1610 次试验。19/21 需要剪刀或多物体资产，单列 unsupported。可用 `--types`、
`--seeds`、`--output` 创建不同名称的小规模验证批次；不能覆盖已有批次。
每个试验保存配置、日志、状态和求解结果；根目录的 `summary.json`、
`success_rates.csv` 与 `manifest.json` 支持统计和回放。初始化失败、求解失败和
超时均留在试验分母；尚未执行的试验有独立状态。

所有默认尺寸由 `retarget/env/grasp/primitives.py` 统一生成。type 10 使用 1–1.5、
步长 .05；其余指定区间均匀五档。16/22 固定宽度只变化厚度，30 固定半径
30 mm、厚度 4–12 mm。元数据中的 `size_axis/size_scale` 描述实际变化轴，
`radial_scale/axial_scale` 则分别记录真实半径和长度倍率。显式半径、长度与网格
覆盖保留原 Cartesian 行为，标准批测使用新的 per-type 策略。

## 公共通过标准

所有新 grasp 结果的 `task_success` 使用 `graspadmm_native_v1`。原有预测稳定性、
五指载荷和 GPU/CPU hold 指标保留为方法诊断，不与新通过判定相与。

公共评测重建真实物体网格和 free-root 手场景，物体质量固定 .1 kg。
各方法优化时的模型、质量、cost 和控制流程保持其原设置；评测不复用各方法的
伺服偏移或控制序列。静态 CEM/MPPI 取直接优化抓姿，FA/SPIDER 取最终执行状态的
手物相对抓姿，ContactAware 取真实优化输出；不会根据评测结果挑选最优轨迹帧。

先使用原版 grasp filter、QP 和 squeeze/pregrasp，然后使用原版 `eval_fc`：

- 禁用重力，物体六个 ±xyz 方向各施加 `mass * 9.8` 的恒力。
- 每个方向重置，执行 pregrasp→grasp→squeeze；摩擦 `[.6, .02]`。
- timestep=.004 s，每方向 10×50 步，共 2 s；每 50 步检查一次。
- 平移严格小于 .05 m、转角严格小于 15°；六方向均通过才成功。

过滤或 pregrasp 失败不进入扰动测试。缺失真实方法输出时记录失败，不能拿初始化
姿态替代。type 9 侧面接触、type 11 掌面接触、16/22 厚度朝向另存诊断；不新增
上游之外的最终通过门槛。

```bash
.venv-graspadmm/bin/python scripts/evaluate_grasp.py --help
.venv-graspadmm/bin/python scripts/reevaluate_grasps.py \
  --source output --output output/grasp_history_graspadmm_eval
# 中断后继续，原始求解结果不改写：
.venv-graspadmm/bin/python scripts/reevaluate_grasps.py \
  --source output --output output/grasp_history_graspadmm_eval --resume
```

历史重评合并 manifest 别名，保留源案例、几何和哈希，使用保存时的实际网格。
旧 type 9 disk 不会被替换为新 cylinder。不同尺寸定义和历史求解预算分开分组，
不能将历史重评统计解释为相同输入上的优化器对照。完整的新尺寸批次、历史重评
是否结束以各输出目录的 `status.json` 为准。

## 本机验证与耗时对照

```bash
.venv-graspadmm/bin/python scripts/benchmark_graspadmm.py --output output/graspadmm_performance
.venv-contact/bin/python run.py --config configs/web_graspadmm.yaml
.venv-contact/bin/python run.py --config configs/web_grasp_evaluation.yaml
```

性能对照固定同一 WUJI2 初值、网格、50×10 预算、CPU 单线程，预热一次后测五次：
未改原版 operation 中位数 1.1619 s，本地完整记录包装 1.3473 s，比例 1.1596；
最终抓姿逐位相同。该基线同样包含必要的 WUJI2 根部和指尖 pad 几何适配，
不代表 Shadow 手模型的论文性能复现。

独立阶段测量：固定模板缓存命中准备 0.0675 s，8192 候选原生初始化 0.6151 s，
公共物理评测 0.9876 s；首次未缓存骨架转换 2.2715 s 单列。
真实测量与逐阶段原始样本在 `output/graspadmm_performance/benchmark.json`。
GraspADMM 专用进程在版本吻合时直接调用同一公共评测器，其他方法仍隔离调用；
两种路径的判据、60次检查及全部轨迹数组已逐元素对照一致。

历史最终重评在 `output/grasp_history_admm_evaluation_final`：515 例，509 例完成
原生判定、6 例缺失最终输出，0 例通过。508 例在 grasp filter 拒绝，1 例进入
六方向测试后失败；无评测运行异常。物理 sweep 总耗时 64.68 s，后续元数据刷新
单独记录。历史失败分布、源文件哈希审计及正例对照证据见该目录 `audit.json`。

专项原生验证 138 项通过；现有配置、求解器、CA、FA、SPIDER、回放兼容回归
215 项通过、2 项可选测试跳过。上游源码的弃用警告不通过修改 vendor 字节消除。

## 完整尺寸结果

[实测结果](graspadmm_results.md) 已完成：31 抓型、161 尺寸、1610 次，524 次通过，
成功率 32.55%。type 9 另将五尺寸十种子全部重测，50/50 初始目标和最终必要远端
接触位于侧面，3/50 通过原生物理测试；以这 50 次完整替换对应对照组，未按成败择优。

最终主结果为 `output/graspadmm_wuji2_results`，原版初始化对照和 type 9 重测分别
保留在 `output/graspadmm_all_types_10seeds_final`、`output/graspadmm_type9_side_seed_10seeds`。
主目录记录两源代码版本、输入策略、实际依赖及计时，不将合并生成时间当作求解耗时。

```bash
.venv-contact/bin/python scripts/plot_graspadmm.py \
  --batch output/graspadmm_wuji2_results \
  --history output/grasp_history_admm_evaluation_final \
  --performance output/graspadmm_performance
```

绘图脚本使用 Matplotlib；统计包含全部失败和超时，不因过滤掉失败而提高分母口径。

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

每个试验采用 `epoch=1, n_final=1`，不回灌本次优化结果，不因失败重采样。
Python、NumPy、Torch 和物面采样都使用显式种子。模板的 Shadow→WUJI2 迁移属于
机器人输入适配。未提供参考的旧路径保留原数值行为；下述新策略明确增加本地输入质量检查和优化软参考，不宣称等价于原版搜索。

`task.options.gesture_policy: whole_hand_v1` 是七类 9/10/12/22/23/30/31 的默认本地策略。
整手联合拟合骨方向、跨指位置与接触法向，再按实际网格调整开口，保留原始接触手指和 required-body 分组。
接触面偏好、关节形状、指间位置与朝向均为有限软代价。ADMM 优化环境添加软势能对应的力，MPPI 读取同一个 `gesture_reference`；外力评测不施加这些力。
原始初始化过滤器的真实通过数、padding 来源与重新检查结果分别记录。输入可通过完整原生初始化门槛，或真实 required-body 接触、原生接触 QP、关节限位和几何检查；初始化容许的最大网格重叠为 1 mm，后续原生严格评测阈值不变。未合格时输出明确失败与拒绝候选。
type 12 的唯一已确认参考来自归档 `t12_disk_s04_seed00` 的 selected MPPI 姿态，复制到批次 `gesture_references/type12.json` 并保存来源哈希；这是显式用户参考，不是原生初始化结果。详见[逐阶段验收](grasp_gesture_review.md)。

`task.options.primitive_seed_policy` 默认 `taxonomy`：type 6–8 的初始化选 distal
掌侧中心附近的真实表面点；ADMM 和 MPPI **允许整个暴露的 distal 表面接触**，
包括指尖、边缘和背侧，不用中心区域裁剪候选。通过有限软代价鼓励接触靠近长度
中点并朝向掌侧，几何接触更好时仍可选择区域外的点；完整碰撞几何保留。
归一化长度 0.3–0.7、掌侧法向点积 >0.5 仅用于初始化参考和结果诊断。
`task.options.contact_center_weight` 默认 0.25，中心代价为
`weight * ((u - 0.5) / 0.2)^2`，另有权重 0.1 的掌侧法向软偏好。
ADMM 投影距离按 2 mm 归一化后与软代价相加，MPPI 在原接触代价中加入相同软偏好。

仅在 `gesture_policy: legacy` 下，type 9 对每个随机 seed 生成一次 **1.5× 原生初始抓姿**，所有尺寸复用它的手指
角度和相对位姿。绕物体中心、掌面 +Y 法向按右手规则旋转物体 +90°，用逆手根变换
表达；重新投影到各尺寸圆柱的侧面。ADMM 每个仿真子步固定手根旋转，MPPI 固定
旋转通道；位置和手指允许优化。公共原生外力评测不施加此约束。

旧 +90° 朝向目前不再作为已确认的手势基准；新策略不锁朝向，先输出方向候选供用户审阅。
这是带接触软偏好和手势参考的本地优化变体，不再等同于原版搜索。
`primitive_seed_policy: upstream` 绕过这些适配。其他抓型和原生通过阈值不变；历史报告仍对应历史源码。
参考初值、来源哈希、旋转矩阵和接触区域诊断分别保存。独立运行的参考缓存默认位于
输出目录的 `type9_seed_references`，批处理在各尺寸 worker 间共享该缓存。

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
步长 .05；type 12 使用 1、1.125、1.25、1.375、1.5，厚度固定 16 mm。尺寸策略版本为 3，旧 type12 s04=1× 不重新解释为新 s04=1.5×。其余指定区间均匀五档。16/22 固定宽度只变化厚度，30 固定半径
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

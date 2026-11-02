# CA 初值、固定物体 MPPI 与自由物体保持

当前流程是 **CA 提议 → 固定物体的 MPPI 接触建立 → 自由物体保持 → 独立 CPU 回放**。CA 的 `success`、激活 mask 和严格 refinement 均不作为 SPIDER 准入门槛。有限但未收敛的 CA 提议继续优化；CA 数值异常时保留进入 CA 前的有限几何初始化，并记录来源。缺失资产、非有限状态、损坏的语义区域才属于输入无效。ForceAware 的旧接口仍只接受通过 CA 验收的参考。

新参考分开存储：`spider_seed.npz/json` 保存 CA 提议、全部模板语义要求和来源；`acquisition/reference/spider_reference.npz/json` 保存经真实几何与独立力闭合验收的 MPPI 参考。模板角度仅提供骨向形状先验，不是最终关节 tracking 目标。

## Power 掌面与指定手指

Power 编号为 **1–5、10、11、15、17–19、26、28、30、31**。19 需要尚不支持的剪刀资产；其余 14 类至少需要一个广义掌面接触。Precision、Intermediate 不额外添加掌面要求，模板本身明确要求的接触仍保留。

广义掌面包括腕部至 MCP 的外露掌侧、中央掌面、鱼际、CMC/MCP 根部壳体，以及下一指节靠近根部的三分之一。候选来自模型表面采样，按掌侧法向、所属实体与射线可见性排除手背、内部遮挡和支架。实际接触使用随实体运动的局部解剖区域；GPU cost 与 CPU 审计使用同一判定。掌面优先计数，同一原生接触不会同时计入手指。根部接触不能替代模板指定远端指腹接触。

六组载荷固定为 `thumb/index/middle/ring/pinky/palm`。要求由全部模板指定手指和 Power 掌面决定，与 CA 当时是否激活无关。type 1–6 仍要求中指真实接触位于长圆柱中央 20%（180 mm 圆柱为轴向 ±18 mm）。

## 优化与验收

固定阶段只优化手腕 6 自由度和 20 个手指关节，所有候选投影至模型关节限位，物体位姿逐位保持初值。使用 SPIDER 加权采样，粗搜索后细化，并对腕部和各手指分块采样。几何雅可比仅生成采样方向，最终仍按 MPPI cost 选取。

姿态 cost 包括语义接近、法向匹配、掌面接触、全手表面穿透、自碰撞、骨向形状、跨尺寸软先验、圆柱中段以及 GPU 批量摩擦锥 wrench support。旧 15 mm/15° 不再裁剪或拒绝候选，只进入软先验。力闭合近似仅使用原生真实几何接触，虚拟 anchor 不贡献 wrench；采样循环不调用 CPU 优化器。

最终均值及至多 8 个最佳候选分别接受独立 CPU 检查：指定语义接触全部建立、穿透 ≤0.5 mm、自碰撞深度 ≤0.2 mm、归一化 wrench rank = 6、圆锥 reserve >1e-4、平衡残差 <1e-6。近似 cost 降低不代表通过。

通过接触建立后，可选有界静态预载、固定物体的物理接触建立及短 MPPI 载荷修整提供释放初值；预载不可行不拦截动态 MPPI。自由物体从零初速度开始，完整测试 **0.4 s**，不剔除释放瞬态。动态 MPPI 保留配置、静止物体轨迹、接触点 tracking，并加入真实六组接触载荷与 wrench-support cost。预测中每 2 ms 检查载荷，每 20 ms 检查近似 wrench support；最终验收每 2 ms 独立检查真实载荷与精确力闭合。

GPU 执行及独立 CPU 控制回放必须同时满足：最大位置误差 **2 mm**、旋转误差 **5°**，每个要求的手指及掌面每步均有 **≥0.05 N** 真实法向载荷，每步真实受载接触通过上述力闭合证书。分别报告接触建立、几何力闭合和动态保持。`completed` 只表示计算完成，须看 `task_success`。

## 首轮与全量后台计算

```bash
.venv-contact/bin/python scripts/run_spider_grasp_batch.py --launch --pilot \
  --batch output/grasp_spider_acquisition_pilot_v03 \
  --cached-batch output/grasp_spider_pose_consistent_five_sizes --workers 2
```

首轮为 Power **1、5、10、15、26、31** 与对照 **6、23**，各 **0.5/1/1.5×**，共 24 例。缓存只复用有限 CA 几何提议，并记录原始路径；不会沿用成功标志作为验收。去掉 `--cached-batch` 可重新运行 CA。每批目录必须全新；代码和配置复制为不可混入后续编辑的 runtime 快照，旧批次保留。

只有首轮全部结束，且圆柱、球体、盘各有一个完整成功案例，才允许全量：

```bash
.venv-contact/bin/python scripts/run_spider_grasp_batch.py --launch \
  --batch output/grasp_spider_acquisition_all_types_five_sizes \
  --promote-from output/grasp_spider_acquisition_pilot_v03 \
  --cached-batch output/grasp_spider_pose_consistent_five_sizes --workers 2
```

需要在首轮结束后自动判断并后台启动全量，可用：

```bash
.venv-contact/bin/python scripts/run_spider_grasp_batch.py --watch-promote \
  --batch output/grasp_spider_acquisition_pilot_v03 \
  --full-batch output/grasp_spider_acquisition_all_types_five_sizes \
  --cached-batch output/grasp_spider_pose_consistent_five_sizes --workers 2
```

门槛未满足时写入 `promotion_gate.json`、`promotion_status.json`，保留各阶段失败诊断，不启动 155 例，也不自动扩大采样量。全量为 31 个支持的主 primitive × 5 个尺寸；19（剪刀）、21（双筷）显式标记不支持。每个抓型仅一个主 primitive，尺寸为 0.5/0.75/1/1.25/1.5×；type 1–6 圆柱全长 0.18 m。

| 文件 | 内容 |
| --- | --- |
| `plan.json` / `launch.json` / `status.json` | 尺寸、冻结代码哈希、PID、队列、实际成功数 |
| `summary.json` | 全部逐例状态，失败也计入分母 |
| `jobs/<id>/attempt_NNN/spider_seed` | 未认证 CA 初值和完整模板接触要求 |
| `.../solve/<id>/acquisition/acquisition.json` | 各候选接触、穿透、自碰撞、独立力闭合诊断 |
| `.../solve/<id>/grip_establishment` | 可选固定物体载荷建立与 MPPI 修整 |
| `.../solve/<id>/trajectory.npz` | 自由物体 rollout、控制、真实六组载荷和接触点 |
| `.../solve/<id>/group_hold/summary.json` | GPU/CPU 每步载荷、保持误差和力闭合证书 |

终态为 `completed`、`input_invalid`、`solve_failed`、`timeout`；历史 `ca_failed` 仍可读取。两个独立 worker 并行，每阶段默认 1800 s 超时。对 supervisor 发 SIGTERM 会保留已有结果；用该批快照脚本 `--supervise --batch <目录>` 恢复时创建新 attempt。

## Viser

```bash
.venv-contact/bin/python scripts/prepare_spider_batch_viewer.py output/grasp_spider_acquisition_pilot_v03 --watch
.venv-contact/bin/python run.py --config configs/web_grasp_spider_batch.yaml
```

打开 <http://localhost:8081>。尺寸用 bar 吸附到已有结果；阶段可切换“接触建立”和“静止保持”。播放接触建立时展示优化迭代记录（不是物理时间），物体固定；保持阶段按 rollout 物理时间播放。青色显示广义掌面候选，橙色显示当前记录的实际几何接触；自由阶段只显示有正载荷的真实接触。失败结果保留，有限初值可预览；非有限输入只展示物体和错误信息。

默认正对实际掌面取景，跨尺寸保留相机相对掌面的视角和缩放。当前首轮覆盖已有同案例历史结果；尚未重算的其他案例继续显示历史来源。历史 `grasp_spider_all_types_five_sizes` 和 `grasp_spider_pose_consistent_five_sizes` 的 CA 严格门控结果保留在原目录，不混称为新流程的结果。

## 本机首轮结果

`grasp_spider_acquisition_pilot_v03` 已完成 24/24 个优化任务，18 个原 CA 失败提议均实际进入 MPPI。全部指定接触建立 18 例；固定姿态完整验收通过 16 例；完整动态保持通过 4 例：Power type 10 的 1× 盘、Power type 15 的 1× 圆柱，以及 Intermediate type 23 的 1×/1.5× 圆柱。缓存 CA 初值的两 worker 并行条件下，逐例墙钟时间平均约 40.0 s（包含接触建立、释放保持和回放），不代表首次运行 CA 或冷编译耗时。

Power 盘的 GPU/CPU 最大位移约 0.853 mm、旋转约 1.237°，六组全程载荷 ≥0.191 N；Power 圆柱约 0.228 mm、1.182°，六组全程载荷 ≥0.330 N。两者均通过 200/200 步独立力闭合检查。另行复核了释放初始姿态的几何、穿透和自碰撞。

球体 1.5× 在默认配置下位移约 0.280 mm、旋转约 0.360°、200/200 步力闭合通过，但掌面于 0.262/0.322/0.352/0.382 s 的原生法向载荷为零。因此整体失败，三形状推广条件尚未满足，155 例全量未启动。针对性对照保持采样预算不变：提高载荷余量惩罚可消除六组卸载，却使位移达到约 2.253 mm；提高位置权重可将位移压至约 0.905 mm，但仍有两步掌面卸载。均未放宽阈值或标记为成功。

逐例统计见 `validation_summary.json`，推广判断见 `promotion_gate.json`，接触丢失诊断见 `contact_loss_analysis.json`，四组受力余量对照见 `force_margin_comparison.json`。`validation/checks.json` 记录 197 个相关回归测试通过、2 个跳过，及 ForceAware 隔离环境的 6 个兼容性测试通过；Viser WebSocket 交互测试覆盖阶段切换、尺寸 bar、两阶段播放、接触显示和历史来源。另验证了物理上通过的 0.1 s 轨迹前缀不会被误认证为完整 0.4 s 保持。

# WUJI2 ContactAware → SPIDER MPPI

新入口读取 `output/grasp_contactaware_feature_normals_final` 中已通过验收的三个 CA 静态参考。CA 的语义 patch、IK 关节配置、五个 active slots、手上局部材料点和物面锚点均直接读取；运行前后核验输入文件哈希，不重新求解 CA，不使用模板关节角作为 tracking 目标。

```bash
.venv-contact/bin/python run.py --config configs/grasp_contactaware_spider_contact_keep.yaml
```

依赖为 `requirements-grasp.txt` 中的普通 CUDA Torch / MuJoCo 3.10 / mujoco-warp 3.10 / Warp 1.16 环境，无需启用 `.venv-forceaware` 或加载 DexForge FA runtime。结果包含 `trajectory.npz`、`iterations.json`、`ca_input_hashes.json`、`reference_manifest.json`、`control_initialization.json`、`replay/replay.json`、`summary.json` 和来源记录。

## 迁移范围

来源为 [SPIDER](https://github.com/facebookresearch/spider/tree/44717007de41cbef7565dff7ff9f4453557a2d3d)，CC BY-NC 4.0。`third_party/SPIDER/dex/UPSTREAM.json` 固定 17 个上游文件的 SHA-256，启动时逐一核验。公式及采样测试直接执行固定版本保留的原函数进行数值对照。

| 部分 | 实现 |
| --- | --- |
| 灵巧手参考 | CA 的 26 维配置：6 个标量手腕坐标和 20 个手指关节 |
| 物体轨迹 | 同一 CA 物体初始姿态在整个预测和执行区间保持不动；位置及四元数旋转均跟踪 |
| 接触点 | CA active mask 选择的手上 body/local point，跟踪参考物体姿态变换后的物面锚点 |
| 动力学 | 完整普通 WUJI2 mesh，原生 `mujoco_warp.step`，implicitfast，20 次 Newton / 50 次 line search |
| 优化 | SPIDER generic dense weighted-mean MPPI，绝对 actuator target 序列；未启用梯度细化 |
| 执行 | 单独的单世界 native 引擎，每 2 ms 记录状态及实际法向接触力 |

每个时刻的正 cost 为：

```text
|| concat(w_hand * (q_hand - q_CA),
          w_pos * (p_obj - p_ref),
          w_rot * quat_sub(q_obj, q_ref)) ||_2
+ w_vel * || v - v_ref ||_2
+ sum_i mask_i * || contact_hand_world_i - contact_ref_world_i ||_2
```

配置误差使用一个合并的非平方 L2 范数；手部与物体诊断范数不能分别相加代替它。默认权重与原版一致：手腕位置 1、手腕旋转 0.3、手指关节 0.003、物体位置 1、物体旋转 0.3、速度 0.0001、末端 1。原版 `contact_rew_scale` 仅决定是否启用接触项，并非数值乘子；本配置设为 1 启用。最后一步的整个 cost 乘 terminal scale，然后对预测步数取平均。没有添加 FA 的平方毫米罚项、force-aware loss 或物体虚拟控制力。穿透由原生接触动力学处理，没有新增独立的穿透 cost。

原版一步 `mjw.step` 后的 qpos/qvel 已积分，而 site/body 变换来自该步积分前。规划评分保持这个原生观察时序。回放中的几何误差则在独立 `MjData` 上计算真实记录状态的 FK，避免影响接触力求解。

MPPI 在 knots 上采样相关噪声，通过上游的全局线性插值网格叠加到密集 nominal controls；选择前 10%，以样本标准差加 0.01 标准化，再经 softmax 加权形成新的密集均值。第一候选为零噪声，末尾 1% 为小噪声 exploitation，按原式退火及 warm shift。保留四个明确的健壮性改动：排除无效 rollout、全无效时报错、单个有限候选标准差取零、避免原版 `[-0:]` 误缩放全部小种群。另对最终加权均值增加一次完整 rollout 检查；仅当它无效时回退到已评估的有限候选。

为了生成可承受重力的初始 actuator targets，默认复用已有静态有界接触力 QP 和 PD 逆映射。这个结果只用作控制序列初值和末尾 control reference，**hand configuration tracking 始终是原 CA IK 配置**。没有调用 FA 的滚动优化、目标函数、adjoint 或 ComFree 引擎。没有更改 CA 初始 qpos/qvel、手部 PD、物体质量、摩擦或 mesh。

## 预算与复现

`grasp_contactaware_spider_mppi.yaml` 是本地短抓取测试配置：128 samples / 128 worlds / 最多 4 轮，预测 0.3 s，规划步长 10 ms，执行步长 2 ms，每 30 ms 重新规划；每尺寸执行 0.4 s。噪声为位置 0.003 m、旋转 0.03 rad、关节 0.08 rad，temperature=1。该小预算不是 SPIDER 官方默认配置。

`grasp_contactaware_spider_fine.yaml` 将规划步长改为 2 ms、预测改为 0.2 s、knot 间隔 0.04 s、最多 8 轮，其他权重和噪声不变。`grasp_contactaware_spider_quality_probe.yaml` 在 fine 配置上增加到 512 samples，仅测试中尺寸，用于检查采样预算的影响。

推荐的 `grasp_contactaware_spider_tracking.yaml` 保留 fine 的动力学、128 samples / 8 轮和全部 CA 参考，只调整 task weights：手腕位置/旋转改为 0.1/0.03（与上游 oakink task override 的手腕尺度一致），物体位置权重改为 10（本地静态保持任务配置）。其余公式与权重不变。该配置允许手腕适度补偿负载，同时更强地限制物体平移；不能把效果归于求解器替换一个因素。

`grasp_contactaware_spider_upstream_budget.yaml` 提供原版 `examples/config/default.yaml` 的预算/时域/噪声参数：1024 samples、32 轮、temperature=0.1、预测 1.6 s、knot/控制周期 0.4 s、规划步长 0.01 s、位置/旋转/关节噪声 0.03/0.03/0.15。为控制显存，每批 128 worlds；输入及 CA 接触项仍是本地 WUJI2 适配。该配置不代表已完成官方数据集基准。

运行包含模型准备、首次 kernel 编译/加载和 CUDA Graph 建图；逐窗 `optimization_seconds` 在 GPU 同步后计时，`case_wall_seconds` 另外包含 setup、实际执行和独立回放。本次使用 RTX 4090，Torch 2.5.1+cu121、CUDA driver 12.2，kernel 缓存已建立；冷启动时间不能与纯优化时间混用。旧驱动不支持 conditional CUDA graph nodes，使用普通固定 solver loops。GPU 并行接触求解和 MPPI 后续控制会产生数值差异，固定 seed 不保证逐位重复；下表是保存结果对应的单次短测，不是统计成功率。

## 验收

严格验收保持原要求：物体最大位移不超过 2 mm、最大旋转不超过 5°，且五根手指在每个执行步均有大于 1e-6 N 的实际法向载荷；GPU 执行及独立 CPU 回放均需通过。几何接近、CA mask、静态 QP 力和 cost 收敛均不作为真实接触的替代。

结果与 FA 的比较是完整后端替换：SPIDER 使用普通 mesh 原生 MJWarp，旧 FA 使用 native ComFree/GS-SDF。即使输入相同，也不能把质量差异仅归因于优化器。0.4 s 静态抓取短测不证明长时间保持或硬件成功。

## 三尺寸结果

推荐 task weights，0.4 s / 200 个执行步，五指顺序为拇指、食指、中指、无名指、小指。结果目录为 `output/grasp_contactaware_spider_tracking`。

| 圆柱半径 mm | 纯优化 s | 含 setup/执行/审计 s | GPU 最大漂移 mm | CPU 最大漂移 mm | GPU / CPU 最大旋转 ° | GPU 五指零载荷步数 | CPU 五指零载荷步数 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 26.25 | 12.82 | 16.33 | 0.893 | 1.104 | 1.440 / 1.436 | 8 / 0 / 94 / 134 / 135 | 7 / 0 / 97 / 146 / 145 |
| 35.00 | 12.21 | 15.06 | 0.419 | 0.419 | 1.246 / 1.245 | 0 / 2 / 4 / 53 / 83 | 0 / 1 / 4 / 53 / 82 |
| 43.75 | 12.38 | 15.18 | 0.918 | 0.963 | 2.040 / 2.001 | 2 / 4 / 13 / 77 / 75 | 2 / 4 / 12 / 75 / 78 |

物体位移及旋转保持：GPU **3/3**，独立 CPU **3/3**。每步五指均有载荷：**0/3**。全部状态有限，CPU MuJoCo 警告为零。每例保存 34 个输入文件/实际 mesh 资产的运行前哈希，结束后逐一验证不变。这里的“物体保持”不能替代“稳定五指抓取”。

控制补偿也带来手姿偏差：小/中/大尺寸的手腕平移坐标相对 CA 的最大 L2 差为 3.315/3.196/4.427 mm，20 个手指关节中单关节最大绝对偏差为 6.815/5.904/5.528°。这些是跟踪执行误差，CA reference 本身没有改动；完整 hand deviation 统计在汇总 JSON 中。

| 配置 | 平均纯优化 s/尺寸 | CPU 最大漂移 mm（小 / 中 / 大） | 严格五指通过 |
| --- | ---: | --- | --- |
| 固定静态 QP 控制，无优化 | 不适用 | 5.860 / 3.778 / 4.672 | 0/3（仅 CPU） |
| SPIDER 128×4，10 ms 规划，原始权重 | 3.42 | 10.135 / 6.199 / 4.924 | 0/3 |
| SPIDER 128×8，2 ms 规划，原始权重 | 10.97 | 4.129 / 3.701 / 5.611 | 0/3 |
| SPIDER 512×8，2 ms，原始权重，仅中尺寸 | 61.17（单例） | — / 2.902 / — | 0/1 |
| SPIDER 128×8，2 ms，task weights | **12.47** | **1.104 / 0.419 / 0.963** | **0/3** |
| 旧 FA40 快速配置，64 starts | 37.27 | 2.529 / 1.802 / 3.671 | 0/3 |
| 旧 FA200 质量配置，64 starts | 单独记录 | 1.371 / 0.545 / 2.043 | 1/3（中尺寸） |

相较旧 FA40 快速配置，推荐 SPIDER task weights 的本机纯优化约快 3 倍，物体 tracking 更好；旧 FA200 的中尺寸曾通过五指验收，新配置未通过，所以不能说完整抓取稳定性更好。增加候选预算虽能降低原始权重的物体漂移，也没有解决五指持续受力。

原始 tracking 目标的限制是：接触点距离只奖励几何位置匹配，不规定每根手指都持续产生正法向载荷，也不直接限制最小接触力、载荷分配或滑移。上表控制可以用部分手指维持物体，同时让其他手指失载。这些对照说明原始公式尚不能替代“每步五指受力”的约束；没有把几何误差较小报告为五指稳定。下面的本地扩展直接针对这项不足。

原始结果：`output/grasp_contactaware_spider_{mppi,fine,quality_probe,tracking}/summary.json`；固定控制对照：`output/grasp_spider_control_baseline/summary.json`。汇总及轨迹/载荷图由 `scripts/analyze_spider_grasp.py` 生成到 `output/grasp_spider_comparison`。受影响的新后端、真实 GPU rollout、原公式与采样一致性、CA 输入保持、配置路由及统一 solver 测试合计 150 项通过、2 项跳过。

## CA 手指脱离接触惩罚

`configs/grasp_contactaware_spider_contact_keep.yaml` 在上面的 tracking 配置上启用一个本地 MPPI cost 扩展。CA 的 IK、物体目标和局部接触点全部不变，物体/手配置/接触点 tracking、采样预算与动力学参数也不变。新增配置为：

```yaml
environment:
  spider:
    cost:
      contact_keep_rew_scale: 0.05
      contact_min_force_n: 0.05
```

每个预测步计算：

```text
c_keep(t) = weight * sum_f m_CA[f] * relu(1 - max(F_normal[t,f], 0) / F_min)^2
```

`m_CA` 从 accepted CA active contact bodies 的手指归属得到。同一手指有多个 CA 接触点时只惩罚一次；CA 没有指定的手指不受此项约束。手掌不是手指，固定传感指垫则归入其所在手指。现有三例 CA 都指定五根手指。

`F_normal` 是每一步原生接触求解后的物体–手指法向载荷之和，单位 N，包含该手指的固定指垫。不是 CA 静态 QP 的拟合力，也不是由点间距离推算的力。完全失载时，该手指贡献完整的 `weight`；低于 0.05 N 时按归一化缺失量平方处罚，达到阈值后为零，不继续奖励更大夹持力。它同时鼓励未建立接触的 CA 手指重新接触，并惩罚之后失载；不是只在接触状态切换瞬间加一次 cost。

载荷在每个 MPPI 预测步由 GPU 上的 `mujoco_warp.contact_force` 解码，按真实 world ID 和 object/finger geom 归属汇总，随后复制进逐步 force trace。采样前清空复用缓冲区，忽略无效和未激活的 contact/efc rows，避免上一候选或上一时刻的载荷残留。没有主机回读、额外 `forward` 或动力学步，兼容整段 CUDA Graph。force 与原生接触点观察一样属于该步积分前的 solver stage；实际执行和独立 CPU 审计保持原规则。

该项和原始 tracking 相加后，使用相同的 terminal scaling 与 H 步平均。`contact_keep_rew_scale` 是真正的乘法权重，与原版仅作开关的 `contact_rew_scale` 不同。默认值为零，关闭时不收集预测接触力、原始 SPIDER 数值目标保持一致。新增项属于本地扩展，不冒称 SPIDER 原公式，也没有接入 DexForge FA 优化器。它是软惩罚，是否持续接触仍以实际逐步力验收。

实现：`retarget/task/grasp/spider_cost.py`、`retarget/env/grasp/spider_contact.py`、`retarget/env/grasp/spider_rollout.py`。输出新增 `contact_retention.json` 和 summary 中的 `contact_retention`，记录 CA 手指 mask、阈值、低于阈值的步数、五指同时有载荷比例及新增目标的实测均值。

```bash
.venv-contact/bin/python run.py --config configs/grasp_contactaware_spider_contact_keep.yaml
```

完整三尺寸对照每例仍执行 0.4 s / 200 步。下面的比例为 GPU 实际执行步中，CA 指定的五指**同时**有大于 1e-6 N 法向载荷的比例；它不是每例全部通过，也不代表持续保持成功率。

| 半径 mm | 五指同时承载比例：原 tracking → 新 cost | CPU 五指同时承载比例：新 cost | CPU 最大漂移 mm：原 → 新 | CPU 最大旋转 °：新 cost | 纯优化 s：原 → 新 |
| --- | --- | ---: | --- | ---: | --- |
| 26.25 | 13% → **69%** | 69.5% | 1.104 → 1.769 | 4.828 | 12.82 → 16.36 |
| 35.00 | 52.5% → **81.5%** | 82.5% | 0.419 → 1.323 | 2.556 | 12.21 → 17.47 |
| 43.75 | 35% → **86%** | 86% | 0.963 → 0.908 | 2.267 | 12.38 → 14.09 |

GPU 五指零载荷步数（拇/食/中/无名/小）由 `8/0/94/134/135`、`0/2/4/53/83`、`2/4/13/77/75` 降至 `0/0/28/19/53`、`0/6/2/13/26`、`2/2/0/13/23`。并非每根手指都改善：例如中尺寸食指出现 6 个失载步；总体五指同时受力区间明显增加，仍有接触切换。新 cost 后三例 GPU/CPU 的物体保持均通过，完整逐步五指验收仍为 **0/3**。

平均纯优化从 12.47 s 增至 15.97 s。最大预算均为 128 samples × 8 轮；由于提前停止行为随 cost 改变，三例实际迭代总数从 69/62/63 增至 89/94/75，因此这项耗时增加不能全部归因于接触力解码开销。物体位移/旋转约束仍通过，但小、中尺寸的 tracking 误差有所增加。

还单独测试了中尺寸 `contact_keep_rew_scale=0.2`：五指同时承载比例为 GPU 88.5% / CPU 88%，但 CPU 最大漂移 3.691 mm、旋转 7.219°，物体保持失败。因此保留 0.05 为推荐权重；不能通过无限加大保持权重解决全部目标的冲突。该探针配置为 `grasp_contactaware_spider_contact_keep_strong_probe.yaml`，结果独立保存，没有替换三尺寸主对照。

原始结果：`output/grasp_contactaware_spider_contact_keep/summary.json`。CPU 分析脚本 `scripts/analyze_spider_contact_keep.py` 生成 `output/grasp_spider_contact_keep_comparison/{summary.json,contact_keep_comparison.png,contact_keep_comparison.svg,artifacts.json}`。前后每对结果的 CA 参考数组及 34 个源输入/mesh 哈希完全一致，tracking 设置和请求的采样预算一致。比较中的 baseline 物理载荷仅离线计算新增 cost 的反事实分数，没有重新优化或改写 baseline。GPU 接触力采集与 CPU `mj_contactForce` 的单步对照最大差为 1.34e-7 N；真实 GPU rollout 测试覆盖分离物体、CA 手指子集去重、状态恢复、默认/独立 stream 和关闭扩展的原版兼容性。

扩展后的受影响测试合计 **188 项通过、2 项跳过**，日志保存在 `output/grasp_spider_contact_keep_comparison/tests.log`。

多抓型输入可只指定部分手指或包含掌面接触。SPIDER 的静态控制预载允许使用该 CA 实际接触集合，原 FA 调用仍默认要求五指。SPIDER 执行验收显式使用 `required_contact_finger_mask` 检查 CA 指定手指，每一步都要求正载荷；同时继续独立报告完整五指指标，不把二指 pinch 误称五指保持。旧三尺寸输入的 mask 都是五指，所以它们的验收标准没有改变。

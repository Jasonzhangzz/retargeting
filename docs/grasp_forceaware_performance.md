# WUJI2 ForceAware 的快速配置与性能对照

当前速度入口是 [grasp_contactaware_forceaware_fast.yaml](../configs/grasp_contactaware_forceaware_fast.yaml)。恢复固定上游的 40 次 Adam、64 starts、5 knots / 30 个预测物理步；保留原损失、优化器和 CUDA Graph。此前的 [fine_samples.yaml](../configs/grasp_contactaware_forceaware_fine_samples.yaml) 是显式 200 次迭代的质量验证配置，不代表原版默认速度。

```bash
.venv-forceaware/bin/python run.py --config configs/grasp_contactaware_forceaware_fast.yaml
```

## 同机有接触窗口的实测

RTX 4090 独占、40 次迭代、64 starts、30 步预测，计时包含原 optimizer 的 candidate evaluation / selection，模型和首次 graph 准备另记：

| 场景 | 单窗稳态求解 |
| --- | --- |
| 固定上游的 LEAP/querypoint + SugarBox GS 资产 | 2.854 s |
| WUJI2 + primitive mesh，三尺寸平均 | 2.662 s |
| WUJI / 原资产基线 | 0.933× |

原资产基线保持原模型、质量、伺服及上游 native 积分，64/64 starts 有效，第一候选每步有 7–11 个实际接触，不是无接触空场景。当前缺少原版 MANO 模型和已求解 CA 产物，因此使用明确标记的合成静态接触参考。**这是原版资产的工作量代理，不是完整快速示例端到端或抓取质量复现。** [基线报告](../output/grasp_forceaware_speed_audit/original_benchmark.json) 和 [提取清单](../output/grasp_forceaware_speed_audit/original_fixture/extraction_manifest.json) 保留这一范围。

## 完整三尺寸运行

同一种 120 mm 高圆柱，半径分别为 26.25、35、43.75 mm，每例执行 0.4 s、14 个 MPC 窗口：

| 半径 mm | 纯优化 s | 端到端 s | 严格五指保持 |
| --- | --- | --- | --- |
| 26.25 | 37.220 | 47.901 | 未通过 |
| 35 | 37.145 | 47.556 | 未通过 |
| 43.75 | 37.434 | 47.651 | 未通过 |
| 平均 | **37.266** | **47.702** | 0/3 |

端到端计时包括场景/参考/预载准备、native finite baseline、原求解和视频、独立 mesh 回放、native/mesh 逐指受力审计。三例全部完整有限求解；整批墙钟 143.258 s。计时从原 `rollout.npz/mpc_step_sec` 按窗口求和，避免把 recorder 的重复逐状态计时当成额外求解时间。[完整结果](../output/grasp_contactaware_forceaware_fast/summary.json) 与 [性能比较](../output/grasp_forceaware_speed_audit/comparison.json) 保存机器可读数值。

该配置满足本机所测“与原版资产基线相近耗时数量级”，但没有同时解决五指稳定性。200 次验证配置的中尺寸曾通过；恢复 40 次后其无名指又出现卸载，不能继承 200 次的 success。额外按原版首窗口允许初态修正的策略检查中尺寸，耗时 37.356 s / 48.295 s，仍有短暂卸载；见 [策略对照](../output/grasp_contactaware_forceaware_quickstart_policy/summary.json)。没有放宽验收，也没有缓存上一轮 200 次的答案作为初值。

## 改动和保留项

- 只恢复原 40 次预算，不减少 64 starts、horizon、损失项、物理皮肤采样或实际执行时长。
- worker 的 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 与上游 `run_example.py` 对齐为 1。
- 快速配置跳过已做过的反复 finite-difference 诊断，但保留 native baseline、原独立 replay、全部逐指载荷审计。不会把跳过诊断标作梯度通过。
- WUJI 官方手指 PD、原质量/惯量、完整手皮肤、当前 mesh 物体和本地 clamp-aware PD 积分适配不变；仍采用 10 ms planning / 2 ms executor。这些 WUJI 输入/积分适配不等于与原 LEAP 物理参数相同。
- 原 pinned vendor 的 139 个文件哈希仍通过。[修复边界](grasp_forceaware_repair.md) 解释本地积分扩展和 CA 输入。

## 为什么没有更换几何或 PD

之前在同一 WUJI 模型的单窗对照中，40 次为 2.665 s、200 次为 13.017 s，约 **4.884×**；预算差异已解释主要退化。CUDA Graph 没有丢失。

固定点 graph profile 每迭代约为 forward 17.35 ms、loss 3.30 ms、backward 41.81 ms，组合约 61.49 ms。额外 PD 积分矩阵 forward/backward 共约 0.405 ms，占组合约 0.66%；没有证据为了速度改变 PD 物理。分项 graph 和单次 kernel event 是不同采样，不能直接相加；[详细 profile](../output/grasp_forceaware_speed_audit/phase_profile_wuji_mesh.json) 保留原始数值。

原 LEAP 手部 12,024 点、WUJI 16,384 点，也不是数量级差异。原资产基线已与当前同预算相当，因此没有启用 GS 物体拟合、物理点裁剪或 querypoint 几何替换。相关研究仅保存在 output-local 调查脚本中。


## 按论文重新核对计时和成功定义

已核对 [论文 v1 附录 H、图 8/9](https://arxiv.org/html/2610.06331v1)：单卡 RTX 4090 D、64 候选、最多 40 次迭代、0.06 s 控制 knot、0.3 s 预测。图 8 只测前向碰撞查询；图 9 使用整段优化秒数，横轴覆盖 0–600 s。不能把前者的毫秒与整段 FA 求解直接比较，也不能直接比较不同长度的任务。

论文 Sec.4.2.1 采用平均位置误差 <10 mm、平均旋转误差 <0.1 rad。对本地 0.4 s 静止任务另按这一公式计算：

| 半径 mm | native 平均位置误差 mm | native 平均旋转误差 rad | 论文 tracking 门槛 | 原严格保持门槛 |
| --- | --- | --- | --- | --- |
| 26.25 | 1.532 | 0.01331 | 满足 | 不满足 |
| 35 | 1.152 | 0.00785 | 满足 | 不满足 |
| 43.75 | 1.445 | 0.01234 | 满足 | 不满足 |

两种口径并列保存，`task_success` 没有改。前文的 0/3 指最大误差和五指逐步承力的严格条件，不是论文 tracking 门槛失败。这里也不等同论文跨手型、长序列成功率复现。见 [论文口径的独立分析](../output/grasp_forceaware_speed_audit/paper_comparison.json)；原三尺寸控制轨迹未改变。

### 质量折中检查

保留原算法、使用 80 次迭代并提高可行的静态预载下限，中尺寸纯优化 73.62 s、总耗时 84.30 s；native 五指 200 步全承力，但独立 mesh 的无名指仍有 5 步卸载，因此严格验收仍失败。40 次加强无名指预载的检查也未完整通过。这些对照均与 40 次速度基线分开，不推荐为已解决稳定性的快速模式。

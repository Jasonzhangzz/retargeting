# 原生 Shadow GraspADMM 复测

此复测独立于 MANO＋LM 适配实验。固定使用 Dexonomy 提交 `d561ff9df8545deef7a3f9e7cfb4045ef66f6913` 与其 MuJoCo Menagerie 子模块 `bf04290ac7911fa7a39339c7e507792fd464c438`。直接导入官方 Python 包，调用原生模板转换、初始化、GraspADMM 和六方向评测函数，不修改上游源码或替换手步、接触查询、QP、filter。

服务器实验目录：

```
/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget_native_graspadmm_20261104
```

入口为 [native_graspadmm_retest.py](../scripts/native_graspadmm_retest.py)。独立环境使用 [requirements-graspadmm.txt](../requirements-graspadmm.txt)，补齐上游声明的 Matplotlib 3.10.3 和 USD 25.2。安装脚本通过官方索引/PyPI 的 SHA256 验证下载文件；环境不继承系统 site-packages。

## 固定协议

- 从原诊断样本读取16个物体与尺度，保留场景及其网格。只用 Shadow type7 官方初始模板。
- 原生初始化：种子20261104、5 epoch、1024采样点、8旋转、100 matcher步、每场景每epoch最多10输出，总上限800。保留官方候选填充行为，不按最终失败补采样。
- 原生精修：Clarabel＋MuJoCo，50轮×10物理子步。保留接触重选、对偶更新、fallback、hybrid模板更新、filter、squeeze和pregrasp。不执行模板自举循环。
- 全部实际初始化输出进入分母。单独记录无初始化输出场景、缺失最终状态、异常、原生拒绝、fallback和六方向评测失败。
- 质量运行用只读函数返回事件采集最后状态；运行前尚未更新的输入姿态不标记成精修输出。主测速直接调用原函数，不启用观察器。后处理独立于优化。
- 单条：首条输入预热1次、实测5次。吞吐：场景轮询最多128条，1与32 workers分别预热后实测3轮，每个worker线程数1。每轮独立目录。另做非测速状态一致性核对。
- GPU1只在初始化阶段短暂暂停身份核验后的本任务worker，并用独立watchdog和finally恢复。CPU精修与评测不暂停GPU生产worker。初始化前后保存负载记录。

## 阶段与证据

`prepare → template → initialize → quality → timing → consistency → diagnostics → profile → report`

CPU进程使用`spawn`。本机固定Torch2.5.1在`fork`子进程的autograd backward段错误退出（复现退出码-11）；`spawn`复现正常，最终姿态哈希与独立单进程一致。仅调整进程启动层，原生单条operation与所有算法源码不变。首轮fork运行整体作为基础设施异常单独归档，沿用同一批800条初值重新执行，无失败重采样。

`MANIFEST.json` 固定官方源码/资产、场景、脚本、依赖规格和预算；`OFFICIAL_FILES.json` 来自本地固定Git检出，用于校验传输。各阶段检查哈希；`stage_receipts.json` 记录完成阶段，按同一指纹接续。`STOP` 文件阻止继续派发后续作业，不会自动启动新预算。

主要产物：

| 文件 | 含义 |
|---|---|
| `initialization.json` | 16场景输出数及全部初始姿态指纹 |
| `quality_operation.json` | 所有原生尝试、拒绝阶段与fallback |
| `quality/cases/*/final_state.npz` | 可取得的最后状态及最终姿态，非优化历史 |
| `native_evaluation.json` | 原生六方向物理GSR，全部初始化输出为主分母 |
| `diagnostics.json` | 目标/手点投影闭合、槽位覆盖、原生接触组、CLN、CDC和穿透 |
| `latency.json` / `throughput.json` | 原函数预热后完整operation计时 |
| `consistency.json` | 含失败状态的1/32 worker非测速一致性检查 |
| `phase_diagnostic.json` | 独立带观察器的分段诊断，不用于提速结论 |
| `per_input_results.json` / `.csv` | 逐输入统一结果 |
| `representatives.json` | 成功及各失败类别的回放索引 |
| `report.md` / `summary.json` | 最终报告与机器摘要 |

投影闭合后处理调用已有生产 `ProjectedQPClosure`，摩擦0.6、13工况、96步QP，保持原阈值。槽位覆盖为距离±2mm、法向30°、至少80%；原生Shadow槽位数逐条记录，不能与MANO四槽直接混算。原生GSR仍由上游六方向MuJoCo评测决定。

[论文v2的时间分析](https://arxiv.org/html/2603.13832v2#S4.SS6)报告约19条/秒，此处只与吞吐对照，不将其倒数当成单条延迟。四卡生产CEM配置保持不变。结果只适用于这16个物体—尺度组合及type7。

## 本地 Viser 回放

只读导出入口 `scripts/export_native_graspadmm_viewer.py` 在原生环境重建初始/最终状态，复用官方网格、MuJoCo FK 和正式诊断数据；不运行优化。共享网格与每条刚体变换保存为 `geometry.npz` / `poses.npz`，800条均保留，不筛掉失败。

```bash
.venv-graspadmm/bin/python scripts/serve_native_graspadmm.py \
  --data output/graspadmm_native_20261104/viewer --port 8103
```

打开 http://127.0.0.1:8103/ 。支持16场景与成功/失败分类筛选、初始化/最终/并排对照、接触槽/物面投影/ADMM目标与间隙、法向和逐槽明细。所有指标明确对应最终状态，初始槽位灰色且不重复判分。服务默认只绑定本机。

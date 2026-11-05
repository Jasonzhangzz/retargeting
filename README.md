# Retarget

求解方法仅有 **当前 v5 patch CEM** 和 **GraspADMM**。模型、数据、几何、动力学及评测工具与求解入口分开。

| 方法 | 入口 | 唯一默认参数 |
| --- | --- | --- |
| CEM | `retarget/solver/cem.py:solve_patch_cem` | 同文件 `DEFAULTS`：128 候选、24 轮、精英比例0.05、温度0.1、最终噪声比例0.2 |
| GraspADMM | `retarget/solver/graspadmm.py:solve` | 同文件 `DEFAULTS`：50轮、10子步、rho=1000、物面步长0.0002 |

所有正式入口读取上述默认参数。YAML 的 `solver` 仅接受 `type: cem` 或 `type: graspadmm`，不接受另一份算法参数。命令行保留设备、文件路径和批量资源选项；MANO 默认 pose batch=1024、候选评分分块=2048。

## MANO patch CEM

Shadow FK、掌部对齐与 MANO 骨架 IK 初始化后，使用 `mano_patch_consistent_spring_history_h5_v5`：实际手面 patch 的贴合、穿透和五步末速度目标。贴合与验收采用相同 witness；可行候选独立保留，满足联合条件后逐 pose 提前停止。

```bash
python scripts/retarget_mano_comfree_campaign.py \
  --source /path/to/packed_shadow_results \
  --dataset /path/to/dexonomy \
  --output results/mano_patch_cem \
  --model /path/to/MANO_RIGHT.pkl \
  --mapping /path/to/fixed_contact_mapping.json \
  --shadow-xml /path/to/shadow_assets/right.xml \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

多抓型按 batch 轮询。单分片入口为 `scripts/solve_mano_comfree.py`。数值运算使用 GPU；资产构建、Python 调度与文件编码单列。STOP 后不自动恢复；续跑须验证配置、输入与代码指纹。只保存最终 pose、关键点、接触、物体信息和指标。

[目标与验收](docs/mano_solver.md)、[接触模型](docs/mano_patch_history.md)、[Shadow 接触映射](docs/shadow_mano_contact_mapping.md)。私有模型权重、审核后的映射和完整数据集需另行提供。

## GraspADMM

`solver/graspadmm.py` 提供 `NativeProblem` 与 `GPUProblem` 两种运行输入，统一调用 `solve`。原版 CPU 操作和 GPU 数值适配使用同一套算法参数。保留物面搜索、接触 QP、虚拟力手步、对偶更新、fallback、filter 和 pregrasp；不增加姿态软参考或历史结果回退。

```bash
bash scripts/setup_graspadmm.sh
.venv-graspadmm/bin/python run.py --config configs/graspadmm.yaml
```

上游固定提交 `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`；原源码哈希与许可证保留。接触覆盖、碰撞和动态 GSR 分别报告。[方法说明](docs/graspadmm.md)、[GPU 适配](docs/native_mano_gpu.md)。

## 检查和回放

```bash
python -m pytest -q
python scripts/solve_mano_comfree.py --help
python scripts/build_mano_patch_history_review.py --help
python scripts/serve_native_graspadmm.py --help
```

CEM 使用当前三维对照页面；GraspADMM 使用独立 Viser。输入和模型缺失的集成检查明确跳过。加载、首次图准备、优化、验收和读写时间分别记录。

人体与 Dex 任务也统一调用 CEM；硬件采集、模型拟合、初始化和独立评测不提供额外抓取 solver。实验产物位于 `output/`、`results/`，不纳入 Git。[维护说明](docs/code_maintenance.md)、[来源及许可](THIRD_PARTY.md)。

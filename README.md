# Retarget

抓取求解仅保留 **CEM＋batched QP** 和 **GraspADMM**。Dex、humanoid 与硬件采集工具独立保留；它们使用的 MPPI、SCSP、ForceAware 不作为抓取求解入口。

## MANO：Shadow pose → CEM＋batched QP

沿用 Shadow FK、掌部 SVD 对齐与 15 步 MANO 骨架 IK 初始化，然后使用仓库 CEM（128 候选 × 24 轮）优化 pose。每次候选评分重新查询接触与碰撞，并执行原有四射线、13 工况、96 步 batched QP。几何目标和权重未因代码整理而改变。

```bash
python scripts/retarget_dexonomy_mano.py \
  --source /path/to/packed_shadow_results \
  --output results/mano_cem_qp \
  --model /path/to/MANO_RIGHT.pkl \
  --mapping /path/to/reviewed_shadow_mano_mapping.json \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --workers-per-device 1 --batch-size 2048 \
  --candidate-chunk 2048 --population 128 --iterations 24
```

每卡 pose batch 为 2048；候选分块 2048 是评分缓冲大小。输入打包用 `scripts/pack_dexonomy_batches.py`；MANO 完成后再求 WUJI2 的顺序控制器为 `scripts/run_mano_then_wuji.py`。同目录结果通过配置指纹及校验后续跑，禁止混用不同配置的缓存。

仅保存最终 pose、关键点、接触、物体信息和最终指标；失败结果保留标记。[数值方法](docs/mano_solver.md)、[MANO 查看器](docs/mano_results_viewer.md)、[接触对应](docs/shadow_mano_contact_mapping.md)。MANO 私有权重和完整数据集须单独提供，不随代码上传。

## GraspADMM

`third_party/Dexonomy` 固定上游提交 `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`，保留上游哈希及许可证。原版初始化、QP、MuJoCo 虚拟力推进、filter、squeeze/pregrasp 与评测由独立环境运行。

```bash
bash scripts/setup_graspadmm.sh
.venv-graspadmm/bin/python run.py --config configs/graspadmm.yaml
```

`retarget/env/grasp/native_admm.py`、`native_batched_qp.py` 和 `native_mjwarp.py` 保留原生刚体 MANO 的 GPU 适配。该适配与 CPU 原版分开验证；不能将数值验证通过当作已达到原版 GSR。当前默认生产方法仍是 CEM＋batched QP。[原版复测](docs/native_graspadmm_retest.md)、[GPU 适配边界](docs/native_mano_gpu.md)。

## 环境和检查

使用已有 CUDA 环境即可运行 CEM；依赖见 `requirements.txt`。`requirements-graspadmm.txt` 是原版复测的独立数值环境，不要求替换生产环境。GPU 版 GraspADMM 还需要与模型兼容的 MuJoCo/MJWarp 版本及接触适配。

```bash
python -m pytest -q
python scripts/serve_mano_results.py --help
python visualize.py --config configs/web_graspadmm.yaml
```

可选的 MANO 权重、已审核映射、完整 Dexonomy 数据或硬件 SDK 不存在时，相应集成测试明确跳过；CPU/GPU 单元测试另行计数。首次运行的模型加载与 Graph 编译不属于稳态求解时间。

## 代码布局

| 位置 | 内容 |
| --- | --- |
| `retarget/solver/cem.py` | 原 CEM 与独立 batch 维度适配 |
| `retarget/env/grasp/pose_objective.py` | 共享几何评分，无独立 MPPI 求解器 |
| `retarget/env/grasp/projected_qp.py`、`qp_projection.py` | 历史四射线 batched QP 与 GPU 约束投影 |
| `retarget/env/grasp/mano_solver.py`、`mano_adapter.py` | MANO 初始化、解码和 CEM 接口 |
| `retarget/env/grasp/graspadmm_*`、`native_*` | 原版 GraspADMM 桥接及独立 GPU 适配 |
| `retarget/env/grasp/batched_physics.py`、`collision.py` | 共享物理与碰撞工具 |
| `retarget/env/{dex,humanoid}` | 独立操控与人体任务 |
| `moxian_web` | 硬件数据采集和 MANO 拟合工具 |

实验输出放在 `output/`、`results/`，均不纳入 Git。旧实现、旧结果和清理前未提交代码已在仓库外归档；详见 [清理说明](docs/code_maintenance.md)。来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

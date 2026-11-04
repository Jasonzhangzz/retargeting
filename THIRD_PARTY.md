本仓库保留当前本机实现。DexForge 分支仅加载本仓库固定版本的源码，不查找外部 Panda 或用户机器上的其他 checkout。

| 位置 | 来源与用途 |
| --- | --- |
| `third_party/Dexonomy`、`retarget/env/grasp/graspadmm_*`、`retarget/env/grasp/grasp_admm_runner.py` | Dexonomy/GraspADMM `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`。原版初始化/RELUQP、ADMM/Clarabel、MuJoCo、filter、squeeze/pregrasp 与六方向评测；逐文件 SHA-256 见 `GRASPADMM_UPSTREAM.json`，保留 CC BY-NC 4.0 声明。本地 WUJI2 free-root/模板/FK/完整 rigid pad 几何与输出适配独立记录。 |
| `retarget/env/grasp/grasp_evaluation.py`、`scripts/reevaluate_grasps.py` | 各 grasp 方法的共同原版 GraspADMM 评测桥接；仅传递实际最终手物相对抓姿及真实网格，重建统一测试场景，保留历史源结果。 |
| `third_party/DexForge/forceaware` | DexForge `fa6d671948ef03d18514062748b94e445b63cd49`，原版 ForceAware 求解器及 bundled ComFree/native adjoint。逐文件 SHA-256 见 `UPSTREAM.json`；保留原 MIT、Apache 2.0 和 ComFree Academic Research 许可。该清单范围不包含 ContactAware、MANO 或上游数据。G1 参考/目标/场景适配在 `retarget`，不修改上游求解器或反传。 |
| `third_party/DexForge/contactaware` | 同一固定提交的原版 ContactAware 源码，独立来源及逐文件 SHA-256 见 `CONTACTAWARE_UPSTREAM.json`。保留原 MIT 和 bundled manopth 的许可证；不含 MANO 模型权重、数据或手资产。仅保留为 SuperMimic 下游评测的静态接触力 QP 依赖；本仓库不再提供 ContactAware 抓姿求解入口。 |
| `retarget/task/humanoid`、`retarget/env/humanoid` | Meta SPIDER，commit `44717007de41cbef7565dff7ff9f4453557a2d3d`。原始 reward 函数体、参考插值、MPC rollout、MJWarp 参数和状态恢复。CC BY-NC 4.0，见 `licenses/SPIDER.txt`；函数核验见任务包 `source.json`。 |
| `data/humanoid` | SPIDER 公开 retarget_example 数据，revision `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63`。G1 模型许可保留在各任务 assets 中。sprint 接触标签为本地推导，见 `contact_provenance.json`。 |
| `retarget/dynamics/comfree` | ComFree Warp commit `ba8b9969f5f7859d51b1f738f458f8d2f49be080` 的三个原始接触响应 kernel；只保留响应和接口，不包含引擎或碰撞实现。完整来源与 AST 核验见 source.json；许可见 `licenses/comfree_core-LICENSE`。MJWarp 通过已安装的外部依赖调用。 |
| `retarget/initialization/dexonomy`、`data/grasp/templates` | Dexonomy 原始输入初始化方法和必要模板，CC BY-NC 4.0，见 `licenses/Dexonomy-NOTICE.txt`。只保留输入初始化的依赖，未保留原项目任务优化器。 |
| `data/grasp/assets/wuji` | 本地 Wuji 手模型、网格和表面查询点；见 `licenses/Wuji-LICENSE`。 |
| `data/grasp/assets/wuji2` | 官方 WUJI Hand 2.2，commit `5c4dfa16917355d073e2e1d7e3d03897b6cf9508`，MIT。原始模型、许可证与 adapter 核验记录保留在资产目录。 |
| `retarget/task/dex` | 本地已准备的 DexYCB 主目标、参考和控制平滑项，来自迁移前 ForceAware 工作区。迁移保留数值目标，移除任务反向优化。 |

MANO 权重仍是仓库外资源。本次不实现原始 MANO 序列到完整运行输入的自动转换，也不复制或删除外部模型资源。

抓取正式求解仅保留 GraspADMM 与 CEM＋历史 `wrench_qp` 的 GPU 批量适配。后者的测试参考保存在 `retarget/env/grasp/reference/wrench_qp.py`；原几何评分迁移到 `pose_objective.py`。DexForge/ComFree 的公共依赖继续服务于人体及 Dex 任务，未作为额外抓取入口。

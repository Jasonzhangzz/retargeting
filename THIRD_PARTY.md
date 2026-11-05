
| 位置 | 来源与用途 |
| --- | --- |
| `third_party/Dexonomy`、`retarget/env/grasp/graspadmm_*`、`retarget/env/grasp/grasp_admm_runner.py` | Dexonomy/GraspADMM `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`。原版初始化/RELUQP、ADMM/Clarabel、MuJoCo、filter、squeeze/pregrasp 与六方向评测；逐文件 SHA-256 见 `GRASPADMM_UPSTREAM.json`，保留 CC BY-NC 4.0 声明。本地 WUJI2 free-root/模板/FK/完整 rigid pad 几何与输出适配独立记录。 |
| `retarget/env/grasp/grasp_evaluation.py`、`scripts/reevaluate_grasps.py` | 各 grasp 方法的共同原版 GraspADMM 评测桥接；仅传递实际最终手物相对抓姿及真实网格，重建统一测试场景，保留历史源结果。 |
| `retarget/task/humanoid`、`retarget/env/humanoid` | Meta SPIDER，commit `44717007de41cbef7565dff7ff9f4453557a2d3d`。原始 reward 函数体、参考插值、MPC rollout、MJWarp 参数和状态恢复。CC BY-NC 4.0，见 `licenses/SPIDER.txt`；函数核验见任务包 `source.json`。 |
| `data/humanoid` | SPIDER 公开 retarget_example 数据，revision `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63`。G1 模型许可保留在各任务 assets 中。sprint 接触标签为本地推导，见 `contact_provenance.json`。 |
| `retarget/dynamics/comfree` | ComFree Warp commit `ba8b9969f5f7859d51b1f738f458f8d2f49be080` 的三个原始接触响应 kernel；只保留响应和接口，不包含引擎或碰撞实现。完整来源与 AST 核验见 source.json；许可见 `licenses/comfree_core-LICENSE`。MJWarp 通过已安装的外部依赖调用。 |
| `retarget/initialization/dexonomy`、`data/grasp/templates` | Dexonomy 原始输入初始化方法和必要模板，CC BY-NC 4.0，见 `licenses/Dexonomy-NOTICE.txt`。只保留输入初始化的依赖，未保留原项目任务优化器。 |
| `data/grasp/assets/wuji` | 本地 Wuji 手模型、网格和表面查询点；见 `licenses/Wuji-LICENSE`。 |
| `data/grasp/assets/wuji2` | 官方 WUJI Hand 2.2，commit `5c4dfa16917355d073e2e1d7e3d03897b6cf9508`，MIT。原始模型、许可证与 adapter 核验记录保留在资产目录。 |
| `retarget/task/dex` | 本地已准备的 DexYCB 主目标、参考和控制平滑项，来自迁移前 ForceAware 工作区。迁移保留数值目标，移除任务反向优化。 |

MANO 权重仍是仓库外资源。本次不实现原始 MANO 序列到完整运行输入的自动转换，也不复制或删除外部模型资源。

抓取正式求解仅保留当前 v5 patch CEM 与 GraspADMM。当前 CEM 使用本地 Kelvin–Voigt 弹簧阻尼和切向历史适配，不将其表述为上游 ComFree 原公式。GraspADMM 的算法内部 QP 及官方来源校验保留。DexForge/ComFree 公共依赖继续服务于人体及 Dex 任务，未作为额外抓取入口。

独立评测数值工具 `retarget/integrations/supermimic/force_margin.py` 的来源提交与文件哈希见 `third_party/DexForge/NUMERICS_SOURCE.json`；对应许可证及声明继续保留。该工具不提供 pose 求解入口。

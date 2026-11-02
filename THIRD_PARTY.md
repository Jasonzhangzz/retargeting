本仓库保留当前本机实现。DexForge 分支仅加载本仓库固定版本的源码，不查找外部 Panda 或用户机器上的其他 checkout。

| 位置 | 来源与用途 |
| --- | --- |
| `third_party/Dexonomy`、`retarget/env/grasp/graspadmm_*`、`retarget/env/grasp/grasp_admm_runner.py` | Dexonomy/GraspADMM `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`。原版初始化/RELUQP、ADMM/Clarabel、MuJoCo、filter、squeeze/pregrasp 与六方向评测；逐文件 SHA-256 见 `GRASPADMM_UPSTREAM.json`，保留 CC BY-NC 4.0 声明。本地 WUJI2 free-root/模板/FK/完整 rigid pad 几何与输出适配独立记录。 |
| `retarget/env/grasp/grasp_evaluation.py`、`scripts/reevaluate_grasps.py` | 各 grasp 方法的共同原版 GraspADMM 评测桥接；仅传递实际最终手物相对抓姿及真实网格，重建统一测试场景，保留历史源结果。 |
| `third_party/DexForge/forceaware` | DexForge `fa6d671948ef03d18514062748b94e445b63cd49`，原版 ForceAware 求解器及 bundled ComFree/native adjoint。逐文件 SHA-256 见 `UPSTREAM.json`；保留原 MIT、Apache 2.0 和 ComFree Academic Research 许可。该清单范围不包含 ContactAware、MANO 或上游数据。G1 参考/目标/场景适配在 `retarget`，不修改上游求解器或反传。 |
| `third_party/DexForge/contactaware` | 同一固定提交的原版 ContactAware 源码，独立来源及逐文件 SHA-256 见 `CONTACTAWARE_UPSTREAM.json`。保留原 MIT 和 bundled manopth 的许可证；不含 MANO 模型权重、数据或手资产。Primitive 适配直接调用 SQP/QP 核心，不加载 MANO 前端。 |
| `retarget/task/humanoid`、`retarget/env/humanoid` | Meta SPIDER，commit `44717007de41cbef7565dff7ff9f4453557a2d3d`。原始 reward 函数体、参考插值、MPC rollout、MJWarp 参数和状态恢复。CC BY-NC 4.0，见 `licenses/SPIDER.txt`；函数核验见任务包 `source.json`。 |
| `third_party/SPIDER/dex`、`retarget/solver/spider_mppi.py`、`retarget/task/grasp/spider_cost.py`、`retarget/env/grasp/spider_*.py` | 同一 SPIDER 固定提交的灵巧手 cost、generic MPPI、插值与 native MJWarp 适配。17 个上游原文件及 SHA-256 见 `UPSTREAM.json`；CC BY-NC 4.0，保留 Meta 版权及许可。WUJI2 静态 CA 输入桥接、完整状态恢复、CUDA Graph、无效样本防护与 CPU 回放为本地适配，不调用 FA 优化器或添加物体虚拟执行器。 |
| `data/humanoid` | SPIDER 公开 retarget_example 数据，revision `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63`。G1 模型许可保留在各任务 assets 中。sprint 接触标签为本地推导，见 `contact_provenance.json`。 |
| `retarget/dynamics/comfree` | ComFree Warp commit `ba8b9969f5f7859d51b1f738f458f8d2f49be080` 的三个原始接触响应 kernel；只保留响应和接口，不包含引擎或碰撞实现。完整来源与 AST 核验见 source.json；许可见 `licenses/comfree_core-LICENSE`。MJWarp 通过已安装的外部依赖调用。 |
| `retarget/initialization/dexonomy`、`data/grasp/templates` | Dexonomy 原始输入初始化方法和必要模板，CC BY-NC 4.0，见 `licenses/Dexonomy-NOTICE.txt`。只保留输入初始化的依赖，未保留原项目任务优化器。 |
| `data/grasp/assets/wuji` | 本地 Wuji 手模型、网格和表面查询点；见 `licenses/Wuji-LICENSE`。 |
| `data/grasp/assets/wuji2` | 官方 WUJI Hand 2.2，commit `5c4dfa16917355d073e2e1d7e3d03897b6cf9508`，MIT。原始模型、许可证与 adapter 核验记录保留在资产目录。 |
| `retarget/task/dex` | 本地已准备的 DexYCB 主目标、参考和控制平滑项，来自迁移前 ForceAware 工作区。迁移保留数值目标，移除任务反向优化。 |
| `retarget/task/grasp/tracking_cost.py` | 复用本仓库 dex cost 的四项公式，为静态抓姿的预测轨迹提供适配；不声称官方 SPIDER 灵巧手 cost。默认权重来源为保留的 DexForge ForceAware 配置，grasp 配置显式覆盖接触深度阈值。 |
| `retarget/env/grasp/contactaware.py` | DexForge ContactAware 单静态 key 的 primitive/WUJI2 适配；直接调用固定版本的 elastic SQP、Clarabel QP、topology 与 actual-point wrench 公式。表面查询和手部几何适配使用本仓库资产，不伪造 MANO 演示。 |
| `retarget/env/grasp/forceaware_*.py` | 直接调用原版 FA workflow、优化器、损失和滚动执行；适配 WUJI2 拓扑、完整手部 GS 几何和 CA 静态参考。可显式选择本地 `dexforge_servo_integration.py` 扩展 actuator 阻尼的隐式积分，不能声称该 runtime 与原 native Euler 相同。active-only CA、最近特征法向、控制预载与可选固定初态作为输入适配记录；可选 PD 增益变体独立标记。native 与独立 mesh 的真实逐步载荷分开审计，上游原文件 SHA-256 保持不变。 |
| `retarget/dynamics/explicit.py` | 本地轻量接触预测，设计参考 Complementarity-Free-Dexterous-Manipulation `87b320ffb4b40ab4f3aa344371f6e5e8d619aa7e`。准静态位移度量，不冒称完整惯性动力学。 |

MANO 权重仍是仓库外资源。本次不实现原始 MANO 序列到完整运行输入的自动转换，也不复制或删除外部模型资源。

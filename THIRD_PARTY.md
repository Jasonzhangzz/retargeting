本仓库保留当前本机实现，不在运行时查找外部源码或修改 Python 导入路径。

| 位置 | 来源与用途 |
| --- | --- |
| `retarget/task/humanoid`、`retarget/env/humanoid` | Meta SPIDER，commit `44717007de41cbef7565dff7ff9f4453557a2d3d`。原始 reward 函数体、参考插值、MPC rollout、MJWarp 参数和状态恢复。CC BY-NC 4.0，见 `licenses/SPIDER.txt`；函数核验见任务包 `source.json`。 |
| `data/humanoid` | SPIDER 公开 retarget_example 数据，revision `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63`。G1 模型许可保留在各任务 assets 中。sprint 接触标签为本地推导，见 `contact_provenance.json`。 |
| `retarget/dynamics/comfree` | ComFree Warp commit `ba8b9969f5f7859d51b1f738f458f8d2f49be080` 的三个原始接触响应 kernel；只保留响应和接口，不包含引擎或碰撞实现。完整来源与 AST 核验见 source.json；许可见 `licenses/comfree_core-LICENSE`。MJWarp 通过已安装的外部依赖调用。 |
| `retarget/initialization/dexonomy`、`data/grasp/templates` | Dexonomy 原始输入初始化方法和必要模板，CC BY-NC 4.0，见 `licenses/Dexonomy-NOTICE.txt`。只保留输入初始化的依赖，未保留原项目任务优化器。 |
| `data/grasp/assets/wuji` | 本地 Wuji 手模型、网格和表面查询点；见 `licenses/Wuji-LICENSE`。 |
| `data/grasp/assets/wuji2` | 官方 WUJI Hand 2.2，commit `5c4dfa16917355d073e2e1d7e3d03897b6cf9508`，MIT。原始模型、许可证与 adapter 核验记录保留在资产目录。 |
| `retarget/task/dex` | 本地已准备的 DexYCB 主目标、参考和控制平滑项，来自迁移前 ForceAware 工作区。迁移保留数值目标，移除任务反向优化。 |
| `retarget/dynamics/explicit.py` | 本地轻量接触预测，设计参考 Complementarity-Free-Dexterous-Manipulation `87b320ffb4b40ab4f3aa344371f6e5e8d619aa7e`。准静态位移度量，不冒称完整惯性动力学。 |

MANO 权重仍是仓库外资源。本次不实现原始 MANO 序列到完整运行输入的自动转换，也不复制或删除外部模型资源。

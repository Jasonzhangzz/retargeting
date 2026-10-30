本次按用户要求只做可运行性短测，没有重新执行完整搬箱、跑步或抓取矩阵，也不据此报告完整任务成功。

2026-10-30 的 ComFree 清理将 `retarget/dynamics/comfree` 从 56 个 Python 文件、约 27,374 行收缩为 **3 个 Python 文件、162 行**（含注释和空行）。只保留三个上游原始接触响应 kernel 与接口；来源固定为 ComFree Warp commit `ba8b9969f5f7859d51b1f738f458f8d2f49be080`。核验信息在该目录 source.json，许可证在 licenses/comfree_core-LICENSE。旧内置 MJWarp、自写碰撞/BVH、运动学、质量求解、积分、softplus 和多轮线性化 ComFree 路径已移除。

`retarget/env/mjwarp.py` 直接调用已安装的 MJWarp：碰撞、约束行、FK、质量矩阵、平滑动力学、质量求解和积分均由引擎执行。ComFree 使用这些约束行的 J、pos、D，默认 stiffness=0.2、damping=0.001；solimp 宽度按上游固定为 0.01。仅接触响应 kernel 与上游一致，不声称整个本机引擎与上游锁定环境逐位相同。

模型物理碰撞从旧自写 querypoint/SDF 转为标准 MuJoCo mesh 凸碰撞表示，因此此前“模型接触与网格数组完全一致”的迁移结论不能套用于此次修改。NPZ 手部采样点未改动；env/surface.py 使用 Warp 内置 mesh 查询为任务 cost 提供表面测量，不负责生成物理接触。Grasp 的轻量准静态 explicit.py 保持独立，不冒称上游 ComFree Warp；其 FK 也已改为调用 MJWarp。

验证机器：NVIDIA GeForce RTX 5060 Ti（15 GiB 可用），Python 3.13、Torch 2.13.0+cu130、Warp 1.16.0、MuJoCo 3.10.0、MJWarp 3.10.0.3。

全部测试 **33 passed**：

- 三个接触响应 kernel 的 AST 哈希与取自上游的记录一致。
- CPU/GPU 接触响应与独立 NumPy 公式一致，覆盖非负截断、不同 timestep、多 world、无活动行及重复调用后的清零。
- 集成测试监测到 MJWarp collision 调用；接触产生正法向力并可积分，分离后接触力清零。
- 两代手模型加载；MJWarp FK 的 body pose、关节 anchor/axis 与 CPU MuJoCo 对照通过。
- 完整 grasp 批量评分与独立 20 维伺服偏移检查通过。
- SPIDER 冻结 reward、完整 48 步聚合、参考索引、控制插值及评分后的物理状态/参数恢复检查通过。
- YAML、输入校验、输出覆盖、CEM/MPPI 更新与退火、提前停止、非有限候选、硬约束和网页筛选检查通过。

受影响的四组 GPU 短测：population=20、iterations=2、worlds=20、seed=0；均评估 41 次、采样 40 次，退出码为 0。

| 任务 | 求解器 | 执行范围 | 首次求解记录（秒） | 最终评分 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| grasp | CEM | 1 个 case | 0.360 | 28,262,308 | 硬约束通过；预测稳定未通过 |
| grasp | MPPI | 1 个 case | 0.432 | 48,630,400 | 硬约束通过；预测稳定未通过 |
| dex | CEM | 1 个窗口、3 个实际步 | 32.924 | 4,628,031 | 状态有限 |
| dex | MPPI | 1 个窗口、3 个实际步 | 18.824 | 4,703,558 | 状态有限 |

抓取输入初始化使用 4 个采样点、4 个平面角、10 次迭代，是短测预算。Dex 的首次求解包含懒加载 kernel JIT 和 Python 调度；测试期间也运行了单元检查，上表不能用于比较 CEM/MPPI 性能。没有测稳态吞吐，也没有据此得出接触模型提高任务成功率的结论。Humanoid 的实现此次未改；本轮通过其 GPU 回归测试，没有重复求解两种 humanoid 任务。

四组结果、解析配置和逐轮日志分别在 output/smoke_grasp_cem、smoke_grasp_mppi、smoke_dex_cem、smoke_dex_mppi。搬箱结果仍为前次短片段。新 grasp/dex 模型可由统一回放器加载，已用 EGL 成功导出两者的视频；网页服务沿用 8081。清理记录与日志在 output/validation/contact_cleanup。

复现测试：

```bash
python -m pytest -q
```

复现短测：使用相应 output/smoke_*/config.yaml（运行时会覆盖同名输出，因此需要先复制配置到别处），或复制 configs 示例并设置上述预算；dex 另设置 environment.frames=1。省略 frames 则执行完整参考。

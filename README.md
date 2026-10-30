本仓库通过一个 YAML 接口运行 grasp、dex 和 humanoid，任务求解器仅有 GPU CEM 与 MPPI。

```bash
pip install -r requirements.txt
python run.py                         # 根目录 task.yaml：搬箱 + CEM
python run.py --config configs/grasp.yaml
python run.py --config configs/dex_sugar_box.yaml
python run.py --config configs/humanoid_sprint.yaml
python run.py --config configs/web.yaml
```

CUDA 是求解所需条件。当前验证环境为 NVIDIA RTX 5060 Ti、MuJoCo 3.10.0、MJWarp 3.10.0.3、Warp 1.16.0、Torch 2.13.0。requirements 中的本机引擎版本用于重现本次验证，不代表 SPIDER 上游锁定环境。

统一配置示例：

```yaml
mode: solve                    # solve / replay / web
task:
  type: humanoid               # grasp / dex / humanoid
  name: move_largebox          # humanoid 另支持 sprint
  data: data/humanoid/move_largebox
solver:
  type: cem                    # cem / mppi
  population: 1024
  iterations: 16
  worlds: 128
  elite_fraction: 0.05         # CEM
  temperature: 0.1
  final_noise: 0.2
environment:
  device: cuda:0
  # frames: 1                  # 仅用于短测；省略时执行完整参考
seed: 0
output:
  name: humanoid_move_largebox_cem
visualization:
  live: false
  save_video: true
```

输入路径相对于 YAML 所在目录解析，示例配置位于 configs，因此使用 `../data/...`。抓取 `task.options` 中显式指定的资源路径也相对于 YAML 解析；内置默认资源相对于 `task.data`。切换求解器只需修改 `solver.type`；省略 temperature/final_noise 时，CEM 默认 0.1/0.2，MPPI 默认 1.0/0.1。跑步示例显式采用上游 temperature=0.6。

输出固定为仓库内 `output/<name>`。配置和输入校验通过后会替换同名目录，其他结果目录保留；名称禁止包含路径或跳转，同名符号链接也会拒绝。结果包含解析后的 config.yaml、status.json、manifest.json、summary.json、轨迹、逐轮日志和来源记录。抓取批量任务使用 case 子目录。失败初始化或无硬约束可行候选会保留失败诊断，不能作为成功抓取导出。

目录职责：

| 目录 | 内容 |
| --- | --- |
| `retarget/solver` | 任务无关的候选采样、CEM/MPPI 更新和评分协议 |
| `retarget/task` | 各任务 cost、参考、搜索变量和验收指标 |
| `retarget/env` | 模型、接触查询、控制解码、rollout、执行与记录 |
| `retarget/dynamics` | ComFree 接触响应、独立的轻量显式接触预测 |
| `retarget/initialization` | 输入 IK、保留的 Dexonomy 初始化数值实现 |
| `retarget/config`、`common`、`viewer` | 配置、结果与来源记录、回放和网页查看 |
| `data/dexycb` | 两批共十条原始打包序列、原始资产/标定和 prepared 示例 |
| `data/humanoid` | move_largebox、sprint 与共享 G1 场景资产 |
| `data/grasp` | 抓取模板、两代手资产、冻结手腕参考 |

CEM 使用 CARD 精英成本标准化后的 softmax 均值更新，精英数量向上取整，不拟合精英方差；噪声按 `sigma * final_noise ** (i / iterations)` 退火，每轮包含历史最佳与均值。MPPI 使用前 10% 候选加权、同形式退火和可配置提前停止。两者分离 population 与 worlds，并通过 Torch/Warp 共享数组分批评分。日志记录实际迭代数、采样数与额外评分数。

Humanoid 使用原始 SPIDER reward、terminal reward、参考插值和 post-step 时域聚合；搬箱保留加权非平方 L2，权重为根位置/根旋转/关节/箱体位置/箱体旋转 `1/1/1/1/0.3`，无附加平衡或接触目标。48 步预测、6 步执行、8 个控制 knots、CPU 初始化 pre-step、DR 与执行前参数恢复均保留。只优化关节控制增量，未添加控制裁剪。版本和函数 AST 哈希在任务包 source.json，迁移前冻结评分样本在 tests/fixtures。独立的 3 秒初始化预检会记录失败并继续，实际初态不受其影响。

Sprint 保留原 qpos/qvel/ctrl 数组，原始文件单独保存在 reference_raw.npz。实际 reference.npz 中两脚接触标签由参考 FK 的足部碰撞几何有符号地面间隙生成，阈值 35 mm；目标为 left_foot/right_foot site 的世界坐标。原始接触字段中的占位数据被替换，生成来源和阈值见 contact_provenance.json。这是推导标注，未冒称 SPIDER 官方标注。

Grasp 的两种求解器直接评分完整几何项、手腕参考、显式物体 pose cost 和硬约束，搜索手腕 6 维、手指 20 维与独立伺服偏移 20 维。显式预测对每个候选分批执行，无几何优化后的梯度细化。保留原输入初始化、接触区域、指腹中心策略和 WUJI1/WUJI2。预测采用轻量准静态接触模型，summary 中明确区分预测稳定与实际物理测试。更换手型使用 `task.options.hand_model: wuji1` 或 `wuji2`；更换抓型使用 task.templates，尺寸使用 radius_scales/height_scales。冻结手腕参考缺失时会记录输入失败，不会自动搜索历史输出。当前冻结库是 WUJI2 的已有参考；WUJI1 可提供匹配的冻结参考，或显式设置 `task.options.wrist_pose_prior: false`。

ComFree 只实现接触响应：`retarget/dynamics/comfree/{kernels,response}.py`。三个 kernel 的函数体来自 [上游 ComFree Warp](https://github.com/asu-iris/comfree_warp/tree/ba8b9969f5f7859d51b1f738f458f8d2f49be080)，固定 commit 和 AST 哈希见同目录 source.json。原始响应为预测速度、预测侵入、非负行接触力与 `J.T @ force`；没有 softplus 或多轮线性化。`retarget/env/mjwarp.py` 调用已安装的 MJWarp 完成碰撞检测、约束行装配、FK、质量矩阵、平滑动力学和积分，仅替换接触求解阶段。模型 YAML 的 `contact_physics.stiffness/damping` 默认为 `0.2/0.001`，行阻抗宽度按上游固定为 `0.01`。不支持的双侧等式和 frictionloss 约束会明确报错。

物理输入使用标准 MuJoCo mesh；原自定义 querypoint/sdfmesh、GS 碰撞、BVH 和整套内置引擎已经删除。MJWarp 的 mesh 碰撞采用凸碰撞表示，替代了旧的自写 SDF 接触几何，不能宣称新旧动力学结果一致。手部 NPZ 采样点和 `retarget/env/surface.py` 的 Warp 内置 mesh 距离查询仅为任务 cost 提供表面测量，不生成物理接触。Grasp 的 `explicit.py` 仍是独立的轻量准静态预测，并非上游 ComFree Warp。

Dex 默认使用 prepared/sugar_box，保留原主目标公式、参考和控制插值，动力学采用上述 MJWarp + ComFree 响应。初态固定，搜索动作 knots，MANO 接触指导固定为参考；没有 consensus/dual、接触点交替优化或物体虚拟控制。原始序列尚未包含完整 prepared 输入时会明确报错；本次不实现 MANO 自动转换。仓库外 MANO 模型不受本次整理影响。

回放和网页都读取 manifest.json：

```bash
python run.py --config configs/replay.yaml
python run.py --config configs/web.yaml  # http://localhost:8081
```

回放配置的 output.name 指向已有结果；不会清空该目录。`live: false, save_video: true` 可用于无显示器的视频导出。网页自动扫描 output，支持任务、抓型、手型及 CEM/MPPI 筛选。统一入口没有旧脚本或命令行选项的转发兼容层。

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

验证仅使用短测，没有重新完成整套任务求解。本次对受影响的 grasp、dex 分别完成 CEM/MPPI 短测，固定 SPIDER reward 样本逐元素一致，新增接触响应及 MJWarp 委托验证；旧的模型数组一致性结论不适用于这次碰撞几何替换。具体预算、结果、耗时与验证限制见 [验证记录](docs/validation.md)。这些结果证明新流程可运行，不构成任务成功率提升或完整搬箱成功的结论。来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

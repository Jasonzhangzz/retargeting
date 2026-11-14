# Retarget

仅保留 moxian 当前采集与 WebUI、CEM／GraspADMM 实现，以及必要的运行依赖、模型资产和验证工具。

- `moxian_web/`：EMF 骨架（Moxian25＋四指轴向旋转，标定后使用原生编译的 Studio 同类有界 QP）、官方 SDK 对照、Wuji原生六动作标定、触觉和 Wuji2 实时映射。
- `retarget/solver/cem.py`：当前 MANO patch CEM；默认参数统一由 `DEFAULTS` 提供。
- `retarget/solver/graspadmm.py`：当前 GraspADMM，包含原版 CPU 与 GPU 输入接口。
- `retarget/env/`、`retarget/task/`：两种求解器的模型、几何、动力学和任务适配。
- `third_party/Dexonomy/`：GraspADMM 固定版本源码，保留来源哈希和许可证。

moxian：

```bash
bash scripts/setup_moxian.sh
bash scripts/setup_moxian_official.sh
bash scripts/start_moxian_web.sh
```

网页地址：`http://127.0.0.1:8890`。SDK 使用独立 Python 环境；厂商目录只保留串口 SDK 和 V1 解码器。MANO 权重通过 `MOXIAN_MANO_MODEL` 指定，仓库不包含私有权重。

默认个体标定使用 **Moxian 原生25自由度＋四指指根轴向旋转及原始Qnbot mesh**，沿用Wuji六个静态动作与其ELF内的原生L-BFGS。只优化15段骨长和36维EMF外参，固定掌部、宽厚、关节轴与蒙皮权重。六动作自动切换，完成后点击拟合；已有六动作记录可复用。Wuji内置人手模型及21→25显示适配已停用，旧候选仅保留诊断。目标函数在Moxian FK和接触曲面上求值，不能宣称与Wuji内置21自由度目标逐值相同。实现及旋转方向检查见 [Moxian六动作标定](docs/moxian_six_calibration.md)。

Qnbot 历史结果另有“分阶段最小二乘标定”分支：低维尺寸、外参、运动学、触觉注册和表面基按证据开放，离线交替求姿态，实时固定前向。默认优化预算600秒，完整验证另计，结果仅保存为独立预览候选。接口、历史数据命令与导出资产见 [最小二乘标定说明](docs/qnbot_least_squares.md)。

Wuji2 触觉重定向默认使用独立的 `moxian_unified_graph_slsqp_v3`：460 个 SID 先绑定到 Moxian 原始 SKN 表面，每个激活 patch 提取表面代表点，再与21个骨架节点构成统一图。材料点连接所属骨架，跨语义 link 接触边使用高权重，同指同节或同属 palm 的点不加直接接触边。NLopt 两阶段 SLSQP 固定源腕部、求20关节，计算单个混合 Laplacian 残差并保留参考骨架辅助项，不增加法向或绝对接触位置 tracking；穿透时以零有符号间距约束做最小关节修复，再稠密验收。网页可切换 SLSQP/CEM，离线通过 `--solver` 选择；旧 MANO 录制需要原始 EMF 和匹配的 Moxian 标定重建源 mesh。依赖、接口、命令、容差和评估见 [Moxian 触觉 SLSQP](docs/moxian_wuji_contact_slsqp.md)。

Moxian 源默认使用 `moxian_studio_native_qp_v1`，保留当前29自由度和原有标定，迭代 FK／Jacobian／有界小 QP 在 C++ 中完成；不是官方固定21自由度二进制。构建入口为 `.venv-moxian-ik/bin/python scripts/setup_moxian_studio_qp.py`。数值一致性、真实输入延迟、接触录制与后端切换验证见 [原生源与实时验证](results/moxian_wuji_realtime_native_20261113/REPORT.md)。

保留的 CEM 后端为 `card_wuji2_v9_source_motion`：从当前源手21节点生成 Robokit 初解，CEM 保留源表面接触位置、方向和相对压力，按语义连杆及段内位置搜索原生可接触表面。它通过独立几何适配支持 Moxian SKN，历史 MANO 输入仍走原路径；CEM 的 cost、solver 和预算保持独立。默认不推断对指或替换目标中点，`experimental_self_contact=False`。

默认 `cem_backend='robokit'` 将 CARD 四组代价及在线运动一致性项实现为 Robokit 的 cost-only Task，使用 `PopulationSolver`／`GaussianSampler`、批量 FK、网格拟合胶囊和整次 CUDA Graph。保持1024×16、5%精英、标准化代价的0.1温度和固定退火；轴角向量直接加权，无额外旋转均值内迭代。`solver.backend_timing_ms` 区分搜索与最终候选验收。`cem_backend='torch'` 保留独立参考路径；显式力锥／对指实验目前明确回到参考后端并报告原因。实现与回放入口见 [Robokit CARD](docs/moxian_robokit_card.md)。

CEM 的±0.25 rad／20 mm精化范围以本帧 Robokit 初解为中心。Robokit 用自己的上一帧关节解热启动，当前 MANO 提供腕部和全部目标，不读取 CEM 历史。CEM 重新评价搜索范围内的上一帧有效解及随当前源腕部搬移的版本；这些候选必须通过当前代价、物体约束和网格验收，不是未经检查的历史回退。整批不可行时仍显示当前约束下违约较小的有限候选并保留超限标记。只检查候选最终姿态，不查询帧间扫掠轨迹。胶囊安全间距只作软惩罚；实际手部网格穿透仍做最终验收。离线滤波另外保留其平滑和验收步骤。

接触指保留style姿态项；源动作一致性代价约束CEM修正量的突变，当前Robokit真实动作仍直接跟随。默认每轮896个全局候选与128个当前动作附近的小范围候选共用1024预算及相同穿透验收。该在线扩展不使用输出低通滤波，细节和对照回放见 [Robokit CARD](docs/moxian_robokit_card.md)。

`robokit_qpos`、`initial_qpos` 记录本帧初解，`solver.output_selection`、`solution_feasible` 区分可行结果与带警告的显示候选。源接触点误差独立于机器人指间闭合距离；没有采用两指闭合约束时 `contact_constraints_satisfied=null`，不把空约束报告为闭合成功。安装依赖使用 `bash scripts/setup_moxian_mano.sh`，Robokit版本固定在 `requirements-moxian-mano.txt`。

当前 `material_uv_rows_v2` 仍是解剖插值，460个SID均有三角面／重心坐标并不代表完成了实物传感器注册；Wuji2上的SID也是可视化用的虚拟投影，默认接触目标函数不把它当成逐税元实测对应。应先核验个性化MANO、手套安装和接触区域，再评价CEM。论文与本地实现核对见 `output/card_contact_audit_20261109/REPORT.md`。

实测税元注册入口为 `scripts/register_moxian_contacts.py`：在固定MANO上用独立参考接触点／法向拟合固定面及重心坐标，每个SID至少2帧训练、2帧留出，最大位置误差2 mm、法向误差30°。通过后可原子导入或恢复，未覆盖SID保留未注册状态。手型改变时材料坐标保留、实测验收失效；手侧、权重、设备或SID分区不兼容时不沿用。`source_contact_geometry` 将源税元注册、个性化手型和EMF安装状态与CEM可行性分开。操作和数据格式见 [接触注册说明](docs/moxian_contact_registration.md)。

触觉方向实验使用 `contact_force.py` 的压力加权法向合力，明确标为相对ADC响应、未测剪切、未标定牛顿。默认仍是CARD的 `contact_direction_model='card_alignment'`。可选 `coulomb_force` 在每个CEM候选上解析拟合满足 `Fn>=0, ||Ft||<=mu*Fn` 的相对力；`normal_force_weight` 控制法向分量拟合。这是给定几何下的力兼容性实验，不是机器人实际力反馈控制，也不修改源MANO或推断对指。离线对照入口为 `scripts/benchmark_moxian_contact_force.py`，报告入口为 `scripts/report_moxian_contact_force.py`；须同时检查原始方向兼容、接触点到位及穿透，不能把投影力天然满足锥约束当作接触成功。

Robokit 初值与 CEM 搜索使用基于原生网格拟合的胶囊软代价；它不能单独保证无穿透。CEM 保存全部轮次候选，保留低代价、初解与早期候选，最多256个，先作原生网格粗检，再按代价顺序稠密验收并选择首个通过者。输出及离线滤波仍须复核；验收预算内未找到可行解时显示本帧有限候选并明确超限。自穿透超过0.5 mm不得标为可行；触觉接触不豁免。跨手指所有连杆、同指非相邻连杆及掌指非连接表面参与检查，MCP共轴机械连接仅排除按原生父连杆测得的局部连接圈。`collision.self_mesh` 公布逐连杆穿透，`object_max_penetration_mm` 独立记录物体SDF穿透；旧的物体3 mm门槛不作为手部自穿透门槛。此为离散曲面采样及winding检查，不是连续三角形相交的完整证明。

已提供的真实物体网格保留3 mm侵入硬门槛。由未匹配压力点推测的物体凸包只作软约束，避免把未知接触围成实心障碍而拒绝所有候选；`collision.object_hard_gate_enabled` 明确区分二者。

Wuji2 对照在下一帧求解期间保留上一帧完整画面，源数据延迟单独提示。输入心跳超时为1.5秒；源与结果超时为至少5秒、按单帧耗时3倍调整且最多15秒。实际断流、重启、错误或等待映射时清除接触指示并标注保留画面，完整新结果到达后替换。新结果仍须通过碰撞验收，不对关节做未经验证的插值。

MANO patch CEM：

```bash
python scripts/solve_mano_comfree.py --help
python scripts/retarget_mano_comfree_campaign.py --help
```

GraspADMM：

```bash
bash scripts/setup_graspadmm.sh
.venv-graspadmm/bin/python run.py --config configs/graspadmm.yaml
```

YAML 的 `solver.type` 接受 `cem`、`graspadmm` 或 `contact_cem`；算法参数分别位于对应模块的 `DEFAULTS`。原有人体与 Dex 任务沿用 CEM。外部输入数据和私有模型需另行准备。

接触空间 CEM 基于随仓库迁入的 Robokit 原始 SimMPC、PopulationSolver／GaussianSampler、IK／LM 和 MjWarpSim，增加 Sharpa 转笔、翻币及 H1 jog 任务：GPU 搜索候选，独立 CPU MuJoCo 4 进程开启 speculative contacts 复核并闭环执行。场景和资产随仓库提供，支持完整任务、多种子对比及冻结 H1 参考轨迹的扰动跟踪。安装、运行、后端差异和严格验收见 [contact_cem 文档](docs/contact_cem.md)。实验是否成功以对应 `summary.json` 和报告为准，短时 smoke run 不计作成功。

当前设备标定保存在本机 `output/moxian/`，保留有效官方标定、对应八动作采集和当前手型；历史结果、标定副本、缓存及编译中间文件已清除。moxian 正在使用的四个虚拟环境保留，闲置 GraspADMM 环境按需由安装脚本重建。保留实现的注释和 docstring 不作批量删除。

骨架与 MANO 的[连续联合标定](docs/moxian_joint_calibration.md)复用八动作初始化，再采集活动范围及三轮对指；第三轮和末尾复测独立验收。固定几何、安装参数和个体骨架通过后整体应用，失败候选保留诊断。

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

# Retarget

包含 Moxian 当前采集与 WebUI、五个实时重定向入口、MANO CEM／GraspADMM，以及必要的运行依赖、模型资产和验证工具。

- `moxian_web/`：EMF 骨架（Moxian25＋四指轴向旋转，标定后使用原生编译的 Studio 同类有界 QP）、官方 SDK 对照、Wuji原生六动作标定、触觉和 Wuji2 实时映射。
- `retarget/solver/cem.py`：当前 MANO patch CEM；默认参数统一由 `DEFAULTS` 提供。
- `retarget/solver/graspadmm.py`：当前 GraspADMM，包含原版 CPU 与 GPU 输入接口。
- `retarget/env/`、`retarget/task/`：两种求解器的模型、几何、动力学和任务适配。
- `third_party/Dexonomy/`：GraspADMM 固定版本源码，保留来源哈希和许可证。

CARD、wuji-retargeting、GeoRT 的实现统一位于 [`retarget/baseline`](retarget/baseline/README.md)。`scripts` 仅保留当前入口及其依赖。

moxian：

```bash
bash scripts/setup_moxian.sh
bash scripts/start_moxian_web.sh
```

网页地址：`http://127.0.0.1:8890`。SDK 使用独立 Python 环境；厂商目录只保留串口 SDK 和 V1 解码器。MANO 权重通过 `MOXIAN_MANO_MODEL` 指定，仓库不包含私有权重。

`wuji-retargeting` 使用当前已标定 Moxian 骨架的21个节点，直接调用用户指定远端的原版 Wuji2 两阶段重定向，保留原版目标函数、热启动和低通。原版源码与6份URDF逐字节保留，完整 CQQ／EMF 链路保留为独立离线入口，见 [接入与一致性说明](docs/moxian_wuji_reference.md)。

GeoRT 动作录制、官方网络训练及虚拟 Wuji2 实时／回放预览见 [GeoRT 使用说明](docs/moxian_geort.md)，依赖安装入口为 `bash scripts/setup_geort.sh`。

CART 提供对指、precision grasp、power grasp、wrap、prismatic 五类、8个子动作的引导录制。Primitive 搜索仅使用 Dexonomy 的 3、5、6、7、8、11、13、26、28 类先验，按当前手型估计 cylinder／sphere 尺寸，并允许有限的 delta geometry；置信度不足或关键手指未全部接触时放弃该帧拟合。录制、网络训练和验收范围见 [CART 使用说明](docs/moxian_cart.md)。

默认个体标定使用 **Moxian 原生25自由度＋四指指根轴向旋转及原始Qnbot mesh**，沿用Wuji六个静态动作与其ELF内的原生L-BFGS。只优化15段骨长和36维EMF外参，固定掌部、宽厚、关节轴与蒙皮权重。六动作自动切换，完成后点击拟合；已有六动作记录可复用。Wuji内置人手模型及21→25显示适配已停用，历史候选已清理。目标函数在Moxian FK和接触曲面上求值，不能宣称与Wuji内置21自由度目标逐值相同。实现及旋转方向检查见 [Moxian六动作标定](docs/moxian_six_calibration.md)。

Qnbot 历史结果另有“分阶段最小二乘标定”分支：低维尺寸、外参、运动学、触觉注册和表面基按证据开放，离线交替求姿态，实时固定前向。默认优化预算600秒，完整验证另计，结果仅保存为独立预览候选。接口、历史数据命令与导出资产见 [最小二乘标定说明](docs/qnbot_least_squares.md)。

Wuji2 触觉重定向默认使用 `moxian_discrete_contact_graph_slsqp_v8`：460 个 SID 先绑定到 Moxian 原始 SKN 表面，每个激活 patch 提取表面代表点，再与21个骨架节点构成统一图。材料点连接所属骨架，跨语义 link 接触边使用高权重，同指同节或同属 palm 的点不加直接接触边。NLopt 两阶段 SLSQP 固定源腕部、求20关节，计算单个混合 Laplacian 残差并保留参考骨架辅助项，不增加法向或绝对接触位置 tracking；穿透时以零有符号间距约束做最小关节修复，再稠密验收。网页保留 CARD、SLSQP 离散接触图＋Laplacian、wuji-retargeting（标定骨架 → 原版机器人重定向）、GeoRT 和 CART 五个选项，离线通过 `--solver` 选择；旧 MANO 录制需要原始 EMF 和匹配的 Moxian 标定重建源 mesh。依赖、接口、命令、容差和评估见 [Moxian 触觉 SLSQP](docs/moxian_wuji_contact_slsqp.md)。

Moxian 源默认使用 `moxian_studio_native_qp_v3_calibration_seeds`，保留当前29自由度和原有标定，迭代 FK／Jacobian／有界小 QP 在 C++ 中完成；不是官方固定21自由度二进制。构建入口为 `.venv-moxian-ik/bin/python scripts/setup_moxian_studio_qp.py`。数值一致性、真实输入延迟、接触录制与后端切换验证见 [原生源与实时验证](output/reports/moxian_wuji_realtime_native_20261113/REPORT.md)。 错误分支恢复与876帧回放见 [源IK修复验证](output/reports/moxian_source_branch_fix_20261116/README.md)。 六动作标定会自动生成、保存并重载校验源IK分支初值，见 [标定初值与旧骨架清理](output/reports/moxian_calibration_seeds_20261116/README.md)。

保留的 CEM 后端为 `card_wuji2_v10_paper_objective_solver`：默认按 CARD 式(2)、(3)、(10)–(13)与表5实现 objective 和 CEM，保留现有 Robokit 初值、Moxian SKN／MANO 源适配及在线／离线预处理。两后端共享语义指腹、胶囊和物体采样几何；默认没有对指推断、源动作一致性项或网格穿透软代价。

默认 `cem_backend='robokit'` 使用 cost-only Task、PopulationSolver／GaussianSampler 和完整 CUDA Graph；`torch` 提供独立参考。固定1024×16、51个有限精英、标准化softmax温度0.1及固定退火，不拟合方差。CEM从上一可行解与当前初值的等权均值开始，速度限制相对上一输出生效；首帧没有初值附近的局部搜索边界。旋转在连续轴角图中直接采样和加权。上一解重新评分，返回全部已评分候选中的可行最优者，不使用局部源动作采样或256候选截断。

掌指胶囊6 mm间距、伪物体和实测物体3 mm穿透门槛均为硬约束；指间胶囊只有软代价。原生网格0.5 mm检查仅为诊断，不改变论文可行性或历史。全部不可行时明确报告，不通过违约优化或历史回退制造可行解。接触优先级按当前帧最大压力归一化；所有手指保留style。保留的显式实验配置单独标记。详见 [CARD objective 与 CEM](docs/moxian_robokit_card.md)。

当前 `material_uv_rows_v2` 仍是解剖插值，460个SID均有三角面／重心坐标并不代表完成了实物传感器注册；Wuji2上的SID也是可视化用的虚拟投影，默认接触目标函数不把它当成逐税元实测对应。应先核验个性化MANO、手套安装和接触区域，再评价CEM。论文与本地实现核对见 [本地验证报告](output/reports/card_paper_alignment_20261116_hep154uy/REPORT.md)。

实测税元注册入口为 `scripts/register_moxian_contacts.py`：在固定MANO上用独立参考接触点／法向拟合固定面及重心坐标，每个SID至少2帧训练、2帧留出，最大位置误差2 mm、法向误差30°。通过后可原子导入或恢复，未覆盖SID保留未注册状态。手型改变时材料坐标保留、实测验收失效；手侧、权重、设备或SID分区不兼容时不沿用。`source_contact_geometry` 将源税元注册、个性化手型和EMF安装状态与CEM可行性分开。操作和数据格式见 [接触注册说明](docs/moxian_contact_registration.md)。

触觉方向实验使用 `contact_force.py` 的压力加权法向合力，明确标为相对ADC响应、未测剪切、未标定牛顿。默认仍是CARD的 `contact_direction_model='card_alignment'`。可选 `coulomb_force` 在每个CEM候选上解析拟合满足 `Fn>=0, ||Ft||<=mu*Fn` 的相对力；`normal_force_weight` 控制法向分量拟合。这是给定几何下的力兼容性实验，不是机器人实际力反馈控制，也不修改源MANO或推断对指。离线对照入口为 `scripts/benchmark_moxian_contact_force.py`，报告入口为 `scripts/report_moxian_contact_force.py`；须同时检查原始方向兼容、接触点到位及穿透，不能把投影力天然满足锥约束当作接触成功。

`collision.self_mesh` 公布原生网格逐连杆穿透诊断，`mesh_diagnostic_exceeded` 表示超过0.5 mm诊断阈值；`object_max_penetration_mm` 单独记录物体穿透。论文可行性由掌指胶囊间距和物体SDF硬约束决定，因此可行结果仍可能存在网格穿透。只检查候选端点，不检查帧间扫掠轨迹。

CARD 的真实物体网格与推测凸包均执行3 mm侵入硬门槛，`collision.object_hard_gate_enabled=True`；没有就绪物体场时不产生物体代价。

Wuji2 对照在下一帧求解期间保留上一帧完整画面，源数据延迟单独提示。输入心跳超时为1.5秒；源与结果超时为至少5秒、按单帧耗时3倍调整且最多15秒。实际断流、重启、错误或等待映射时清除接触指示并标注保留画面，完整新结果到达后替换。新结果仍须通过碰撞验收，不对关节做未经验证的插值。

MANO patch CEM：

```bash
python scripts/solve_mano_comfree.py --help
python scripts/retarget_mano_comfree_campaign.py --help
```

GraspADMM：

```bash
python3.11 -m venv .venv-graspadmm
.venv-graspadmm/bin/python -m pip install -r requirements.txt
.venv-graspadmm/bin/python run.py --config configs/graspadmm.yaml
```

YAML 的 `solver.type` 接受 `cem`、`graspadmm` 或 `contact_cem`；算法参数分别位于对应模块的 `DEFAULTS`。原有人体与 Dex 任务沿用 CEM。外部输入数据和私有模型需另行准备。

接触空间 CEM 基于随仓库迁入的 Robokit 原始 SimMPC、PopulationSolver／GaussianSampler、IK／LM 和 MjWarpSim，增加 Sharpa 转笔、翻币及 H1 jog 任务：GPU 搜索候选，独立 CPU MuJoCo 4 进程开启 speculative contacts 复核并闭环执行。场景和资产随仓库提供，支持完整任务、多种子对比及冻结 H1 参考轨迹的扰动跟踪。安装、运行、后端差异和严格验收见 [contact_cem 文档](docs/contact_cem.md)。实验是否成功以对应 `summary.json` 和报告为准，短时 smoke run 不计作成功。

当前设备标定保存在本机 `output/moxian/`，保留有效官方标定、当前六动作标定采集和当前手型；历史结果、标定副本、缓存及编译中间文件已清除。moxian 正在使用的三个虚拟环境保留，闲置 GraspADMM 环境按需由安装脚本重建。保留实现的注释和 docstring 不作批量删除。

骨架与 MANO 的[连续联合标定](docs/moxian_joint_calibration.md)复用八动作初始化，再采集活动范围及三轮对指；第三轮和末尾复测独立验收。固定几何、安装参数和个体骨架通过后整体应用，失败候选保留诊断。

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

仓库目录、依赖与清理边界见 [目录说明](docs/repository_layout.md)。

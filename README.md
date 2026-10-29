# MANO → Wuji Retarget

根据 MANO 手部参考与接触模板优化 Wuji 手部动作，跟踪物体轨迹。默认使用 **AdamW + 接触点探索 + ComFree**，并开启 viewer。

## 运行

需要 Linux、NVIDIA GPU/CUDA 驱动；viewer 需要桌面显示环境。

```bash
python -m pip install -r requirements.txt
python run.py                         # 默认示例，开启 viewer
python run.py --no-viewer             # 无界面运行
python run.py --duration 3.6 --seed 0  # 指定仿真时长与随机种子
```

viewer 中空格暂停/继续，`X` 切换透明显示。结果保存在 `outputs/<executor>/<case-name>/`；可用 `--case-name NAME` 命名。

```bash
python run.py --replay --case outputs/comfree/你的运行名称
python run.py --replay --case outputs/comfree/你的运行名称 --video result.mp4
```

## 可选分支

| 参数 | 功能 |
| --- | --- |
| `--adamw`（默认） | 接触点、hand action 与 dual 交替更新，每窗口默认一轮、一次 hand 更新 |
| `--contact-solver qp` | 将默认采样 LM 接触更新替换为 QP |
| `--no-contact-exploration` | 固定物体模板，保留 ADMM 一致性项和 dual |
| `--no-admm` | 固定模板，仅更新 hand action，关闭 ADMM |
| `--fa` | L-BFGS 基线 |
| `--sqp` | 简化接触模型的 SQP-RTI，需额外配置 acados；当前尚未通过稳定抓取验证 |
| `--executor mujoco` | 用 MuJoCo 执行求得的动作，默认执行器为 ComFree |
| `--noise` | 扰动 MANO 手指，通过 SDF 最近投影重新生成接触模板和手部参考 |

```bash
python run.py --noise --noise-finger thumb --noise-amplitude 0.6
python run.py --noise --noise-amplitude 0.6 --no-admm
python run.py --contact-solver qp
```

噪声默认全手、0.35 rad；`--noise-finger` 可选 `all/thumb/index/middle/ring/pinky`。`--noise-seed N` 使用可复现的随机关节偏移。生成噪声需要 Torch、chumpy、manopth 和本地 MANO 模型；本机已配置，其他机器可用 `--noise-mano-models DIR --noise-manopth DIR` 指定资源。噪声输入保存在 `outputs/noise_inputs/`，回放自动读取对应输入。

手部参考由扰动 MANO 的掌部与骨链直接做 IK 生成，不读取原可行 Wuji 轨迹；扰动从第一帧生效。原接触时序与语义对应仍保留。

其他参数见 `python run.py --help`。SQP 的 acados 路径可用 `ACADOS_SOURCE_DIR` 指定。

## 独立静态抓取与 primitive 尺寸实验

`--templates` 支持 1–33 的数字编号，例如 `--templates 6`；多个类型写作 `--templates 1 6 27`，全部类型使用 `--templates all`。完整模板名称仍兼容。

`run_grasp.py` 单独测试不同物体、抓取类型及尺寸下的抓取。输入由 Dexonomy 初始化构建。GPU 默认使用 CARD 退火 CEM，搜索手腕与 20 个手指关节；接触、style、平滑、碰撞四组 cost 之外，默认增加轻量柔顺接触动力学预测的物体 pose cost，并联合接触吸引、手型和平滑成本的梯度优化手姿及 20 个独立压紧偏移。每步重查允许皮肤区域的接触，1 mm 激活范围之外不产生接触力。物体目标始终是初始 pose；默认预测 4×0.1 s，接触力由关节刚度、控制压紧及重力驱动的耦合接触求解计算，不要求外部力输入，也不额外增加受力平衡目标。CEM 默认预算为 1024 个候选、16 轮、5% 精英；之后进行 32 次梯度细化及 GPU 回溯（先压紧控制、再联合姿态），仅接受降低完整目标且满足原硬约束的候选。`--no-explicit-pose` 恢复纯几何 CARD 对照。`--grasp-optimizer adamw` 保留公共动力学求解器。导出静态优化姿态、单独的预测末端、所选接触点、模型力与分项代价；独立审计和真实重力回放继续单独验收。公式、参数、模型适配与限制见 [CEM 抓取优化说明](docs/grasp_cem.md)。

抓取默认改为 **WUJI Hand 2.2 右手**（`--hand-model wuji2`），采用官方关节变换、限位、惯量、位置执行器和五个传感器指腹网格；一代手可用 `--hand-model wuji1`。二代手资产及固定版本见 [assets/wuji2/source.json](assets/wuji2/source.json)，重新生成命令为 `python scripts/build_wuji2_assets.py`（离线）。手腕保留原有六自由度接口，手指四关节按真实二代手运动学重新初始化，8880 个接触/穿透采样包含固定指腹和安装件。二代手扁平 MCP 外壳采用完整网格包围盒分离证书辅助胶囊硬检查，软碰撞代价保持胶囊形式。已有一代手结果继续按归档 XML 回放，不把旧关节角套到二代手上。

默认开启实时可视化和图片保存；GPU 模式仅做静态检查，物理保持测试使用 `--legacy-serial --drop-test`。Dexonomy 原始 sampler/matcher 仍输出物体 pose、接触点和手姿；这些输出单独归档，不再次投影。Shadow → Wuji 的模板拟合增加关节余量先验，候选选择增加执行器可行性检查，并排除接触距锐边小于 0.1 mm 的候选，避免端盖/侧面法向歧义；保留匹配目标排序，距锐边的余量排序作为失败后的备选。原始初始化的计算方法保持不变；候选筛选是本地扩展，不能称作完整复现上游生成流水线。

AdamW 输入默认启用 `--load-conditioning`（CEM 保留初始化侧检查，但不使用载荷 warm start 或关闭参考）：CUDA 静态载荷 QP 同时考虑物体重量、手自身重力、接触摩擦锥、真实位置指令范围和力矩限制，计算 `tau = gravity_bias + J.T @ force`，再构造 `command = q + tau/kp`。原手姿作为初始状态和恒定状态参考，载荷指令通过 previous action 和 warm correction 输入公共求解器，不能把载荷指令误当作已执行的关节角。这是输入构建，不改变公共 ADMM/AdamW 的损失、权重、更新规则或预算。`input_00/load_conditioning.json` 和 `load_input.npz` 保存力、约束残差、指令及 warm start。

以下准备、稳定和释放流程属于 `--legacy-serial --drop-test`：准备阶段暂时固定物体，在完整 9.81 重力下从手自身重力补偿指令平滑增加到载荷指令，默认斜坡 0.3 s（`--preload-ramp-seconds`），建立接触共 1 s（`--formation-seconds`）；随后固定指令，至少稳定 0.5 s。稳定须连续 0.2 s 满足速度阈值，或满足位置平台条件；硬接触小幅关节抖动还需通过全手 query 包络运动不超过 0.2 mm 的保守界限。3 s 内不能稳定则记录 `not_settled`，不运行释放测试。`--settle-mode free` 可禁用准备阶段的物体夹具。夹具只用于建立抓取，释放后完全自由，没有地面、物体位置修正或外部支撑。

默认 `--hold-controller mpc` 在释放后使用与 `run.py` 共用的 `forceaware.workflow.RollingController`：每 30 ms 从实际执行状态求解、执行三条 10 ms 控制，平移 warm 并反馈最后实际控制。初始载荷 warm 从实际稳定状态和保持指令恢复，清理过期求解历史。公共配置仍是 1 轮 ADMM、1 次 AdamW、8 个并行候选。物理执行默认使用 CUDA Graph 重放相同子步，`--no-physics-graph` 可关闭做逐位对照；规划求解器不变。物体目标恒定，无抬升轨迹。物理执行使用公共 ComFree CUDA 硬接触后端，默认 0.25 ms 步长以稳定细圆柱的硬接触；求解时间网格仍为 10 ms，二者分别计时。

以稳定时的手坐标系为基准，重力方向相对移动达到 30 mm 记为 `dropped`，其他方向达到 30 mm 记为 `escaped`，首次越界后保持失败记录。`drop_test.retained` 只表示两秒受控保持通过；`formed_geometry` 独立检查实际稳定状态的接触、法向、穿透、自碰撞和几何力闭合；`verified_grasp` 要求二者均通过。失败时默认最多再构建两次输入（`--initialization-retries 2`）：先在相同采样上切换匹配/锐边排序，再更换种子重采样；`--init-selection` 指定首选策略。所有失败目录、采样 RNG 状态、策略和尝试计数均保留。不能把重试后的通过当作首次成功率。

摩擦、solref/solimp 默认继承 `run.py` 的 `data/example/sequence/scene/wuji_right/sugar_box_sdf.xml`。新载荷初始化默认 `impratio=10`，以减小少量指尖接触的切向正则化滑移；它是摩擦/法向阻抗比例，不等于物体材料刚度。单独调整它无法补足不存在的夹持力。`--physics-reference XML` 或显式 `--impratio` 优先；回放默认保留归档参数，覆盖后另存 `drop_test_scene.xml`。规划和执行使用相同接触输入，报告保存实际编译参数。`--no-load-conditioning` 对照路径保留旧闭合参考和参考 XML 的 `impratio`；`--closing-depth-mm`/`--closing-seconds` 仅控制该旧参考，原始表面接触不变。

`run.py` 虽没有实际力幅值下限，但位置伺服、重力接触动力学与连续闭合参考可以隐式产生夹持力。公共目标还有压入深度及归一化 wrench 方向项；归一化的几何力闭合并不证明已实现牛顿级预载。当前修复的控制对照、物理结果、耗时与限制见 [夹持力修复报告](reports/dexonomy/grasp_repair_20261020.md)。

```bash
git clone --depth 1 https://github.com/JYChen18/Dexonomy.git third_party/Dexonomy
python run_grasp.py --list-types

# 固定圆柱高度，改变半径（变粗/变细）
python run_grasp.py --primitives --templates 6 \
  --primitive cylinder --radius-scales .75 1 1.25 --height-scales 1 --render

# 独立改变圆柱半径和高度，执行参数组合
python run_grasp.py --primitives --templates 1 3 \
  --primitive cylinder --radius-scales .75 1 1.25 --height-scales .75 1 1.25

# 球体大小；也支持 disk、cuboid、plate 的对应尺寸扫描
python run_grasp.py --primitives --templates 27 --radius-scales .75 1 1.25

# 自己的网格，单位 m；毫米网格增加 --scale .001
python run_grasp.py --object /path/to/object.obj /path/to/another.stl \
  --templates 1 14
```

primitive 默认物理尺寸来自本项目显式配置，可用 `--radius`、`--height` 覆盖，单位为米。圆柱的两个缩放参数分别控制半径和完整高度，disk 的 height 是厚度，cuboid/plate 的 radius 控制两个平面尺寸而 height 控制厚度，sphere 只有 radius 生效。`--templates all` 枚举全部编号模板，计算量会明显增加。

**来源区别：** Dexonomy 没有逐类型 primitive 硬定义；[其模板](https://github.com/JYChen18/Dexonomy/tree/main/assets/hand/shadow/raw_anno) 定义手姿态和接触。本项目依据它引用的 [GRASP taxonomy](https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf) 及[形状属性目录](https://www.cs.cmu.edu/~jialiu1/database.html) 建立可审计的测试关联，见 `retarget/grasp_primitives.py`。具体半径/高度是本项目的实验设置，未冒充上游标注。#19 剪刀、#21 双筷保留为特殊类别，primitive 模式写入 `skipped.json`，需要专门网格或多物体模型，不会自动替换成一根圆柱。

初始化直接调用外部 checkout 中的 [Dexonomy 原始代码](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/op/gen_init.py)：`sample_init_pose` 采样物体 pose，`HandTemplateLoader._load_data` 建立手接触坐标系，`HandObjMatcher.forward` 使用原始 Torch Adam 与 Warp 表面查询匹配 pose 和物体接触。每次原始 matcher 调用中手指关节固定。安装额外依赖：`python -m pip install -r requirements-grasp.txt`，并使用支持本机 GPU 的 CUDA PyTorch；`--init-device cuda:0` 选择设备。

Wuji 输入模板先做骨链 IK，再由 `retarget/grasp_template.py` 在 CUDA 上联合拟合手根与关节，让真实 query 表面点的位置和法向保持 Shadow 的接触布局，避免拇指越过四指。圆柱/球体扫描按直径调整相向开口。第一次原始初始化产生物体表面目标后，最多进行 `--template-refinements 2` 次 GPU 手姿/实际表面点细化，每次再调用原始 matcher；达到初始化有效且接触间隙不超过 2 mm 时提前结束。最终传入 ADMM 的 pose、物体接触仍是最后一次原始 matcher 的未改动输出；拟合失败则保留较好的上一组输入。`--template-refinements 0` 仅保留第一阶段接触布局拟合，`--no-template-fit` 可复现旧骨链映射作对照。这些手模板适配是本项目新增的输入处理，不是上游模板进化或额外 ADMM。输入网格在 45° 锐边处分离法向，避免圆柱端盖污染侧面插值法向；三角形几何保持不变。

`--init-points`、`--init-inplane`、`--init-iterations` 控制原始初始化预算，默认分别为 24、8、60；原上游默认预算是 1024、8、100，可显式指定。所有候选都由原始 matcher 批量优化，旧 `--init-refine` 已移除。优先选择同时满足上游逐点 loss 阈值及 GPU 全手穿透阈值的原始结果，再按匹配损失排序；没有修改原始 pose 或重新投影接触。没有合格初值时仍保留候选供公共求解器细化/诊断，明确记录 `initialization_valid=false`，终端打印失败，不能当作有效初始化。这里核对了上游 `_loss_filter` 的同一不等式和 YAML 阈值，不运行完整 `InitFilter`、无效样本补齐或模板进化。

`retarget/grasp_initialization.py:from_dexonomy_result` 接收原始 `grasp_pose`、`grasp_qpos`、`obj_pose`、`obj_cp_w`、`obj_cn_w`。它只将世界系变换为物体固定系，并把内法向取反为外法向，不再次投影接触点。`retarget/grasp_runtime.py` 将其写成公共求解器的闭合/保持手参考、静止物体参考和接触指导。原始接触进入 `query_position_object`，由公共代码初始化 ADMM 的 c/z；手姿态作为初始状态和参考。公共求解器没有显式的模板法向对齐项；Dexonomy 的法向损失只在初始化中生效，不能声称它继续进入了 AdamW。结果报告保存初始和求解后的逐点法向残差、夹角及模板接触面的有向间距。物体具有自由关节，初始速度为零，没有地面；目标是抵抗重力并保持该物体 pose。

求解与 `run.py --adamw` 使用同一工厂、同一配置转换、同一目标函数、ComFree 前向/解析伴随、GPU sampled LM 和 scaled dual 更新。默认同为 1 轮 ADMM、每轮 1 次 AdamW、8 个并行 solver starts、8 个接触候选、3 次 LM、5 mm 接触搜索半径。`--starts` 单独控制 Dexonomy 初值数量，默认 1；`--multistarts` 控制公共求解器并行候选。`--rounds`/`--hand-iterations` 是 `--contact-rounds`/`--iterations` 的兼容别名。旧 CPU 静态求解器及 `--hand-polish`、`--hand-contact-radius-mm` 已移除。

输出与检查使用公共求解器的 `plan.next_qpos`，即下一执行间隔末端（默认 0.01 s）的预测状态，接触 x/z/dual 与该时刻一致。0.3 s 长预测末端另外保存，不能当成当前抓取 pose。用 CUDA 网格 BVH 检查实际接触与全手 query surface 穿透，并用 CUDA float64 原始/对偶 LP 检查 `rank(G)=6` 及严格正的归一化 `Gα=0` 证书。LP 未收敛不会报告力闭合。`force_closure` 与 `feasible` 分开报告；后者还要求每个标注手指/掌部组有接触，以及物体穿透、指间胶囊自碰撞在阈值内。自碰撞几何与公共 GPU 目标使用同一组胶囊；分组采用 Dexonomy 手指/掌部语义。摩擦点模型无扭转力矩，两个 pinch 点不能构成完整空间力闭合。单次预算不足时会明确报告失败。

默认结果位于 `outputs/grasps/<name>/grasps.npy`，通过固定结果目录自动发现。文件仅含数值：IK/优化后两个阶段各 20 个手指关节角、物体在真实 wrist（`right_palm_link`）坐标系下的 `xyz + quaternion(wxyz)`，以及用于识别类别、形状、尺寸和显示审计状态的数值字段。单位为米、弧度。每个批次不再保存 XML、OBJ/STL、接触点历史或逐轮预测；手模型从共享 `assets/wuji*` 读取，primitive 按尺寸生成，自定义物体作为共享输入资产仅缓存一次。求解器中间文件放在临时目录，在独立 FK/导出/力闭合审计完成后清理；无需 pickle，NPY 支持内存映射快速读取。最新 15 个样本连同 IK 共 9913 字节。

需要完整诊断、重新精化或动力学回放时显式加 `--archive-results`，保留旧格式的 `summary.json`、`grasp.npz`、`contacts.npz`、场景与模型输入。旧归档继续可用；`scripts/refine_grasp_results.py` 等完整归档比较工具仍读取此格式。精简结果不保存保持指令或速度，不将其当作原动力学回放数据。

Viser 浏览器可视化通过 [configs/grasp_viewer.yaml](configs/grasp_viewer.yaml) 配置：

```bash
python scripts/view_grasp_results.py --config configs/grasp_viewer.yaml
# 无浏览器检查筛选后的所有 IK / 优化姿态：
python scripts/view_grasp_results.py --config configs/grasp_viewer.yaml --check-all
```

默认打开 `http://127.0.0.1:8081`。YAML 只需设置 `categories: [2, 6, 13, 14, 17]`、`methods: [cem_explicit]`、`hands: [wuji2]`；各项可写 `all`。结果从固定目录 `outputs/grasp_results/<方法>/<手型>/<批次>/` 自动发现，不需要 `results` 或 `path` 配置。方法名为 `cem_explicit`、`cem`、`adamw`，手型名为 `wuji1`、`wuji2`；加载时核对保存的真实手型。`run_grasp.py` 及调用它的批量脚本会自动登记新批次并更新 `latest` 链接，原结果保留在各自的输出目录。已有一代、二代手及最新 15 样本批次已接入固定目录。

每种方法/手型自动选最后登记或放入目录的、已保存结果的批次；新批次尚未产出时继续显示上一批，有结果后刷新页面即可读取新批次。不同批次的类别和尺寸不会混在一起，侧栏信息会标明实际批次。批次目录只需一个 `grasps.npy`；旧归档仍兼容。`python scripts/compact_grasp_results.py` 可把已登记的旧归档转换为数值记录，原归档保留在原目录。

侧栏的“结果阶段 / Stage”可切换“初始化 IK”和“优化后结果”。精简数据在共同的 wrist 坐标系下显示，两阶段保留同一视角；优化验收指标只在优化结果视图展示。“尺寸”使用带实际尺寸刻度的 bar，拖动会吸附到已求解尺寸，不插值手姿；只有一个尺寸时禁用滑条。YAML 的 `stage: ik` 或 `stage: optimized` 决定初始阶段；修改筛选条件后刷新网页即可加载。`server.host/port` 改动需要重启服务。重力播放仅适用于保留指令和速度的完整归档，精简记录与 IK 视图禁用该按钮。

批量求解可指定 `python scripts/run_grasp_batch.py --run-dir outputs/grasp_batch --templates 2 6 13 14 17 --primary-shape-only --radius-scales .75 1 1.25 --iterations 16`。`--primary-shape-only` 为每类只取第一种关联形状，以上正好生成 15 个样本；省略时会覆盖每类的所有关联形状。进度保存在批次目录的 `status.json`。

```bash
python run_grasp.py --replay outputs/grasps/批次名/物体目录/抓取类型 --viewer
# 默认对保存的抓取重新做稳定/重力测试；仅看静态结果：
python run_grasp.py --replay outputs/grasps/批次名/物体目录/抓取类型 --no-drop-test
# 对比初始化或长预测，在同一条回放命令中增加：
# --stage initial
# --stage prediction
# --stage formed    # 实际稳定后的手与物体
```

逐 case 的 `drop_test.json.phase_timings` 分别记录 preload/settling/gravity 的仿真秒数、墙钟秒数及 solver/executor/diagnostic/viewer 分项；控制台同步打印。实时 viewer 每两秒显示累计仿真时间。逐 case 的 `hold_windows.json` 保存逐窗口耗时及 prepare/contact/hand 分项；`drop_test.npz` 保存建立接触、稳定和重力阶段的 qpos/qvel、时间、阶段标记（−1 建立接触、0 稳定、1 重力测试）、逐步实际控制、接触力、关节扭矩和受力手指；`drop_test.json` 保存稳定判定、掉落时间、相对位移及独立的物理 setup/仿真/视频耗时。视频为 `drop_test.mp4`，末帧为 `drop_test_final.png`；`formed_grasp.npz` 和 `formed.png` 是实际稳定后的状态。重新测试已有抓取会同步更新其 `result.json` 和 batch summary；`grasp.npz` 的规划结果保留。缺少旧控制记录的历史结果采用保存手姿作为位置指令，并在报告中注明。

0.25 半径缩放的初值失败、锐边过滤及 GPU 执行提速见 [阶段耗时与细圆柱诊断](reports/dexonomy/grasp_scale025_20261020.md)。

模板 1 的圆柱稠密包络另用 CUDA 输入拟合处理全部指节/掌部接触、端面和指间碰撞，并保留拟合姿态供原始 matcher 评估；没有修改公共 ADMM。`--templates 1 --primitive cylinder --radius-scales .75 2` 的实测结果为：.75 倍（直径 52.5 mm）通过力闭合及 2 秒重力验证；2 倍（直径 140 mm）尚未找到可行初值，报告 `initialization_failed`，跳过无效物理阶段。失败原因、接触合力分离证书及具体限制见 [模板 1 修复报告](reports/dexonomy/template1_20261020.md)。

运行验证与限制见 [静态 primitive 抓取报告](reports/dexonomy/README.md)。

## 更换物体

人形运动与人形双手操作另见 [SPIDER 人形 ADMM 说明](data/spider/README.md)。`python run_humanoid.py` 默认在本机引擎上复用固定版本 SPIDER 的搬箱 cost、参考插值、MPC 和 MJWarp 源码，使用动作副本一致性 ADMM。底层引擎版本未与上游锁定环境对齐，且关闭了默认初始化预检，因此不能称为“完整环境相同、只换 solver”；证据见 [环境复核报告](reports/humanoid/SPIDER_ENVIRONMENT_AUDIT_20261029.md)。原接触点 ADMM 改为显式 `--cost spider-contact`；`python run_humanoid.py --cost spider-contact --task humanoid --frames 30` 运行本地 AMASS G1 参考实验。

人形入口现在默认使用 **SPIDER 位姿 cost + 与 `run.py` 共用的接触采样/ComFree 力 LM 和 AdamW**。根部、关节与物体位姿进入同一个加权 L2 范数；箱体位姿跟踪已启用。物体接触点在参考表面点周围 5 mm 内选择，action 通过 MJWarp 接触动力学与 ADMM 一致性更新，每窗口顺序为接触点 → AdamW action → dual。预算保持 1 轮 ADMM、2 次 action 更新、学习率 0.3。当前完整跑步/搬箱回放仍会失衡，尚不能作为稳定控制配置；实现、测试与完整结果见 [SPIDER cost + AdamW 验证](reports/humanoid/SPIDER_COST_ADAMW.md)。

2026-10-28：人形预测与执行已统一为 **MJWarp implicitfast**，移除了人形 native ComFree 动力学、Euler/PD 修正和动力学伴随。AdamW 使用 MJWarp 中心差分，默认迭代预算保持不变；运行时间与原解析伴随不同。实现、测试及历史实验边界见 [MJWarp 后端说明](reports/humanoid/MJWARP_BACKEND.md)。

`--cost legacy` 保留此前的关键点、balance/upright、动态 wrench 等目标，也用于其他实验 action solver，例如 `--cost legacy --action-solver acados-gn`。旧目标的结果见 [此前修改验证](reports/humanoid/TASK_FIX.md)。

使用另一套已准备好的实验目录：

```bash
python run.py --experiment data/my_object
```

参考 `data/example/` 准备以下内容：

```text
data/my_object/
  forceaware.yaml
  demonstration_visual.npz           # 与新轨迹对应的 MANO 可视化
  contactaware/                      # Wuji 参考、物体轨迹、接触指导及配置
  sequence/
    mano_raw/meta.json               # {"object": "005_tomato_soup_can"}
    model/                          # 新物体网格、纹理
    scene/wuji_right/
      tomato_soup_can_sdf.xml        # ComFree 场景
      tomato_soup_can_mesh.xml       # viewer / MuJoCo 场景
```

场景前缀由 `meta.json` 的 `object` 去掉数字前缀得到。更换网格时，同步更新 XML 的质量、惯量、碰撞几何及资产路径，并保留 Wuji 手部结构和 `obj`、`obj_joint`、`obj_collision` 等命名。`contactaware/` 中的手/物体参考与模板必须对应新物体，坐标系及数组格式沿用示例；仅替换 mesh 不会自动生成抓取轨迹。使用 `--noise` 还需提供匹配的原始 MANO 参数和语义顶点对应。

[data/dexycb/](data/dexycb/README.md) 包含五种物体的原始 MANO 数据，尚需准备为上述实验目录。代码与资产来源及许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。

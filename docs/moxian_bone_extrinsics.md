# Moxian 静态 17 动作：骨长与 EMF 外参

> 历史方案：当前默认已改为 [Moxian原生六动作标定](moxian_six_calibration.md)。本页的25自由度、17动作、Moxian objective和独立复测门槛只用于解释旧记录。离线重算旧记录时须显式传入 `--method bone_extrinsics`。

旧方法为 `bone_extrinsics`，模型版本为 `moxian_bone_extrinsics_v1`。
原八动作官方初始化保留。新流程需要重新采集静态动作；旧连续记录、宽厚及
Delta 候选仅用于显式历史对比，不作为新拟合的形状或姿态初始化。

## 参数与 mesh

只有 51 个共享参数：每指三段长度共 15 维、五个 RX 安装刚体变换共 30 维、
TX→wrist 全局刚体变换 6 维。拇指三段为 CMC→MCP、MCP→IP、IP→tip；
其余手指为 MCP→PIP、PIP→DIP、DIP→tip。其余掌骨长度、根位置、关节轴、
宽厚、自由顶点位移、SID 材料坐标和 skin weights 固定。

Moxian 原生 25 个关节自由度保留。离线每个保持姿态都有自己的 25 维辅助角度，
它们不成为网络、不进入在线模型。新初始化从官方八动作观测重新计算，不能继承
上一套个体模型在采集时输出的角度。

原始 SKN 文件不修改。15 个长度同时进入 FK 和 `QnbotSurface.rest()`：
沿原生骨轴确定性拉伸静止皮肤，并同步更新绑定变换，再做 LBS。
这允许没有独立 tip 蒙皮骨的 Qnbot 末端皮肤随末节长度变化。
这不是独立 mesh 优化，也不是 Wuji GLB 的逐顶点复刻。

## 采集和 objective

`contact_static_v5` 保留 17 类动作，拟合采集、独立复测各一遍。第一遍完成后停止采集，等待用户点击“开始 EMF 与骨架拟合”。候选保存后，用户可单独开始独立复测；完整复测完成后点击验收，复用已拟合参数，不再次运行 L-BFGS。未完成复测不自动应用。第 17 类张开、并拢
分别保持，所以每遍有 18 个聚合观测。每帧先检查当前目标的静态姿态约束，
满足后才计入保持／采集；不要求往返运动、两次重复或轨迹极小值。
最近 20 帧五指 XYZ 的最大轴向方差
不超过 `0.0002 m²` 后，直接聚合最近20帧，无额外2秒保持；不稳定、无效数据或超过
250ms 断流会重置当前窗口。超时 120 秒暂停；已完成保持片段保留。

四个对指保留 Wuji 的 RX 间距 ≤50mm 条件，指腹距离与法向不作采集硬门槛。
其余动作以固定八动作参考判断区域接近：一般 ≤60mm、掌面 ≤40mm；
非掌面相向角 ≥45°，多指簇 ≥30°，掌面角度仅诊断。五指采用 RX 连通簇和收拢比例检查。
张开和并拢要求伸直距离比例 ≥0.45、四指离掌面角 ≤45°；相邻指方向角
分别要求 ≥6°、≤20°。这些是 Moxian 适配值。参数来源是
`calibration-contact-targets.json` 中 `contact_static_v5.pose_constraints`。

这些额外阈值是 Moxian 的粗姿态筛选，不冒充 Wuji 原来的全部动作约束，
也不代替最终毫米级接触验收。UI 显示 `constraints_ok` 和具体不满足的条件。

位置求均值，四元数采用 `Σqqᵀ` 的最大特征向量。原始有效帧、被接受的
采集区间和代表观测一并保存。八动作 open 观测提供平手约束。
训练观测估计噪声、选择接触材料点与多指连接，随后全部冻结；复测不参与这些步骤。

变换约定：`T_wrist_TX` 把 TX 坐标映射到 wrist，`T_distal_RX` 把 RX 映射到
对应末端骨骼。长度、平移保存为米，旋转保存为矩阵。

```text
预测 RX(TX系) = inverse(T_wrist_TX) · FK_wrist_distal(L,q) · T_distal_RX
cost = EMF位置/朝向的归一化平方误差
     + 固定材料点的接触位置/相反法向约束
     + 平手平面/伸直姿态约束
     + 骨长、比例、外参及弱姿态零空间先验
```

位置 sigma 从训练保持片段估计，限制在 1–10mm；旋转 sigma 为 0.05–0.3rad。
接触尺度 3mm、法向 15°；骨长及比例先验 10%；外参先验 10mm、0.15rad。
骨长 log 增量范围 ±0.15，RX 平移 ±20mm、旋转 ±0.3rad；
全局平移 ±20mm、旋转 ±0.2rad。关节角使用原生限位。

## 直接复用 Wuji 原生 solver

默认**不调用另行安装的 Python nlopt 包**。`wuji_calibration_solver.py`
直接调用此前分析的 Wuji SDK 2026.7.1 ELF 内的 NLopt 2.9.1 `LD_LBFGS`，
即 calibration 调用的同一组原生创建、边界、objective、停止和 optimize 入口。
设置与 Wuji 标定一致：算法枚举 10、`ftol_rel=1e-10`、`maxeval=20000`。

该二进制未导出 NLopt 名称，因此 adapter 使用已恢复的 RVA，并且只允许
SHA256 为 `8041b5ee059de6e8aa44641f5e7f323d0997604ff908d2dc108d18976631300c`
的字节一致 ELF；不同版本拒绝执行。取消或 Python objective 异常先调用
原生 `nlopt_set_force_stop`，待 C 返回后再抛出原异常；始终调用原生 destroy。
加载不执行 PyO3 模块初始化，不创建设备或 SDK 会话。

```bash
.venv-contact/bin/python scripts/setup_wuji_calibration_solver.py
```

安装脚本优先使用已有固定版本 wheel，否则从其固定 PyPI URL 下载；同时校验
wheel 和 ELF SHA256。库及依赖解包到 `moxian_web/vendor/wuji-calibration-native/`。
也可用 `MOXIAN_WUJI_CALIBRATION_ELF` 指定该版本 ELF，需保留同目录结构下的依赖。
当前 adapter 支持 Linux x86_64。

**复用范围是原生数值 solver；objective 是 Moxian 的 25 自由度、17 动作适配。**
原 Wuji caliber objective 绑定其 21 自由度和内置动作，未取得可接受任意 Moxian
模型/动作的公开接口。报告明确记录 `native_solver_reused=true`、
`native_caliber_objective_reused=false`，不把自定义先验权重标为 Wuji 原始数值。

离线重算不会发布模型：

```bash
.venv-contact/bin/python scripts/calibrate_qnbot_glove.py \
  --capture output/moxian/hand_model/<设备hash>/qnbot/contact-static-v5.json \
  --output output/moxian/debug/bone-extrinsics
```

## 实时应用与验收

`emf_qp.py` 逐指求解位姿残差的有界增量 QP，active-set＋Cholesky，
关节限位和速度界、阻尼、静止姿态及历史项。先最多 40 次；误差超过 5mm 或
5° 时从同一起点用两倍、四倍速度各重试 36 次。默认速度 `2π rad/s`。
这是依据 Wuji QP 结构在 Moxian 模型上的实现，和上述直接调用的离线原生
calibration solver 是两个组件。

模型内容先保存为不可变文件，再原子替换 `qnbot/active.json`。骨骼 worker 和
mesh worker 读取同一模型 hash；不一致时停止发布 mesh。第一次输入、250ms
断流、模型切换及回滚清空历史。官方八动作指纹与当前个体 hash 分开管理。

独立复测逐帧用原始 EMF 求实时 QP，不传动作目标、不使用离线辅助姿态，
也不根据动作标签重置 IK。动作标签仅用于接触评分。全部复测帧都检查原始
曲面的穿透；接触评分只针对接受的保持区间。

接触均值 ≤3mm、最大 ≤5mm、法向 ≤15°、穿透 ≤2mm；每指 EMF RMSE ≤15mm，
平均朝向 ≤15°，相对基线位置退化不得超过 5%+0.5mm、朝向不得超过 2°；
展开后的目标材料点距离不得小于 2mm。优化未收敛、取消或完整曲面验收不可用
均不自动应用。没有旧版训练的 60 秒硬门槛。

目前的合成检查不等于真人戴手套复测；实际效果和实时延迟须由新静态采集验证。


静态采集宽松门槛：四种对指只要求原始 RX 距离 ≤50mm（本机 Wuji 2026.7.1 恢复值），指腹距离与法向仅显示诊断。其余 Moxian 动作采用区域距离 ≤60mm、掌面目标 ≤40mm、相向角 ≥45°（多指簇 ≥30°）；伸直比例 ≥0.45、离掌面角 ≤45°、并拢 ≤20°、张开 ≥6°。后者为 Moxian 宽松默认值，并非 Wuji 原值。姿态通过后界面显示绿色文字，显示稳定窗口进度；最终独立复测验收标准不变。


五指采集只检查五个 RX 能以 ≤80mm 的边连成一簇，并检查四指平均伸直比例 ≤0.85 以排除平掌；不要求小指直接接触拇指，也不要求五指法向同时相向。这是采集的宽松识别，最终接触复测标准不变。首次启动后自动切换并采集后续动作；切换时至少一个当前有效手指位移5mm或旋转0.1rad后才开始新的稳定窗口，避免同一姿态连续通过多个动作。

指腹接触掌面/指根区域的动作取消法向角度采集门槛；掌面法向仅作为诊断显示，采集仍检查区域接近距离及20帧稳定性。


## 目标函数执行性能

原生 Wuji L-BFGS 的 objective 默认使用 float64 CUDA Graph 重放。全部姿态和静止几何共用一次 FK，接触点使用等价的批量 LBS 仿射混合，损失／梯度／诊断一次回传 CPU；无需 torch.compile 编译。CUDA 不可用或无法捕获时保留 eager 路径，并在报告中记录执行后端及原因。求解器、目标项、边界与停止精度不变。

原始观测只读，拟合时复制聚合观测而不再深拷贝整个原始记录；原始记录与拟合分区的 SHA256 共用一次逐行 JSON 编码，输出与原算法相同。

本机同一份采集记录的离线完整拟合：约77.1秒降为13.4秒，其中优化迭代约5秒；单次目标/梯度从约99ms降至8.6ms。新旧均以 ftol_rel=1e-10 正常收敛。运行时间受设备负载、动作数量和观测影响；此时间不包含后续独立复测验收。完整结果见 `output/reports/bone_fit_speed_20261112/summary.json`。基准只保存诊断候选，没有替换活动模型或已保存的用户候选。

可复现命令：

```bash
.venv-contact/bin/python scripts/benchmark_bone_extrinsics.py \
  --capture output/moxian/hand_model/<设备目录>/qnbot/motion-capture.json \
  --output output/reports/bone_fit_benchmark --backend cuda_graph --compare
```

界面只显示真实评估次数、目标值与评估速度；20000是评估上限，不代表完成百分比。

# Qnbot 分阶段最小二乘标定

方法名为 `least_squares`，只支持 `surface_backend='qnbot'`。这是现有三个训练方法之外的独立候选分支，不训练网络、不使用 Adam 或 EMA。现有分支和 `current.json` 不变。默认优化预算 600 秒，完整验证另计，60 秒性能目标只作诊断。

## 调用与结果

```bash
.venv-contact/bin/python scripts/calibrate_qnbot_least_squares.py \
  --device-home output/moxian/hand_model/348640923d9e2f3f \
  --output output/moxian/debug/qnbot-least-squares-20261111-v4 \
  --seconds 600
```

命令行先在隔离 SDK 解释器中，从匹配的原始初始化重建当前 IK 策略指纹；过期骨架必须从原始 EMF 完整回放，不能直接替换版本标签。`seed-policy.json` 保存实际策略及配置来源。默认输入设备根目录的 `motion-capture.json`；也可用 `--capture` 指定。输出目录必须为空，每次运行独立保存。完整候选原子注册到设备的 `qnbot/methods/least_squares/candidate.json`，网页“已训练权重 → 分阶段最小二乘标定 → 预览此权重”读取它。预览结束恢复原几何、安装、材料绑定及原有 Delta。该分支不会自动应用。

统一 Python 入口为 `moxian_web.mano_calibration.fit_capture(..., method='least_squares', surface_backend='qnbot', solver_options=...)`。原有 `training_options` 不用于该分支。界面提供单独的预算、每阶段最大迭代与阶段开关。

`solver_options` 默认值：`max_seconds=600`、`max_iterations=6`、`pose_iterations=2`、`shared_batch_size=96`、`chunk_size=48`、`relative_tolerance=1e-5`、`step_tolerance=1e-4`、`rank_rtol=1e-3`；阶段顺序固定为外参、尺寸、运动学、触觉、表面。取消在数据分块与迭代边界检查。

## 参数与数值方法

- 15 维尺寸：每指长、宽、厚，同指可动指节共享 log 尺度，掌骨固定。
- 36 维外参增量：五指安装位置／朝向和统一坐标配准。
- 18 维运动学：固定中指基准，其他四个指根各三维位移，三条拇指 CMC 轴各两个切向倾斜坐标。
- 60 维触觉注册：20 区域共享的两个重心坐标 log 比值与一个法向偏移。保持面片、SID 归属和重心坐标单纯形；法向偏移合并表示厚度与佩戴间隙。
- 最多 20 维表面系数：图 Laplacian 平滑区域法向基，投影去除与刚体、尺寸和运动学静态位移重复的方向；保留原始拓扑与蒙皮权重。

坐标总容量 149，不代表全部可以辨识。长度 log 增量范围 ±0.15、宽厚 ±0.1、指根 ±2 mm、轴倾斜坐标 ±5°；安装位置 ±20 mm、安装旋转坐标 ±0.3 rad、配准旋转坐标 ±0.2 rad、配准平移 ±20 mm。触觉法向偏移 ±2 mm，表面单系数 ±1 mm，并有整体幅度、平滑与边长约束。

PyTorch 前向自动微分计算 Jacobian，逐帧有界 LM 与共享参数 LM 交替。共享线性化采用按动作／窗口分层、迭代轮换的最多 96 个独立训练帧；接受与拒绝共享更新使用整个训练集的真实目标。姿态扫描处理全部有效训练帧；时序邻居跨计算分块保持，窗口、分区和大于 250 ms 的断流重置。

阶段开放前，测量 Jacobian 消去逐帧姿态方向，不把先验加入可辨识性信息。仅沿支持的右奇异子空间更新；在支持子空间内用 SLSQP 求解小规模凸二次子问题以满足坐标边界，再作全训练集线搜索；它不训练手模型，避免单个到界坐标阻塞所有方向或逐坐标裁剪引入零空间分量。报告局部噪声归一化线性诊断，不声称恢复唯一解剖参数。触觉与表面阶段缺少证据或有效秩时冻结。零偏、耦合、个人限位、压力到真实力的增益及时间偏移不作虚假辨识。

## 观测与留出

只用 `partition='train'` 且 `training_eligible` 的帧更新参数；初始化记录只用于外参／姿态参考，全部 `validation` 帧保留。噪声统计仅由训练轨迹的稳健高通统计和明确尺度下限得到。

接触正标签来自 25 ms 内的双方 patch 和 taxel 激活；只凭 `contact_confirmed`、动作名称或轨迹极小值不会产生等式。双方有效且均不激活时可提供负标签，单侧和未知观测跳过。虚拟 MCP 掌面没有直接 SID 证据时无法验证，不冒充真实触点。上游骨架只是 EMF 派生种子。当前历史记录没有物体位姿，物体 SDF 与真实力监督关闭。

损失包含鲁棒、噪声归一化的 EMF 位置／SO(3) 旋转，材料接触距离／相向法向、触觉 BCE、弱姿态先验与按时间戳的修正加速度。碰撞见证来自原始三角面；训练抽查最多三帧并刷新见证，验证检查所有留出帧的原始顶点和面心。目标接触对训练穿透容差 2 mm，其余对为零；最终最大穿透仍受 2 mm 门槛约束。这是离散曲面检查，不是连续三角形无相交证明。

分别导出标准模型固定前向、候选固定前向和候选离线 EMF 重建。离线验证的姿态求解关闭接触及动作观测掩码；标签只用于事后评价。逐帧姿态不进入候选或实时模型，实时效果由实际上游关节角固定前向决定。

几何验收沿用接触平均／最大 3／5 mm、法向 15°、穿透 2 mm、逐指 EMF RMSE 15 mm 及相对基线退化检查。缺失的接触覆盖明确记录为无法验证，不会因此放行；预算耗尽标为未收敛。`report.json` 包含逐指、逐动作、逐帧误差及秩诊断。

## 导出资产

每次运行保存 `candidate.json`、`report.json`、`REPORT.md`、`validation.json`、`hand_shape.json`、`kinematics.json`、`emf_extrinsics.json`、`tactile_map.json`、`contact_geometry.json`、`hand_mesh.obj`、原始三角面 `collision_mesh.obj`、`skin_weights.npy`、`surface_basis.npz`、`offline_poses.npz` 和真实迭代记录。手型明确为 Qnbot 参数，没有 MANO β。

命令行额外保存输入、当前模型和既有分支的前后 SHA-256 检查到 `integrity.json`；完整结果验证完成后才注册新候选。网页拓扑携带 `sid_normal_offset_m`，材料点按当前三角面法向评估，随 mesh 一起运动。

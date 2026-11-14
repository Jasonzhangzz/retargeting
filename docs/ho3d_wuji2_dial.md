# HO-3D → Wuji2：MJWarp 动态重定向

`scripts/retarget_ho3d_dial.py` 恢复 HO-3D train 的 MANO 与物体轨迹，并以实际 Wuji Hand 2.2 模型求解带自由物体的采样式控制优化。抓取建立、并行候选 rollout、闭环执行、IK 控制基线以及回放几何 FK 都在 CUDA MJWarp 上运行。参考姿态 IK 使用 CUDA Torch 的有界 LM。没有 CPU MuJoCo stepping 或 CPU 仿真回退；`MjModel` 用于模型编译，`MjData` 用于状态上传及序列化。官方 `mjw.put_data` 上传时会进行一次 CPU kinematic 初始化，这不参与轨迹动力学积分。

这是一套离线物理优化入口，未连接真实硬件。默认质量、摩擦及重力方向是假设，不能把当前指标当作实机成功率。

## 数据与坐标

- 只读取 `train.txt` 中的姿态标签。缺帧、无效姿态和物体切换均拆分片段，不补造连续帧。
- 用官方 MANO RIGHT v1.2 权重、实际 shape 和完整 48D 轴角恢复网格。`handTrans` 是 MANO 模型平移参数，不能直接作为 wrist 位置。
- 21 个关节采用 HO-3D 官方 FK 和指尖顶点 `[744,333,444,555,672]`，再转换为 wrist、thumb、index、middle、ring、pinky 顺序。
- 物体使用原始 YCB mesh 的局部原点，禁止独立居中。MANO 与物体应用同一个恒定刚体变换；OpenGL camera 转 Z-up，并让第一帧物体原点位于 `(0,0,0.3)`。
- 相机 Y 方向被假设为重力向上；帧率默认 30 Hz。数据未提供标定桌面，默认不添加地面，`--floor-z` 只供已知支撑面时显式设置。
- 接触与距离字段来自 `handVertContact` / `handVertDist`，保留有效性掩码。抓取阶段定义为至少两根手指各有至少三个接触顶点。旧版恢复 NPZ 可通过 `--mano` 使用原始 skinning weights 推导该阶段；标签缺失时不施加最低抓力目标。

```bash
.venv-contact/bin/python scripts/retarget_ho3d_dial.py prepare \
  --source /path/to/HO3D_v3.zip --mano /path/to/MANO_RIGHT.pkl \
  --output /path/to/recovered

.venv-contact/bin/python scripts/retarget_ho3d_dial.py solve \
  --motion /path/to/recovered/MC1_0000_0341.npz \
  --models /path/to/YCB/models --mano /path/to/MANO_RIGHT.pkl \
  --output output/my_dynamic_case --config configs/ho3d_wuji2_dial.json
```

`--models` 可指定一个物体 mesh，也可指定包含 `<objName>/textured_simple.obj` 的目录。准备脚本支持 ZIP 和解包目录，完成的 NPZ 与 provenance receipt 可断点跳过。

## 物理与目标

手部使用仓库原生 Wuji2 URDF 转换模型的关节约束、质量、惯性、PD 及有限驱动力。26 个受控关节包含腕部 6D 根关节与 20 个手指关节；物体是完全自由的 7D freejoint，不接受轨迹驱动或外力补偿。默认控制 30 Hz、物理 300 Hz。

先在禁用重力、每个物理子步固定物体的条件下，由张开手逐渐闭合 12 个控制帧，随后恢复重力并完全释放物体。固定物体只用于建立初始抓取。优化与 IK 基线从相同 MJWarp 状态出发。

每条候选控制轨迹通过 Robokit `MjWarpSim` 并行积分。目标包括物体位置和 SO(3) 角度跟踪、手姿及控制参考、控制平滑、穿透、过大法向力，以及有标签抓取阶段的双指最低法向载荷。每次评估重置所有持久积分状态；无效或容量溢出的候选不参与权重更新。

接触采用 MuJoCo mesh 凸包和 elliptic friction cone。非凸物体的凹槽、孔洞可能被凸包填充；这不是论文的 SuperDex capsule/SDF 碰撞实现。质量默认 0.1 kg、摩擦默认 0.8，均写入结果供审查。法向力统计来自每个控制区间最后一个物理子步的 MJWarp constraint，不代表 300 Hz 的峰值力。

## DIAL 适配

参考 [DIAL-MPC 官方实现](https://github.com/LeCAR-Lab/dial-mpc/blob/main/dial_mpc/core/dial_core.py)：对控制 knot 加高斯扰动，近端噪声较小，逐迭代退火，按候选总体 cost 标准差与 temperature 计算 softmax 权重，更新控制均值，并在执行后移动 warm start。

这里没有学习式 diffusion 网络。默认 128 个随机候选加一个 incumbent，6 轮、16 帧 horizon、每 4 帧一个 knot、每次执行 2 帧。实现采用线性 knot 插值，并最终选择经物理评估的最优候选或均值；这与原仓库 spline、默认首 knot 固定的细节不同。所有参数在 JSON 中可配置。

目标和抓取建立方式参考 [GNR 论文](https://arxiv.org/abs/2610.12440)。结果中的 `accepted` 是适配后的保守检查：所有帧位置误差 <150 mm、角度误差 <40°、平均最大穿透 ≤5 mm、抓取阶段双指载荷中位数 ≥1 N、各指法向力 p95 ≤100 N。它不是论文原实验的复现指标，不代表整套数据已成功。

Warp 会将 rollout、接触目标和执行辅助 kernels JIT 编译为 `.cu` / PTX；物理 kernels 使用 MJWarp CUDA 实现，重复轨迹通过 CUDA graph 运行。无需用 CPU MuJoCo 评估候选。

## 结果、回放及批处理

每个目录包含 `scene.xml`、`reference_qpos.npy`、`ik_controls.npy`、`initial_state.npz`、`rollout.npz`、`progress.json`、`result.json`。结果记录源数据及 mesh SHA256、全部 solver/cost/physics 参数、运行设备、耗时和误差。开启 viewer 时还导出 `index.html`、`replay.json` 与本地 Three.js 依赖。

```bash
.venv-contact/bin/python scripts/batch_ho3d_dial.py \
  --motions output/ho3d_dial_20261112/all_motions \
  --models output/ho3d_dial_20261112/models --mano /path/to/MANO_RIGHT.pkl \
  --output output/ho3d_dial_20261112/batch_mjwarp \
  --config configs/ho3d_wuji2_dial.json
```

批处理为每段启动独立进程，逐段记录完成、失败、超时或短片段跳过状态，不混合不同配置的历史结果。默认仅优化至少 30 帧的片段；更短片段仍完整保存在恢复数据中。`status.json` 与逐段 `progress.json` 可查看进度。验证入口：

```bash
.venv-contact/bin/python -m pytest tests/dynamic -q
```

物理测试将 `mujoco.mj_step` / `mj_forward` 替换成抛异常函数，再检查 MJWarp 抓取释放、自由落体、状态恢复、批量 rollout 与单世界执行一致性；不靠 CPU 仿真作为 oracle。

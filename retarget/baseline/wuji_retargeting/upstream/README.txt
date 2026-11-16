EMF → 人手 IK → Wuji1 / Wuji2 独立重定向项目
==========================================

当前入口是 emf_retarget；生产运行不导入或调用 wuji_sdk。
适用范围：已解码的五个 EMF 接收器位姿、项目 CQQ 人手模型、左右手、两代内置机械手。
输入不是原始磁场强度。目标 SDK 对照版本为 2026.8.31。

一、运行

Frank 上已具备 conda retarget 环境。进入：
  conda activate retarget
  cd /mnt/cpfs/zbl-cpfs-new/USERS/frank/code/emf_wuji_retarget

命令行：
  python -m emf_retarget /path/to/wuji_glove_offline.h5 --side both --models wuji1 wuji2 --output /tmp/emf-retarget-result.npz

短片段：在上述命令中加 --start 0 --limit 64。
每次运行重新初始化状态；--start 是从该帧冷启动，不会隐式求解更早帧。
默认不覆盖现有输出；显式 --overwrite 可覆盖。
同时生成 JSON，记录模型、URDF SHA256、输出形状、帧数和运行时间。

安装为命令行工具（可选，当前目录下 python -m 已可用）：
  python -m pip install . --no-build-isolation --no-deps
  emf-retarget /path/to/wuji_glove_offline.h5 --output /tmp/result.npz

已验证环境：Python 3.10，NumPy 2.2.6，NLopt 2.11.0，h5py。
HDF5 读取需要 h5py；SDK数值验证脚本另需 SciPy 和指定版本的 wuji-sdk。
依赖没有 wuji-sdk；SDK 只用于独立验证脚本作为对照。

二、Python API

  from emf_retarget import EmfFrame, EmfRetargetPipeline

  pipeline = EmfRetargetPipeline(side='right', models=('wuji1', 'wuji2'))
  frame = EmfFrame(
      positions=positions_5x3,                # TX坐标系，米
      orientations_xyzw=quaternions_5x4,      # XYZW，非零；内部归一化
      confidence=confidence_5,
      seq=frame_index,
      timestamp_us=timestamp_us,
      frame_id='right_emf')
  result = pipeline.process(frame)
  human_q21 = result.human_q                 # float64，弧度
  skeleton21x3 = result.skeleton_position     # wrist坐标系，float32，米
  wuji1_q20 = result.robot_q['wuji1']          # float32，弧度
  wuji2_q20 = result.robot_q['wuji2']
  pipeline.reset()                            # 同时清除人手/机器人状态与帧缓存

一只手、一条时间流使用一个实例。五指顺序为 thumb/index/middle/ring/pinky。
同一输入（含header和float32位模式）连续重复调用复用缓存；换header就作为新帧。
两代手分别维护各自 warm start 和低通状态，共享一次人手 IK/FK。
输出数组是缓存的副本，调用者修改结果不会污染后续状态。

三、文件输入/输出

HDF5 使用 ego_fast_retarget 原有格式：
  left/ 或 right/ 下 emf_poses/position、orientation_xyzw、confidence；
  source_seq、source_timestamp_ns、source_frame_id。
按记录的原生 EMF 顺序逐帧求解；不做相机时间对齐、插帧或最近邻重采样。

NPZ 单手输入要求：
  positions (N,5,3)、orientations_xyzw (N,5,4)、confidence (N,5)。
可选 seq、timestamp_us、frame_id，形状均为(N,)；调用必须指定 --side left/right。

输出 NPZ 的数组键以前缀 left_ / right_ 区分：
  human_q (N,21)、human_confidence (N,5)、human_position_error_m (N,5)
  skeleton_position (N,21,3)、skeleton_orientation_xyzw (N,21,4)
  skeleton_confidence (N,21)、tip_position (N,5,3)、tip_orientation_xyzw (N,5,4)
  wuji1_q / wuji2_q (N,20)、对应 *_joint_names
  human_joint_names、skeleton_names、seq、timestamp_us、frame_id。
人手角度与机器人角度不是同一套关节，必须连同名称/型号使用。

四、代码结构

  emf_retarget/          统一类型、帧缓存、状态事务、H5/NPZ适配、CLI
  human_ik_reference/   独立人手模型、SDK姿态分支、QP、迭代IK、重试与历史
  retarget_reference/   独立机器人预处理、FK/J、目标函数、两阶段SLSQP、低通
  tests/                统一入口契约与原始EMF到两代机器人的端到端验证
  docs/                 当前架构和实现/验证缺口说明
  pyproject.toml        可安装包及 emf-retarget 入口；包含6份URDF和许可说明

根目录中的 probe-*、focused-*、反汇编、旧报告和 JSON 是研究/验证证据，
不属于生产导入链。旧报告只代表各次实验状态，当前状态以 docs/ 中的新报告为准。
retarget_reference 中的 nalgebra 衍生实现保留 LICENSE-nalgebra.txt 和 NOTICE.txt。
人手/机器人模型资产来源见各组件报告；没有据此为厂商资产重新授予许可。

五、验证与边界

  python -m unittest discover -s tests -p 'test_*.py' -v
  python tests/verify_end_to_end.py --help

当前数值结果和未实现项见 docs/implementation-status-zh.txt。
本项目提供独立离线转换，不是整个 SDK 的替代绑定：不提供设备连接/控制、
在线订阅、IMU融合、标定训练、按SN自动发现标定模型或完整WujiException兼容。
人手默认使用明确打包的CQQ URDF，不模拟SDK省略URDF时的builtin/calibration选择。
自定义人手URDF需保持当前关节/link约定；其它人体模型及RX校正未获同等验证。

Numerical compatibility
The implementation preserves quaternion FK, TX-to-wrist composition order, scalar accumulation and float32 state boundaries. Do not replace these with reassociated BLAS products or zero small coordinates.
Linux uses standard system libm sincos, without loading the Wuji SDK. Other platforms fall back to separate sin/cos calls; bitwise equality is not guaranteed there. SDK comparisons require identical URDFs, continuous state and runtime environment.

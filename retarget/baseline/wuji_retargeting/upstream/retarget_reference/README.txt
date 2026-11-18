RetargetSession 独立 Python 参考实现

目标版本：wuji-sdk 2026.8.31，Linux x86_64。
这是依据固定SDK二进制证据重建的实现，不是SDK官方源码。
包运行时不导入wuji_sdk、不读取SDK内存，也不需要设备连接。
输入为米制21×3 MediaPipe关键点（或按C顺序展平的63个元素）；输出为固件顺序20个float32关节角，单位rad。

Frank上的运行位置：
  /mnt/cpfs/zbl-cpfs-new/USERS/frank/tools/wuji-sdk-analysis
依赖已在conda retarget中存在：numpy2.2.6、nlopt2.11.0。
不要通过改名此包来覆盖已安装的wuji_sdk。

在上述目录、retarget环境的Python中：

  import numpy as np
  from retarget_reference import RetargetSession, HandModel, Handedness

  session = RetargetSession.for_hand(HandModel.WujiHand, Handedness.Right)
  # points为真实的一帧21x3人手骨架位置，单位m。
  q = session.step(points)
  # 后续帧继续用同一session，保留warm/filter状态。
  q_next = session.step(next_points)
  session.reset()

也可传入字符串 "WujiHand" / "WujiHand2" 和 "Left" / "Right"。
session.kinematics.joint_names给出20个输出坐标的名称。
session.last_trace提供最近一帧预处理S、q_ref、两阶段status/评估cost/候选q、未滤波结果，便于解释实现流程。

模块：
  preprocess.py：float32输入转换、wrist中心化、掌面坐标、左右映射、Hand2旋转、pinch、L/D目标图。
  eigen.py：3×3特征分解、列排序与QR，用于复现共线掌部的法向量选择。
  kinematics.py：固定SDK提取URDF的独立FK与解析Jacobian。
  objectives.py：方向、Laplacian、平滑、拇指相对角、防反转项及梯度。
  session.py：两阶段NLopt SLSQP、限位、失败回退、warm和float32低通、reset。
  assets/：四个SDK嵌入URDF及其提取位置/SHA256。

从父目录运行验证：
  python compare-retarget-reference.py --frames 64
  python test-reference-session.py
  python test-reference-failure-status.py
  python compare-degenerate-session.py
  python test-reference-degenerate-holdout.py
前两个测试将SDK作为被测对照，输入只共享21点；参考算法不读取SDK中间量。
失败注入测试依据机器码恢复的负status分支检查自己的封装，并不声称原生SDK动态触发了所有错误状态。

边界：本包实现RetargetSession的两套内置配置。并未实现原始EMF→人手IK；该环节仍由原SDK生成骨架输入用于数据集对照。任意自定义URDF/配置和非零backoff不属于原SDK这个公开构造接口，也不在本次验证范围。精确误差和退化输入限制见父目录的retarget-reconstruction-report-zh.txt。

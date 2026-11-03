# WUJI Glove 个性化 MANO 标定

入口：[calibrate_wuji_mano.py](../scripts/calibrate_wuji_mano.py)。使用官方 WUJI
SDK 的原始 `emf_poses()`，采集摊平、四组对指、钩指、虚握拳等十个动作。
不使用已依赖 SDK 手型的 `hand_skeleton()` 作为真实人体尺寸标签，也不控制机器人。

## 模型与尺寸

MANO 官网：https://mano.is.tue.mpg.de/ 。原始权重需要在官网注册、接受许可后下载，
本仓库不分发权重。使用 `MANO_RIGHT.pkl` 或 `MANO_LEFT.pkl`。
`--side left` 必须配套左手权重与左手套，不自动镜像右手。

脚本支持原版 MANO 的 10 维形状 `betas`、姿态修正和线性混合蒙皮，并在 rest mesh
上增加整体比例、五指独立纵向比例和五指独立径向比例。骨架与 mesh 一起调整。
径向比例同时改变宽和厚；未单独测量的厚度由 MANO 形状先验及该假设决定。
这属于 **MANO 扩展**，只保存 `betas` 无法在原版 decoder 中复现结果。
每帧姿态是 15 个局部 axis-angle（绝对旋转，不再额外加 `hands_mean`）。
长度比例会调整每条手指链；拇指链起于 CMC，但拇指长度的量尺定义从 MCP 开始。

本机已有两处相同的右手文件（SHA256 均为
`45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`）：

```
/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano/MANO_RIGHT.pkl
/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_RIGHT.pkl
```

这些是本机发现的已有文件，不是脚本自动下载的文件。随附说明要求从官网下载，
且 v1_2 目录包含 MANO 原始许可；没有官网压缩包的可信摘要，故不声称完成官网字节校验。
脚本兼容这些旧版 PKL 内的 chumpy 数值节点，不需要安装旧 chumpy。

## 使用

建议 Python 3.10–3.12 环境；需使用官方 SDK 支持的平台和 Python wheel。
离线拟合只需 numpy、scipy、torch。

```bash
python -m pip install -r requirements-wuji-mano.txt
python scripts/calibrate_wuji_mano.py init --output output/my_hand
```

编辑 `output/my_hand/measurements.json`，填写 `widths_mm` 下五指的宽度及
`palm_width_mm`。数值单位 **毫米**，未测项用 `null`：

- 指宽：脱下手套，量近节指骨中段左右宽度；四指为 MCP–PIP 中点，拇指为 MCP–IP 中点。
- 掌宽：腕横纹到中指 MCP 约 60% 位置的左右外缘距离，不包含拇指。
- 可选指长：伸直时 MCP 关节中心到指尖；拇指不包含掌骨。
- `measurement_sigma_mm` 是量取与 mesh 截面定义误差的容差，默认 2 mm。

这些测量截面是低分辨率 mesh 上的近似，不等同于精确软组织扫描。
周长不能直接填入宽度字段。

连接手套后运行（同网段且设备可被 SDK 发现）：

```bash
python scripts/calibrate_wuji_mano.py calibrate \
  --mano-model /home/zz/GraspSONIC/thirdparty/DexForge/assets/mano/MANO_RIGHT.pkl \
  --measurements output/my_hand/measurements.json \
  --output output/my_hand --visualize
```

多设备时加 `--sn SERIAL_NUMBER`。每一步按回车后保持动作；默认要求约 1.5 秒的
稳定窗口、位置标准差小于 2 mm、姿态 RMS 波动小于 5°、置信度至少 0.7。
每个动作最多等待 45 秒；超时、断流、坐标系改变和取消都会保留已采集数据。
传感器顺序按官方 SDK 示例：thumb/index/middle/ring/pinky；位置使用米，四元数为 xyzw。
整个采集期间应保持传感器和掌背发射器的安装位置不变，避免金属干扰。
不能通过重新佩戴传感器来“改善”单个动作；那会破坏跨动作的共享安装变换。

## 实时可视化

`viser==1.1.1` 已加入依赖。直接连接手套并自动打开浏览器：

```bash
python scripts/calibrate_wuji_mano.py view
```

默认页面是 `http://127.0.0.1:8095`。多设备加 `--sn SERIAL_NUMBER`；左手加
`--side left`。不需要 MANO 权重也可以查看五个原始 EMF 传感器的坐标轴、置信度及
SDK 解算的 21 点骨架。SDK 骨架采用其当前手型，不表示已完成个性化标定。
两种坐标系分开展示，避免将不同 frame_id 的数据误叠加。

完成标定后，可同时显示随动作变化的个性化 MANO mesh：

```bash
python scripts/calibrate_wuji_mano.py view \
  --mano-model /path/to/MANO_RIGHT.pkl \
  --parameters output/my_hand/personalized_mano.npz
```

在线只优化关节姿态，固定已辨识的手型、传感器安装偏移和掌背发射器坐标变换。
页面显示传感器拟合 RMS，红线连接实测和模型预测的传感器位置。它是每帧少量迭代的
warm-start IK，不保证硬实时速率；可用 `--fps 20 --ik-steps 3` 调整显示与优化预算。
低置信度时冻结 MANO，断流时显示过期状态。不要在标定后重新佩戴或移动传感器而
仍沿用旧的安装参数。Ctrl+C 会关闭页面服务并释放本脚本的手套连接。

`capture` / `calibrate` 加 `--visualize` 可边做标定动作边看原始观测及 SDK 骨架；
视图在采集结束后关闭，离线拟合完成后用上面的 `view --parameters` 查看个性化 mesh。
没有手套时可以回放实际保存的数据：

```bash
python scripts/calibrate_wuji_mano.py view \
  --capture output/my_hand/capture.json \
  --mano-model /path/to/MANO_RIGHT.pkl \
  --parameters output/my_hand/personalized_mano.npz
```

回放会明确标注为“非实时手套”，按 `--fps` 播放帧；不会恢复动作之间未记录的运动。
可选 `--no-browser` 只打印地址，`--port` 更改端口，`--duration 10` 运行十秒后退出。
服务默认仅监听本机；SSH 场景可转发相应端口后在自己的电脑打开。

也可分开采集和拟合：

```bash
python scripts/calibrate_wuji_mano.py capture --output output/my_hand
python scripts/calibrate_wuji_mano.py fit \
  --capture output/my_hand/capture.json \
  --mano-model /path/to/MANO_RIGHT.pkl \
  --measurements output/my_hand/measurements.json \
  --output output/my_hand_fit
```

拟合默认 CPU、1200 次 Adam 更新；可用 `--steps`、`--threads`、`--device cuda`
显式调整。已有 capture 或 calibration 报告不会被覆盖，重新实验应选择新目录。

## 输出与复用

- `capture.json`：原始五指 EMF 位姿、置信度、设备时间戳、动作声明和 SDK 版本。
- `calibration.json`：拟合参数、每个动作的误差、尺寸误差、源模型与输入 SHA256、耗时。
- `personalized_mano.npz`：所有形状/姿态/安装参数，以及 rest 和各动作 mesh、关节、三角面。
- `personalized_rest.obj`、`personalized_flat.obj`：可在 mesh viewer 中检查的米单位网格。

可在 Python 中复用扩展 decoder：

```python
import numpy as np
import torch
from scripts.calibrate_wuji_mano import PersonalizedMANO

model = PersonalizedMANO('/path/to/MANO_RIGHT.pkl')
p = np.load('output/my_hand/personalized_mano.npz')
t = lambda name: torch.tensor(p[name], dtype=torch.float64)
vertices, joints, rotations = model(
    t('betas'), t('global_scale').log(),
    t('finger_length_scales').log(), t('finger_radial_scales').log(),
    t('poses_axis_angle'),
)
# 上述是 MANO 模型坐标。转到采集时 EMF 坐标还要应用 root_axis_angle 和 translation_m。
```

拟合共同优化手型、各动作关节姿态、一个共享的 MANO→EMF 刚体变换，以及每个传感器
相对远节指骨的共享安装旋转和偏移。对指接触使用局部 mesh 距离作为弱约束；没有假定
两个 EMF 传感器中心会重合，也没有把传感器中心直接当成皮肤接触点。

`fit_residuals_passed` 仅表示位置、方向、测量和声明对指的表面间隙达到所声明阈值，**不表示完整解剖
辨识或真实硬件精度已验证**。姿态、安装偏移和长度可能互相补偿；未量取的尺寸仍依赖
先验。对指约束基于操作者的动作声明，不自动证明已经真实接触。脚本不做全手自碰撞
检测；对最终 mesh 应检查明显畸变，并在未参与拟合的动作上检验误差。
不通过时仍导出候选和报告，返回退出码 2；异常返回 1，取消返回 130。

## 本次离线检查

右手和左手在不启用显式比例扩展时，与 SMPL-X 0.1.28 原版 MANO LBS 的随机姿态
前向结果比较，最大顶点误差低于 2e-10 m。零角度旋转、形状梯度、SDK 字段解析、
稳定性筛选、取消清理与错误对指拒绝都有测试。
合成 EMF（包含安装偏移与旋转）端到端拟合使用 1200 次 CPU 更新，位置 RMS 约
0.059 mm、方向 RMS 约 0.082°，最大尺寸误差约 0.016 mm。这是同模型生成的合成检查，
不代表真实手套精度，也不证明参数唯一恢复。真实 WUJI Glove 尚未连接验通。

## 来源

- [MANO 官方模型](https://mano.is.tue.mpg.de/)
- [WUJI SDK](https://github.com/wuji-technology/wuji-sdk)，API 对照 2026.10.26 Python wheel。
- [官方动作采集与标定示例](https://github.com/wuji-technology/wuji-sdk/blob/b6be39b1c923b822f627ac8564dd91ceb38b303b/examples/python/wuji_glove/5.calibration.py)
- [官方 EMF 字段与顺序示例](https://github.com/wuji-technology/wuji-sdk/blob/b6be39b1c923b822f627ac8564dd91ceb38b303b/examples/python/wuji_glove/3.offline_pipeline.py)
- [SMPL-X MANO 顶点标记](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py)

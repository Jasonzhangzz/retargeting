# 墨现手套 SDK 与 WebUI

本机 Ubuntu 24.04 的访问地址：**http://127.0.0.1:8890**。

WebUI 直接运行厂商包内 `sdk/qnbot_sdk` 的 `QnBotSDK` 与 `GloveModule`，
使用包内 `emf_v1.py` 解码 V1 数据、校验 CRC，再通过 SDK 生成指尖位姿与触觉数据。
厂商目录保持原样，Linux 适配在 `moxian_web/`。页面参考本地 `frontend-design` skill，采用浅色实验室控制台布局，前端无需联网。

## 接触分布与去皮

02 默认使用“接触分布”：完整手形底图保留，未接触区灰显，仅高亮接触 patch。
可切换“触觉分布”查看原 ADC 数据；两种模式都支持手形／矩阵布局。

1. 戴好左手手套，保持**无外部接触**，点击 02 的“去皮 · 3 秒”。
2. 系统保存原始 460 单元的中位数基线和 MAD 噪声，至少需要 240 个有效帧；失败或断线保留旧配置。
3. 逐区轻触，确认高亮位置。展开“校正分区映射”，在矩阵拖动选择 SID，选择 patch 并“分配所选 SID”；
   触碰核对后点击“已核对当前区”，最后“保存映射”。“恢复默认”清除自定义映射与确认标记，保留去皮。
4. 重新穿戴、温漂明显时重新去皮；去皮是当前无接触姿态的噪声参考，不能保证消除所有弯曲造成的信号。

共 20 区：拇指 distal/proximal；其余四指 distal/intermediate/proximal；掌心 2×3。
初始末端为串口 rows 19–22，中段 17–18，近端候选 15–16；拇指 proximal 合并 15–18。
掌心 rows 0–14 按 5 行×10 列分为六区。近端和掌心分区为**初始示意映射，需逐区触碰校正**。
矩阵屏幕自上而下 `sid=(22-row)*20+col`；460 个 SID 各归属一个 patch，任何 patch 都不能留空。

算法：`σ=max(1,1.4826×MAD)`，先对**带符号残差** `raw-baseline` 进行 EMA，再取 `d=max(EMA,0)`。
EMA 时间常数为 40 ms，`α=1-exp(-Δt/0.040)`，按原始帧间隔计算；120 Hz 时 α 约 0.188。
先平滑再取正避免把零均值噪声整流成正偏置；断流超过 250 ms、去皮或映射改变时清空平滑和滞回状态。
响应仍为 `clip(d/max(12,6σ),0,1)`，无逐帧最大值缩放。归一化响应仅供内部检测和记录，不表示力或概率。接触分布采用二值显示：已确认接触为绿色 1，其他已就绪分区为灰色 0；不使用响应梯度、候选边框或冲突配色。未知仍单独显示。
轻触进入阈值降为 `max(1.5,1.5σ)` ADC，退出阈值为 `max(0.6,0.75σ)` ADC；同 patch 至少两个四邻接单元支持，
进入持续 30 ms，退出持续 80 ms。EMA 会带来随信号幅度变化的额外延迟，需在实物上继续核对误报与响应。
未去皮、去皮中、数据过期或断线时显示未知，不将未知写成无接触。

旧“原始触觉基线”只供触觉 ADC 模式使用，不替代接触去皮。两套基线均按设备和手侧保留。

## MANO 手部模型

01 另可选择 MANO 曲面；EMF 观测是五个传感器原点，不是曲面指尖。每指独立存储末节骨骼到 EMF 的固定 SE(3)，与指长／曲面末端延伸分开标定。先佩戴平摊、稳定一秒并“校正平摊参考”，仅初始化坐标与方向，位置仍为先验；多动作和接触数据通过留出检查后才应用安装位置和尺寸。骨架仅提供初始化信息；拟合 15 个节段轴向比例（0.8–1.2）和 5 个手指径向比例（0.75–1.3），应用后静息网格固定，运行时仅求关节姿态。
手型标定与官方 IK、触觉去皮独立；02 完全按传感器检测输出二值结果，不依赖 MANO、几何或标定状态。01 以固定亮度绿色点标记确认接触位置；实时动力学和局部压缩均停用，碰撞检查不向关节施力。实时 EMF 求解也不再使用触觉贴合约束；接触位切换不会改变姿态。
穿透软惩罚默认从 5 mm 开始，可设置 0–10 mm；超过 5 mm 不硬性拒绝；十动作采集完成后自动拟合，通过检查后自动应用尺寸。详见 [MANO 标定与模型说明](moxian_mano.md)。

## 官方骨骼 IK

01 可切换“骨骼姿态／指尖姿态”。骨架来自**官方 Core/Glove 1.7.0 Linux SDK 内置 Skeleton**，
八动作采集／标定通过公开 `ExternalConnection`、`ExternalGloveFrame` 接入；标定就绪后停止该 SDK 引擎，使用同一官方库的公开 C ABI 读取纯运动学骨架和同帧腕部指尖目标。派生运行配置关闭动力学，保留原始官方标定文件。
串口由现有进程独占；独立 `.venv-moxian-ik` 进程接收有效、未滤波五指米制位置和 XYZW 四元数。
SDK 日志与进程间结构化事件分开，最新帧缓冲限制为一帧，IK 积压或退出不会阻塞串口采集。

首次使用按 01 下方网页提示依次摆好八个动作，然后分别点击“已就位，采集此动作”：

1. 五指自然并拢。
2. 五指自然张开。
3. 四指根关节约 90°，其余指节伸直。
4. 自然握拳。
5. 拇指与食指指腹轻触。
6. 拇指与中指指腹轻触。
7. 拇指与无名指指腹轻触。
8. 拇指与小指指腹轻触。

每阶段按 SDK 要求采满 30 帧；不自动替佩戴者确认动作，不跳过或伪造标定。
页面随当前动作切换示意图，并可展开八动作总览。并拢、张开、根关节弯曲／握拳对照直接复用厂商包
`qnbot_hand_openxr_hand/apps/images` 的三张 PNG，图片字节未修改；对照图左侧为根关节弯曲，右侧为握拳。
四个对指动作暂用明确标注的本地补充示意，**不宣称为官方原图**。
图片来源及 SHA256 记录在 `moxian_web/static/gestures/sources.json`。
2026-11-04 检查时，官网索引中的手势总览／重采截图原图返回 404，当前在线 SDK 采集页仅含文字，不能直接热链复用完整八图。
网页提供确认、重试、取消和重启；取消后停止解算准备，不因持续输入自动重开采集。
兼容记录自动复用。当前操作者为 `local`，SDK 数据目录按设备序列号散列隔离。

成功后绘制 wrist-local **25 节点／24 骨段**，依照名称连线。位置原值为米、方向为 XYZW，网页显示毫米。
骨架样本时间单位为毫秒，独立保存序列号和年龄，不以相同序列号假定与触觉配对。
本次只显示腕部局部骨架，不提供全局手腕位置或蒙皮模型。未标定、数据过期或异常会显示实际状态，指尖视图仍可用。

官方参考：
[骨骼输出](https://docs.qnbot.com/en/docs/vetra-glove-pro/models/retargeting/latest)、
[骨骼可视化](https://docs.qnbot.com/zh-CN/docs/vetra-glove-pro/sdk/python/latest/hand-visualization)、
[外部输入示例](https://github.com/qnbot-technology/qnbot-sdk/blob/main/examples/glove/python/src/external_input_run_background.py)。

实际设备已由佩戴者确认是**左手**，序列号 `2DF5F3B00DA5`。USB PID `5740` 与实际及固件手侧冲突，
保留提示，不修改固件。串口会自动重新发现，当前可能为 `/dev/ttyACM1`。

## 启动与停止

安装 SDK 的独立 Python 环境：

```bash
cd /home/zz/retarget
scripts/setup_moxian.sh  # 同时安装独立官方 IK 环境
```

当前已通过用户级 systemd 服务启动，不需要额外开一份：

```bash
systemctl --user status moxian-web.service
systemctl --user restart moxian-web.service
systemctl --user stop moxian-web.service
journalctl --user -u moxian-web.service -n 30
```

服务文件：`~/.config/systemd/user/moxian-web.service`。当前只启动服务，未设置登录自动启动。
若当前终端缺少用户运行目录，可在命令前加 `XDG_RUNTIME_DIR=/run/user/$(id -u)`。

也可先停止服务，再以前台方式启动（Ctrl+C 停止）：

```bash
scripts/start_moxian_web.sh
# 指定端口或设备：
scripts/start_moxian_web.sh --port 8891 --device /dev/ttyACM1
```

默认仅监听本机。通过 SSH 远程使用时，在自己电脑建立隧道：

```bash
ssh -L 8890:127.0.0.1:8890 <你的主机>
```

## GloveIPC

`GloveIPC_pack/GloveIPC_V1.0.1.exe` 是 Windows Qt 应用。已在项目缓存目录安装独立 Wine 9，
不依赖 sudo，不修改系统 Wine。已验证程序窗口“压感手套上位机 V1.0.1”启动成功。
目前实时采集由原生 SDK 负责；GloveIPC 的硬件采集未与 SDK 同时启用或验证。

重新安装与启动：

```bash
scripts/setup_gloveipc.sh  # 面向 Ubuntu 24.04，校验固定版本包的 SHA256
scripts/start_gloveipc.sh
```

两个程序不能同时占用串口。要把串口交给 GloveIPC，先在 WebUI 点击“断开手套”，
再在 GloveIPC 中选择对应 COM 口；退出 GloveIPC 后，回到网页点击“连接手套”。
也可通过下面的启动选项先释放 SDK 串口：

```bash
scripts/start_gloveipc.sh --take-serial
```

Wine 会自动映射 Linux 串口；当前映射可查看：

```bash
ls -l .cache/moxian-wine/prefix/dosdevices/com*
```

## 文件与接口

- 原生后端：`moxian_web/server.py`；V1 兼容：`moxian_web/protocol.py`。
- 页面：`moxian_web/static/`。
- Python 环境：`.venv-moxian/`（原始采集）与 `.venv-moxian-ik/`（官方 IK 1.7.0）；可单独运行 `scripts/setup_moxian_ik.sh`。
- 旧基线：`output/moxian/baselines.json`；去皮与映射：`output/moxian/contact_profiles.json`。
- IK 标定：`output/moxian/ik/<serial-hash>/`；运行日志：`output/moxian/skeleton-worker.log`。
- 录制：`output/moxian/recordings/*.jsonl`，文件名采用 UTC 时间。
- Wine：`.cache/moxian-wine/root/`；隔离 Windows 环境：`.cache/moxian-wine/prefix/`。
- `GET /api/state`：当前姿态、触觉、连接状态、帧率与新鲜度。
- `GET /api/ports`：识别到的墨现手套串口。
- `GET /api/mapping`：20 区定义、460 项 `assignments`（patch 索引 0–19）、已核对 patch ID 列表与映射版本。
- `POST /api/command`：JSON `{"command":"connect"}`；支持 `connect`、`disconnect`、`baseline`、`clear_baseline`、`record_start`、`record_stop`。
- 新命令：`tare`、`clear_tare`、`mapping_save`（`assignments` 和 `confirmed`）、`mapping_restore`；
  `ik_confirm` 传最新 `request_id`；`ik_cancel` 传最新 `stage_id` 和当前 `request_id`（采样期间可为空），`ik_retry` 传最新 `stage_id`，`ik_restart` 重建骨架进程。
- `/api/state.contact`：`ready/status`、`tare`、归一化 460 值、patch 接触结果、映射版本、采样时刻与 `detector`（算法版本、EMA 时间常数、阈值、滞回时间）；
  `/api/state.skeleton`：官方 pose、chains、capture/calibration、状态、新鲜度与独立毫秒时间戳。
- JSONL 保留旧字段，并增加上述 contact/skeleton 内容；录制为最多约 100 Hz 的状态快照，非完整 120 Hz 串口流。
- `GET /api/recording`：下载当前服务会话最近一份已停止的录制。历史录制仍在上述目录中。

## 验证

```bash
.venv-moxian/bin/python -m unittest discover -s tests -p 'test_moxian_*.py' -v
node --check moxian_web/static/app.js
```

自动测试覆盖 V1 分包和 CRC、20 区覆盖及映射持久化、去皮统计、2 ADC 轻触、EMA 抗短脉冲／带符号噪声、
不同采样率下的时间一致性、断流清空状态、单点抑制、滞回、离线未知，
以及官方 worker 的初始化、采集状态、过期控制拒绝、取消、背压、进程退出和恢复。
浏览器检查模式切换、SID 选择、移动端布局及无 JavaScript 异常；模拟数据仅用于显示与接口测试，不作为实物准确率证据。

2026-11-04 本机佩戴者已完成左手八动作采集及官方 Skeleton 标定；服务升级后成功复用记录，
实测 25 节点持续更新、原始流约 120 Hz、CRC 错误为 0。EMA 40 ms 与新阈值已从实时 API 核实生效。
自动测试共 20 项通过；单元测试验证 2 ADC 持续轻触可进入接触，同时保留短脉冲与单点抑制。
现场观测记录位于 `output/moxian/ema-live-verification.json`；没有独立接触真值，不能据此报告误报率或实际接触延迟。

实物验收需佩戴者完成去皮和官方八动作标定，并逐区触碰：核对接触位置、释放恢复、张手／握拳／对指方向。
完成后观察一段确认无接触时间，记录误报率；接触响应延迟需与独立的实际接触时刻对齐测量。
尚未取得这类实物标签时，不声明已验证检测精度或骨架拟合精度。

MANO v9：十动作采集完成后自动拟合，通过独立留出检查即自动应用并显示“已标定 ✓”；旧采集可用“重新拟合并自动应用”复算，无需重做官方 IK 或去皮。并拢动作加入相邻侧面与平行方向约束，实时仍为纯 EMF 位姿求解。详细变量、求解器和滤波说明见 [MANO 说明](moxian_mano.md)。

MANO v11：实时曲面与骨骼视图共用官方 QnBot IK 的关节中心及旋转，经固定骨段轴变换后蒙皮；不运行第二次 IK、额外平滑或接触贴合。保留已标定静息局部表面参数，动态关节间距来自官方骨架，旧验收数字标为历史。派生运行资源限制四指指间关节物理反伸约 3°，关闭官方关节 EMA 和反复从 open 重求；原始标定与安装包保持原样。曲面与较低频的接触查询分开更新。

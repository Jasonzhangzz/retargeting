# Retarget · Moxian 手套与网页标定

本项目提供 Moxian 手套的实时姿态、触觉显示、个人手型标定，以及向 Wuji2 的重定向。本文按首次使用的顺序介绍：安装厂商 ZIP → 连接手套 → 启动网页 → 完成标定 → 找到并备份标定文件。

## 1. 准备环境

当前安装脚本面向 **Linux x86_64**。默认 Qnbot 连续手部模型需要 NVIDIA GPU 和可用的 CUDA 驱动。准备：

- `git`、`unzip`、`g++`、Eigen3 开发头文件（`/usr/include/eigen3`）。
- 支持 `venv` 的系统 Python，以及计算环境用的 `python3.11`。
- `/usr/bin/python3.12`：现有原生标定依赖安装脚本会检查此路径。
- 厂商提供的 `MoxianEMF-Windows-20260917.zip`。ZIP 不随仓库分发；虽然文件名带 Windows，本项目使用其中的 SDK Python 源码和 V1 解码器。

Ubuntu／Debian 上可先安装通用工具；Python 3.11 和 3.12 请按自己的系统准备：

```bash
sudo apt-get update
sudo apt-get install git unzip build-essential libeigen3-dev python3-venv

git clone https://github.com/Jasonzhangzz/retargeting.git
cd retargeting
```

后续命令都在仓库根目录执行。若已克隆仓库，进入现有目录即可。

## 2. 解压 Moxian ZIP

将下面的 ZIP 路径替换成自己的实际位置：

```bash
mkdir -p moxian_web/vendor
unzip /path/to/MoxianEMF-Windows-20260917.zip -d moxian_web/vendor
```

解压后必须是以下层级，避免多套一层同名目录：

```text
moxian_web/vendor/
└── MoxianEMF-Windows-20260917/
    └── MoxianEMF/
        ├── sdk/
        │   └── qnbot_sdk/
        │       └── core/sdk.py
        └── server/
            └── qnbot_hooks/
                └── emf_v1.py
```

可以检查这两个关键文件：

```bash
test -f moxian_web/vendor/MoxianEMF-Windows-20260917/MoxianEMF/sdk/qnbot_sdk/core/sdk.py
test -f moxian_web/vendor/MoxianEMF-Windows-20260917/MoxianEMF/server/qnbot_hooks/emf_v1.py
```

### 安装默认 Qnbot 模型所需依赖

默认“Qnbot 官方连续 mesh”不加载 MANO 权重，可按以下步骤安装。

先创建串口／网页环境，并注册 ZIP 内的 SDK：

```bash
python3 -m venv .venv-moxian
awk '/^# Compute/{exit} {print}' requirements.txt | .venv-moxian/bin/python -m pip install -r /dev/stdin

.venv-moxian/bin/python - <<'PY'
import site
from pathlib import Path
sdk = Path('moxian_web/vendor/MoxianEMF-Windows-20260917/MoxianEMF/sdk').resolve()
Path(site.getsitepackages()[0], 'moxian-sdk.pth').write_text(str(sdk) + '\n')
PY

.venv-moxian/bin/python -c 'from qnbot_sdk.core import QnBotSDK; print("Moxian SDK ready")'
```

再安装独立的骨架 SDK、编译源手求解器，并建立计算／标定环境：

```bash
bash scripts/setup_moxian_ik.sh

python3.11 -m venv .venv-contact
.venv-contact/bin/python -m pip install -r requirements.txt
.venv-contact/bin/python scripts/setup_wuji_native_calibration.py

.venv-contact/bin/python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
```

最后一条应显示 `CUDA available: True`。三个环境分别负责串口与网页、独立骨架 SDK、GPU 手部模型与拟合；使用启动脚本时无需手动激活环境。

如果还需要历史 MANO 功能，并且已有获许可的 MANO 权重，也可以使用现有一键安装脚本：

```bash
export MOXIAN_MANO_MODEL=/absolute/path/to/MANO_LEFT.pkl
export MOXIAN_MANO_MODEL_RIGHT=/absolute/path/to/MANO_RIGHT.pkl
bash scripts/setup_moxian.sh
```

该一键脚本包含 MANO 文件检查；仅使用默认 Qnbot 时，按上面的分步安装即可。

## 3. 插上手套

1. 按厂商说明接好手套和 EMF 配件，将手套通过可传输数据的 USB 线接入电脑。
2. 关闭正在占用同一手套的厂商程序或其他采集进程。
3. 查看串口，通常是 `/dev/ttyACM0` 或 `/dev/ttyUSB0`：

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

如果连接时报 `Permission denied`，先检查设备权限。使用 `dialout` 组的系统可执行下面的命令，然后**注销并重新登录**：

```bash
sudo usermod -aG dialout "$USER"
```

程序会按支持的 USB 设备信息自动识别手套及左右手。连接多只手套时，启动服务时用 `--device` 指定本次使用的串口。

## 4. 启动网页

```bash
bash scripts/start_moxian_web.sh --port 8890
```

默认监听 `0.0.0.0:8890`。本机或同一局域网的电脑打开 **`http://运行服务的电脑IP:8890`**（例如 `http://10.x.x.x:8890`）；本机也可访问 <http://127.0.0.1:8890>。服务会尝试连接手套；如未连接，点击页面上的“连接手套”。确认设备在线、帧数持续增加，且五指 EMF 数据有效。

指定串口的启动示例：

```bash
bash scripts/start_moxian_web.sh --device /dev/ttyACM0 --port 8890
```

需要只允许本机访问时：

```bash
bash scripts/start_moxian_web.sh --host 127.0.0.1 --port 8890
```

终端保持运行；按 `Ctrl+C` 停止服务。

## 5. 在网页中进入标定

### 首次使用：先完成八动作初始化

在姿态视图下拉菜单选择 **“SLSQP 骨架”**，也可以直接打开：

<http://127.0.0.1:8890/?view=skeleton>

没有当前设备的初始化记录时，页面会显示八动作采集提示。依次按照当前动作图示摆好手势，点击 **“已就位，采集此动作”**，保持姿态直到该动作采集完成，再按提示进入下一动作。动作不合适时用“重试此动作”。

全部完成后等待骨架标定计算，直到显示骨架在线。已有有效记录时会自动加载，无需每次重做。确需重采时点击 **“重做八动作初始化”**；“重启骨架”会复用已有记录。

### 个人手型：完成六动作标定并拟合

1. 将姿态视图切换为 **“手部模型／联合标定”**，或打开 <http://127.0.0.1:8890/?view=mano>。
2. “模型基础”选择 **“Qnbot 官方连续 mesh”**，等待模型就绪。
3. 点击 **“开始六动作标定”**。如果骨架初始化尚未完成，页面会先引导回到八动作初始化。
4. 按图示就位，点击 **“开始自动六动作采集”**。按钮在不同状态也可能显示“就位，开始自动六动作采集”。
5. 按页面顺序做六个动作：拇指与食指、中指、无名指、小指分别对指；四指根部弯曲 90°且指节伸直；最后伸直并拢、手掌展平。
6. 每个动作通过姿态和稳定性检查后，保持 1 秒、采集 1 秒，再自动切换。未通过时按页面提示调整；超时暂停后可以继续当前动作。
7. 显示六动作采集完成后，点击 **“拟合 Moxian 骨长与EMF外参”**，或主流程按钮中的“开始 Moxian 六动作 EMF 与骨架拟合”。**采集完成后仍需点击拟合。**
8. 等待拟合及回放检查。候选通过检查后自动应用，页面显示“已标定”；未通过时保留原模型，并显示原因。需要回退时使用 **“恢复上一版本”**。

当前方法为 **“Moxian 六动作 · 25＋4侧旋自由度”**。骨架视图中的“Wuji 原生六动作标定”按钮也会跳转到这里；当前默认拟合的是 Moxian 手型。

触觉基线单独操作：在“接触感知”面板点击 **“去皮 · 3 秒”**，保持手套不与物体接触，等待采集完成。它与上述骨长／EMF 外参标定分别保存。

## 6. 标定文件保存在哪里

所有路径都相对于仓库根目录。文件自动保存在 **`output/moxian/`**，按设备区分：

```text
设备目录名 = SHA256(设备序列号) 的前 16 位十六进制字符
```

下面用 `<设备hash>` 表示该目录名，用 `<记录ID>` 表示 SDK 生成的初始化记录目录。

| 路径 | 内容 |
| --- | --- |
| `output/moxian/ik/<设备hash>/glove/calibrations/<记录ID>/artifacts/hand_system.yaml` | 八动作初始化生成的骨架及安装标定配置 |
| `output/moxian/hand_model/<设备hash>/qnbot/wuji-calibration-capture.json` | 六动作原始观测和采集进度，可用于重新拟合 |
| `output/moxian/hand_model/<设备hash>/qnbot/methods/moxian_wuji/candidate.json` | 六动作拟合候选模型 |
| `output/moxian/hand_model/<设备hash>/qnbot/methods/moxian_wuji/report.json` | 拟合、检查与应用结果 |
| `output/moxian/hand_model/<设备hash>/qnbot/current.json` | 当前已应用的个人手型、骨长、EMF 外参及源 IK 初值 |
| `output/moxian/hand_model/<设备hash>/qnbot/previous.json` | 上一个已应用版本，首次应用时可能没有 |
| `output/moxian/hand_model/<设备hash>/qnbot/active.json`、`models/` | 运行时活动模型指针及其引用的模型文件 |
| `output/moxian/contact_profiles.json` | 各设备的触觉去皮、噪声及接触检测配置 |

网页标定面板会显示“自动保存位置”，也可点击 **“下载已保存的标定数据 JSON”** 导出采集数据。该下载不包含全部模型与初始化文件。

**备份／迁移时保留整个 `output/moxian/`，至少同时保留该设备的 `ik/<设备hash>/` 和 `hand_model/<设备hash>/`。** 仅拷贝 `current.json` 不足以保留初始化记录、活动模型和重拟合所需数据。重启后，同一设备会自动加载匹配的已保存记录；换人佩戴、调整传感器安装或更换手套后应重新标定。

`output/` 和整个 `data/` 都被 Git 忽略，**不会随 `git push` 上传**；请自行备份。首次只做 Moxian 显示与标定不需要下载抓握数据集；使用 Wuji2 重定向等功能时，还需在本地准备相应的 `data/` 模型资产。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 提示缺少厂商 SDK 或 `emf_v1` | 检查 ZIP 是否按第 2 节的层级解压，确认两个关键文件存在。 |
| 网页显示未发现手套 | 检查 USB 数据线、供电及串口；确认其他程序未占用设备，必要时使用 `--device`。 |
| 网页能打开，手部模型不可用 | 确认 `.venv-contact/bin/python` 存在且 CUDA 可用；查看 `output/moxian/hand-model-worker.log`。 |
| 原生求解器编译失败 | 确认 `g++` 和 Eigen3 已安装，再运行 `.venv-moxian-ik/bin/python scripts/setup_moxian_studio_qp.py`。 |
| 六动作按钮不可用 | 先确认设备在线、八动作初始化完成、Qnbot 模型就绪，并结束历史权重预览。 |
| 动作一直不能通过 | 对照当前手侧的图示调整姿态，保持五个接收器有效，查看页面的姿态／稳定性提示。 |
| 服务提示端口被占用 | 停止已有服务，或改用 `--port 8891` 并访问对应地址。 |

算法与进一步使用：[六动作标定](docs/moxian_six_calibration.md) · [接触注册](docs/moxian_contact_registration.md) · [SLSQP 重定向](docs/moxian_wuji_contact_slsqp.md) · [CARD／CEM](docs/moxian_robokit_card.md) · [wuji-retargeting](docs/moxian_wuji_reference.md) · [GeoRT](docs/moxian_geort.md) · [CART](docs/moxian_cart.md)。

目录与来源：[目录说明](docs/repository_layout.md) · [baseline 代码](retarget/baseline/README.md) · [第三方来源与许可](THIRD_PARTY.md)。

网页功能与维护范围：[当前功能清单](docs/moxian_web_features.md)。RVQ 独立训练工具保留，网页中的蓝色重建、grasp type 与旧 MANO 触觉预测入口已移除。

Wuji2 随机姿态、原生网格碰撞采样与直接关节角条件化 MLP：[Configuration Neural SDF](docs/wuji2_configuration_sdf.md)。

# 目录和依赖

本次清理保留的五个求解入口：

| ID | 用途 |
| --- | --- |
| `cem` | CARD objective 与高斯 CEM |
| `slsqp_graph` | 离散接触图选择＋Laplacian，默认 |
| `wuji-retargeting` | 已标定 Moxian 骨架 → 原版 Wuji2 两阶段重定向 |
| `geort` | 当前 GeoRT 模型推理 |
| `cart` | 独立的接触驱动网络推理 |

`data/grasp` 和 `data/contact` 是运行所需的机器人网格、MJCF、模板等资产；原 `datasets` 已迁入 `data/datasets`，存放成对抓握数据和独立查看器资源。保留 31 类、1,247 组成对姿态及查看器需要的文件，删除较早的 `rl-refinement` 试验。迁移后 `verify.py` 校验保留文件的 SHA256 与姿态配对。

所有生成结果统一写入 `output`；简短历史报告位于 `output/reports`，编译缓存位于 `output/runtime`。`output/moxian` 保存设备当前标定、当前手型、选中 GeoRT 模型及运行状态。2026-11-16 清理删除了旧录制、非当前模型、训练中间权重、候选历史和大体积调试输出；当前六动作标定采集、当前模型及仍供覆盖图使用的两段动作录制保留。已删除录制不会随模型保留，缺失训练数据时需要重新录制。

`moxian_web/vendor` 存放用户提供的厂商 SDK、原始 Windows 压缩包及固定版本 Wuji 原生标定库，均不纳入 Git。`scripts/setup_moxian.sh` 检查迁移后的 SDK 路径。原 `.cache` 中的原生库已迁出，`.cache`、`.pytest_cache` 和 Python 字节码可重建，不应提交。

根目录 `requirements.txt` 是唯一依赖清单。Python 3.11 计算环境安装完整清单；串口和官方骨架 SDK 的安装脚本只安装清单共享区，再安装各自固定版 SDK，避免同名 SDK 包互相覆盖。可选 MuJoCo 4 验证环境由 `scripts/setup_contact.sh` 单独安装其引擎版本。

保留三个正在使用的环境：`.venv-moxian`（串口/Web）、`.venv-moxian-ik`（独立官方 SDK）、`.venv-contact`（GPU/CARD/GeoRT）。它们均可在停止服务后删除并重新安装，但当前删除会中断运行，故本次保留。`tests` 仅保留求解器、当前标定、传输和界面的核心回归。

`licenses` 保存仍在使用的第三方源码和资产的许可及来源声明；对应关系见根目录 `THIRD_PARTY.md`。它不属于缓存，本次保留。`.git` 是版本历史，本次未清理。

核心 GPU 回归验证速度边界、穿透诊断和可行性标记。离散接触图在极深穿透初态下不保证45帧内恢复；仍有穿透时保持不可行，不以固定帧数替代稠密验收。

保留独立的 CART 网络求解入口及其录制、训练和预览功能。旧接触 edges solver 已移除。

`retarget/baseline/wuji_retargeting/upstream` 保存用户指定 SSH 目录的原版生产链路、6份URDF、测试和许可。`SOURCE.json` 逐文件记录 SHA256；默认模型、目标、梯度、预算和低通均不改写。已删除不可满足实时性的 SLSQP·CARD 入口及专用代码。

CARD、wuji-retargeting、GeoRT 的全部算法实现分别位于 `retarget/baseline/card`、`retarget/baseline/wuji_retargeting`、`retarget/baseline/geort`。Web中的旧模块只是兼容导入；当前工厂、测试、安装和训练命令使用新路径。`scripts` 的Python文件从98个精简为21个；按当前依赖与使用路径保留，删除77个历史实验／轨迹／报告脚本。清单见 `output/reports/wuji_reference_integration_20261116/script_cleanup.json`。

再次清理移除了未使用的完整 `third_party/GeoRT` 副本；实际依赖的固定版本源码、许可证和文件校验清单仍保留在 `retarget/baseline/geort/upstream`。Python／pytest 缓存、旧回放诊断原始帧和渲染截图已清除，简短报告保留。`output`、SDK、虚拟环境都不提交到远端，其他机器需按安装文档准备厂商依赖并完成设备标定。

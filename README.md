# Retarget

Moxian 采集与 WebUI、实时手部重定向、MANO CEM／GraspADMM，以及运行所需的模型资产和验证工具。

## 启动

```bash
bash scripts/setup_moxian.sh
bash scripts/start_moxian_web.sh
```

网页地址：<http://127.0.0.1:8890>。厂商 SDK、私有 MANO 权重和设备标定需在本机准备；MANO 权重通过 `MOXIAN_MANO_MODEL` 指定。

## 当前功能

网页提供五个重定向选项：

| 选项 | 实现与说明 |
| --- | --- |
| CARD／CEM | [CARD objective 与 CEM](docs/moxian_robokit_card.md)；代码位于 `retarget/baseline/card` |
| SLSQP 离散接触图 | [Moxian 触觉 SLSQP](docs/moxian_wuji_contact_slsqp.md)；默认后端 |
| wuji-retargeting | [标定骨架到原版 Wuji2 重定向](docs/moxian_wuji_reference.md)；代码位于 `retarget/baseline/wuji_retargeting` |
| GeoRT | [录制、训练与回放](docs/moxian_geort.md)；代码位于 `retarget/baseline/geort` |
| CART | [五类抓取录制、训练与预览](docs/moxian_cart.md) |

默认源手使用 Moxian 原生25自由度加四指指根轴向旋转，共29自由度，保留 Qnbot 原始 SKN 网格。当前源 IK 为 `moxian_studio_native_qp_v3_calibration_seeds`，在 C++ 中执行 FK、Jacobian 和有界 QP；标定初值由六动作流程保存及重载校验。详见 [六动作标定](docs/moxian_six_calibration.md)。

原生求解器按源码哈希自动编译，也可手动构建：

```bash
.venv-moxian-ik/bin/python scripts/setup_moxian_studio_qp.py
```

接触材料坐标仍须通过独立实测注册验收；460个 SID 的表面绑定和机器人虚拟投影不代表完成传感器注册。操作见 [接触注册](docs/moxian_contact_registration.md)。

## 离线求解

MANO patch CEM：

```bash
python scripts/solve_mano_comfree.py --help
python scripts/retarget_mano_comfree_campaign.py --help
```

GraspADMM：

```bash
python3.11 -m venv .venv-graspadmm
.venv-graspadmm/bin/python -m pip install -r requirements.txt
.venv-graspadmm/bin/python run.py --config configs/graspadmm.yaml
```

YAML 的 `solver.type` 接受 `cem`、`graspadmm` 或 `contact_cem`，参数由对应模块的 `DEFAULTS` 提供。接触空间 CEM 的安装、任务和验收见 [contact_cem](docs/contact_cem.md)。外部输入数据和私有模型需另行准备。

## 检查与目录

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-contact/bin/python -m pytest -p no:cacheprovider -q
```

生成结果写入被 Git 忽略的 `output/`。本机当前标定、手型、GeoRT 模型和录制保存在 `output/moxian/`；运行环境与厂商依赖保留。可重建的缓存、编译中间文件、旧报告及已废弃的实验代码已清理。

目录与依赖见 [目录说明](docs/repository_layout.md)，算法位置见 [baseline说明](retarget/baseline/README.md)，来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

# dex_grasping primitives · 2026-11-15

本文件夹保存这批完整的 primitive 姿态映射结果、几何诊断、RL 小实验和 Viser 查看所需的冻结模型资源。

- 数据源：<https://huggingface.co/datasets/zhezhang92/dex_grasping> 的 `primitives`。
- 31 种抓型，1,247 组尺寸；每组包含 MANO 来源姿态、Moxian glove 映射、Wuji2 映射。映射输出共 2,494 个，不是 2,494 个独立原始样本。
- Glove 使用新上传的 Moxian 29 自由度模型，指纹 `f0dd73b5a88b6ae0b6f2c0cc68fccfd69d5195ab39fec52db6a00fd36c463c3c`。其原始拟合验收标记仍为 `false`。
- Wuji2 使用原生 20 关节顺序。两种映射均由 Robokit 直接从 MANO 独立计算。
- 姿态、坐标与物体尺寸使用米和弧度；四元数为 **WXYZ**。

## 内容

| 路径 | 内容 |
|---|---|
| `glove/grids/` | 31 个 NPZ，1,247 个 glove 姿态及配对 MANO、物体数据 |
| `wuji2/grids/` | 31 个 NPZ，1,247 个 Wuji2 姿态及配对 MANO、物体数据 |
| `glove/glove-model.json` | 冻结的 Moxian 个体模型 |
| `source-manifest.json`、`source-mano_geometry.npz` | 来源清单、源文件指纹、MANO 拓扑 |
| `source-wuji2_model.json` | Wuji2 模型参考快照 |
| `size-index.csv`、`type-summary.csv` | 尺寸与抓型索引 |
| `physics-audit-final/` | 初始手物几何检查 |
| `rl-staged-gravity/` | 稳定门槛、虚拟扳手训练与独立物理验证 |
| `rl-refinement/` | 较早的 PPO 试验；其中原生碰撞失败结论已更正，见该目录说明 |
| `viewer_runtime/` | 与本批个体模型匹配的冻结模型代码，隔离于应用代码 |
| `viewer_resources/` | 查看器所需的 Qnbot 描述和蒙皮资源，不含 SDK 可执行文件或采集记录 |
| `FORMAT.md` | NPZ 字段格式与映射方法 |
| `MANIFEST.sha256.json` | 本目录各文件 SHA-256 |

## 查看

在仓库根目录运行：

```bash
.venv-moxian-mano/bin/python visualization/viser_primitive_retarget.py --port 8767
```

打开 <http://127.0.0.1:8767>。默认加载本文件夹，支持 8 个快捷样例、抓型/尺寸选择、透明度和 RL 末帧对照。不需要 GPU、远端目录或现有 SDK 安装；查看器使用本文件夹冻结的模型代码与资源，以及仓库内的 Wuji2 模型。

新环境可安装 `viewer-requirements.txt` 中的依赖后用 `python` 运行同一命令。依赖版本对应本批验证环境，现有 `.venv-moxian-mano` 已可直接运行。

```bash
python -m pip install -r datasets/dex_grasping_primitives_20261115/viewer-requirements.txt
python visualization/viser_primitive_retarget.py --check
python datasets/dex_grasping_primitives_20261115/verify.py
```

两个 `model_final.pt` 使用 Git LFS。首次拉取请确保安装 Git LFS，并运行：

```bash
git lfs pull --include="datasets/dex_grasping_primitives_20261115/**/*.pt"
```

普通 NPZ 数据与 Viser 查看均不需要 checkpoint。

## Python 读取

```python
from pathlib import Path
import numpy as np

root = Path('datasets/dex_grasping_primitives_20261115')
z = np.load(root / 'wuji2/grids/10_Power_Disk.npz', allow_pickle=False)
q = z['target_qpos'][27]                # 20 个关节角，顺序见 joint_names
base = z['target_base_pose'][27]        # XYZ + WXYZ
mano_vertices = z['mano_vertices'][27]  # 世界坐标
object_pose = z['object_pose_world'][27]
radius_m = z['radius_m'][27]
height_m = z['height_m'][27]
```

## 结果边界与复现

这些主要是运动学映射参考，不能整体标成稳定抓握数据。`valid`、`grasp_passed` 等继承字段是源 MANO 诊断，不能当作新目标的成功标签。106 条薄板的完整物体尺寸尚不明确，因此 Viser 显示另外 1,141 组。

RL 仅覆盖 `10_Power_Disk` 第 27 行这个初态，不是 128 种独立抓型。最终虚拟模式复测的两指接触保持数为 118/128，严格虚拟扳手检查通过数为 0；独立真实扰动每个环境只测试一个方向。完整指标和准备/加载流程见 `rl-staged-gravity/README.md`。

旧报告中原生网格动态测试漏恢复相邻连杆碰撞排除规则，导致虚假自碰撞；该结论已更正，后续测试已修复。几何闭合、有限预载承载能力和实际扰动保持是不同指标。

子目录中的绝对路径保留为原始实验来源记录；它们不用于定位当前查看器数据。模型、代码与依赖版本冻结在本文件夹。Wuji2 描述的许可见仓库 `data/grasp/assets/wuji2/LICENSE`。

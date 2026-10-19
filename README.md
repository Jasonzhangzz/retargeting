# MANO → Wuji Retarget

根据 MANO 手部参考与接触模板优化 Wuji 手部动作，跟踪物体轨迹。默认使用 **AdamW + 接触点探索 + ComFree**，并开启 viewer。

## 运行

需要 Linux、NVIDIA GPU/CUDA 驱动；viewer 需要桌面显示环境。

```bash
python -m pip install -r requirements.txt
python run.py                         # 默认示例，开启 viewer
python run.py --no-viewer             # 无界面运行
python run.py --duration 3.6 --seed 0  # 指定仿真时长与随机种子
```

viewer 中空格暂停/继续，`X` 切换透明显示。结果保存在 `outputs/<executor>/<case-name>/`；可用 `--case-name NAME` 命名。

```bash
python run.py --replay --case outputs/comfree/你的运行名称
python run.py --replay --case outputs/comfree/你的运行名称 --video result.mp4
```

## 可选分支

| 参数 | 功能 |
| --- | --- |
| `--adamw`（默认） | 接触点、hand action 与 dual 交替更新，每窗口默认一轮、一次 hand 更新 |
| `--contact-solver qp` | 将默认采样 LM 接触更新替换为 QP |
| `--no-contact-exploration` | 固定物体模板，保留 ADMM 一致性项和 dual |
| `--no-admm` | 固定模板，仅更新 hand action，关闭 ADMM |
| `--fa` | L-BFGS 基线 |
| `--sqp` | 简化接触模型的 SQP-RTI，需额外配置 acados；当前尚未通过稳定抓取验证 |
| `--executor mujoco` | 用 MuJoCo 执行求得的动作，默认执行器为 ComFree |
| `--noise` | 扰动 MANO 手指，通过 SDF 最近投影重新生成接触模板和手部参考 |

```bash
python run.py --noise --noise-finger thumb --noise-amplitude 0.6
python run.py --noise --noise-amplitude 0.6 --no-admm
python run.py --contact-solver qp
```

噪声默认全手、0.35 rad；`--noise-finger` 可选 `all/thumb/index/middle/ring/pinky`。`--noise-seed N` 使用可复现的随机关节偏移。生成噪声需要 Torch、chumpy、manopth 和本地 MANO 模型；本机已配置，其他机器可用 `--noise-mano-models DIR --noise-manopth DIR` 指定资源。噪声输入保存在 `outputs/noise_inputs/`，回放自动读取对应输入。

手部参考由扰动 MANO 的掌部与骨链直接做 IK 生成，不读取原可行 Wuji 轨迹；扰动从第一帧生效。原接触时序与语义对应仍保留。

其他参数见 `python run.py --help`。SQP 的 acados 路径可用 `ACADOS_SOURCE_DIR` 指定。

## 更换物体

使用另一套已准备好的实验目录：

```bash
python run.py --experiment data/my_object
```

参考 `data/example/` 准备以下内容：

```text
data/my_object/
  forceaware.yaml
  demonstration_visual.npz           # 与新轨迹对应的 MANO 可视化
  contactaware/                      # Wuji 参考、物体轨迹、接触指导及配置
  sequence/
    mano_raw/meta.json               # {"object": "005_tomato_soup_can"}
    model/                          # 新物体网格、纹理
    scene/wuji_right/
      tomato_soup_can_sdf.xml        # ComFree 场景
      tomato_soup_can_mesh.xml       # viewer / MuJoCo 场景
```

场景前缀由 `meta.json` 的 `object` 去掉数字前缀得到。更换网格时，同步更新 XML 的质量、惯量、碰撞几何及资产路径，并保留 Wuji 手部结构和 `obj`、`obj_joint`、`obj_collision` 等命名。`contactaware/` 中的手/物体参考与模板必须对应新物体，坐标系及数组格式沿用示例；仅替换 mesh 不会自动生成抓取轨迹。使用 `--noise` 还需提供匹配的原始 MANO 参数和语义顶点对应。

[data/dexycb/](data/dexycb/README.md) 包含五种物体的原始 MANO 数据，尚需准备为上述实验目录。代码与资产来源及许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。

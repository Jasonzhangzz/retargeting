# DexYCB MANO 测试样本

保留五条不同物体的右手 MANO 序列，源自 `dex-ycb-20210415` 版本、`20200709-subject-01`、相机 `836212060125`，帧率 30 Hz。没有时间降采样或重新拟合，保留无效标注帧。

| 物体 | 序列 | 总帧数 | 有效帧数 |
| --- | --- | ---: | ---: |
| `002_master_chef_can` | `20200709_141754` | 72 | 63 |
| `003_cracker_box` | `20200709_142211` | 72 | 72 |
| `004_sugar_box` | `20200709_142517` | 74 | 74 |
| `005_tomato_soup_can` | `20200709_142853` | 74 | 69 |
| `006_mustard_bottle` | `20200709_143211` | 72 | 72 |

每个 `<object>/<sequence>/` 包含 `mano.npz`、`meta.json`、原始 `source_meta.yml` 和相机内参 `camera_intrinsics.yml`。`index.json` 汇总路径、帧数和 SHA256。

`mano.npz` 字段（N 为帧数，O 为该场景中的物体数）：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `hand_pose_51` | `(N, 51)` | 原始 `pose_m`：前 48 维为 MANO 姿态 PCA 表示，后 3 维为相机坐标系平移，米；不是逐关节 axis-angle 数组 |
| `hand_shape_10` | `(10,)` | MANO shape/betas |
| `object_transform_camera` | `(N, O, 3, 4)` | 所有物体原始 `pose_y`，`[R | t]`，平移单位米；顺序见 `source_meta.yml` 的 `ycb_ids` |
| `object_pose_camera_7` | `(N, 7)` | 被抓物体的 `[qx,qy,qz,qw,tx,ty,tz]`；从对应 `pose_y` 转换 |
| `joint_3d_camera` | `(N, 21, 3)` | 原始 `joint_3d`，相机坐标系、米；无标注帧可含 `-1` |
| `extrinsics_4x4` | `(4, 4)` | 官方标定中此相机的外参矩阵，未修改其方向约定 |
| `frame_indices` | `(N,)` | 原始标签帧编号 |
| `times` | `(N,)` | 从零开始的秒数，`frame_indices / 30` |
| `frame_valid` | `(N,)` | 被抓物体旋转有效且手姿有限、非零 |

无效物体旋转转换为单位四元数占位，必须结合 `frame_valid` 使用。完整原始物体矩阵仍保存在 `object_transform_camera`。

读取示例，在仓库根目录执行：

```python
import json
from pathlib import Path
import numpy as np

root = Path('data/dexycb')
for item in json.loads((root / 'index.json').read_text()):
    with np.load(root / item['path'], allow_pickle=False) as sample:
        valid = sample['frame_valid']
        hand = sample['hand_pose_51'][valid]
        target = sample['object_pose_camera_7'][valid]
        times = sample['times'][valid]
        print(item['object'], hand.shape, target.shape, times[[0, -1]])
```

这些数据用于 MANO/轨迹读取和预处理测试。它们尚未生成 Wuji 参考手姿、接触指导和各物体的物理场景，不能直接作为 `run.py --experiment` 的输入。可直接运行的完整 MPC 示例在 `data/example/`。本仓库不包含 MANO 模型权重；将 PCA 参数还原成网格时，需要自行取得 MANO 模型并使用匹配的 PCA 配置。

数据来源及字段定义：[DexYCB 官方网站](https://dex-ycb.github.io/)、[官方 Toolkit 的标签说明](https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples)。数据按 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 使用，归属 Yu-Wei Chao 等 DexYCB 作者。这里的修改为：将逐帧标签按原顺序打包，附加四元数形式、时间和有效掩码；未重新授权数据。

```bibtex
@inproceedings{chao2021dexycb,
  title={DexYCB: A Benchmark for Capturing Hand Grasping of Objects},
  author={Chao, Yu-Wei and Yang, Wei and Xiang, Yu and Molchanov, Pavlo and Handa, Ankur and Tremblay, Jonathan and Narang, Yashraj S. and Van Wyk, Karl and Iqbal, Umar and Birchfield, Stan and Kautz, Jan and Fox, Dieter},
  booktitle={CVPR},
  year={2021}
}
```

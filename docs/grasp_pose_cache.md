# 紧凑抓姿缓存

`results/grasp_current_20261103` 是当前正式数据：31 个 grid、640 个最终求解点、39,007 个插值姿态。
包含 Type 6–8/24 圆柱中部接触、Type 18 高度扫描及新的 1× 解、Type 26 食指接触和中指居中修正。

```bash
.venv-contact/bin/python scripts/verify_grasp_cache.py results/grasp_current_20261103
.venv-contact/bin/python visualize.py --port 8089
```

首次运行自动创建 `output/grasp_current_20261103` 链接，不需要求解器或旧 output 数据。

## 存储

- `scene.xml`：唯一的公共手部场景，物体网格由读取器在内存中提供。
- `geometry.npz`：按形状共享的单位物体几何，没有逐尺寸 XML 或 OBJ。
- `poses.npz`：640 个 float64 原始手姿、原生手姿、真实尺寸、案例名和源投影点。
- `dense/*.npz`：每个 grid 的插值手姿与尺寸。每段分为 64 段，根旋转使用 SO(3) 最短弧插值。
- `templates.json.gz`、`metrics.json.gz`：共享的固定接触模板与最终指标。
- `manifest.json`、`dense_pose_index.json`：案例及 grid 索引。

原始 float64 数值保持不变。线性插值投影只存源点，完整插值序列可逐位重建，无需重复保存。
输入的约 240 MB 最终记录及插值整理为约 6 MB；NPY 不压缩，当前使用标准 NPZ 无损压缩。
完整重复接触点云、初始化依赖链、优化轨迹、重试和日志不属于此姿态缓存。

```python
import numpy as np
from pathlib import Path
from retarget.viewer.snapshot import get_snapshot
from retarget.viewer.dense_pose import load_dense_grid

path = Path('results/grasp_current_20261103')
poses = np.load(path / 'poses.npz', allow_pickle=False)
q = poses['qpos']                 # [640, 26]
dimensions = poses['dimensions_m']
snapshot = get_snapshot(path)
model = snapshot.model(0)        # 公共 XML + 该案例尺寸，直接在内存中构建
grid = load_dense_grid(path, '18_Extensior_Type', 'disk')
sample = grid.sample(1.25)
```

`qpos` 前 3 维为根平移（米），接下来 3 维为内禀 XYZ 根旋转（弧度），最后 20 维为手指关节（弧度）。
物体形状及尺寸按案例记录，`grasp_qpos` 为原生 27 维手姿。原始点和源投影值按 SHA256 校验。
共享几何与原始展开式模型的最大顶点重建误差为 1.31e-8 米；手姿和尺寸没有精度缩减。

`source_verification.json` 记录清理前对全部 640 例的源数据、FK、碰撞、力证书以及全部插值数组的核对。
`verification.json` 是无需旧结果的独立缓存检查，`checksums.json` 覆盖发布文件。
模型共享的仓库资产及其哈希记录在 `provenance.json`；历史来源路径仅为来源元数据。

插值姿态未重新求解或做物理验证。612/640 的投影闭合是虚拟接触力证书，原生动态评测未运行。
原始失败案例仍保留。许可及第三方资产见仓库 [THIRD_PARTY.md](../THIRD_PARTY.md)。

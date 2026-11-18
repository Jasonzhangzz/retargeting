# Wuji2 primitive 抓取增强

独立生成器，复用 Moxian 网页 GeoRT 的原生 20 关节顺序、禁止负向屈伸范围和真实网格自穿透检测。不会替换当前网络、修改录制或把增强数据自动送入 GeoRT 训练。

## 运行

使用仓库 `.venv-contact/bin/python`。依赖包括 PyTorch、Warp、MuJoCo、Trimesh、Manifold 3.5.2 和 HDF5。默认使用 CUDA，正式目标为每类 50,000 条，共 1,550,000 条。

```bash
# 全尺寸接触校正 + 每类 200 条试运行 + 独立密集验收
.venv-contact/bin/python -m retarget.env.grasp.primitive_augmentation generate \
  --output data/datasets/wuji2_primitive_aug_pilot_v1 --per-type 200

# 复用同配置、同代码、同资产的试运行校正与基线结果
.venv-contact/bin/python -m retarget.env.grasp.primitive_augmentation generate \
  --output data/datasets/wuji2_primitive_aug_v1 \
  --bootstrap-from data/datasets/wuji2_primitive_aug_pilot_v1

# SIGINT / SIGTERM 安全停止后恢复；配置或代码变化会明确拒绝混用
.venv-contact/bin/python -m retarget.env.grasp.primitive_augmentation resume \
  --output data/datasets/wuji2_primitive_aug_v1

# 独立复查；--repair 删除不合格记录，后续 resume 补采
.venv-contact/bin/python -m retarget.env.grasp.primitive_augmentation audit \
  --output data/datasets/wuji2_primitive_aug_v1 --repair
.venv-contact/bin/python -m retarget.env.grasp.primitive_augmentation report \
  --output data/datasets/wuji2_primitive_aug_v1

# 原 primitive 查看入口新增 --augmentation；可筛抓型、尺寸和 mask
.venv-contact/bin/python visualization/viser_primitive_retarget.py \
  --augmentation data/datasets/wuji2_primitive_aug_v1 --port 8767
```

查看已经保存的试运行数据及接触法向：

```bash
.venv-contact/bin/python visualization/viser_primitive_augmentation.py \
  --data data/datasets/wuji2_primitive_aug_pilot_v1 --port 8782
```

青色为展示时按物体表面积重新采样的表面点，绿色为文件中保存的体积点；两者可单独显示。橙色表示有效接触见证点在物体表面的投影周围的邻域，默认半径 3 mm，体积点还须位于外侧 1 mm 层内。这是可视化邻域估计，不是补写的逐点训练标签，也不会强行把距离接触很远的最近体积点标成接触。红色标记为真实物体表面接触投影，黄色箭头为该处物体网格外法向；可另外打开紫色手部法向。表面点、接触投影和箭头均不修改保存数据。

查看器提供抓型、增强方式、尺寸和 mask 筛选，以及上一条、下一条、随机 pose 和回放。页面同时展示真实任务状态；`incomplete_stalled` 表示未达到目标数量而停止，不能解释为生成完成。

`calibrate` 可单独校正全部 1,247 个尺寸。`generate/resume --max-jobs N` 在 N 个提议批次后安全暂停；它不表示生成了 N 条合格数据。默认 batch 64、接触迭代 60、固定 patch 候选点 128、体积点数 1,024；每类 40% 形状、30% 姿态噪声、30% 表面滑动。

每类连续 100,000 次提议没有新的合格姿态时，以 `incomplete_stalled` 停止并保留诊断，不降低接触要求或复制姿态。磁盘剩余不足 5 GiB 时以 `paused_disk` 停止。运行期检查点保留每个尺寸的校正进度和随机状态；已提交 HDF5 分片可在检查点写入中断后恢复。

## 几何与验收

物体先用封闭 primitive 三角网格表示。球形工具从外表面种子向内扩大，布尔切削的实际移除体积不超过 50%。细长物体使用外侧相切工具，降低直接切断整个截面的概率；仍产生碎片、内部孤立空洞或不封闭网格的提议会被拒绝。极细圆柱先在单位尺度建模再缩放，避免端面退化删除。

体积点从原 primitive 的均匀分布提出，再对保存的切削网格做 winding 内部判定。因此保存的是变化后实体内的条件均匀点云，不给点云额外加高斯噪声。16、22 类薄板采用仓库 benchmark 规则 `[2r, 1.4r, h]` 重建；这是明确记录的尺寸约定，不是补出的原始标注。

模板接触锚点映射到真实 Wuji2 三角形。各 patch 为同一刚体表面内、锚点周围 10 mm 的连通测地区域，候选点有固定三角形编号和重心坐标。每个 patch 只判断是否存在 `-0.005 < d <= 0` 的候选点；至少 `ceil(0.8 * N)` 个 patch 命中。没有面积覆盖率要求。滑动样本另外要求 `-0.0001 <= d <= 0`、相对法向误差不超过 30°；优化距离目标为零，数值求解目标向内偏置 20 微米以避免微小正间隙。

手物穿透使用双向真实表面查询：全部手网格顶点与面心对物体、全部物体顶点与面心对手。自穿透复用 GeoRT 原生检测及机械连接排除规则。手物最大穿透与整手最大自穿透均不得超过 5 mm。无效查询、非法关节和不合格接触均不写入训练分片。这是密集采样验收，不是连续三角形无交叉证明，也没有动力学稳定性标签。

## 数据格式

- `config.json`：单位、原生 `joint_names`、关节范围、源文件与代码指纹、固定阈值。
- `templates/*.npz`：固定 patch、映射、局部原始网格；这里的顶点/面顺序定义接触三角形编号。
- `geometry/*.npz`：共享切削几何、球形工具、种子和实际体积比例。
- `calibration/*.npz`：逐尺寸校正结果；`accepted=false` 的行不是训练数据。
- `baseline/*.npz`：独立的近模板高斯扰动合格基线，不计入正式配额。
- `shards/<type>/*.h5`：原子提交、无损压缩、逐行对齐的合格训练数据。
- `status.json` / `checkpoint.json`：真实计数、随机状态、有限姿态池、拒绝原因。
- `audit_report.json` / `report.json` / `manifest.json`：密集复查、覆盖诊断、文件 SHA256 和是否完成。

HDF5 主要字段：`qpos[N,20]` 为 GEORT 原生顺序弧度；`target_base_pose[N,7]` 与 `object_pose_world[N,7]` 为米和 WXYZ；`pose_object_local[N,26]` 为求解器辅助姿态（XYZ、intrinsic XYZ 欧拉角、20 关节），不能当成 GeoRT pose 使用。

`volume_points_m[N,1024,3]` 默认在物体局部坐标。`grasp_type` 保留原分类编号（1–33 中的 31 类，不包含 19、21）。接触字段包含 `contact_owner`、`contact_face_id`、`contact_barycentric`、`contact_local`、世界坐标接触位置及法向、`patch_hit`、有符号间距与法向点积。未命中 patch 的见证点同时保存，但不能当作有效接触点；训练时必须读取 `patch_hit`。

```python
from retarget.env.grasp.primitive_augmentation.storage import Dataset
ds = Dataset('data/datasets/wuji2_primitive_aug_v1')
path, row = next(ds.records())
sample = ds.sample(path, row, world=True)  # 点云转换至保存的世界坐标，float32 米
q = sample['qpos']
contacts = sample['contact_position_world'][sample['patch_hit']]
```

## 覆盖率的含义

全域 Sobol 提议、可累计远离模板的连续游走、共享姿态池的稀疏补采及跨类型关节提议都必须重新投影并通过目标抓型验收。按 `1e-4 rad` 量化去除同类重复关节姿态。

`report.json` 记录归一化原始 20 维空间的逐关节分箱、熵、最近邻距离和变异系数、类型间邻接及有限样本 epsilon 图连通分量。图边阈值为 L∞≤0.1 且 L2≤0.3，不能解释为已验证的连续可行路径。PCA 图只用于查看，不表示覆盖率。接触可行域可能真实断裂，报告不会宣称整个 20 维可达域被均匀覆盖。

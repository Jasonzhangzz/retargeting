# Shadow → MANO 接触映射检查

当前用户要求先检查可视化，尚未启动全量 MANO MPPI。

- 交互页：http://192.168.0.32:8099/
- 静态图集：http://192.168.0.32:8100/
- 输出：output/dexonomy_shadow_mano_mapping_20261103
- 入口：scripts/inspect_shadow_mano_mapping.py
- 静态渲染：scripts/render_shadow_mano_mapping.py
- 检查：python -m unittest tests.test_shadow_mano_mapping -v

31 类、每类模板与一条真实缓存姿态，共 62 个姿态；259 个模板标注点。
原始 Shadow 数据没有逐样本接触标签，多数原始 grasp 无零间隙接触。
因此这份映射的源点是仓库中原始抓型模板 YAML 标注，在各自手指/指节局部坐标重放，
不是伪造的数据集接触真值。缓存姿态页另外列出实际几何接触数和到物体的投影距离。

中性姿态下，先按同名手指/指节建立解剖骨架坐标，按骨长转移目标点与法向，
然后仅在对应 MANO 皮肤权重区域选择三角面并保存重心坐标。
换姿态时使用同一三角面及重心坐标重建点，不重新在所有手指上选最近点。
全手 LBS 和骨架初始化使用已上传的标准右手 MANO，beta=0、单位比例。
当前展示是骨架 IK 初始化，尚未运行物体 MPPI。

中性映射偏差中位数 1.64 mm，95% 分位约 5.02 mm，最大 10.19 mm。
6 类 #5、7 类 #4、8 类 #3 是共用拇指末端标注，超过 10 mm，需要检查。
6 个真实模型一致性测试通过（含姿态后法向无翻转），但这不构成所有解剖映射或抓取可行性已确认。

MANO 权重位于 data/grasp/assets/mano/MANO_RIGHT.pkl，附原始 LICENSE.txt 和来源摘要；
权重来自用户本地已有文件，经明确要求上传，SHA256：
45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767。

## 全量 MPPI 后台任务（用户已明确启动）

输出目录：results/dexonomy_mano_mppi_full_20261103
主进程/worker PID 以该目录的 process.json、worker_*.json 为准。
设备 cuda:1、cuda:2；batch 64、population 512、48 轮 MPPI、
120 步骨架 IK 初始化、候选分块 32768；每卡 Torch 显存预算最高 28 GiB，
同时限制为启动时空闲显存的 80%。遇到 Torch OOM 时减半 pose batch 重试。

进度页：http://192.168.0.32:8100/progress.html
原始状态：results/dexonomy_mano_mppi_full_20261103/status.json
日志：results/dexonomy_mano_mppi_full_20261103/run.log

全量入口 scripts/retarget_dexonomy_mano.py 已消费可视化中固定的 MANO 三角面/重心坐标。
所有抓型的映射冻结在输出根目录 fixed_contact_mapping.json，并参与配置指纹。
6/7/8 类的拇指复核标记保留，不静默修改或排除。
目标包含骨架保真、固定接触槽到原始物体表面的距离/法向、
相对于 Shadow 标注投影的接触位置保真及物体穿透惩罚。
该目标不包含动态稳定性或力闭合求解，task_success 仍为 null。

每个源分片在对应 grasp_type/shards/index 下输出与 primitive 相同类别的容器：
poses.npz、manifest.json、templates.json.gz、metrics.json.gz，
另外 contacts.npz、geometry.npz、mano_model.json、provenance.json、
checksums.json 和 verification.json。
MANO 维度为 qpos 51 / grasp_qpos 52，定义记录在 mano_model.json，
不能套用 Wuji 的 20 维手指角解释。

poses.npz 的 projected_points/contact_counts 对应固定映射槽；
geometric_contact_counts 单独记录所有 778 个皮肤顶点的实际近表面查询数量。
contacts.npz 同时保存固定映射点、物体投影点、法向/距离/物体三角面重心坐标，
以及另行采样的皮肤顶点接近点（ragged offsets）。
模板源点是抓型标注重放，并非原数据逐样本接触真值，来源均明确保存。

geometry_accepted 仅表示：物体 watertight、全顶点最大穿透不超过 2 mm、
至少两个几何接近点、至少 80% 固定槽同时达到 2 mm 间隙和 30° 对向法向。
它不是抓取稳定性或力闭合认证。质量不达标的候选同样保存；
运行异常写 failure.json 并保留源分片/样本引用。

使用 --watch 持续消费上游新增的已完成 Shadow 分片，直到上游 completed。
主计数仅包含已验证并写盘的分片；active/*.json 记录当前分片的批内进度。
重复启动同一输出由文件锁拒绝；以相同配置重启会重用已验证结果，
配置、模型或映射指纹不兼容则报错。
每个批次种子、实际 batch 和源权重/数据指纹记录在 provenance.json。

验证：两卡 12 个真实样本、6 个抓型的 smoke 已完成，运行错误为 0；
python -m unittest tests.test_mano_mppi_mapped tests.test_shadow_mano_mapping -v
共 10 项通过，包括固定槽重建、源 FK、对象变换/重心坐标、保存格式、
MPPI 代价不增、断点重用及不兼容权重拒绝。
源码快照与验证记录保存在全量输出的 implementation/ 和 validation.json。

为处理 CPFS 偶发 ESTALE，后台启动器 scripts/run_dexonomy_mano_background.py 对读写/哈希/分片读取增加有界指数退避。该适配层不修改求解参数或配置指纹；科学计算源码未变，原已验证分片继续重用。当前实际 PID 以 launch.json/process.json 为准。

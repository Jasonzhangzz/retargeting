# 七类抓型逐阶段验收

当前策略为 `whole_hand_v1`，覆盖 9/10/12/22/23/30/31。手势正确性待用户审阅；原生六方向物理评测独立报告，不能替代手势验收。

初始化先在 CUDA 上联合拟合二十个关节，以骨方向、指间位置和接触法向为软参考，再使用真实三角网格与 MuJoCo 碰撞调整开口。保持接触手指及 required-body 分组；type 23 保留食指/中指侧面接触，type 31 保留手指环抱结构。直指参考对末端弯曲使用较高但有限的软权重，没有硬姿态或皮肤区域裁剪。

局部几何拟合保持 0.7 mm 目标接触间隙，反复选择暴露的真实手表面点。筛选区分原始标注槽位检查与真实接触检查，并保存 padding 的真实来源。完整原生初始化检查合格，或真实 required-body 接触、原生 QP、关节限位及本地几何检查合格，才进入 ADMM。初始化的重叠上限为 1 mm；后续原生 grasp filter 的零穿透和六方向测试不变。未合格时保存拒绝候选，不冒充可用初始化。

ADMM 使用软势能对应的关节、指间位置、法向及朝向力；MPPI 使用相同来源的 `gesture_reference`，并对接触面添加有限软偏好。实际碰撞检测覆盖全部手部。独立参考不会被本次 ADMM 输出覆盖；原生外力评测不安装软参考或朝向锁。

type 12 的参考明确取自归档 `t12_disk_s04_seed00`，策略 `native_validation_fallback` 所选的 MPPI 输出。保存精确原始姿态、尺寸、物体相对掌面变换和来源哈希。1× 保留该姿态作为初始候选，其余尺寸适配开口；参考朝向贯穿优化。新尺寸是 1、1.125、1.25、1.375、1.5，半径 45–67.5 mm，厚度 16 mm。新旧尺寸索引通过策略版本区分，历史记录不重新标注。

type 9 的旧 +90° 固定朝向被撤回为未确认参考。审阅页提供 1.5× 下绕掌面 +Y 的 0/90/180/270° 方向对照；这些对照仅用于选方向，不计为合格试验。

## 复现与审阅

使用 pi0 派生的 `.venv-graspadmm`（原生 ADMM/评测）与 `.venv-contact`（MPPI）。两者继承 pi0 的 Torch 2.10.0+cu128；宿主 CUDA 为 RTX 5060 Ti。需要允许 GPU 设备访问，沙箱内不可见不代表宿主缺少 CUDA。

```bash
.venv-graspadmm/bin/python scripts/run_graspadmm_batch.py --output grasp_gesture_review_admm_seed0_v3 --types 9 10 12 22 23 30 31 --seeds 0 --workers 2 --timeout 300
.venv-contact/bin/python scripts/refine_graspadmm_mppi.py --source output/grasp_gesture_review_admm_seed0_v3 --output output/grasp_gesture_review_mppi_seed0_v3
.venv-graspadmm/bin/python scripts/review_grasp_gestures.py --source output/grasp_gesture_review_admm_seed0_v3 --refined output/grasp_gesture_review_mppi_seed0_v3 --output output/grasp_gesture_review_seed0_v3
python run.py --config configs/web_grasp_gesture_review.yaml
```

审阅目录包含 `index.html`、`review.json`、`audit.json`、每例双视角对照图及独立 viewer manifest。对照包括原始标注、旧逐指 Wuji 模板、新整手模板、初始化、ADMM 和 MPPI。被拒绝候选明确标红，后续缺失输出保留空缺；历史发布包没有保存的初始化明确标为缺失。

开发校验批次 v1、v2 保留供追踪；v2 在统一接触面软偏好接通后主动停止。正式审阅使用完整冻结的 v3，所有尺寸均只运行 seed 0。不存在按物理测试结果重采样或挑选 seed 的流程。

当前正式 viewer 的配置及历史结果保留。只有手势确认后才运行 seed 0–9 共 410 例，并另行审计、接入正式 viewer。

## 本轮结果

41 例均完成记录，32 例通过初始化几何检查。原生物理通过为 ADMM **17/41**、MPPI **15/41**；单独报告的原生验证后回退策略为 **17/41**。MPPI 保留 15 例、丢失 2 例，没有恢复失败例；没有把回退结果计入直接 MPPI 成功数。

| 类型 | 例数 | 初始化合格 | ADMM 物理通过 | MPPI 物理通过 |
|---|---:|---:|---:|---:|
| 9 | 5 | 5 | 1 | 1 |
| 10 | 11 | 9 | 9 | 8 |
| 12 | 5 | 5 | 4 | 3 |
| 22 | 5 | 5 | 0 | 0 |
| 23 | 5 | 5 | 0 | 0 |
| 30 | 5 | 0 | 0 | 0 |
| 31 | 5 | 3 | 3 | 3 |

仍未解决的部分明确保留：type 10 的 1.45/1.5×、type 30 的全部五档、type 31 的 0.5/0.75× 没有合格初始化；type 22/23 有初始化但尚未通过物理评测。不得将这些结果表述为七类已经全部修好或手势已经验收。

type 12 的新 1× 初始关节角与指定来源逐值一致，完整姿态的最大浮点表示差异约 `2.13e-8`。五档均保存同一来源的姿态与朝向参考。来源、预算、原生物理判据和失败分母审计通过；187 个审阅阶段的模型与姿态均已加载检查。

当前本地服务：[双视角审阅页](http://localhost:8085/)；[独立 3D 预览](http://localhost:8088/)。3D 中可通过“结果”选择同一实例的各阶段，正式 viewer 仍在原端口与批次。

# Type 6–9 软中心偏好与统一参考姿态实测

2026-11-02 在 pi0 的 CUDA 环境上实际完成 20 例 smoke 和 200 例完整配对试验。
完整批次覆盖四类、五个尺寸、seed 0–9；没有运行异常、超时或按结果重采样。

6–8 的初始化采用 distal 掌侧中心点，ADMM/MPPI 保留整个暴露 distal 的候选，
使用有限软偏好 `0.25 * ((u - 0.5) / 0.2)^2` 及掌侧法向偏好。
0.3–0.7 仅作为中心接触诊断带，不是硬约束，也不影响原生通过判据。
smoke 中选出的优化锚点实际覆盖 u=0.37–0.91，确认区域外点仍可被选择。

9 对每个随机 seed 复用一次 1.5× 原生初始抓姿，在掌面内绕物体中心旋转 +90°，
所有尺寸复用参考朝向。10 个参考初值均通过来源哈希、关节角逐位一致及跨尺寸
一致性检查；ADMM 仿真记录与 MPPI 输出的最大朝向误差为 0.00001132°。
外力评测保持自由运动。

| Type | 例数 | ADMM 物理通过 | MPPI 直接物理通过 | 独立验证后回退策略 | MPPI 实际中心接触诊断 |
|---|---:|---:|---:|---:|---:|
| 6 | 50 | 4 | 5 | 7 | 33/50 |
| 7 | 50 | 8 | 6 | 9 | 17/50 |
| 8 | 50 | 1 | 1 | 1 | 1/50 |
| 9 | 50 | 0 | 0 | 0 | 不适用 |
| 合计 | 200 | 13（6.5%） | 12（6.0%） | 17（8.5%） | 51/150 |

“实际中心接触”要求该例全部必要手指均有接触处于诊断带，单独报告，不用于裁剪
候选或替代物理成功。允许区域外接触是软偏好的预期行为；目前 type 8 的中心诊断
比例和 type 9 的物理稳定性仍较低。MPPI 保留 8 个原成功、损失 5 个、修复 4 个，
其余 183 个仍失败；这些失败全部留在结果中。

通过判据仍为 `graspadmm_native_v1` 的原版过滤、squeeze/pregrasp 和六方向外力
测试。MPPI 固定每例 4096 个采样，不在优化内调用物理评测选解。
MPPI 纯优化合计 49.80 秒、每例中位 0.249 秒；含准备、I/O、评测和汇总的
串行墙钟为 75.51 秒，均不包含前置 ADMM 批次时间。

GPU 为 RTX 5060 Ti，Torch 2.10.0+cu128，Python 3.11.15。ADMM/评测使用
MuJoCo 3.3.3、Warp 1.8.0；MPPI 使用 MuJoCo 3.10.0、MJWarp 3.10.0.3、
Warp 1.16.0。两个隔离环境继承 pi0 的 Torch，pi0 本身未修改；与历史
RTX 4090/Torch 2.5.1 的数值和时长不作同环境比较。

结果目录：

- `output/graspadmm_soft_center_200_pi0`：原生初始化、ADMM 及其实际评测。
- `output/graspadmm_soft_center_mppi_200_pi0`：全部 MPPI 输出、原生评测、逐例统计与图表。
- 后者的 `audit.json`、`source_audit.json`、`region_pose_audit.json` 均通过。
- 先前硬区域 smoke 和中断的 149/200 批次保留在独立目录，不混入这份结果。

`configs/web_grasp_best.yaml` 优先展示这 200 例的原始 MPPI 结果，包括失败案例；
其余抓型继续显示 `graspadmm_mppi_best` 历史回放。同一例不会因新结果失败而退回
历史抓姿。上表的“验证后回退策略”仅作为独立统计，不冒充 MPPI 直接输出。

复核命令：

```bash
.venv-contact/bin/python scripts/audit_graspadmm_mppi.py --batch output/graspadmm_soft_center_mppi_200_pi0
.venv-contact/bin/python scripts/audit_grasp_region_pose.py --batch output/graspadmm_soft_center_mppi_200_pi0
.venv-contact/bin/python run.py --config configs/web_grasp_best.yaml
```

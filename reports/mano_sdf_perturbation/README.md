# MANO 手指扰动后的 AdamW 端到端操作验证

历史协议：本报告保留旧的差分参考输入和结果。当前 `--noise` 已更改为直接 MANO 重定向、首帧即施加扰动；新结果与耗时见[直接参考测试](../mano_direct_reference/README.md)。

输入先验核对：本实验保留了原手腕轨迹、接触时序/手部语义对应，并以原可行 Wuji 参考做差分扰动。9/9 不能解释为无可行先验的自主抓取恢复；objective 确实接收扰动后的物体坐标和手指参考，详见[输入审计](../noise_input_audit/README.md)。

本实验的验收对象是实际执行的 hand action：输入扰动后的 MANO 手部参考和由其投影得到的错误模板，是否仍能抬起物体并完成原物体轨迹。接触点恢复情况不作为成功条件。

完成 33 组 ComFree 闭环仿真：9 组扰动 × 3 种方法，加原始模板和零扰动 SDF 投影两个基线 × 3 种方法。默认 `--adamw --contact-solver sampled-lm`、QP 和固定错误模板的 AdamW 对照均在 9/9 组扰动中完成抬升与整段轨迹运动。

进一步要求起升时序同步时，采样 LM 和 QP 各有 8/9 组满足下面的严格验收条件。未满足的是食指 0.60 rad：采样 LM 在参考已离桌 5 mm 后还有连续 8 个采样点低于此高度，QP 为 10 个点，分别约滞后 80 ms 和 100 ms；随后都完成抬升，运动终点分别抬升 173.8 mm 和 169.3 mm。没有把这两组记作严格通过。

固定错误模板的对照也能完成操作，说明这些结果证明了 AdamW 动作优化对这组输入的容错能力；单凭这些结果不能把成功归因于接触点更新，也不能保证任意错误模板都能恢复。

后续已补跑真正关闭一致性惩罚、dual 和接触记忆的 `--no-admm`：同样 9/9 完成抬升与整段运动，7/9 满足严格起升同步要求，见[完全无 ADMM 对照](../mano_no_admm/README.md)。

## 输入生成与公平对照

- 解码与示例完全对应的原始 MANO 参数，解码顶点与原始可视化最大差异为 1.23e-07 m。
- 拇指、食指、全手三种模式；分别对所选手指三个关节的局部 y 轴旋转加 0.15 / 0.35 / 0.60 rad 扰动。扰动在参考帧 21–23 渐增到全幅度；原接触 mask 从帧 21 开始激活。
- 重新解码 MANO；保留原来的 19 个语义接触顶点索引，用三角网格精确最近特征的 SDF 投影 `p - phi(p) grad(phi(p))` 得到新的物体模板。没有对模板进行抓取修复。
- 扰动 MANO 通过手指链方向变化的差分运动学拟合映射为 Wuji 参考。手腕、未扰动手指、物体参考轨迹、接触 mask/权重/slot 对应保持原样；映射过程中没有接触约束优化。
- 每窗口 1 次 hand 更新、1 次交替轮次，8 个初值，seed=0，物体侧半径 5 mm。各方法使用同一份扰动参考和模板。
- 固定模板对照通过实验子类仅冻结物体接触变量，保留相同的物理 wrench、ADMM 一致性项和 5 mm 归一化参数，避免用 radius=0 同时关闭其他项造成混淆。
- 一次闭环 3.6 s，完整参考运动到 2.37 s，随后保持终点约 1.23 s。求解器与示例配置的 SHA256 已记录，矩阵执行期间保持不变。

## 验收规则

运动区间使用接触激活后至 2.37 s 的实际执行误差：位置 RMSE ≤20 mm、姿态 RMSE ≤15°，且运动终点位置误差 ≤20 mm、姿态误差 ≤15°；终点物体中心相对初始抬升至少 50 mm。这些条件定义 `lift_and_motion_complete`，三种方法均为 9/9。参考终点抬升约 169.7 mm。

更严格的 `operation_pass` 还要求：在参考物体离桌 5 mm 以上的运动采样点中，实际物体至少 95% 的采样点也离桌 5 mm 以上。采样 LM 与 QP 各为 8/9，固定模板对照为 9/9。

这是明确的实验验收阈值，并非零误差跟踪。运动完成后的保持误差单独列于 CSV，不混入运动通过率；通过运动验收不代表终点已经完全静止。

## 运动区间结果

表内数值为位置 RMSE（mm）。模板偏移是相对同一 MANO 零扰动最近投影模板，在激活接触中的最大位移；原始模板一行不适用。

| 输入 | 模板最大偏移 mm | 固定模板 | 采样 LM | QP |
|---|---:|---:|---:|---:|
| original | — | 7.49 | 7.91 | 8.83 |
| projected_zero | 0.00 | 10.60 | 9.22 | 9.23 |
| thumb_150 | 4.49 | 9.21 | 10.40 | 9.93 |
| thumb_350 | 19.93 | 8.61 | 7.84 | 9.99 |
| thumb_600 | 28.54 | 9.27 | 8.15 | 6.74 |
| index_150 | 3.51 | 7.82 | 17.05 | 7.60 |
| index_350 | 8.01 | 11.06 | 9.08 | 13.25 |
| index_600 | 13.50 | 14.16 | 11.42 | 17.62 |
| all_150 | 11.00 | 8.39 | 11.70 | 11.51 |
| all_350 | 25.18 | 9.25 | 5.75 | 6.46 |
| all_600 | 41.63 | 8.20 | 9.50 | 8.77 |

## 强扰动下实际求出的操作

| 输入 | 方法 | 终点抬升 mm | 运动位置 RMSE mm | 姿态 RMSE ° | 运动终点位置误差 mm | 验收 |
|---|---|---:|---:|---:|---:|---|
| thumb_600 | sampled-lm | 178.00 | 8.15 | 3.19 | 8.45 | 通过 |
| thumb_600 | qp | 172.91 | 6.74 | 3.03 | 8.63 | 通过 |
| all_600 | sampled-lm | 175.71 | 9.50 | 4.30 | 7.85 | 通过 |
| all_600 | qp | 180.54 | 8.77 | 5.09 | 15.24 | 通过 |

回放中的橙色半透明手为扰动 MANO，青色物体为原始示范目标，实体手与纹理物体为实际执行状态。视频也包含运动结束后的保持区间：

- [拇指 0.60 rad，采样 LM](videos/thumb_600_sampled_lm.mp4)
- [全手 0.60 rad，采样 LM](videos/all_600_sampled_lm.mp4)
- [全手 0.60 rad，QP](videos/all_600_qp.mp4)

## 数据与复现

- [完整操作指标](operation_results.json)、[简表](summary.csv)、[运动误差图](tracking_matrix.png)。
- `inputs/<condition>/mano_input.npz` 保存扰动后的 MANO 参数与投影对应；`contactaware/` 保存实际输入参考。
- `runs/comfree/<condition>__<method>/rollout.npz` 保存求得的 `ctrl_traj` 和实际执行的 `qpos_traj`/`qvel_traj`；`contacts.npz` 保存对应接触变量。
- `manifest.json` 记录输入生成、求解器版本与参数。其 LP 可行性字段和另存的局部 wrench 诊断是辅助的四棱锥模型结果，不作为本次端到端验收依据，也不等同于圆形摩擦锥的严格不可行性证明。

复用已准备输入运行整个矩阵（已有完整结果会跳过）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_mano_sdf_perturbation.py --run-only
python scripts/analyze_mano_sdf_perturbation.py
```

直接运行强扰动下的默认分支：

```bash
python run.py --adamw --experiment reports/mano_sdf_perturbation/inputs/all_600 --multistarts 8 --iterations 1 --contact-rounds 1 --contact-solver sampled-lm --duration 3.6 --seed 0
```

交互回放已保存的动作：

```bash
python run.py --replay --experiment reports/mano_sdf_perturbation/inputs/all_600 --case reports/mano_sdf_perturbation/runs/comfree/all_600__sampled-lm
```

从原始 MANO 重新准备输入时需另设输出目录，并提供本地有许可的 MANO 模型与 manopth；本仓库不复制模型权重。脚本参数可通过 `--help` 查看。

范围限制：只有 sugar box 一条示例、一个随机种子及上述九种扰动；这是 ComFree 模型中的闭环执行验证。此实验不构成跨物体、任意扰动或真实硬件成功率结论。投影解析盒体测试和现有回归测试合计 57 项通过；保存的视频也已检查。

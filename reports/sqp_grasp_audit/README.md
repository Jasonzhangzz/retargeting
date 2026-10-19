本次检查的结论是：原实现同时存在接触规划不一致和 action 路径错误。结构错误已修正，但当前 SQP 仍未通过 box 的稳定抓取验证。

- 原 SQP 的 `tips + J @ (hand - x[7:7+nu])` 中两个 hand 是同一变量，对齐项对手指的梯度恒为零。前四个有效接触槽还会漏掉拇指；第 23 帧只覆盖三根手指，第 24 帧只覆盖两根。原始探针见 [old_action_structure.json](old_action_structure.json)。
- 原 SQP 使用的 `SurfaceContactObjective` 与默认 AdamW 后续执行的 `LocalContactExplorer` 不是同一个物体子问题。旧的“同一个 AdamW 类，所以接触点一致”的判断不成立。
- 当前 SQP 复用 `LocalContactExplorer` 的真实网格 patch、硬半径、wrench、一致性和位移正则。AdamW 保留 rollout + dual 的三轮交替更新；SQP 使用当前测量的手部点和零 dual，避免 forward ComFree rollout。因此只保证完全相同子问题输入的结果一致，不能保证两条完整分支的最终点一致。

第 23 帧固定相同初态、19 个槽、31 个阶段、单候选：相同输入的物体子问题输出逐元素一致，最大偏差 0 mm；完整 AdamW 三轮交替后的接触点与 SQP 测量输入的计划最大差 5.338 mm。数组和设置见 [mesh_contacts.json](mesh_contacts.json)、[mesh_contacts.npz](mesh_contacts.npz)。复现：

```bash
python scripts/audit_sqp_contacts.py --contact-rho .1
```

当前 action 修复保留全部 19 个槽，使用世界坐标查询点计算 MuJoCo Jacobian，补入物体相对 Jacobian，并让模型中的累积 servo target 与执行的绝对 action 相同；同时修正实际执行器增益/惯量输入、正向 softplus 梯度以及地面支撑预测。预测窗口为 30 × 10 ms。CasADi 对这个简化模型求解析导数；该模型仍不等于原生 Warp ComFree，目标也不是完整 AdamW 目标，不能宣称已经完全移植 trigrasp。

两次各 5.4 秒的闭环对照（默认 ComFree 执行器，rho=.1）：

| SQP 物体点设置 | 最后窗口拇指对齐误差 | 最终物体位置误差 | 参考帧 | 稳定抓取 |
|---|---:|---:|---:|---|
| 固定 MANO 点，半径 0 | 77.85 mm | 5.04 mm | 23 | 未通过 |
| 共享网格搜索，半径 5 mm | 74.46 mm | 9.01 mm | 23 | 未通过 |

两次 180 个窗口的 QP 都返回 status=0，但均没有通过接触门控；QP 成功不能当作物理任务成功。详细末端误差和接触门控见 [closed_loop_ablation.json](closed_loop_ablation.json)。末端误差取最后一次规划前的测量手部查询点到该窗口目标接触点的欧氏距离。

```bash
python run.py --sqp --contact-rho .1 --contact-radius-mm 0 --duration 5.4 --case-name audit_fixed_contacts_final --output-base /tmp/retarget_grasp_audit
python run.py --sqp --contact-rho .1 --contact-radius-mm 5 --duration 5.4 --case-name audit_shared_mesh_final --output-base /tmp/retarget_grasp_audit
```

另外，固定最后窗口的物体与手掌，仅优化手指关节，精确 FK/Jacobian 的运动学拟合能把拇指误差从 74.46 mm 降到 8.12 mm，见 [finger_kinematic_feasibility.json](finger_kinematic_feasibility.json)。此对照没有碰撞约束，只用于确认 action 尚未实现可得到的关节对齐，不能当作抓取可行性证明。固定点与表面更新两组都失败，结合原 action 的零梯度和漏指问题，说明当前主要瓶颈在 action 模型及目标实现，不能仅靠 5 mm 表面调整解决。

验证：32 项测试通过，包含全部五指覆盖、世界点 Jacobian 有限差分对照、指关节梯度能降低真实 FK 对齐误差、servo 预测/执行一致和接触子问题同输入输出一致。当前未宣称稳定包络已修复。

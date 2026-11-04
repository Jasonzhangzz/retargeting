# 原生刚体 MANO 的 GPU 接触适配

`native_mjwarp.NativePhysics.projected_closure` 使用当前模型的 geom margin、gap 和投影所属 geom ID，计算 margin 内有效槽的投影闭合。隐式接触对沿用 MuJoCo 3.3.3：

```text
margin = max(hand.geom_margin, object.geom_margin)
gap = max(hand.geom_gap, object.geom_gap)
detected = signed_distance < margin
force_active = detected and signed_distance < margin - gap
```

显式 contact pair 使用自己的 margin/gap。距离单位为米，外部分离为正、穿透为负；模型加载可以在 CPU，筛选、投影闭合 QP 和结果计算在 GPU。缺失 geom、非有限距离或无有效接触均不能获得闭合认证。2 mm 穿透验收与 margin 接触激活分别判断。

QP 仍为既有摩擦 0.6、13 工况、96 次更新、误差阈值 0.05 的投影诊断。每个有效接触的压力约束和原目标不变；无效槽压力固定为零，数值结果核对为直接移除这些槽后的相同 QP。margin 不移动物面投影点，也不修改力矩臂。

调用 `ProjectedQPClosure.solve_with_geom_margins` 返回 `(ClosureResult, ContactMarginSelection)`，后者含有效槽 mask、检测 mask、组合后的 margin/gap 和有限性。`NativePhysics.projected_closure` 从模型读取这些参数，并处理显式 pair。调用方须提供实际投影所属的 hand/object geom ID，不可用任意 geom 的 margin 代替。

评测最终 grasp 时须恢复该 grasp 的模型 margin 与姿态。pregrasp 阶段临时增大的 45 mm margin 不能用于最终 grasp 的闭合统计。

`ProjectedQPClosure.solve` 未指定 mask 时继续给出全部投影槽的假想闭合诊断；两类结果必须分列。投影目标自身闭合、margin 内实际手点投影闭合和原生动态 GSR 不能互相替代。当前四卡 CEM 的 objective 与已运行 worker 保持原配置，不因诊断接口修改自动切换。

检查：`tests/mano/test_projected_contact_margin.py` 覆盖旧版 CPU 接触激活对照、max 与 sum 的区别、gap/边界、显式 pair、无接触、非有限输入、有效槽子集等价性、GPU 投影及 CUDA Graph 重放、模型 margin 动态读取。测试仅做固定输入验证，不更新 pose。

该文件记录接触适配接口；完整 GPU ADMM 的端到端质量与大规模启动仍需另行验收。

## 重启预检

`scripts/restart_native_mano_gpu.py` 在固定的 16 条已有 Shadow 来源 MANO 初值上执行 GPU ADMM，再执行原版 filter/squeeze/pregrasp 和 margin 内投影闭合诊断。这批输入用于验通端到端路径，不等于启动全量数据集。

模型参数由原版 CPU 3.3.3 环境静态导出；输入姿态的变换、物面查询、QP、接触梯度、虚拟力推进和评测均在 GPU。`native_model_assets.py` 按 body 名称恢复物理参数，验证父子关系、geom/joint 所属 body 和 equality 绑定，避免 XML 序列化将 mocap 移到末尾后错配质量/惯量。Weld 的原 11 个 `eq_data` 字段完整保留；新版本第 12 字段保留编译器默认值。

当前重启显式使用 eager CUDA MuJoCo Warp 步进，避免默认 stream 的 Graph 捕获错误；没有 CPU solver fallback。50×10、原六系数 ADMM QP 和原 filter QP 不变。Graph 调度优化单独验证，不把它作为更换物理模型或目标的理由。

服务器预检目录为 `retarget_mano_mjwarp_20261104/restart_margin_gpu_preflight_03`。前两次分别在模型布局检查、Graph 准备时停止，均未提交 ADMM 更新；失败记录和相同输入清单保留。查看 `status.json`、`worker.log` 与实际提交的 `solve_receipt.json` 区分准备、求解和结果。首次求解的手面 BVH 准备仍包含在 ADMM 段计时中，不能作为充分预热后的稳定性能数字。

输出 `poses.npz` 为最终状态，不保存完整优化历史。现有输入槽的法向覆盖先标记为 transported-input 诊断；实际更新手面法向、六外力/六纯扭矩动态验收尚需完成。预检未通过前，`production_launch_allowed=false`，生产 CEM 与 CPU 大规模 STOP 状态不变。

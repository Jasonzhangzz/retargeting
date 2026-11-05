# GraspADMM 刚体 MANO 的 GPU 适配

`native_admm.py` 保留原接触搜索、接触力目标、对偶及弹簧公式；`native_mjwarp.py` 使用 MJWarp 推进刚体手，`native_batched_qp.py` 求解 GraspADMM 自身的优化 QP 和原生 filter QP。

内部 QP 保留每接触点六个系数、原目标和约束：优化系数下界 0.01、每点和不超过1、总和至少1。filter 使用其原生约束。GPU 求解器独立核对原始/对偶可行性、stationarity 和 complementarity；不以增加正则或 CPU fallback 掩盖失败。相关验证见 `tests/grasp/test_native_batched_qp.py`。

模型加载与资产导出可以在 CPU；姿态推进、物面查询、接触梯度、批量 QP 与动态评测使用 GPU。`native_model_assets.py` 按名称和所有关系恢复物理参数，保留 equality/weld 信息并检查绑定。原版 CPU Clarabel＋MuJoCo 路径与 GPU 适配分别维护。

入口包括 `scripts/restart_native_mano_gpu.py`、`scripts/run_native_mano_gpu_cohort.py` 和独立数据集/原版复测工具。原生输出、接触覆盖、碰撞、六方向外力 GSR 和扭矩稳定性分别记录；几何联合通过仅表示覆盖及碰撞检查，不代表动态稳定性。

最终状态保存后再做独立物理验收。STOP、来源和配置指纹仍生效；代码整理不会恢复或启动生产任务。原生内部 QP 是 GraspADMM 算法的一部分，并未作为 CEM 评分使用。

# GraspADMM

唯一方法入口为 `retarget/solver/graspadmm.py:solve`，默认参数由同文件的冻结对象 `DEFAULTS` 定义。YAML 只设置 `solver.type: graspadmm`；执行时不覆盖算法参数。

```python
from retarget.solver.graspadmm import solve, NativeProblem
result, status, trace = solve(NativeProblem(initial_data, scene, output_directory))
```

GPU 调用使用 `GPUProblem`，传入物理状态、接触数据、物面查询和 QP 后端；不在业务脚本里重写 ADMM 循环。CPU 路径直接执行固定上游 operation，GPU 路径保留相同物面更新、原接触 QP、虚拟力手步和对偶公式。模型接口、刚体几何与执行后端的适配单独验证。

默认50轮，`step_obj=1`、`step_hand=1`、`substep=10`、`rho_admm=1000`、`lr_obj=0.0002`、`square_res=True`、`fix_hand_local_cp=False`、优化摩擦参数 `[0.1,0]`。默认初始化采用一次 epoch、一个最终候选、1024采样点、8个面内旋转和100个 matcher 步。初始化预算与原生/GPU方法参数均从同一默认对象读取。

初始化骨架映射与手模型适配保留；没有历史姿态回退、姿态软参考力或区域偏好目标。内部 QP、对偶重置、fallback、filter、squeeze 和 pregrasp 属于算法本体。过滤 QP 与优化 QP 分工不同，各自的约束属于同一默认流程。

上游提交 `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`，加载时核对 `third_party/Dexonomy/GRASPADMM_UPSTREAM.json`。失败输入保留失败阶段及最后有效状态；接触覆盖和碰撞不替代动态 GSR。运行使用新目录与来源指纹，STOP 后不自动恢复。

# 原生 GraspADMM 数值与质量验证

`native_graspadmm_retest.py` 是原生 operation 的固定输入验证工具。优化和初始化参数均读取 `solver/graspadmm.py:DEFAULTS`，不维护独立预算配置。手模型、场景、尺度、模板、随机种子和输出目录是测试输入。

上游固定提交 `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`。原版初始化、接触搜索、QP、虚拟力推进、对偶、fallback、filter、squeeze/pregrasp 和六方向评测均沿用固定源码，运行前后核对哈希。CPU原版与GPU执行适配分别核对，不把数值验证当作质量提升。

验证阶段为准备、模板、初始化、质量、计时、一致性、几何诊断和报告。全部实际初始化输出进入质量分母；无输出场景、异常、原生拒绝、fallback 和缺失最终状态分别记录。计时直接调用原函数，观察器采集独立运行；失败尝试也计时，进程启动与预热分列。

`MANIFEST.json` 保存输入、配置和源码指纹；`initialization.json` 保存实际初始化数；`quality_operation.json` 和逐输入结果保存拒绝及最终状态；`native_evaluation.json` 保存动态 GSR；`diagnostics.json` 保存覆盖、接触组、CLN、CDC和穿透；延迟与吞吐分别报告。

只读回放入口是 `export_native_graspadmm_viewer.py` 和 `serve_native_graspadmm.py`。显示实际输入和最后可取得的最终状态，不用输入冒充成功结果。STOP 后不恢复，不修改生产任务；重新运行须使用匹配的代码与配置指纹。

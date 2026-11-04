# 测试

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

- `mano`：共享 CEM、四射线 batched QP、GPU 投影、MANO IK/FK、分片恢复与执行顺序。
- `grasp`：原生 GraspADMM、GPU 约束 QP、模型参数迁移与 Shadow→刚体 MANO 初始化。
- `dexonomy`：输入坐标、静态接触、缓存及 QP 证书。
- `core`：公共配置与查看器。
- `test_moxian_*`：独立硬件与协议工具，需要相应 requirements 和 SDK。

历史 MPPI、ComFree 闭合、流形学习和 LM/AdamW 抓取求解对照测试已随实现移除。LM 仍用于输入 IK，不是可选 grasp pose solver。

外部资产与完整历史数据不纳入 Git；对应集成测试明确跳过。`RETARGET_SHADOW_XML`、`RETARGET_RIGID_MANO_XML` 可指定原版刚体手资产。MANO 权重、接触映射和数据集检查不会因缺失输入生成替代成功结果。

`fixtures/grasp_objective_fingerprints.json` 固定清理前工作区中核心几何项、CEM 和 QP 的 AST 指纹，用于核对此次整理未改变公式。QP 同时与保留的历史 NumPy 数值参考比较；该参考仅用于测试。

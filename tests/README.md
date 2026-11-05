# 测试

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

- `mano`：当前 v5 patch CEM、末速度与穿透验收、GPU patch/三角形查询、切向历史、MANO IK/FK、个性化模型及抓型轮询。
- `grasp`：GraspADMM 原版来源、内部 GPU 批量 QP、模型参数迁移、Shadow→刚体 MANO 初始化和动态评测扩展。
- `dexonomy`：输入坐标、静态接触与缓存。
- `core`：公共配置与查看器。
- `test_moxian_*`：独立硬件与协议工具，需要相应依赖和 SDK。

外部资产与完整历史数据不纳入 Git；相关集成测试明确跳过。`RETARGET_SHADOW_XML`、`RETARGET_RIGID_MANO_XML` 可指定原版刚体手资产。MANO 权重、接触映射或数据集缺失时，不生成替代成功结果。

当前数值检查覆盖 GPU/参考几何一致性、接触力及 Jacobian、物理响应、CUDA Graph 重放、联合早停与可行候选保留。GraspADMM 内部 QP 与原版 CPU 实现分别核对目标、约束、wrench 和接触梯度。

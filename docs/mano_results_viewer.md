# 已求解 MANO 抓取可视化

交互页面：http://192.168.0.32:8101/
静态预览：http://192.168.0.32:8100/solved.html

页面读取 `results/dexonomy_mano_mppi_full_20261103` 中已验证完成的分片。按 grasp type、几何验收筛选，每页 100 条，支持随机浏览、刷新和手掌／手背相机。状态每 20 秒刷新；点击“刷新已完成结果”载入新增姿态。显示实际缓存的 778 顶点 MANO 表面、缩放后物体及其世界位姿、固定映射槽／物体投影、2 mm 近表面顶点、法向与原始 Shadow 骨架。可以隐藏物体或调整透明度检查被遮挡的手指。

“求解完成”不表示“几何达标”或“力闭合通过”。保留失败记录，并显示覆盖率、穿透量、MPPI cost、骨架误差和映射待复核槽。自碰撞、MANO 关节限位与动力学稳定性未评估。

力闭合在选择后用 CPU 按需计算，结果单独写入 `results/dexonomy_mano_results_viewer_20261103/closure`。使用仓库 `dexonomy_replay.closure` 与 `ProjectedComFreeClosure` 的原有判据，沿用原始缩放物体的质量、质心、惯量，默认滑动摩擦 0.6、扭转系数 0.02 m、每点名义预载上限 20 N。

分别显示：
- 固定映射槽在物体表面的投影闭合；即使手未贴合，也可能成立。
- 2 mm 内 MANO 顶点的近表面点闭合；顶点不是独立实测接触，也不代表动力学稳定。
- 几何验收与投影闭合联合结果。

保存 13 个外载荷场景的证书力、内禀扭矩与残差；投影证书通过后额外独立审计合力／力矩残差。可选择外载荷场景显示证书力箭头，这些力不是仿真实测值。未计算／计算失败明确标注，不修改原始求解缓存和指纹。

启动：
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  python scripts/serve_mano_results.py --port 8101
```

服务信息与日志：`results/dexonomy_mano_results_viewer_20261103/server.json`、`server.log`。离线图由 `scripts/render_mano_results.py` 生成，是真实 MPPI 最终结果。新增 smoke test：`python -m pytest -q tests/test_mano_results_viewer.py`（6 项，包括真实 Viser API、世界坐标与接触重建、分页筛选、证书缓存、独立残差审计和原始缓存不变）。

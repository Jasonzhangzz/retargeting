# 个人 MANO 与 WUJI2 抓取尺寸图谱

本地入口：http://127.0.0.1:8892 。左侧为保存的个人 MANO 镜像右手，右侧为缓存 WUJI2。两侧使用同一抓型、同一物体尺寸及同步视角。滑条控制直径或厚度，支持扫描、线框、透明物体与绿色二值接触区域标记。

## 资产与版本

原始不可覆盖快照位于 `output/mano_wuji_primitives_20261105/saved_left`，保存当时已应用的左手静息几何、778 顶点、16 关节、蒙皮权重、指尖锚点、材质、权重和源码副本。颜色为 `#d4ba96`；摘要记录在快照 `manifest.json`。页面可下载包含 OBJ/MTL 的存档。

修正图谱位于 `output/mano_wuji_primitives_cem_20261105_recomputed`。失败试算保留在各自目录，没有覆盖原图谱或原标定。运行时不改变手型长度、粗细、蒙皮或配色；镜像采用位置 `[-x,y,z]`、旋转向量 `[rx,-ry,-rz]` 和反向三角面绕序。此页面与实时手套 :8890 服务独立。

## 本次修正


新图谱直接 import 仓库最新 `retarget.solver.cem.solve_patch_cem` 和 `retarget.env.grasp.mano_patch_consistent.PatchConsistentAdapter`。保持原 cost、碰撞实现、默认权重、可行解保留与提前停止，不再使用图谱自定义 adapter，不实现另一份 CEM。`source/` 及 `MANIFEST.json` 保存实际使用的 solver、cost、几何和 rollout 源码及摘要。已有试算属于退役版本，不复用其优化结论。

图谱只提供固定个人几何、27 个解剖关节坐标、锚点和问题边界。四指相邻指尖间距误差是额外展示的诊断，6 mm RMS 为页面对比门槛，不修改共享 CEM 的 cost 或提前停止条件。页面分别报告 CEM 接触检查和间距诊断。

每种抓型先优化一次参考尺寸的手腕，范围为初始位置 ±25 mm、旋转向量 ±0.35 rad，之后锁住全部六个手腕变量。邻近尺寸从已求解姿态继续，解剖关节增量不超过 0.1 rad；没有在播放中施加排斥力或执行手部自由动力学。

WUJI 保留缓存手指关节角，并将各尺寸的手腕变换到同一个参考姿态。物体同时执行相同刚体变换，严格保留缓存手物相对几何；仅把手腕六个值替换掉而不变换物体会破坏抓取。两侧共享变换后的物体姿态。

物体目标来自参考尺寸的 WUJI 接触槽，投影一次后固定物体三角面和重心坐标。随尺寸缩放同一材料区域，不在每个尺寸重新选择最近落点。MANO 接触槽按同一手指、同一指节映射为固定 10 mm 曲面 patch；此解剖映射尚未人工逐区核验。

碰撞与表面查询完全沿用当前共享 adapter，包括 optimized 几何后端、原穿透项和五步 ComFree。图谱不覆盖碰撞深度、材质、力学或验收公式。之前自定义的曲面复核和额外评分已经从活动代码移除。

## 网格、显示和有效性

缓存来自清理备份中的 固定历史参考缓存：31 类、183 个原始尺寸点；19/21 无对应单刚体缓存。每个源区间分成 8 段，共 1,247 个实际优化 MANO 网格点。原 WUJI 完整姿态保存在 `wuji_qpos_original`，用于核对；网页显示固定手腕坐标中的等价抓取。

参考点使用已有固定锚点 LM 96 步初始化，其他尺寸使用已有有界 LM 48 步初始化。实际 CEM 采用当前入口的默认预算：人口 128、最多 24 次更新，统一读取 `cem.DEFAULTS`。达到联合条件时提前停止。LM 仅提供候选初值，最终选择使用当前 CEM cost 与通过条件。

滑条采用局部关节 SLERP 后 FK/LBS，保持骨段和静息几何固定；物体位置/尺寸线性插值，物体旋转 SLERP。绿色点表示已贴合 patch 对应的固定物体参考区域，不是精确力作用点；中间帧只显示两端均贴合的区域。二值显示，不使用连续力颜色，也不施加碰撞力。

插值帧未重新做碰撞或 cost 检查。页面数字来自邻近求解点；仍有未贴合、稳定性失败或间距误差较大的尺寸分别显示未通过；非有限 cost 输出为 `null` 并标为不可用，不算成功。数据生成、浏览器渲染正确不等于全部抓型或尺寸通过物理验收。

## 复现与验证

使用 `.venv-contact` 运行优化及数值检查，`.venv-moxian` 打包、服务和 Playwright 检查。生成器具有独占锁、原子进度和逐尺寸检查点；相同配置可续接。`STOP` 文件仅停止相应运行，改变源码/配置需要新目录，不能静默覆盖历史。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-contact/bin/python scripts/fix_personalized_grasp_grid.py --previous output/mano_wuji_primitives_20261105 --output output/mano_wuji_primitives_cem_20261105_recomputed
.venv-moxian/bin/python scripts/publish_personalized_grasp_viewer.py --output output/mano_wuji_primitives_cem_20261105_recomputed
.venv-moxian/bin/python scripts/check_personalized_grasp_browser.py --output output/mano_wuji_primitives_cem_20261105_recomputed
.venv-contact/bin/python scripts/validate_personalized_grasp_grid.py --output output/mano_wuji_primitives_cem_20261105_recomputed
.venv-contact/bin/python scripts/report_personalized_grasp_quality.py output/mano_wuji_primitives_cem_20261105_recomputed
```

检查覆盖快照摘要、全部网格 FK 重建、手腕固定、归一化物体锚点/重心坐标、WUJI 原始点保存与坐标等价、浏览器顶点和刚体姿态、两侧物体尺寸/姿态一致、滑条与播放连续性及移动端。`validation.json`、`browser_validation.json`、`quality_comparison.json` 分别保存数值、浏览器和前后质量证据；物理成功率另列，不能以测试条数代替。

本次完整重算已上线：31 类、1,247 点；586 点通过 CEM 接触检查，122 点同时通过四指间距检查，9 点无有效稳定性结果。详细结果位于 `output/mano_wuji_primitives_cem_20261105_recomputed/RECOMPUTE_REPORT.md`。

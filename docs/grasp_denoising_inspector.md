# Grasp pose 加噪、去噪与收紧检查页

重建入口现在默认使用 `--noise-mode fingers`：随机选 0–5 根手指、每指独立轮次；`--noise-mode all` 保留下面旧报告的全关节规则。`--method diffusion_vq|diffusion_vae` 可接入[新两阶段模型](grasp_pose_flow.md)，页面显示 flow 积分时间及实际各指噪声。下文历史统计属于旧 checkpoint 与全关节加噪，新模型尚未完整训练。

此检查页使用本地 `skills/inspection-webpages` 的生成器、浅灰绿样式、Plotly 离线脚本和页面结构；为避免重复保存网格，使用共享真实手部网格与精确 FK，在切换时生成当前阶段的表面。

## 打开

本次已启动的本地服务：

- 全部缓存 pose 的 Viser：<http://127.0.0.1:8090>。切换抓型和结果可浏览 640 个原始求解姿态；尺寸滑块可浏览 39,007 个缓存插值姿态。插值姿态未重新做物理求解。
- PCA / diffusion 网页：<http://127.0.0.1:8093>。
- 离线 HTML：`output/grasp_denoising_inspector/index.html`，可直接用浏览器打开。

先选择抓型、噪声种子和累计轮数。上方两视图固定为标准参考和同一加噪输入；下方并排显示 PCA 与 diffusion。点击阶段按钮查看去噪、沿尺寸收紧、加回原始噪声。支持网格/骨架、物体开关、透明度、同步视角、逐关节高亮、加噪播放，以及当前 pose JSON 导出。

默认收紧比例已从 10% 提高到 **50%**，可选范围扩大到 **90%**；打开页面即显示收紧后的干净估计。部分抓型会先达到缓存训练尺寸下限，页面会提示实际可减少的比例；在下限处继续增加请求不会产生额外位移。

## 重建与服务

```bash
.venv-contact/bin/python scripts/inspect_grasp_denoising.py
.venv-contact/bin/python visualize.py --batch grasp_current_20261103 --port 8090
# 在另一个终端运行：
.venv-contact/bin/python -m http.server 8093 --bind 127.0.0.1 \
  --directory output/grasp_denoising_inspector
```

Viser 若发现端口被占用会自动选择其他端口；以启动日志为准，重建页面时可用 `--viser-url` 指向实际地址。服务默认只绑定本机；`visualize.py --host` 可显式指定绑定地址。

重建参数示例：

```bash
.venv-contact/bin/python scripts/inspect_grasp_denoising.py \
  --sigma .05 --rounds 5 --seeds 11,23,37 --fractions 0,.1,.2,.35,.5,.65,.8,.9 \
  --viser-url http://127.0.0.1:8090
```

## 数据和计算定义

使用 `results/grasp_current_20261103` 中 31 类各自保存的 **1× 标准节点**，不是随机测试节点。全部原始姿态仍可在 Viser 查看。模型和 PCA 来自 `results/grasp_manifold_20261103`，diffusion 使用保留的 seed 27 checkpoint；不重新训练，也不改变原始缓存或权重。

手腕前 6 维固定，后 20 维手指关节按各自限位中心和半范围归一化。每轮独立采样 `epsilon_k ~ N(0, .05² I)`，累计 `noise_r = sum(epsilon_1 ... epsilon_r)`，输入为 `reference + noise_r`。累计理论标准差为 `.05 * sqrt(r)`。两方法接收完全相同的输入；diffusion 由已知累计 sigma 选择最近的训练时刻，页面分别报告理论 sigma、模型时刻和对应 sigma。推理不重新采样噪声。

每类的两个预测都使用相同的已有 PCA 尺寸曲线：仅在训练节点的 PCA 系数空间进行 PCHIP，目标倍率为 `source * (1 - fraction)`，限制在训练节点范围内。收紧位移 `delta = curve(target) - curve(source)`，施加到该方法的干净估计上。物体仍保留源尺寸；根坐标固定，不随曲线移动。

默认“加回噪声”严格表示 `clean_estimate + delta + noise_r`，完整保留已知的高斯扰动。另有两种独立标记的对照：

- **各方法去除量**：`clean_estimate + delta + (observed - clean_estimate)`，因此两方法最终均为 `observed + delta`。去除量包括预测误差，不等同于已知噪声。
- **PCA 正交残差**：`clean_estimate + delta + (observed - PCA_project(observed))`。仅保留噪声的正交部分，也包括标准姿态的截断残差。

原始候选不做关节限位裁剪或碰撞修复。诊断包含相对源参考的归一化关节 RMSE、模板接触包络 RMS、原生 MuJoCo 手物穿透、自穿透及关节越限。收紧后 RMSE 是相对源标准姿态的偏移，不是目标尺寸真值误差。包络减少是运动学变化，不能解释为夹持力提升或稳定抓持。

## 本次检查结果

生成 31 类 × 3 个噪声种子 × 6 个累计轮数 × 2 种方法 × 8 个收紧比例，共 **8,928 组对照、44,640 个预测阶段姿态**。数值一致性检查结果保存于生成目录的 `verification.json`；全部候选应有限且手腕保持不变。

第 4 轮（累计理论 sigma=0.1）、3 个种子、31 个标准节点的描述性均值：

| 方法 | 加噪输入 RMSE | 去噪预测 RMSE | 收紧 10% 的包络平均变化 | 包络减少数量 |
| --- | ---: | ---: | ---: | ---: |
| PCA | 0.09631 | 0.04126 | -1.32 mm | 73/93 |
| diffusion | 0.09631 | 0.04080 | -1.31 mm | 75/93 |

包络变化以该方法收紧前的干净估计为基准。这是标准节点演示，包含训练节点，不能视为独立测试集性能排名。网页明细同时报告穿透与越限。

检查包括 24 项相关 Python 测试、DOM 控件逻辑、全部 31 类在 Viser 的实际 WebSocket 网格加载，以及尺寸切换（Type 18 高度、Type 24 半径、Type 26 均匀缩放）。当前没有可连接的浏览器，**未做实际 WebGL 渲染或截图验收**。DOM 检查模拟了 Plotly 绘制，不代表渲染验收。

```bash
.venv-contact/bin/python -m pytest -q tests/manifold/test_grasp_manifold.py \
  tests/manifold/test_grasp_manifold.py tests/primitive/test_grasp_primitive_viewer.py tests/primitive/test_grasp_primitive_viewer.py
.venv-contact/bin/python scripts/check_dense_viewer.py --port 8090
# DOM 检查需要本机 Node.js 和 jsdom；可在临时目录安装，不改变项目依赖。
NODE_PATH=/path/to/node_modules node scripts/check_grasp_denoising_page.cjs
```

生成目录还包含 `experiment.json`（噪声增量、各方法预测、指标及来源）、`poses.npz`（全精度 pose 与阶段/抓型/种子索引）、`verification.json`（数值核对和来源哈希）。不需要外网或 CDN。

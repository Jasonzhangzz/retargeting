# 干净的抓取流形结果

每类 PCA 已拟合；字典与 diffusion 各完成了 3 个种子、每种子 3,000 步训练。
这里只保留验证集宏平均 RMSE 最好的两个 checkpoint：字典 seed 17、diffusion seed 27。

- `clean_poses.npz`：640 个原始参考姿态和四种方法在无额外噪声输入上的干净估计；所有 qpos 为 26 维 float64，手腕固定。diffusion 在 sigma=0 时直接返回 PCA 投影；实际去噪须使用非零 sigma，训练状态见 checkpoint 和评估摘要。输出未裁剪，也未保证物理可执行。
- `pca.npz`：31 类的均值、主成分、特征值、保留方差和归一化参数。basis 为 (31,20,19)，每类仅前 component_counts[g] 列有效。
- `dataset.json / dataset.npz`：数据划分、质量标记、PCA 元数据与来源；源快照为相邻的 `../grasp_current_20261103`。
- `models/dictionary_seed17/best.pt`：训练完成的 128 原子字典；同目录 dictionary_codes.npz 为干净训练节点的 128 维编码。
- `models/diffusion_seed27/best.pt`：训练完成的条件 DDPM / DDIM checkpoint；config.json 保留训练配置。
- `evaluation.json`：原三种子评估摘要、六次训练完成记录及保留模型的验证集选择依据。
- `checksums.json`：最终结果校验和。

加噪样本、收紧演示、HTML、图片、逐样本评估与训练历史已清理；代码可重建演示。程序自动选择保留的最佳种子，避免引用已移除的 seed 7。

```bash
.venv-contact/bin/python scripts/grasp_manifold.py demo --model selected
.venv-contact/bin/python scripts/grasp_manifold.py demo --model diffusion --grasp-type 24
```

演示会重新生成噪声和图形，仅在需要时运行；GitHub 提交不含这些可重建文件。完整说明见 `../../docs/grasp_manifold.md`。

# 当前缓存与历史远端求解

最新的可移植缓存为 `results/grasp_current_20261103`，包含修正后的 640 个原始抓姿和
39,007 个插值姿态，所有 grid 共享一份场景 XML。直接运行：

```bash
.venv-contact/bin/python scripts/verify_grasp_cache.py results/grasp_current_20261103
.venv-contact/bin/python visualize.py --port 8089
```

旧 A800 展开式结果已由紧凑缓存替换；历史数据和下面的复现命令保留在
[84331ee 版本](https://github.com/Jasonzhangzz/retargeting/tree/84331ee/results/grasp_template_mppi_a800_20261102)。
以下为历史求解记录，旧目录命令需要该历史版本。当前存储说明见 [grasp_pose_cache.md](grasp_pose_cache.md)。

# 远端 GPU 求解

本轮已完成，并下载到 `results/grasp_template_mppi_a800_20261102`。31 类、640 个尺寸全部保留，其中 31 个标准尺寸沿用已审计结果，609 个相邻尺寸重新求解。投影闭合 610/640，patch 内部贴合率 ≥80% 为 420/640；Type 16 的 17 个尺寸和 Type 24 的 13 个尺寸未通过闭合。Type 9 的 17 个尺寸均通过闭合及贴合检查。总耗时 46.14 分钟，GPU 求解批次耗时 44.39 分钟。

仓库只保留每格选定的最终姿态、轨迹、接触力、指标、必要参考与来源审计，不包含重试输出及中间日志。模型 mesh 和回放 manifest 使用相对路径；最终数组与力平衡数据和远端原始文件逐字节一致。没有超过 3 mm / 3° 的相邻物体姿态跳变。投影力闭合仍是虚拟接触下的证书，原生动态评测未运行。

从仓库根目录复核并回放：

```bash
python scripts/audit_template_mppi_results.py results/grasp_template_mppi_a800_20261102
mkdir -p output
ln -s ../results/grasp_template_mppi_a800_20261102 output/grasp_template_mppi_a800_20261102
python visualize.py --batch grasp_template_mppi_a800_20261102 --port 8089
```

该链接只让 viewer 找到版本化结果，不修改已有 viewer 配置。最终结果导出入口为 `scripts/publish_template_mppi_results.py`；`materialization.json` 保存原始与发布文件的 SHA256，`source_audits/` 保存远端原始审计。来源记录中的远端绝对路径仅用于溯源，不作为本地回放依赖。

以下记录本轮迁移和启动步骤。

本地已停止批量求解。当前保留：

- `output/grasp_template_mppi_current`：31 类、183 个最终结果，包含修正后的 Type 9；最终 q 和投影力平衡数据的 SHA256 在整理前后完全一致。
- `output/grasp_template_mppi_standard26`：Type 26 精确 1× 的额外初始化结果。
- `output/migration_20261102`：清理与停止进程记录。
- `results/`：已有的版本化基线；`output/graspadmm_mppi_best` 是其相对路径链接。

最终结果已经复制成独立快照。场景中的 mesh 路径相对于 `scene.xml`，初始化依赖相对于仓库根目录，运行时不需要任何已删除的实验目录。旧源代码审计记录保存在各快照的 `source_audits` 中；历史来源名称只作为溯源信息保留。

在远端 pi0 中安装：

```bash
python -m pip install -r requirements-template-mppi.txt
```

使用 `/workspace/jasonzhang/miniconda3/envs/pi0/bin/python`。其原有 `libtorch_cpu.so` 校验失败，已使用另一套相同版本（2.10.0+cu128）、安装记录校验一致的二进制修复；全部 Torch 原生库的校验结果保存在 `migration/torch_native_verification.json`。

保留 pi0 已安装的 CUDA Torch／NumPy／SciPy，只固定本项目使用的 MuJoCo、MuJoCo-Warp、Warp 和 Viser 兼容版本。安装前的 pip 版本清单与安装日志保存在远端 `migration/`。

从仓库根目录运行：

```bash
python scripts/run_template_mppi_remote.py --devices 1 2 3 --population 1024
```

每个 GPU 启动独立进程，将类型按照接触数量和密集尺寸数量分配，单个类型的相邻尺寸依赖保留在同一张卡上。每个候选 batch 为 1024（原本为 256）；各 GPU 独立以启动时可用显存的 80% 为预算，先测量真实峰值，再允许相邻节点并发。不会停止服务器上的其他任务。

控制进程记录在 `output/grasp_template_mppi_a800_20261102/remote_job.json`，各卡日志为 `gpuN.log`。所有分片完成后自动执行独立审计、合并 manifest，并生成 `dense_report.md`、`dense_report.json` 和 `dense_results.csv`。原生动态评测保持关闭。

最终仅显示结果：

```bash
python visualize.py --batch grasp_template_mppi_a800_20261102 --port 8089
```

已检查的原始骨骼角保存在 `data/grasp/assets/wuji2/template_bone_poses.json`，不再依赖 `output/template_mapping` 缓存。相邻续算直接读取父参考旁的固定模板，验证注释、骨骼 XML 和查询几何的来源哈希，不重新转换或改变局部点／法向。

# Wuji2 缓存抓姿与统一 SuperMimic policy

该任务从现有优化抓姿开始学习稳定保持，不执行桌面接近或抓取举起。
31 个抓型、161 个主 primitive 尺寸共享一个 Actor、Critic 和世界模型。
手腕固定，物体自由运动，控制输出为 20 个手指关节相对缓存姿态的残差。
19/21 类不属于这个单刚体 primitive 基准。

## 数据

训练读取已生成并冻结的完整抓姿数据集，检查 manifest、数组、网格指纹和训练/验证/测试划分。保留失败输入及其来源记录；历史评测标签不作为策略或世界模型输入。原始完整抓姿另外保存，不能用简化模板替代。

必要接触保持 `required_cbody` 的完整分组，不能简化成五指 mask。
例如第 33 类的四个指节组需要分别承载，尽管只涉及两根手指。
固定指尖 pad 归属其 distal 刚体；第 8 类按注释要求拇指、食指和中指，
不能将抓型名称中的“2 Finger”误解释为连拇指在内总共两指。

## 安装与运行

服务器上激活 `pi0`，然后安装适配入口：

```bash
python /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget/scripts/install_supermimic_grasp.py \
  --dex /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/Dex-Supermimic
cd /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/Dex-Supermimic
dex-supermimic-train --task hand2-grasp-hold \
  --config ../retarget/configs/supermimic_grasp_hold.json \
  --manifest ../retarget/output/supermimic_grasp_dataset_v2/manifest.json
```

安装器备份并检查现有入口源码后，加入新任务的独立分派，并在当前虚拟环境
写入指向 retarget 的 `.pth`。原来的三个任务保留原实现和检查点兼容补丁。
有效配置、代码哈希、数据签名、TensorBoard、JSONL 指标及 `status.json`
写入该轮日志目录。训练将 tqdm 进度输出到 stderr，可通过 `nohup ... >console.log 2>&1 &` 后台运行。`best.pt` 按验证集最差组合成功率、宏平均成功率排序选择，
`last.pt` 是训练终点；验证分数完全相同时取较晚模型，测试集不参与选择。

```bash
dex-supermimic-train --task hand2-grasp-hold --checkpoint <run>/interrupted.pt \
  --manifest ../retarget/output/supermimic_grasp_dataset_v2/manifest.json
dex-supermimic-evaluate-grasp --checkpoint <run>/best.pt --output <run>/test-recheck
dex-supermimic-evaluate-grasp --baseline --output <run>/baseline-recheck
dex-supermimic-play --checkpoint <run>/best.pt --case-id t09_cylinder_s02
```

新任务的播放默认启动 Viser，端口 8082，可在 VS Code 转发。
无界面播放使用 `--no-viser --steps 110`。恢复保留模型、优化器、迭代、随机数和
累计预算，从缓存重新收集 replay；不声称逐物理步完全重现中断现场。

可重复 GPU 预检与训练曲线导出：

```bash
python ../retarget/scripts/check_supermimic_grasp.py \
  --manifest ../retarget/output/supermimic_grasp_dataset_v2/manifest.json \
  --output /tmp/grasp-preflight --device cuda:3
python ../retarget/scripts/summarize_supermimic_grasp.py --run <run>
```

预检覆盖六种代表性场景、有载接触的 NumPy/GPU 归约一致性、
多步训练、策略梯度、验证后采样边界及保存恢复。它不代表正式任务达标。
另检查 `t10_disk_s09_seed01` 的 ADMM 缓存：该原始候选有关节越界，
仍保留在训练清单中，运行时明确计为无效起点；没有替换成其他候选。

## 训练与物理判定

使用 MJLab mesh variants 保留真实多面体及完整 Wuji2 碰撞几何。
物理步长 2 ms，控制 50 Hz，EMA alpha=.5，残差幅度 .5 rad，
关节限位和原生执行器力矩限制始终生效。输入包含抓型、几何尺寸、接触要求、
参考与当前手物状态、接触载荷和动作历史；没有按案例切换的策略权重。

保留 SuperMimic 的 58 维物理状态、20 维动作积分、带 episode mask 的 replay、
多步世界模型监督、独立 failure head、可微想象、lambda returns 和 Critic 更新。
接触头回归必要组载荷、禁止组载荷、256 个方向的 wrench support、穿透量，
另预测自碰撞；连续量不使用二值接触损失。真实和想象奖励调用同一个纯函数。
载荷来自原生约束解；其采样相位与 MuJoCo 的末次积分前接触缓冲一致。

正式配置为每组合 4 个环境，共 644；显存不足可降至每组合 2 个，共 322。
最多 8000 轮或累计训练墙钟 12 小时。先进行 500 轮世界模型预热，外力逐步增强。
每 1000 轮验证，结束后自动评测零残差基线与同一策略。

`wuji2_policy_force_closure_hold_v1` 是新的闭环协议：

- 每组合 10 次测试，分别覆盖六个 ±xyz 外力方向，每方向独立 reset。
- 启用重力；先自由调整 .2 s，再叠加 .98 N 外力保持 2 s。
- 调整期后，每个控制步位移不超过 2 mm、转角不超过 5°。
- 每个必要组至少 .05 N，禁止手指或不允许的掌面各不超过 .01 N。
- 严格力闭合只用指定组的真实承载接触，使用实际摩擦系数。
- 真实求解状态必须为 `Solved`、rank=6、reserve>1e-4、平衡残差<1e-6；
  不将 `AlmostSolved` 重新命名为成功。溢出、NaN、无效起点和求解错误均失败。

GPU wrench-support 只是训练近似，不能代替正式力闭合证据。
逐方向记录失败原因、逐组合 CSV 和全部分母，并保留代表性成功/失败轨迹。
不将结果与旧静态 GraspADMM 的 40.19% 直接比较。

共享盘读取缓慢时，可以使用字节一致的本地磁盘运行副本。
`RETARGET_GRASP_DATA_CACHE` 必须同时指定 `RETARGET_GRASP_EXPECTED_MANIFEST`
进行哈希检查；checkpoint 仍可保存共享盘 manifest 路径供后续复现。
运行状态和实测结果以该轮 `status.json`、测试 `report.json` 为准；
代码存在或训练已启动不表示所有抓型已经成功。

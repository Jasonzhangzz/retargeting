# Shadow → MANO 全量求解

唯一生产配置是 GPU LM IK 15 步和自适应对角 CEM：128 个候选、最多 24 轮、均值步长 2、步长上限 3 个初始标准差。批量默认 64。实现统一在 retarget/env/grasp/mano_solver.py；不再提供 Adam IK、旧 elite 搜索、手动切换求解器或退火实验选项。

保留按物体和尺寸门控的历史解热启动、每设备有界网格/BVH 缓存、逐行早停、GPU 上的 cost 历史与集中 CPU 导出。IK 使用自动微分 Jacobian 和批量阻尼最小二乘，仅对 5 个指尖做蒙皮；CEM 和最终几何验收仍使用 MANO 表面，最后检查完整 778 顶点。

## 启动与恢复

    python scripts/run_dexonomy_mano_background.py \
      --source results/dexonomy_shadow_grasp_full \
      --output results/dexonomy_mano_lm_cem_full_20261103 \
      --resume-from results/dexonomy_mano_mppi_fast_20261103 \
      --devices cuda:0,cuda:1,cuda:2,cuda:3 --watch

只开放设备、batch、显存上限、候选分块、seed、输入输出与运行范围等操作参数。算法使用同一套默认参数。

恢复时展开历史引用，校验 MANO 模型、固定映射、源数据、行数及全部缓存文件 SHA256；原始已验证文件直接引用，不重复计算、不改写。不同实现指纹使用独立结果目录。当前结果入口是 results/dexonomy_mano_current 符号链接。

所有未通过几何验收的求解结果仍保留并标记。数据格式沿用 primitive grasp pose 的 NPZ、JSON/gzip manifest、模板和接触记录；新数据用 cem_iterations、cem_best_cost_history，查看器兼容历史 MPPI 字段。

停止码：0=预算耗尽，1=严格几何通过，2=停滞，3=几何复核未通过。3 不是成功：最终复核推翻中途判断时保留该姿态和诊断，避免将整分片数据丢弃。已有失败分片不会被当作已完成而跳过。

## 安全停止

创建结果目录中的 STOP 文件，或给 process.json 中的父 PID 发送 SIGTERM。四个 worker 都会完成当前分片并写完校验，再退出。重启前删除 STOP 文件，使用相同配置和恢复根目录。

## 查看

[求解结果 Viser](http://192.168.0.32:8101/)，[进度](http://192.168.0.32:8100/progress.html)，[历史速度与收敛报告](http://192.168.0.32:8100/cem_trials.html)。

几何检查包括完整网格穿透、近表面点及固定映射槽距离和法向。此检查不是动力学稳定性或力闭合证书，失败记录不会过滤掉。查看器支持按需计算仓库的投影力闭合并单独缓存证书。

## 清理与历史

保留全部原始数据、MANO 权重、固定接触映射及已验证 MANO/Shadow 缓存。删除重复的基准 NPZ、旧验证缓存和性能剖析中间文件。旧实现、实验脚本、配置和数值报告归档在仓库外 ../retarget_archives/mano_cleanup_20261103，现行源码只保留一个求解模块。

历史单条延迟：模型/BVH 常驻、batch=1，5 个姿态共 10 次，中位 1.65 秒；batch=64 的完整流程摊销 57–62 ms/条。共享 GPU 上的小样本计时不保证全量性能，四卡持续吞吐以当前进度为准。

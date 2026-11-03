# 核心测试

这些是开发回归检查，应用启动不会加载。按近期使用的流程，从原来 101 个文件精简为 21 个测试文件（185 个测试函数），按功能分组。先合并 10 个零散文件，再移除 70 个暂未使用的测试文件；删除不表示对应功能已被移除。

| 目录 | 保留的检查 |
| --- | --- |
| mano | 当前 LM/CEM、Shadow 接触映射、结果缓存、查看器与标定 |
| dexonomy | 静态重建、grasp 目标提取和接触坐标 |
| manifold | VAE/VQ-VAE Flow、去噪检查页及 PCA/旧 diffusion 兼容 |
| grasp | 投影力闭合、姿态快照和资产接口 |
| primitive | 姿态查看、连续尺寸输入和缓存延续 |
| core | 公共配置、统一查看器及 ComFree 力平衡/CUDA |
| fixtures | 当前力闭合检查、仍在使用的 ADMM 基准所需的小型参考 |

humanoid、dex、Force-Aware、ADMM、SPIDER、SuperMimic 等任务的运行代码及入口保留。移除的历史测试可从 Git 提交 a24bc7b2 恢复，原路径为 tests/test_*.py；仓库外同时保存删除文件和校验清单。此处不承诺这些任务仍有完整回归覆盖。

## 使用

只收集测试，不执行求解或训练：

    python -m pytest --collect-only -q

运行核心 CPU 检查（含两步优化的训练接口 smoke）：

    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests

部分检查依赖 MANO 私有资产、已有模型结果、Node 或 CUDA；缺少可选依赖时按各测试声明跳过。不要将单元测试和全量训练/批量求解混同。

共享路径在 _paths.py，MANO fixture 在 mano/helpers.py 和 mano/conftest.py。新增检查优先放入对应文件；临时性能实验不提交，必要数值参考可以保留为小型 fixture。

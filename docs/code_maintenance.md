# 代码与测试清理

保留 grasp、dex、humanoid 及已注册求解器的运行能力、原始数据、正式缓存、第三方来源与许可。

本次删除 8 个只在自身原型链中相互引用、无运行入口/配置/脚本调用的旧 primitive 修正模块，以及 33 份没有调用或文档引用的历史调参预设、1 个已完成任务的一次性 Dexonomy 缓存迁移脚本。

按用户指定的近期使用范围，测试由 101 个文件精简为 21 个：合并 10 个零散文件，删除 70 个暂未使用的测试文件和 8 个不再被引用的小型 fixture。保留 MANO、Dexonomy 静态缓存、VAE/VQ-VAE Flow、primitive 查看、力闭合及公共接口检查。其他任务的运行代码仍保留，但不再具有原先的测试覆盖。详见 [测试说明](../tests/README.md)。

没有仅凭文件名较旧或缺少静态 import 删除运行功能。SuperMimic bridge、humanoid 接触标注生成器和公开的 Force-Aware 诊断接口继续保留。

清理前版本为 a24bc7b2，可通过 git show 恢复文件。远端仓库外归档 ../retarget_archives/repository_cleanup_20261103 保存删除清单、校验和、源文件及测试收集记录。没有启动全量求解或训练。

验证：262 个案例收集成功，保留案例与清理前一致；54 份配置 YAML 和全部 Python 语法检查通过。CPU 短测首次 232 项通过、29 项因可选依赖或 CUDA 跳过，1 项暴露现有 Viser 绘线参数不兼容。将手套/MANO 查看器两处 thickness 改为当前 line_width 接口后，相关 2 项重跑通过。未执行 CUDA 验收或全量任务。

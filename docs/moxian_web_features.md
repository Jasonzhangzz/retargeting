# Moxian Web 当前功能

页面以实时观察、六动作标定、机器人重定向与 CART 为主线。

- 设备连接、EMF 位姿、实测触觉 0/1、去皮及 SID 位置核对。
- Qnbot 官方 mesh，当前 Moxian 六动作骨长／EMF 外参标定；Wuji 内置手型仅作诊断。
- 当前默认 SLSQP 接触图、CARD/CEM、wuji-retargeting、GeoRT 对照及 CART。
- CART 同步录制、观测与 IK 准备、IK 播放、策略训练和验收。
- 真实网格与 Neural SDF 对照，按按钮启用；不随启动自动计算。

GeoRT、CART、标定结果、尺寸及拟合曲线默认折叠，按需打开。
标定曲线折叠时不轮询历史记录。原始数据和已保存标定不因清理而删除。

## 已移除的网页功能

旧 MANO 触觉预测、RVQ/grasp 分类、蓝色重建、小窗插图及专用 worker/静态资源。
旧均值基线采集按钮移除，当前触觉使用去皮。旧 17 动作、FlashSAC、Delta、个体参数及分阶段最小二乘实验不再提供网页启动或结果选择入口。
CART 不提供离线 Tube 求解入口；旧录制协议仍能识别。

共享标定模块仍被当前 worker 或独立命令行工具引用，不能仅因文件名含 `mano` 或历史方法而删除。
这些底层模块保留，网页只开放当前六动作方法。RVQ 独立训练与 primitive augmentation 数据流程保留。

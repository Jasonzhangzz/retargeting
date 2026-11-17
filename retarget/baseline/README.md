# Baseline 实现

当前三个后端的算法代码集中于此；输入、角度、位置单位沿用各自实现。

| 目录 | 实现与入口 |
| --- | --- |
| `card` | 当前 CARD objective、CEM、Warp、胶囊／SDF、Robokit 初值、在线／离线处理。主类 `retarget.baseline.card.retarget.TactileRetarget`。 |
| `wuji_retargeting` | 已标定 Moxian 骨架 → 原版 Wuji2；`upstream` 保存用户指定远端的未修改源码、6份URDF及许可证。主适配类 `retarget.baseline.wuji_retargeting.runtime.WujiReferenceRetarget`。 |
| `geort` | 数据适配、GPU FK、覆盖／可达空间、训练、模型发布、推理、CLI；`upstream` 保存固定版本的原始网络、数据集和损失。CLI 为 `python -m retarget.baseline.geort.cli`。 |

`moxian_web` 保留设备与 HTTP/worker 调度、共享显示曲面和服务接口。所有算法调用直接使用上述路径；迁移前的兼容导入模块和旧目录链接已移除。

网页支持 `cem`、`slsqp_graph`、`wuji-retargeting`、`geort`、`cart`。离散接触图 SLSQP 和 CART 不属于本次迁移的三个 baseline；SLSQP·CARD 已移除。

远端生产代码一致性由 `wuji_retargeting/upstream/SOURCE.json` 的45份文件哈希核对。GeoRT 原始文件哈希由 `geort/upstream/SOURCE.json` 核对。迁移只调整项目适配模块的导入与资源路径，不修改原始算法文件。

使用范围与数值验证见 [wuji-retargeting 接入说明](../../docs/moxian_wuji_reference.md)、[CARD](../../docs/moxian_robokit_card.md)、[GeoRT](../../docs/moxian_geort.md)。

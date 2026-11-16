# Wuji 完整原生标定：已停用的诊断方案

> 本方案在真实数据上出现手型与RX旋转不匹配，已撤下默认流程并禁止实时预览/应用。当前使用 [Moxian原生25自由度六动作标定](moxian_six_calibration.md)。以下记录仅保留调查与诊断，不是当前操作指引。

当前默认方法 `wuji_native`，模型 schema `moxian_wuji_native21_v1`，拟合版本
`wuji_studio_calibration_2026_7_1_v1`。固定对象是此前复原的本机 **Wuji Studio
2026.7.1**，与其同版 SDK/caliber；不以更新版 SDK 替代。原25自由度、17动作方案
仅保留历史查看与显式离线重算。

## 页面与操作

网页：<http://127.0.0.1:8890>。选 Qnbot mesh，点击“开始 Wuji 原生六动作标定”，
再开始一次自动采集。通过后自动切换，六动作全部完成后停止，点击“开始 Wuji
原生 EMF 与骨架拟合”。成功后自动应用；失败或取消保留当前模型。

动作次序、原生约束与左右手处理直接来自固定 ELF 的动作表：

| 次序 | SDK动作名 | 原生采集条件 |
| --- | --- | --- |
| 1–4 | `pinch_index/middle/ring/pinky` | 对应 RX 与拇指 RX 距离 ≤50mm，使用原生 float32 边界 |
| 5 | `four_finger_bend_90` | 四指 RX 的 roll ∈[-135,-45]°、pitch ∈[-38,50]°、yaw ∈[-35,40]° |
| 6 | `flat_open` | 四指 RX 的 roll ∈[-40,35]°、pitch ∈[-30,40]°、yaw ∈[-30,50]° |

角度范围为原版右手 RX 判别坐标中的欧拉角，不是解剖关节角。左手 roll/yaw 区间按原生逻辑镜像。Moxian 接收器安装方向与 Wuji 不同，直接检查 Moxian 原始 TX 欧拉角会误拒绝实际弯曲手势。因此仅在采集判别时，使用同初始化指纹的八动作 open 与 mcp90 原始旋转确定固定判别坐标：`R_gate = C_f (R_RX R_flat,f^T) C_f^T`。其中 `C_f` 把参考弯曲轴旋转到 Wuji 的 roll 轴；角度不缩放，也不把参考角强制变成90°。同一坐标用于弯曲和平掌，沿用原生阈值；只改变接收器安装坐标解释。参考矩阵与来源hash保存在采集记录的 `gesture_frame`。
四元数规范化与欧拉角转换直接调用该 ELF 的原函数。图示采用该版 Studio 原图，
文件来源与哈希见 `moxian_web/static/gestures/wuji_native/SOURCE.json`。
姿态满足时显示绿色文字；稳定通过与采集完成另外计时。

恢复原版规则：最近20帧五指XYZ的最大轴向总体方差；变换手势时方差须超过
`0.00002 m²`，随后稳定阈值为 `0.0002 m²`，稳定保持1秒再采集1秒。
稳定或姿态失效会重新计时；无效输入或250ms断流清空窗口。每动作120秒超时暂停，
已完成动作保留。阈值与计时的机器码证据见
[原生采集器笔记](../output/reports/wuji_studio_rea_20261112/sdk/collector_optimizer_followup.md)。

原始六组采样帧全部交给原生拟合入口，由其求位置均值及 `Σqqᵀ` 主特征向量旋转平均。
显示用聚合记录不参与替代原生聚合。原17动作与连续轨迹不会转换为六动作输入。
当前兼容八动作记录提供设备初始化指纹、输入通道和上述固定采集判别坐标；拟合仍只使用新六动作的原始 EMF，不使用八动作的角度、几何或自定义目标。

## 真正执行的原生 objective 和 solver

入口是同版 SDK 的完整 `calibrate_from_data` 内部函数 `0xbc6d8e`，而不只是它内嵌的
NLopt 数值接口。原生函数负责观测聚合、内置模型初始化、全部变量和约束构造、objective、
梯度、边界、先验、投影及 URDF 生成。Python 没有重写这些代价项或使用 Moxian 自定权重。

原生全向量是61维 HandModel加6×21维辅助姿态，共187维；固定10个掌骨根坐标后，
实际联合优化177维：15段独立骨长、30维 RX 外参、6维 TX 全局外参，以及126维姿态角。
优化器由原函数内部创建 NLopt `LD_LBFGS`，`ftol_rel=1e-10`、`maxeval=20000`。
骨长、外参和姿态的具体参数化、bounds及正则均执行该版原生实现，**不再使用旧Moxian
log骨长±15%、±20mm/±0.3rad等自定配置**。变量与调用链证据见
[联合拟合取证](../output/reports/wuji_studio_rea_20261112/native/fit_notes.md)。

与该版未指定 hand_profile 的默认分支一致，分别生成 `wujihand` 和 `wujihand2` 两套结果，
活动配置为 `wujihand`。两个配置的原始 wrist-root 与 TX-root URDF 原样保存。
报告保存真实 native log、每配置评估次数/耗时/最终cost和二进制指纹；不把20000次上限
绘成完成百分比，不生成原生入口没有公开的梯度、分项损失或训练网络指标。

完整原生调用位于可取消的隔离子进程。原生 `Ok`、URDF结构与显示适配完整性检查、同版实时SDK离线加载检查通过后
接受；这一状态的含义是 **Wuji原生标定成功**，不宣称通过了旧Moxian独立接触/穿透验收。
原17动作的3/5mm、15°、2mm门槛不会再混入原生 objective 或自动应用门槛。
真人效果和延迟仍需实际重新采集后检查。

## 原生21自由度 IK 与 mesh

实时直接调用同版 SDK 的公开 `WujiGlove.offline_pipeline(..., urdf_path=...)`，
输入五个原始 EMF 位姿。不会建立设备管理器或向实体 Wuji Hand 发送指令。
该版公开角度接口把每指补齐到5槽；真正21维为拇指5维加其余每指前4维，补齐零不参与求解。

mesh worker 消费同一模型hash、同时间戳的原生结果：精确URDF FK→固定显示坐标旋转→
Qnbot骨骼基适配→原Qnbot权重LBS。兼容输出中的25个显示节点不是25个参与优化的关节。
原生零姿态的15段骨长同时确定派生静止皮肤和绑定位置；宽厚、原始模板、skin weights、
触觉材料坐标不优化，顶点残差为零。Qnbot掌部辅助骨保持模板状态；Qnbot显示不保证与
Studio的原GLB逐顶点相同，显示结果不回传 objective。

活动模型保存为不可变内容文件，通过同一个 `qnbot/active.json` 原子指针切换；
骨骼与mesh模型版本不一致时停止发布mesh。首次输入、250ms断流、切换和回滚重新建立
SDK离线pipeline，清空历史。八动作指纹与个体模型hash独立。

## 代码入口与安装

| 文件 | 职责 |
| --- | --- |
| `moxian_web/wuji_native_abi.py` | 固定ELF动作表、原生角度函数和完整标定ABI |
| `moxian_web/wuji_native_capture.py`、`wuji_gesture_frame.py` | 六动作静态状态机、Moxian RX判别坐标、保存未变换的原始帧 |
| `moxian_web/wuji_native_calibration.py`、`wuji_native_fit_child.py` | 真实原生拟合、进度与取消 |
| `moxian_web/wuji_native_model.py` | 原生URDF FK、模型hash和显示适配 |
| `moxian_web/wuji_native_ik.py`、`wuji_offline_child.py` | 同版SDK真实21自由度实时IK |
| `moxian_web/qnbot_worker.py`、`ik_worker.py` | 同模型切换与mesh发布 |

```bash
.venv-contact/bin/python scripts/setup_wuji_native_calibration.py
# 或整套依赖
bash scripts/setup_moxian_mano.sh

# 离线重算，仅保存诊断候选，不应用到设备
.venv-contact/bin/python scripts/calibrate_qnbot_glove.py \
  --capture output/moxian/hand_model/<设备hash>/qnbot/wuji-calibration-capture.json \
  --output output/moxian/debug/wuji-native
```

Linux x86_64；离线公开SDK使用 `/usr/bin/python3.12`。安装脚本固定wheel SHA256，
内部ABI仅接受CP310 ELF SHA256
`8041b5ee059de6e8aa44641f5e7f323d0997604ff908d2dc108d18976631300c`。
实时SDK CP312 ELF SHA256为
`dddeba3148f6a1e9eae87e5830eea25fc39f5c8bcb7cb958749db6bce82b621c`。
版本不符拒绝执行，避免内部ABI误调用。

新采集独立保存为 `qnbot/wuji-calibration-capture.json`，新候选位于
`qnbot/methods/wuji_native/`，不会覆盖旧17动作采集文件。
候选与报告携带新方法和独立模型版本；历史候选只读比较，不继承形变或网络参数。

## 验证

`tests/mano/test_wuji_native_calibration.py` 验证原生动作表与float32阈值、左右手镜像、
保持/重置/超时/完成不重采集、实际完整原生双profile拟合、取消、左右手SDK角度与独立
URDF FK一致、模型hash原子读取、同时间戳mesh及断流重置。合成样本验证调用与数值一致性，
不代表人体标定质量。原生内存布局仅用于固定版本，依据已核对的caller与返回值机器码。

原17动作回归测试仍保留，用于验证历史读取和旧显式离线工具。

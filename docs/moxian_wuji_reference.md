# wuji-retargeting

网页求解器 `wuji-retargeting` 使用**当前已标定 Moxian 骨架 → 原版机器人重定向**。不再运行默认 CQQ 人手 IK。实现位于 `retarget/baseline/wuji_retargeting`，直接调用其 `upstream/retarget_reference` 中未修改的 `RetargetSession.for_hand('WujiHand2', side)`。

远端来源：`ssh -p 1016 8.147.67.185`，目录 `/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/emf_wuji_retarget`。引入的45份原版源码、6份URDF、测试和说明文件保持原始SHA256，记录于 `upstream/SOURCE.json`。机器人重定向实现一致；输入骨架使用当前 Moxian 标定，因此不宣称完整 EMF／CQQ 链路复现。

## 输入与一致性范围

- 从 Qnbot `joint_chains` 取共同腕节点及每指末4个节点，顺序 wrist、thumb、index、middle、ring、pinky，共21点。非拇指额外掌骨根节点不进入输入；腕节点按拓扑读取，当前索引为2。
- 保留原始腕部局部坐标和米单位，不额外镜像、缩放、预对齐或滤波。原版 `step()` 自行执行float32转换、腕部中心化、左右手掌坐标对齐、Wuji2固定旋转和pinch预处理。
- 保留原版 Laplacian／姿态／平滑／拇指相对姿态／反向弯曲目标、解析梯度、原生URDF及限位。两阶段NLopt SLSQP首帧预算50/80，后续1/10，绝对停止阈值1e-4。
- 未滤波解按float32保存为下一帧热启动；输出使用原版float32低通α=0.2。负状态保留原版阶段起点回退。没有额外接触目标、碰撞修复、速度盒或输出滤波。
- 输入必须已标定且非预览，手侧、几何、绑定、模型哈希必须匹配，骨架与源帧时间差不超过25ms；非法／退化骨架在推进求解历史前拒绝。
- 重复时间戳与相同float32节点的快照不重复推进状态；新时间戳即使姿势相同也调用原版。时间跳跃不额外清空状态；显式重启、换后端或重建模型才创建新状态。

在线仍使用最新帧队列，忙时跳过中间输入。比较在线和离线时，应使用相同的实际消费帧序列。输入骨架、源网格及输出身份使用同一源时间戳；压缩网格仍严格校验该时间戳。

## 显示与诊断

显示坐标取原版刚性预处理的逆变换，并使用真正腕节点作为平移原点，避免显示手掌坐标与优化目标不一致。左右手均按各自原版URDF执行FK；现有右手网格仅在绘图层镜像绑定到原版左手URDF。26维 `qpos` 末20维等于原版返回角度，前6维仅用于显示。

算法版本为 `wuji_retargeting_moxian_skeleton_v2`；显示修订为 `inverse_original_preprocess_v3`。输出记录输入节点索引、21点输入、当前模型和标定指纹、原版源码与URDF哈希、两阶段状态与实际评价次数。`human_q_rad` 等默认CQQ诊断不再属于在线输出。`solution_feasible=null` 表示原算法未执行 CARD 式接触／碰撞验收。

## 两个离线入口

当前已标定 Moxian 骨架录制使用与网页相同入口；不要求触觉，不执行SG后处理。同一模型／录制会话内的跳帧保留原版状态：

```bash
.venv-contact/bin/python scripts/retarget_moxian_wuji2.py \
  --input recording.jsonl --output output/wuji_moxian.jsonl \
  --solver wuji-retargeting
```

原始EMF录制需要匹配标定，可提供 `--source-model current.json` 重建Moxian骨架。完整原版 EMF → CQQ人手IK → Wuji2 链路作为独立复现入口保留：

```bash
.venv-contact/bin/python scripts/retarget_wuji_reference.py input.npz \
  --side right --models wuji2 --output output/wuji_reference.npz
```

后者支持原版H5／NPZ、`--start`、`--limit`、`--human-urdf`；参数见未改动的 `upstream/README.txt`。全部依赖统一在根目录 `requirements.txt`。

## 切换原因与验证

旧路径的原版默认CQQ模型不适配当前Moxian的TX/RX外参：真实采样曾出现约33/167/186/185/155mm的五指接收器位置残差，多个人手关节达到限位。用户选择使用当前标定骨架，保留原版机器人部分。此前16帧合成EMF的远端逐位一致验证仅证明原码数值一致，不代表设备外参适配；历史记录仍位于 `output/reports/wuji_reference_integration_20261116` 和 `output/reports/wuji_reference_display_20261116`。

本次回归覆盖左右手真实Qnbot拓扑、原版连续角度／热启动／float32滤波／完整优化trace逐位一致、重复帧、重置、非法输入不推进历史、显示逆变换、离线回放、worker后端切换与网页。当前回归入口为 `tests/test_wuji_reference.py`。


## 对指适用边界

2026-11-16 对当前模型的四组静态标定动作复测发现：源骨架指尖间距约11.65／13.27／13.59／18.13mm，均未触发原版9mm pinch修正。骨架末端标记距离不等于指腹表面间距；显示几何中食指／中指复测已出现相交。原版没有指腹接触、法向或非穿透目标，`thumb_free=True`还放松了拇指中间姿态。因此原码数值一致不保证Moxian对指接触质量。


源IK的错误分支恢复已单独修复，版本为 `moxian_studio_native_qp_v2_branch_recovery`；机器人 `RetargetSession` 原始文件保持原SHA256。此前对指审计中的源IK残差属于修复前数据。


标定初值保存已纳入正式六动作流程，当前源版本为 `moxian_studio_native_qp_v3_calibration_seeds`。每次拟合自动生成、序列化并重载验证初值库；在线仅作为原目标函数的求解起点。源运行版本与几何hash分开，旧骨架数据已清理且禁止训练／回放复用。

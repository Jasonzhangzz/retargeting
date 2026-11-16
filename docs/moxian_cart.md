# CART 五类抓取与接触驱动网络

在 Moxian 网页展开 **CART · 五类抓取与三维接触**。使用已应用标定的源手原生网格，在当前帧中估计真实接触或可夹持的三维凸包，再训练独立的 Wuji2 接触网络。

## 分类和采集

| 类别 | 子动作 | 阶段判定和初始估计使用的曲面 |
|---|---|---|
| 对指（opposition） | 拇指分别对食指、中指、无名指、小指，共4项 | 指定 distal 之间的真实手部接触 |
| precision grasp | 1项 | 五根手指的 distal，共5个区域 |
| power grasp | 1项 | 五指 proximal／middle／distal；拇指没有独立 middle，共14个区域 |
| wrap | 1项 | index–pinky 的 proximal／middle／distal，共12个区域；proximal、distal 权重为 middle 的2倍 |
| prismatic | 1项 | index–pinky 的 proximal、distal，共8个必需区域；middle 参与空间范围估计，同时加强两端对向、平行夹持 |

最终 primitive 的必需指节以所选 Dexonomy type 的接触标注为准。内部 `middle` 指节名为 `intermediate`。不要求拇指的虚构 middle 区域。power 的指腹—掌面距离用于闭合阶段判断，不作为其凸包的必需接触区域。

1. 在训练分区依次录制8个动作，每段自然完成两次 **张开→闭合→张开**。
2. 切换独立验证分区，重新录制全部8个动作。训练与验证不能来自同一录制会话。
3. 选择录制，执行“完整接触目标预处理”；训练也会自动重建缺少或过期的预处理。
4. 同时选择训练与验证数据，运行“训练独立 CART 网络”。默认 AdamW、学习率 `1e-4`、200轮、每批64段、每段16帧。
5. 完整训练成功后启用网络，并用独立验证序列评估。测试模型不能激活。

重录生成新资产，原始帧独立于浏览器和几何预览保存。预览可跳过来不及处理的帧，完整预处理仍逐帧重建所有保存的同步帧。对指始终使用真实表面目标；其余动作在指定真实曲面完全闭合时切换手部接触。找不到凸包本身不会触发该切换。

## 按手型估计 primitive 与接触 delta

`cart_contact_v3` 的几何版本为 `cart_primitive_delta_v5`。仅抽取 [Dexonomy](https://github.com/JYChen18/Dexonomy) type **3、5、6、7、8、11、13、26、28**，共297个本地参考姿态。圆柱对应3/5/6/7/8，球对应11/13/26/28；这种 primitive 形状关联是本项目的拟合配置，Dexonomy 原始标注提供的是手型与接触，不提供本项目的尺寸拟合算法。

| CART 动作 | 候选 type |
|---|---|
| precision grasp | 6、7、8、13 |
| power grasp | 3、5、11、26、28 |
| wrap | 3、5、26、28 |
| prismatic | 6、7、8 |

参考骨架及 WXYZ 物体位姿转换到掌部坐标，用当前手型作带尺度的相似配准，再对当前真实指节曲面估计中心、方向、半径及长度。拟合同时惩罚整手材料采样点进入 primitive，避免只贴合关键接触点却穿入其他手指。最终仍使用完整三角形检查。参考文件中的离散尺寸不直接成为输出尺寸。圆柱没有双端盖接触时，长度由指节轴向跨度与径向余量估计，并明确标记 `length_observed_from_contacts=false`，几何置信度乘0.8；该长度不是实际物体尺寸的唯一解。

关键手指与允许接触的指节读取 `data/grasp/templates/assets/hand/shadow/raw_anno/` 的原始接触标注，每根关键手指在其标注曲面中至少一处接触即可，不要求它的 proximal／intermediate／distal 同时接触。type 7为拇指、食指、中指、无名指；type 8为拇指、食指、中指；type 28为拇指、食指、中指、无名指。拇指 middle 映射到其实际 proximal 表面。非指节的掌面标注用于整手碰撞检查，不额外要求掌面接触。上表动作曲面仍用于运动阶段判定和初始空间估计，最终关键手指以所选 type 为准；引导文字明确要求拇指参与 wrap／prismatic 的对向包围。

置信度是几何质量分数，不是统计校准概率：结合当前曲面间距残差、相向法向残差及接触方向支持。模板手型配准误差仅用于候选排序和初值，不直接压低几何置信度。平行且相反的夹持面可约束圆柱，不因法向矩阵秩为1而强制判零。阈值为 **0.55**；单侧、缺少对向接触支持时为0。低于阈值直接放弃该候选，不继续添加 delta，也不回退使用上一帧目标。

高置信度 primitive 通过完整手部三角形碰撞检查后，允许最多5次每次4%的向内修正，再添加当前材料表面的局部切向接触面，形成包含 primitive 网格核心的凸几何。delta 最大幅度为 `max(2 mm, min(12 mm, 0.5 × 半径))`。不再沿自由方向无限扩张；结果同时保存 primitive 参数、delta 顶点与位移、幅度限制及各必需接触见证。球/圆柱用网格近似传输，碰撞与接触验证针对实际输出网格。

**每根关键手指至少一处标注曲面**间距≤2 mm、法向误差≤30°、delta 未超限并通过完整手物碰撞检查，才标为有效监督。缺少整根关键手指的可用曲面、低置信度、碰撞或接触不全时明确返回不可用；仅高置信度但接触未全的几何可作橙色诊断候选。每帧重新拟合并完整验收。实时与离线对单个候选使用相同曲面采样和拟合精度，避免仅离线能找到目标。120 ms限制开始下一候选，已经开始的候选必须完成接触与完整碰撞检查，不在拟合后直接丢弃。失败时跨帧轮换后续类型，防止首个候选长期占满预算；成功后优先从该类型重拟合，仍不回退未验收的历史几何。单候选和网格验证可超出120 ms，实际时延单独报告。

同帧源手估计完成后冻结 primitive 和 delta 的形状、尺寸。网络仅预测20个机器人关节及物体刚体位姿。模板、原始接触标注、完整动作语义、拟合策略和算法版本均进入几何指纹；旧预处理自动失效并在训练前重建，旧几何模型拒绝激活。

## 网络、损失和数据版本

共享两层256单元编码器和128单元因果 GRU，输入84维同步骨架特征。识别头有8个动作和1个过渡状态，阶段头输出张开／闭合／保持／释放。动作数量、过渡索引和混淆矩阵大小从协议导出。凸包描述仍为30维。

损失通过原生可微 FK 对真实曲面接触、法向、包围关系、自碰撞和手物穿透求导。wrap／prismatic 的 proximal、distal 接触与法向权重提高；prismatic 的对向项作用于随 FK 变化的手部法向。power／wrap／prismatic 降低内部角度形状项权重，允许机器人适应物体。开放／释放阶段保留分离趋势，停止物体吸引。

新协议不兼容旧七类／16动作的分类头和标签。旧原始资产保留，但正式训练需要按新分类重新录制或显式重标注；不会自动猜测旧动作属于哪种 grasp。旧模型不能激活。预处理与模型配置包含协议、完整动作语义、估计器版本及模板内容指纹；语义或模板改变会使缓存失效。

数据仍位于 `output/moxian/cart/`：

- `recordings/<id>/` 保存原始帧、骨架、时间戳、模型和拓扑。
- `cart_targets.jsonl` 保存逐帧同步接触目标；有效凸包使用 `contact_mode=hull`，包含 `vertices_m`、`faces`、`planes`、接触见证及 `initialization`。后者记录初值类型、位置、朝向及参考来源，不代表最终形状。
- `cart_preprocess.json` 保存指纹、周期、有效 `hull_frames` 和接触帧统计。
- `models/<id>/` 保存配置、checkpoint、训练日志和评估；`jobs/<id>/` 保存后台任务状态。

现有 `cart_hull`／`cart_robot_hull` 显示字段和预览、录制、训练、回放 API 保持。新增录制模式为 `cart_v3`，下载接口仍可读取旧资产。

```bash
.venv-contact/bin/python -m moxian_web.cart_cli control cart_stage_set --data '{"stage":"precision_grasp","split":"train"}'
.venv-contact/bin/python -m moxian_web.cart_cli control cart_record_start
.venv-contact/bin/python -m moxian_web.cart_cli control cart_record_stop
.venv-contact/bin/python -m moxian_web.cart_cli preprocess --datasets TRAIN_IDS VALIDATION_IDS
.venv-contact/bin/python -m moxian_web.cart_cli train --datasets TRAIN_IDS VALIDATION_IDS --epochs 200
.venv-contact/bin/python -m moxian_web.cart_cli evaluate --model MODEL_ID --datasets VALIDATION_IDS
```

## 验证边界

自动检查覆盖指定9类提取、原始标注中的关键手指、尺寸随当前手型变化、低置信度放弃、全指节接触与完整碰撞、delta 幅度和刚体变换、超时不复用历史、逐帧同步、缓存与旧模型拒绝、网络因果性及合成目标的一轮训练。合成训练检查 checkpoint 和优化器梯度有限，不能替代实际录制和独立验证。

此前 `convex_v3_20261116.json` 是旧自由凸包算法的历史抽查，不能用于证明当前 primitive/delta 实现的有效性。

## 2026-11-16 搜索可用性修复

此前“只有1/9参考手型达标”的结果同时受错误置信度、每指节强制接触、预览采样降质与预算截断影响，不应视为预期的严格筛选结果。修复记录位于 `output/reports/cart_primitive_search_fix_20261116`，包含同一组冻结手型、相同录制片段及实时参数下的重复验收。

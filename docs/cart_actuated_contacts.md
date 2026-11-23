# v9 固定腕部目标

`card_raw_neural_self_contact_v4_actuated` 使用未来十帧实测 EMF 和触觉，以及五指独立 decoder。策略输出 20 个关节增量，腕部固定。

掌部的全局位置与法向不受手指关节控制，因此掌部位置/法向/连续性仅报告物理诊断，不加入策略 loss。手指接触掌部仍保留：可运动手指的接触目标、自接触闭合、所有自碰撞组和活动触觉点的穿透约束均参与训练。真实 mesh 验证仍包含掌部误差。

其他项与腕部版共用同一实现：CARD 跟踪项、触觉位置和法向、材料保持、link 连续性、源穿透产生的自接触闭合、q-only 自碰撞场和 5 mm 超限软惩罚。完整均值经 `C/(C+1e8)` 归一化。

训练使用 `--policy-objective card_raw_neural_self_contact_v4_actuated --policy-loss-scale 100000000 --optimizer muon_sgd`。未指定初始化选项时，策略、SDF、接触投影场、自碰撞场及优化器均从零开始；只有输入数据和离线 IK 缓存被复用。

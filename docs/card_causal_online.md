# 单帧 CARD 在线连续性

本文记录 `card_wuji2_v17_developer_causal_soft_collisions` 的历史单帧对照。
按后续要求，实时网页使用 `card_wuji2_v21_main_future3_rtc_prefix`（当前帧＋真实未来 3 帧，关节平滑权重 30，弧度平方项系数 900）；
当前默认不启用下文的 v9 源动作项。
新窗口首帧与上一实际输出之间的关节平滑项单独按 2 倍计入，不随窗口长度平均；
窗口内部仍为相邻关节差平方。重叠时间戳继承上次预测作为 CEM 搜索初值，
超出旧窗口的未来帧使用新观测初始化，不外推或复制旧末帧。
腕部、姿态、接触及碰撞代价不变，连续性在优化目标中处理，不增加显示滤波。

RTC 风格衔接参考 [Physical Intelligence 官方实现的 get_prefix_weights](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py)：
按实际时间戳对齐上一窗口的剩余预测，利用上次 worker 总耗时估计延迟前缀，
延迟范围权重为 1，后续重叠区使用原版 exponential schedule 衰减；无覆盖的未来权重严格为 0。
前缀一致性为 `wtheta² * sum_h(weight_h * ||q_h - q_old(timestamp_h)||²)`，
融合进原有 GPU 代价归约，不增加 CEM 采样数、迭代、FK、mesh 查询或每轮 kernel launch。
暖启动已包含旧预测，删除原来重复评分的暖启动候选。
这是 CEM 的软前缀代价，不是 flow policy 的 VJP guidance，也尚未改为异步播放整段动作。
若真实 3 帧覆盖时间短于两轮求解间隔，就没有重叠可用；此时仅保留实际输出边界平滑，不外推旧尾部。
来源是 developer 历史提交 [98df8ec5](https://github.com/Jasonzhangzz/retargeting/commit/98df8ec5f3933892fd949bbee062ff9a0417bce6)
的 `card_wuji2_v9_source_motion`。developer 最新提交已经是 v10，不能直接恢复最新分支来获得 v9 的连续性。

`retarget/baseline/card/online.py` 保留历史版的 Robokit GaussianSampler、GPU 归约、
局部源动作采样、当前 IK 周围的搜索范围和历史候选。初始化器与接触几何沿用当前原生模型。
当前接触／姿态／平滑／碰撞能量仍复用 CARD 核心，不恢复历史版的网格硬拒绝；
掌指间距与物体穿透只有软代价，原生网格只诊断选出的一个端点。

## 连续性

```
q_expected = clamp(q_CEM_previous + q_IK_current - q_IK_previous, joint_limits)
E_motion = 300² * ||q_candidate - q_expected||²
```

已有连续历史时，每轮 1024 个候选中的 128 个在期望姿态附近采样，局部标准差为主分布的 0.15 倍。
上一解、随源腕部刚体运动搬移的上一解、源关节运动提案均按本帧目标重新评分，符合本帧搜索范围才参与选择。
真实 IK 动作保留，抑制的是 CEM 修正量的跳变。原始 `wtheta=0.1` 保留。
断流、时间回退、seek、流身份或标定改变会清空对应历史。没有输出 EMA，也不等待未来帧。

显式实例化 `OnlineCardRetarget` 可使用上述配置；实时入口当前使用 `FixedLagCardRetarget`。
离线和 CART 对核心 `TactileRetarget` / `CardSettings()` 的调用仍保留原来的默认权重，
不会因实时防抖把 CART 训练的 motion 权重改成 300。

## 验证记录（2026-11-19）

保持 1024×16 的 CEM 预算。相同冻结真实输入加受控微小运动，剔除前 4 次预热后统计 20 帧：

| 模式 | 整帧均值 ms | 相邻关节变化 RMS ° | 修正量变化 RMS ° |
|---|---:|---:|---:|
| 当前＋未来 2 帧 | 187.2 | 0.000 | 0.062 |
| 普通单帧 v10 | 45.2 | 1.191 | 1.192 |
| developer v9 连续性 | 64.4 | 0.112 | 0.097 |

这段冻结输入没有激活触觉，不能代表有接触时的性能，也不能据此判定全部真实动作的跟随质量；
三帧窗口在这段微小动作中的零变化也不代表更好的跟随。
运行中的触觉输入另抽查 20 个有效结果：整帧均值 117.8 ms、GPU 采样 69.5 ms、
采样时源帧年龄均值 262.5 ms，没有未来帧缓冲、没有碰撞硬拒绝。
真实负载和运行中的其他 GPU 任务会影响墙钟耗时。

原始对照结果在本机忽略目录 `output/moxian/cem_debug/causal-comparison.json`，
实时抽查在 `developer-causal-live.json`。这些是工程回归，尚不能替代独立真实会话的动作验收。

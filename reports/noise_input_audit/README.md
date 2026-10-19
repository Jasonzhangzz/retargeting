# --noise 输入到 objective 的信息核对

历史记录：此审计对应旧的差分手部参考。当前 `--noise` 已切换为直接从扰动 MANO 重生成参考，原可行 Wuji 轨迹不再参与生成，见[新测试报告](../mano_direct_reference/README.md)。

当前是依赖 MANO/已有 Wuji 模板的局部动作优化。物体侧允许在扰动模板附近 5 mm 内更新，hand action 在扰动参考基础上优化。

执行了实际的 `load_sequence -> make_window -> assign_target` 路径，并检查 CPU Warp 的 loss target 缓冲。所用默认幅度 0.35 rad 的准备输入为 `outputs/noise_inputs/noise_cli_hand_only/inputs/noise/`；同当前 `--noise` 默认生成方式。

- 第 40 参考帧的 `query_position_object` 和 `contact_position_object` 均逐项等于噪声输入文件，激活点相对原始模板最大偏移 19.867 mm。因此不是仅修改可视化、objective 仍直接读取原接触坐标。
- `action` 等于噪声手部参考的插值；该输入手指关节相对原 Wuji 参考最大变化 0.2952 rad。
- 手腕参考全程变化为零。
- 接触 mask、权重、手部 link ID、手部局部接触位置、link 标签均继承原指导。SDF 投影只重新计算物体侧接触位置，不重新检测接触时序与手指对应。
- Wuji 手部参考生成使用原本可行的整条 Wuji 轨迹作为差分基准和关节正则目标；没有仅依据扰动 MANO 从头完成通用重定向。MANO 偏移在帧 21–23 渐增，之前接近抓取的参考段基本保留。
- 运行时未发现重新载入原始物体接触位置的旁路；但保留的手腕、参考生成基准和接触语义是实质先验。SDF 投影后也不必然任务不可行。

所以已有的 9/9 操作完成结果，只能证明在这些先验保留时可以补偿所测扰动，不能证明无可行抓取先验地自主找到抓取，也不能作为接触探索必要性的证据。

严格排除这些先验时，应仅由扰动 MANO 重新生成手部重定向、接触判定和对应关系，并将扰动覆盖抓取建立阶段；原始可行 Wuji 参考与原接触 mask 不应参与输入生成。原物体轨迹仍可作为用户指定的任务目标。

完整数组检查：[target_assignment.json](target_assignment.json)。本次仅核对目标赋值路径，没有修改目标函数或重复闭环成功率实验。

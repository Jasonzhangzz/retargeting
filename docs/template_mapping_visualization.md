# Shadow Hand → Wuji2 模板与接触点核对

生成结果在 `output/template_mapping/`：

- `index.html`：双手并排交互检查，包含 33 个编号抓型和 3 个附加模板。
- `all_templates.jpg`：全部 36 个模板的静态双手对照。
- `previews/*.png`：每类抓型的独立高清对照图。
- `preview.png`：Type 6、9、11、24 的重点对照。
- `data.json`：原始 Shadow qpos、Wuji2 固定关节角、两侧接触点和法向、逐点映射、模型 / 标注来源哈希。
- `audit.json`：查询点索引与坐标一致性检查及逐类诊断。

打开本地页面：

```bash
.venv-contact/bin/python -m http.server 8086 --bind 127.0.0.1 --directory output/template_mapping
```

浏览器访问 `http://localhost:8086`，也可以直接打开 `index.html`；Plotly JS 随目录提供，不依赖 CDN。

两侧默认展示 Shadow 原始 YAML grasp pose 和通过生产 `bone_seed_cuda` 得到的 Wuji2 固定模板。该页面仅展示固定输入映射。以各自**中立**掌部解剖坐标系对齐，并保留真实尺寸；两侧视角同步、坐标范围完全一致。也可切换零关节中立姿态检查接触点的 link 局部位置，该模式并非另一个有效抓姿。

“直接映射”使用 `retarget/task/grasp/dexonomy.py::map_contacts`，按中立掌部对齐、骨长缩放、位置及法向选择实际 Wuji2 表面查询点。“当前 ADMM 模板”使用同一映射并应用生产管线的两项显式策略：Type 6–8 移到 distal 掌侧中心；Type 11 添加一个中央掌部标注。新增点以 `＋` 编号，直接映射视图不显示新增点。这一检查页针对当前 ADMM 固定模板输入；后续求解和评测独立进行。

同号表示同一标注，颜色表示手指。点击对应表可高亮两侧同一点，勾选透明表面或隐藏表面可以查看被遮挡的接触点。法向线长 12 mm。点位是模板锚点，未宣称与物体产生真实接触。

诊断含义：

- 直接映射偏差：Wuji2 表面查询点到映射算法期望点的距离；不是源手与目标手的直接空间距离，也不是物体接触误差。大于 8 mm 标色只是检查提示，不是通过标准。
- 中立法向差：两侧中立解剖坐标系对齐后，源标注法向与当前显示目标点法向之间的夹角。大于 45° 标色是检查提示；不代替接触面的语义判断。
- 骨向差：两侧 15 根骨段在上述公共坐标系的方向夹角；最大值用于定位姿态差异。两只手形态不同，非零值不自动代表程序错误。
- 策略位移：当前策略点相对直接映射点的 link 局部位移，和直接映射偏差是不同的量。

生成与渲染命令（需可用 CUDA / EGL）：

```bash
.venv-contact/bin/python scripts/visualize_template_mapping.py
MUJOCO_GL=egl .venv-contact/bin/python scripts/render_template_mapping.py
```

首次运行需提供 Shadow 原始 XML 引用的九个外观网格。当前缓存目录为 `.cache/shadow_hand_visual`，上游为 [MuJoCo Menagerie / shadow_hand](https://github.com/google-deepmind/mujoco_menagerie/tree/b0655057a9df2abe39ea7fc989914ba6f248301c/shadow_hand)，固定提交 `b0655057a9df2abe39ea7fc989914ba6f248301c`。缓存保留 `revision.json`、下载 URL 清单和上游 `LICENSE`。网格仅用于显示，运动学和接触标注仍使用仓库的 Dexonomy XML / YAML。生成脚本检查公共坐标系为正交且行列式为 +1，避免显示时镜像。

本次 Type 6–9 的 20 个固定关节角与 `graspadmm_soft_center_200_pi0` 保存模板比较，最大差异小于 `5e-8 rad`（保存模板使用 float32）。所有导出锚点均逐个验证为对应 query 文件内记录索引的实际坐标。

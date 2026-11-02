# 页面数据合同

`scripts/build_page.py` 接受一个 JSON 对象。Plotly trace / layout 使用原生结构；生成器不解释领域几何。先运行 `--demo --output /tmp/demo/index.html` 可得到最小可执行样例及其 `index.data.json`。

```json
{
  "title": "源模型 → 目标模型",
  "eyebrow": "MODEL / INSPECTOR",
  "subtitle": "比较姿态、对应点及偏差。",
  "views": [{"label": "源模型"}, {"label": "目标模型"}],
  "legend": [{"label": "同一对象", "color": "#0077c8"}],
  "columns": [{"key": "id", "label": "ID"}, {"key": "distance", "label": "偏差 / mm"}],
  "footer": "单位、坐标系及指标定义。",
  "sources": ["数据来源：本地真实结果文件"],
  "cases": [{
    "id": "case-1",
    "label": "案例 1",
    "variants": [{
      "id": "raw",
      "label": "原始阶段",
      "note": "当前阶段说明。",
      "metrics": [{"label": "对应点", "value": 1}],
      "rows": [{"id": "p1", "distance": 1.2}],
      "figures": [
        {"data": [{"type": "scatter3d", "mode": "markers+text", "x": [0], "y": [0], "z": [0], "text": ["1"], "customdata": ["p1"], "marker": {"color": "#0077c8"}}], "layout": {}},
        {"data": [{"type": "scatter3d", "mode": "markers+text", "x": [0.0012], "y": [0], "z": [0], "text": ["1"], "customdata": ["p1"], "marker": {"color": "#0077c8"}}], "layout": {}}
      ]
    }]
  }]
}
```

## 字段及行为

- `views`：一个或多个视图，数量必须与各阶段的 `figures` 数量一致。
- `cases[].id` 和每个案例内的 `variants[].id`：非空且唯一。不同案例可以有不同阶段；切换时优先保留同名阶段，否则选择该案例首个阶段。
- `figures[].data / layout`：Plotly 原生 trace 数组 / layout 对象。支持 2D 与 3D；2D 页面保留指定布局和图表交互。当前模板只处理主 `scene`，多 3D subplot 时需扩展同步逻辑。
- `rows[].id`：非空且在阶段内唯一，是高亮关联键。设置 trace 的 `customdata` 为**同长度的字符串 ID 数组**，表格选择便可在各视图共同高亮；`mode` 中应包含 `markers`。缺失对应点无需伪造。
- trace 的 `meta.role`：可标为 `surface`、`normals`、`bones`。对应开关只在相关图层出现时显示。法向建议作为线段，每段用 `null` 分隔。
- 透明开关只调整 `surface` 图层的不透明度，不移动点或改写数据。标记高亮只更改 marker 样式。
- `columns`：表格列及顺序；row 的其他键不自动显示。数值和单位在 adapter 中按检查需要格式化。通用模板不预设“错误”的阈值。
- `metrics`：`label`、`value`，可选 `unit`。数值为展示结果，计算由 adapter 负责。
- `theme`（可选）：可覆盖颜色 token `bg ink muted line accent warn`，值使用 `#RGB` 或 `#RRGGBB`。默认即可复现原页面。
- `camera`（可选）：Plotly camera 对象；默认从 +Y 方向观察，+Z 向上。新模型应根据其参照系设置。模板会保留用户视角，且在同步开启时联动 3D 视图。
- `sources`、`footer`、标题和表格值均作为纯文本显示；用于来源追溯和指标解释。

## 3D 比例与性能

模板对当前阶段全部 3D trace 的有限坐标求公共立方范围，包含线段端点和表面顶点，两侧使用同一比例。关闭图层不改变取景范围。大点云使用循环求界，不使用对巨型数组的 `Math.max(...values)`。

海量模型不必逐阶段复制 mesh 顶点。可沿用模板的 CSS 和交互，改用共享局部网格 + 刚体变换、懒加载阶段或专用 WebGL renderer。保留单位、稳定 ID、固定公共参照系和来源，不以无标记的缩放 / 网格简化隐藏差异。

数据内嵌到 HTML 时，生成器将 `<`、`>` 和 `&` 编码为 JSON Unicode 转义，避免字段中的 `</script>` 提前终止脚本。页面不需要 `fetch` 或 CDN，双击即可打开。另存的 `<页面文件名>.data.json` 用于追溯，不应包含任务不需要展示的私密字段。

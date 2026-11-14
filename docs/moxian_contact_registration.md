# moxian 接触材料注册

该流程将独立测得的税元落点注册到固定MANO表面，不从“两个手指必须碰到”推断位置，也不修改betas、姿态或机器人接触目标。分区确认、手动点击和机器人虚拟SID均不构成实测验收。

## 厂家资料核查

2026-11-09查阅Qnbot公开文档索引及相关Python SDK、上位机和模型页面（31个中英页面对）。[HMP-MX压感说明](https://docs.qnbot.com/zh-CN/docs/vetra-glove-pro/sdk/python/latest/raw-data#hmp-mx-pressure)提供460路0–255原始值和读取方法，未列逐测点三维坐标、间距或展开图。[坐标系说明](https://docs.qnbot.com/zh-CN/docs/vetra-glove-pro/sdk/python/latest/coordinate-frames)给出设备轴向、外壳基准原点尺寸，并说明EMF根系与算法wrist_local并不等同。这些外壳尺寸不能直接作为税元位置或佩戴后的安装变换。

当前23×20显示与默认分区是本项目先验；厂家公开的460路设备顺序本身不能证明该二维布局对应实物。现有八动作记录可用于手型和运动学验证，不能据此宣称460个实物落点已唯一确定。

## 准备输入

使用有已知落点／法向的探针、几何基准或独立跟踪系统测量，并标定测量系到`mano_wrist_local`的刚体变换。法向为皮肤向外方向，长度统一用米。每个输入帧需要该时刻固定形状MANO的778个顶点；这些顶点是待检验的模型预测，不能把模型预测的税元坐标复制为实测值来生成“真值”。

生成当前设备的空模板（不会生成测量或应用标定）：

```bash
.venv-contact/bin/python scripts/register_moxian_contacts.py \
  --serial 2DDDF3AC0DA5 --template --output /tmp/contact-measurements.json
```

在模板中填写`measurement_source`，以及`frames`。每帧格式为：

```json
{
  "id": "独立采集帧ID",
  "split": "train",
  "vertices_m": [[0.0, 0.0, 0.0]],
  "observations": [
    {"sid": 354, "position_m": [0.01, 0.02, 0.03], "normal": [0.0, 0.0, 1.0]}
  ]
}
```

上例只展示字段结构，`vertices_m`实际必须有778行。每SID在至少2个训练帧、2个独立留出帧中提供观察；留出帧使用`split: "validation"`。同帧不能重复SID，训练与留出不能复用帧ID。一次可注册部分SID，未测的SID不会被标为已验证。

## 拟合及应用

```bash
.venv-contact/bin/python scripts/register_moxian_contacts.py \
  --serial 2DDDF3AC0DA5 --measurements /tmp/contact-measurements.json \
  --output /tmp/contact-registration.json
```

训练只选择面及面内重心坐标；留出帧不更新这些参数。所有请求的SID都须通过训练及留出的最大位置误差≤2 mm、法向误差≤30°。失败候选只保存诊断，不能应用。通过后，在同一命令中加`--apply`，或POST `/api/command`：

```json
{"command":"mano_registration_import","artifact":{}}
```

`artifact`填写完整通过验收的文件内容。设备、左右手、MANO权重、形状、拓扑和SID分区均需匹配；校验先于写入。原文件备份为`contact-registration.previous.json`，新文件原子替换。`mano_registration_get`返回状态及文件；`mano_registration_restore`校验并恢复前一份。原有手动点击在导入覆盖的SID上被替换，其余SID的编辑保留。

## 运行时和机器人对应

- `contact_registration`报告实际覆盖的SID。修改手型后保留材料坐标，但实测验证状态变为过期；换设备、手侧、权重或SID分区后不复用不兼容材料坐标。手动改变落点也不能继承旧实测验证。
- MANO和Wuji2共用深度消歧的CoP投影；单个活动税元直接保留其材料坐标，避免曲面展开重叠时跳面。
- 每个人手区域对应Wuji2的语义连杆。式(2)等价为在其原生pad三角面上投影`(c_local + alpha*psi)/(1+alpha)`；输出`robot_face_id`和`robot_barycentric`可重建所选接触点。
- pad面mask由原生几何、掌侧轴、可见性和边缘筛选得到，`physical_pad_verified=false`；尚未宣称完成机器人实物接触面标定。`/api/wuji2/model`公开逐区域面ID及owner，便于审查。
- `source_contact_geometry.contact_geometry_ready`要求当前活动SID全部完成实测注册、个性化MANO已验收、EMF安装已验收。该字段不与`solution_feasible`混用；CEM仍显示当前最优有限姿态并报告限制。

算法不自动放宽穿透门槛，不从两侧压力猜测对指中点，也不把相对ADC解释为牛顿。

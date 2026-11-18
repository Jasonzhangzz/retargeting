# 目录和依赖

| 目录 | 用途 |
| --- | --- |
| `moxian_web/` | 设备采集、标定、HTTP／worker 调度、网页界面 |
| `retarget/baseline/` | CARD、wuji-retargeting、GeoRT 当前实现及固定版本上游源码 |
| `retarget/env/`、`retarget/task/`、`retarget/solver/` | CEM、GraspADMM、接触空间任务的模型、几何和求解器 |
| `retarget/learning/mano_contact/` | 可选 MANO 接触预测的准备、训练和推理 |
| `scripts/` | 当前安装、启动、标定和离线求解入口 |
| `data/contact/`、`data/grasp/` | 本地提供的机器人网格、MJCF、模板和许可，不纳入 Git |
| `data/datasets/` | 本地数据集与查看器资源，不纳入 Git |
| `visualization/` | 当前抓握数据与 primitive 增强查看器；资源从本地 `data/` 读取 |
| `docs/` | 当前功能和数据接口说明 |
| `third_party/`、`licenses/` | 运行依赖的固定版本源码、许可及来源记录 |

网页提供 `cem`、`slsqp_graph`、`wuji-retargeting`、`geort`、`cart` 五个求解入口。算法直接从 `retarget.baseline` 导入；迁移前的 Web 兼容模块和 GeoRT 目录链接已移除。

根目录 `requirements.txt` 是项目依赖清单。Python 3.11 计算环境安装完整清单；串口和官方骨架 SDK 的安装脚本安装共享区后，再安装各自固定版 SDK，避免同名包互相覆盖。可选 MuJoCo 4 验证环境由 `scripts/setup_contact.sh` 安装。

本机环境 `.venv-moxian`（串口／Web）、`.venv-moxian-ik`（独立官方 SDK）、`.venv-contact`（计算／训练）保留，均被 Git 忽略。`moxian_web/vendor/` 是用户提供的厂商 SDK 和原生标定库；其中的共享库属于运行依赖。

所有生成内容写入 `output/`，不纳入版本管理。`output/moxian/` 保存当前标定、手型、选中模型及仍供覆盖图使用的录制；不得作为普通缓存整目录删除。原生求解器的 `output/runtime/`、Python 字节码、pytest 缓存和构建中间文件可重新生成。旧报告、训练权重快照和失效实验输出已清理；运行中的日志和锁保留。

`dex_grasping_primitives_20261115` 保留查看器读取的姿态与 RL 末帧数组，已删除训练 checkpoint 和历史脚本快照，仅在本地保存。

`moxian_grasp` 保存从远端下载的原生25自由度Moxian抓握数据，每类4,000条，并保留原始手部资产、全部引用物体网格与124,000组精确配对的Wuji2姿态。查看器并列展示两种手抓取同一物体。`contact_retarget_v1/` 另存固定 Wuji2 接触锚点重算的 Moxian 姿态、独立几何验收标记与训练筛选索引，原始关节映射保留用于对照。启动 `visualization/viser_moxian_dataset.py --port 8767` 浏览；数据来源和查看方法见本地数据目录的 `VIEWER.md`。

上游原始算法及其 `SOURCE.json` 校验清单保留；来源与许可证见根目录 `THIRD_PARTY.md`。GitHub `main` 接收基于原 `main` 历史的代码快照，不引入 `developer` 分支新增的数据或 LFS 历史。当前版本不跟踪任何 `data/` 文件。测试目录、独立 benchmark／核验脚本和第三方演示代码已移除；运行期几何与输入校验仍保留。

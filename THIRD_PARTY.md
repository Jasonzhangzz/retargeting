# 来源与许可

| 保留部分 | 来源 |
| --- | --- |
| `retarget/baseline/geort/upstream`、`retarget/baseline/geort/training.py` | Meta GeoRT，commit `fb2075e2a3272be372016580c0b6ebd6120228f0`；CC BY-NC 4.0。原始文件、许可证及 SHA256 见该目录 `SOURCE.json`；设备／MuJoCo适配与网页实现单独组织。 |
| `third_party/Dexonomy`、GraspADMM 适配 | Dexonomy/GraspADMM，commit `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`；CC BY-NC 4.0。原版初始化、ADMM、QP、filter、pregrasp 和评测源码哈希见 `third_party/Dexonomy/GRASPADMM_UPSTREAM.json`。 |
| `retarget/env/humanoid`、`retarget/task/humanoid` | Meta SPIDER，commit `44717007de41cbef7565dff7ff9f4453557a2d3d`；CC BY-NC 4.0，见 `licenses/SPIDER.txt` 和对应 `source.json`。 |
| `retarget/dynamics/comfree` | ASU IRIS ComFree Warp，commit `ba8b9969f5f7859d51b1f738f458f8d2f49be080`；只保留接触响应 kernel 和接口。来源见同目录 `source.json`，许可见 `licenses/comfree_core-LICENSE`。 |
| `retarget/initialization/dexonomy/util/torch_util.py` | 经 Dexonomy 保留的 PyTorch3D 数学工具；原始出处为 `https://github.com/facebookresearch/pytorch3d/tree/main`，Dexonomy 声明见 `licenses/Dexonomy-NOTICE.txt`。 |
| `data/grasp/assets/wuji` | Wuji 模型、网格和查询点，见 `licenses/Wuji-LICENSE`。 |
| `data/grasp/assets/wuji2` | WUJI Hand 2.2，commit `5c4dfa16917355d073e2e1d7e3d03897b6cf9508`；MIT，资产目录保留许可证和来源记录。 |
| `data/grasp/templates` | Dexonomy 原始输入模板；见 `licenses/Dexonomy-NOTICE.txt`。 |
| `moxian_web/static/vendor` | Three.js 与 OrbitControls，MIT；许可证声明保留于 Three.js 文件头。 |
| 本机厂商串口 SDK、V1 解码器 | 来自用户提供的 `moxian_web/vendor/MoxianEMF-Windows-20260917`；只保留当前 moxian 的依赖，不纳入 Git。官方骨架 SDK 1.7.0／1.8.1 通过安装脚本安装。 |

当前 patch CEM 的 Kelvin–Voigt 弹簧阻尼和切向历史属于本地实现。MANO 模型通过外部授权权重加载。保留实现的许可证、来源声明、注释与 docstring 均保留。
# Contact-space benchmark additions

- `third_party/emf_wuji_reference` contains objective mathematics and the source-graph preparation excerpt from the user-authorized remote `emf_wuji_retarget/retarget_reference`, targeting SDK 2026.8.31. `SOURCE.json` records the source path and upstream/local SHA256 values; notices and the upstream nalgebra license are retained. No SDK binary, URDF, MANO preprocessing or nalgebra eigensolver is copied. This is an independent reference subset, not an official SDK release.

- Vendored library `third_party/robokit` (original SimMPC, GaussianSampler, PopulationSolver, IK/LM, Task terms and MjWarpSim), plus migrated task scene/reference/scoring definitions, derive from [OpenRoboKit](https://github.com/openrobokit/robokit), commit `736f112df93eb40930219501246032605202684a` (MIT). Licenses: `third_party/robokit/LICENSE` and `retarget/task/contact/ROBOKIT-LICENSE`; library per-file hashes: `third_party/robokit/SOURCE.json`; source task hashes: `retarget/task/contact/SOURCE.json`.
- Sharpa Wave MJCF and meshes are copied from the pinned `openrobokit/robokit` asset revision `045838f22292b6931f7bc7490959925b1495a77c` (Apache-2.0); license included in `data/contact/assets/sharpa/LICENSE`.
- Unitree H1 MJCF and meshes derive from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), commit `98a70c569c4c1b6da467a20012f766ee1f8dc521`; Unitree BSD license included in `data/contact/assets/h1/LICENSE`.
- MuJoCo 4 speculative-contact behavior was studied at tag `4.0.0`, commit `ac60eae30fa7e0d31315c38d9476752fb81be67e` (Apache-2.0). The independent predictor uses public Python APIs; no internal C function is vendored or called. Engine constraint insertion remains native.

- `moxian_web/mano_calibrate_flashsac.py`：参考 Holiday-Robot/FlashSAC 的设备常驻 tensor buffer 与编译更新组织，固定 revision `87edc9061150ae9e962dd84e6544e27a1554b3ab`（MIT）。此处是直接损失参数辨识实验，不是 SAC；来源见 `licenses/FlashSAC-SOURCE.json`，许可见 `licenses/FlashSAC-LICENSE`。

- Wuji 完整原生标定固定使用官方 `wuji-sdk==2026.7.1` 发行wheel（SDK项目MIT许可证见 [上游许可证](https://github.com/wuji-technology/wuji-sdk/blob/main/LICENSE)，已解包的依赖保留原声明）；原生库不纳入仓库。`moxian_web/static/gestures/wuji_native/` 的12张动作图及 `tests/fixtures/wuji_2026_7_1_wrist.urdf` 来自本机同版 Wuji Studio，保留原始内容及 `SOURCE.json` 来源/哈希；用于当前本地界面和离线验证，不将这些素材声明为本项目原创。实现与版本边界见 `docs/moxian_wuji_native_calibration.md`。

- `retarget/baseline/wuji_retargeting/upstream`：用户指定的 `ssh -p 1016 8.147.67.185:/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/emf_wuji_retarget` 原版生产 EMF、人手 IK 和机器人重定向链路。45份原始文件逐字节引入，来源和 SHA256 见目录内 `SOURCE.json`；保留 `retarget_reference/NOTICE.txt`、`LICENSE-nalgebra.txt` 及 URDF 来源清单。资产许可范围沿用原声明，不据此重新授予许可。输入与网页显示适配位于 `retarget/baseline/wuji_retargeting`。

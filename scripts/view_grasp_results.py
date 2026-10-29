#!/usr/bin/env python3
"""Browse saved grasp results and replay cached native-Warp gravity tests.

Run with --config configs/grasp_viewer.yaml, then open the configured URL.
Choose saved IK or optimized poses, without interpolating robot joints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import mujoco
import numpy as np
import viser
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.grasp_results import METHODS, HANDS, RESULTS_ROOT, discover_result_sources
DEFAULT_REPORT = ROOT / "reports/grasp_validation_20261028"
DEFAULT_CONFIG = ROOT / "configs/grasp_viewer.yaml"
STAGES = {'ik': '初始化 IK', 'optimized': '优化后结果'}


class PlaybackManager:
    """Share cached trajectories; queue one GPU simulation worker at a time."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gravity-playback')
        self.lock = threading.RLock()
        self.jobs = {}

    def request(self, case, saved_velocity=False):
        from retarget.grasp_playback import playback_paths
        path, report, _ = playback_paths(case, saved_velocity=saved_velocity)
        key = str(path)
        with self.lock:
            if path.exists() and report.exists():
                result = Future()
                result.set_result((path, json.loads(report.read_text())))
                return result
            future = self.jobs.get(key)
            if future is None or (future.done() and future.exception()):
                future = self.executor.submit(self._build, case, saved_velocity, path, report)
                self.jobs[key] = future
                future.add_done_callback(lambda completed: self._forget(key, completed))
            return future

    def _forget(self, key, completed):
        with self.lock:
            if self.jobs.get(key) is completed:
                self.jobs.pop(key, None)

    @staticmethod
    def _build(case, saved_velocity, path, report):
        path.parent.mkdir(exist_ok=True)
        command = [sys.executable, str(ROOT/'scripts/build_grasp_playback.py'), '--case', str(case)]
        if saved_velocity:
            command.append('--saved-velocity')
        log = path.with_suffix('.log')
        with log.open('w') as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode or not path.exists() or not report.exists():
            raise RuntimeError('重力仿真未完成：' + log.read_text()[-1200:])
        return path, json.loads(report.read_text())


def read_rows(path: Path) -> list[dict]:
    if (path/'grasps.npy').is_file():
        from retarget.grasp_compact import read_rows as compact_rows
        return compact_rows(path)
    if path.is_file():
        sources = [path]
    elif (path/'summary.json').is_file():
        sources = [path/'summary.json']
    else:
        sources = sorted([*path.glob('type_*/summary.json'), *path.glob('node_*/summary.json')])
    rows = []
    for source in sources:
        try:
            rows.extend(json.loads(source.read_text()))
        except (OSError, ValueError):
            # Background writers may still be completing this node's summary.
            continue
    return rows


class Results:
    def __init__(self, gpu: Path | None = None, legacy: Path | None = None, extra=(), *, config=None):
        self.initial_stage = 'optimized'
        if config is not None:
            self.rows = self._configured_rows(Path(config))
        else:
            self.rows = {"GPU": read_rows(gpu)}
            if legacy is not None:
                self.rows["Legacy"] = read_rows(legacy)
            for label, path in extra:
                if Path(path).exists():
                    rows = read_rows(Path(path))
                    if rows:
                        self.rows[label] = rows
        self.index = {}
        for method, rows in self.rows.items():
            if not rows:
                raise ValueError(f"No saved results for {method}")
            for row in rows:
                p = row["primitive"]
                hand = row.get('hand_model', 'wuji1')
                key = (method, hand, row["template"], p["kind"], float(p["radial_scale"]))
                if key in self.index:
                    raise ValueError(f"Ambiguous saved case: {key}")
                self.index[key] = row
        self.initial_method = (next(iter(self.rows)) if config is not None else
            next((name for name in reversed(self.rows) if name not in ('GPU', 'Legacy')), 'GPU'))

    def _configured_rows(self, path):
        config = load_config(path)
        self.initial_stage = config.get('stage', 'optimized')
        methods, hands = selector(config.get('methods', 'all')), selector(config.get('hands', 'all'))
        categories = selector(config.get('categories', 'all'))
        rows = {}
        loaded = set()
        for source in discover_result_sources():
            method, hand = source['method'], source['hand']
            if (methods is not None and method not in methods) or (hands is not None and hand not in hands):
                continue
            if (method, hand) in loaded:
                continue
            location = source['path']
            saved_rows = read_rows(location)
            if not saved_rows:
                continue
            loaded.add((method, hand))
            for row in saved_rows:
                actual_hand = row.get('hand_model', 'wuji1')
                if actual_hand != hand:
                    raise ValueError(f'Hand mismatch in {location}: directory specifies {hand}, saved case uses {actual_hand}')
                template = row['template']
                if categories is not None and not any(str(value) in (template, template.split('_')[0]) for value in categories):
                    continue
                case = Path(row['case']).expanduser()
                if not case.is_absolute():
                    base = location.parent if location.is_file() else location
                    case = ROOT/case if (ROOT/case).exists() else base/case
                rows.setdefault(method, []).append(dict(row, case=str(case.resolve()), result_batch=source['batch']))
        if not rows:
            raise ValueError(f'No saved results match YAML categories / methods / hands in {RESULTS_ROOT}')
        return rows

    def hands(self, method):
        return sorted({r.get('hand_model', 'wuji1') for r in self.rows[method]})

    def types(self, method, hand):
        return sorted({r["template"] for r in self.rows[method] if r.get('hand_model', 'wuji1') == hand},
                      key=lambda name: int(name.split("_")[0]))

    def shapes(self, method, hand, template):
        return sorted({r["primitive"]["kind"] for r in self.rows[method]
                       if r["template"] == template and r.get('hand_model', 'wuji1') == hand})

    def scales(self, method, hand, template, shape):
        return sorted(key[4] for key in self.index if key[:4] == (method, hand, template, shape))


def selector(value):
    if value == 'all':
        return None
    return value if isinstance(value, list) else [value]


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('Viewer YAML must be a mapping')
    if 'results' in config:
        raise ValueError('Remove YAML results: sources are now automatically discovered in outputs/grasp_results/<method>/<hand>/<batch>')
    unknown = set(config)-{'server', 'categories', 'methods', 'hands', 'stage'}
    if unknown:
        raise ValueError(f'Unknown viewer YAML fields: {sorted(unknown)}')
    if config.get('stage', 'optimized') not in STAGES:
        raise ValueError('stage must be ik or optimized')
    for name, available in [('methods', METHODS), ('hands', HANDS)]:
        values = selector(config.get(name, 'all'))
        if values is not None and (not values or any(v not in available for v in values)):
            raise ValueError(f'Invalid {name}: {values}; available: {sorted(available)}')
    return config


def available_stages(case):
    from retarget.grasp_compact import is_compact_case, record_for_case
    if is_compact_case(case):
        valid = record_for_case(case)['stage_valid']
        return [stage for stage, enabled in zip(STAGES, valid) if enabled]
    with np.load(Path(case)/'grasp.npz', allow_pickle=False) as saved:
        return [key for i, (key, field) in enumerate([('ik', 'initial_qpos'), ('optimized', 'qpos')])
                if field in saved.files and ('stage_valid' not in saved.files or saved['stage_valid'][i])]


@lru_cache(maxsize=8)
def load_geometry(case: str, stage: str = 'optimized'):
    """Use compiled MuJoCo mesh coordinates and FK from the saved qpos."""
    from retarget.grasp_compact import is_compact_case
    if is_compact_case(case):
        return compact_geometry(case, stage)
    model = mujoco.MjModel.from_xml_path(str(Path(case) / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(Path(case) / "grasp.npz", allow_pickle=False) as saved:
        data.qpos[:] = saved["qpos"]
        if stage not in STAGES:
            raise ValueError(f'Unknown pose stage: {stage}')
        key = 'initial_qpos' if stage == 'ik' else 'qpos'
        if key not in saved.files or ('stage_valid' in saved.files and not saved['stage_valid'][list(STAGES).index(stage)]):
            raise ValueError(f'No saved {key} for {case}; cannot display {STAGES[stage]}')
        selected_qpos = saved[key].copy()
    mujoco.mj_kinematics(model, data)
    obj = data.xpos[model.body("grasp_object").id].copy()
    palm = data.xpos[model.body("right_palm_link").id].copy()
    center = .7 * obj + .3 * palm
    # A common center and camera let IK/output comparisons preserve placement.
    data.qpos[:] = selected_qpos
    mujoco.mj_kinematics(model, data)
    meshes = []
    for geom in range(model.ngeom):
        # Group 1 contains the original visual meshes; group 0 duplicates them
        # with collision meshes. Do not draw both coincident surfaces.
        if model.geom_group[geom] != 1 or model.geom_rgba[geom, 3] == 0:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"Unexpected non-mesh visual geom in {case}: {geom}")
        mesh = model.geom_dataid[geom]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = model.mesh_vert[va:va + vn].astype(np.float32).copy()
        faces = model.mesh_face[fa:fa + fn].astype(np.uint32)
        rgba = model.geom_rgba[geom].copy()
        if model.geom_matid[geom] >= 0:
            rgba = model.mat_rgba[model.geom_matid[geom]].copy()
        is_object = model.geom_bodyid[geom] == model.body("grasp_object").id
        if not np.isfinite(vertices).all() or faces.max() >= len(vertices):
            raise ValueError(f"Invalid mesh in {case}: {geom}")
        quaternion = Rotation.from_matrix(data.geom_xmat[geom].reshape(3, 3)).as_quat(scalar_first=True)
        meshes.append((vertices, faces, tuple((rgba[:3] * 255).astype(int)), is_object,
                       data.geom_xpos[geom].copy()-center, quaternion, geom))
    direction = obj - palm
    azimuth = np.arctan2(direction[1], direction[0]) + np.deg2rad(200)
    elevation = np.arctan2(direction[2], np.linalg.norm(direction[:2]))
    distance = max(.32, 2.0 * (np.linalg.norm(direction) + .065))
    camera = distance * np.array([np.cos(azimuth) * np.cos(elevation),
                                  np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])
    return meshes, camera, center, model


def compact_geometry(case, stage):
    from retarget.grasp_compact import record_for_case, hand_model, object_geometry
    record = record_for_case(case)
    index = list(STAGES).index(stage)
    if not record['stage_valid'][index]:
        raise ValueError(f'No saved {stage} pose for {case}')
    model = hand_model(f"wuji{int(record['hand_id'])}")
    data = mujoco.MjData(model)
    data.qpos[:6] = 0.
    data.qpos[6:26] = record['hand_config'][index]
    mujoco.mj_kinematics(model, data)
    wrist = model.body('right_palm_link').id
    rw = data.xmat[wrist].reshape(3, 3)
    pw = data.xpos[wrist]
    camera_index = 1 if record['stage_valid'][1] else 0
    direction = record['object_pose_wrist'][camera_index, :3]
    center = .7*direction
    meshes = []
    for geom in range(model.ngeom):
        if model.geom_group[geom] != 1 or model.geom_rgba[geom, 3] == 0:
            continue
        mesh = model.geom_dataid[geom]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        rgba = model.geom_rgba[geom]
        if model.geom_matid[geom] >= 0:
            rgba = model.mat_rgba[model.geom_matid[geom]]
        meshes.append((model.mesh_vert[va:va+vn].astype(np.float32),
            model.mesh_face[fa:fa+fn].astype(np.uint32), tuple((rgba[:3]*255).astype(int)), False,
            rw.T@(data.geom_xpos[geom]-pw)-center,
            Rotation.from_matrix(rw.T@data.geom_xmat[geom].reshape(3, 3)).as_quat(scalar_first=True), geom))
    vertices, faces = object_geometry(record)
    pose = record['object_pose_wrist'][index]
    meshes.append((vertices, faces, (64, 166, 217), True, pose[:3]-center, pose[3:], -1))
    azimuth = np.arctan2(direction[1], direction[0])+np.deg2rad(200)
    elevation = np.arctan2(direction[2], np.linalg.norm(direction[:2]))
    distance = max(.32, 2.*(np.linalg.norm(direction)+.065))
    camera = distance*np.array([np.cos(azimuth)*np.cos(elevation),
        np.sin(azimuth)*np.cos(elevation), np.sin(elevation)])
    return meshes, camera, center, model


def playback_geometry(path, model, meshes, center):
    """Render actual recorded qpos; upload meshes once and animate transforms."""
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    if arrays['qpos'].shape[1] != model.nq:
        raise ValueError('Playback model and saved qpos dimensions disagree')
    ids = np.array([mesh[6] for mesh in meshes], int)
    data = mujoco.MjData(model)
    positions, quaternions = [], []
    for qpos in arrays['qpos']:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        positions.append(data.geom_xpos[ids].copy()-center)
        quaternions.append(Rotation.from_matrix(data.geom_xmat[ids].reshape(-1, 3, 3)).as_quat(scalar_first=True))
    arrays['geom_positions'] = np.array(positions)
    arrays['geom_quaternions'] = np.array(quaternions)
    return arrays


def details(row, stage='optimized'):
    p = row["primitive"]
    flag = lambda value: "✅ YES" if value else "❌ NO"
    batch = f"批次：`{row['result_batch']}`\n\n" if 'result_batch' in row else ''
    if stage == 'ik':
        initial = row.get('initialization', {})
        depth = initial.get('initial_penetration_mm')
        shape_text = ''
        if initial.get('contacts_optional'):
            shape_text = ('IK 验收以手型与碰撞间隙为准，允许部分手指不接触；力闭合由优化后结果另行验收。\n\n'
                + f"禁止区域最小间隙：**{initial.get('forbidden_region_min_gap_mm', 0.):.3f} mm**  \n"
                + f"骨方向最大偏差：**{initial.get('bone_direction_max_deviation_deg', 0.):.2f}°**\n\n")
        failure = ('初始化未通过，已跳过优化。原因：`'
                   + ', '.join(initial.get('failure_reasons', [])) + '`。'
                   if row.get('termination') == 'initialization_failed' else
                   '优化后验收指标请切换到‘优化后结果’查看。')
        return (f"**{row['template'].replace('_', ' ')} · 初始化 IK**\n\n"
            f"手型：**{row.get('hand_model', 'wuji1')}** · 形状：`{p['kind']}` · 尺寸 **{p['radial_scale']:.2f}×**\n\n"
            + batch +
            "显示保存的优化器输入配置 `initial_qpos`：包含初始化阶段的 IK 拟合与物体对齐。\n\n"
            + (f"初始化物体穿透：**{depth:.3f} mm**\n\n" if depth is not None else '')
            + shape_text
            + failure)
    prediction = row.get('solver_diagnostics', {}).get('explicit_prediction')
    prediction_text = ''
    if prediction:
        settings = prediction.get('settings')
        duration = f" **{settings['horizon']*settings['dt']:.2f} s**" if settings else ''
        shift = prediction.get('predicted_max_displacement_mm', prediction.get('predicted_displacement_mm'))
        angle = prediction.get('predicted_max_rotation_deg', prediction.get('predicted_rotation_deg'))
        prediction_text = (f"显式模型预测{duration}：物体最大位移 **{shift:.2f} mm**，"
            f"最大旋转 **{angle:.2f}°**。目标为初始 pose。\n\n")
        if 'predicted_hold_feasible' in prediction:
            prediction_text += f"模型持握验收：**{flag(prediction['predicted_hold_feasible'])}**。\n\n"
    return (
        f"**{row['template'].replace('_', ' ')}**\n\n"
        f"灵巧手：**{'WUJI Hand 2.2' if row.get('hand_model') == 'wuji2' else 'WUJI Hand 1'}**\n\n"
        + batch +
        f"形状：`{p['kind']}` · 尺寸 **{p['radial_scale']:.2f}×**\n\n"
        f"半径/平面尺度：**{1000 * p['radius_m']:.2f} mm**  \n"
        f"高度：**{1000 * p['height_m']:.2f} mm**\n\n"
        f"| 检查 | 当前结果 |\n|---|---|\n"
        f"| 近似力闭合 | {flag(row['force_closure']['force_closure'])} |\n"
        f"| 综合合格 | {flag(row['feasible'])} |\n"
        f"| 指定手指接触 | {flag(row['required_groups_satisfied'])} |\n"
        f"| 有效接触数 | {row['active_contacts']} |\n"
        f"| 物体穿透 | {row['max_object_penetration_mm']:.3f} mm |\n"
        f"| 手指胶囊重叠（独立审计） | {row['max_self_penetration_mm']:.3f} mm |\n"
        f"| 优化器硬约束 | {flag(row.get('optimizer_hard_constraints_satisfied', False))} |\n\n"
        + prediction_text +
        "静态保存指标；重力仿真结果见‘重力播放’面板。\n\n"
        "尺寸选项对应实际测试档位；高度保持原测试设置。"
    )


def attach_client(client, results, playback_manager):
    """Keep selections and scene private to each browser tab."""
    client.scene.set_up_direction("+z")
    client.scene.world_axes.visible = False
    lock = threading.RLock()
    handles = []
    state = {"updating": False, "row": None, "camera": None, "generation": 0,
             "playing": False, "playback": None, "loading": False, "frame": -1}
    closed = threading.Event()
    with client.gui.add_folder("抓取结果 / Grasp results"):
        initial_method = results.initial_method
        method = client.gui.add_dropdown("方法 / Method", list(results.rows), initial_value=initial_method)
        hand = client.gui.add_dropdown("手型 / Hand", results.hands(method.value))
        types = results.types(method.value, hand.value)
        grasp = client.gui.add_dropdown("抓取类别 / Grasp type", types, initial_value=types[0])
        shape = client.gui.add_dropdown("物体形状 / Shape", results.shapes(method.value, hand.value, grasp.value))
        scales = results.scales(method.value, hand.value, grasp.value, shape.value)
        size = client.gui.add_slider("尺寸 / Size ×", min=scales[0],
            max=scales[-1] if len(scales) > 1 else scales[0]+.001, step=.001,
            initial_value=1. if 1. in scales else scales[0], disabled=len(scales) == 1,
            marks=tuple((s, f'{s:g}×') for s in scales))
        stage = client.gui.add_dropdown("结果阶段 / Stage", list(STAGES.values()),
            initial_value=STAGES[results.initial_stage])
        opacity = client.gui.add_slider("物体不透明度", min=.1, max=1., step=.05, initial_value=1.)
        reset = client.gui.add_button("重置视角 / Reset view")
    with client.gui.add_folder("重力播放 / Gravity playback"):
        play = client.gui.add_button("▶ 播放重力仿真")
        rewind = client.gui.add_button("回到抓取姿态 / Rewind")
        timeline = client.gui.add_slider("播放时间 / Time (s)", min=0., max=3., step=.02,
                                        initial_value=0., disabled=True)
        speed = client.gui.add_slider("播放速度 / Speed ×", min=.25, max=2., step=.25, initial_value=1.)
        velocity = client.gui.add_checkbox("使用保存的初始速度", initial_value=False,
            hint="默认从静止释放；勾选后使用求解器保存的 qvel。")
        playback_status = client.gui.add_markdown("点击播放：从当前抓取姿态静止释放，保持保存的关节指令。"
            "物体自由受重力作用，无物体夹具、无额外挤压或 MPC 补救。首次生成轨迹后自动缓存。")
    with client.gui.add_folder("静态指标 / Static checks", expand_by_default=False):
        status = client.gui.add_markdown("")
        client.gui.add_markdown("拖动旋转 · 滚轮缩放 · 右键拖动平移。每个浏览器窗口可独立选择。\n\n"
            + " · ".join(f"{method}: {len(rows)}项" for method, rows in results.rows.items())
            + "。后台生成期间刷新页面可加载新增结果。")

    def reset_camera():
        if state["camera"] is not None:
            client.camera.up_direction = (0., 0., 1.)
            client.camera.position = state["camera"]
            client.camera.look_at = (0., 0., 0.)
            client.camera.near = .001
            client.camera.far = 10.

    def refresh(*, reset_view=False):
        row = results.index[(method.value, hand.value, grasp.value, shape.value, float(size.value))]
        stages = available_stages(row['case'])
        if not stages:
            raise ValueError(f"No saved configurations in {row['case']}")
        previous = stage.value
        stage.options = [STAGES[key] for key in stages]
        stage.value = previous if previous in stage.options else stage.options[-1]
        stage_key = next(key for key, value in STAGES.items() if value == stage.value)
        meshes, camera, center, model = load_geometry(row["case"], stage_key)
        state['generation'] += 1
        state.update(playing=False, playback=None, loading=False, frame=-1,
                     model=model, meshes=meshes, center=center, stage=stage_key)
        with client.atomic():
            for handle, _ in handles:
                handle.remove()
            handles.clear()
            for i, (vertices, faces, color, is_object, position, quaternion, _) in enumerate(meshes):
                handle = client.scene.add_mesh_simple(
                    f"/grasp/mesh_{i}", vertices, faces, color=color,
                    opacity=opacity.value if is_object else 1., side="double",
                    position=position, wxyz=quaternion)
                handles.append((handle, is_object))
            state["row"], state["camera"] = row, camera
            status.content = details(row, stage_key)
            playback_enabled = stage_key == 'optimized' and not row.get('compact', False)
            play.label, play.disabled = '▶ 播放重力仿真', not playback_enabled
            velocity.disabled = not playback_enabled
            rewind.disabled = not playback_enabled
            timeline.value, timeline.disabled = 0., True
            playback_status.content = ('精简记录仅保存关节角与物体相对 wrist 的 pose，不含动力学回放指令。'
                if row.get('compact') else
                '当前显示初始化 IK；切换到‘优化后结果’可播放保存指令的重力仿真。' if stage_key == 'ik' else
                '从当前保存姿态释放；重力 **9.81 m/s²**，保持保存的关节指令。'
                '物体无外部支撑，不运行 MPC。点击播放生成或读取3秒真实仿真。')
            if reset_view:
                reset_camera()
        print(json.dumps({"client": client.client_id, "method": method.value,
                          "hand": hand.value, "stage": stage_key,
                          "type": grasp.value, "shape": shape.value, "size": size.value,
                          "case": row["case"]}), flush=True)

    def selection_changed(event):
        with lock:
            if state["updating"] or event.client is None:
                return
            state["updating"] = True
            try:
                if event.target is method:
                    previous = hand.value
                    hand.options = results.hands(method.value)
                    hand.value = previous if previous in hand.options else hand.options[0]
                if event.target in (method, hand):
                    previous = grasp.value
                    grasp.options = results.types(method.value, hand.value)
                    grasp.value = previous if previous in grasp.options else grasp.options[0]
                if event.target in (method, hand, grasp):
                    previous = shape.value
                    shape.options = results.shapes(method.value, hand.value, grasp.value)
                    shape.value = previous if previous in shape.options else shape.options[0]
                if event.target in (method, hand, grasp, shape):
                    previous = size.value
                    scales = results.scales(method.value, hand.value, grasp.value, shape.value)
                    size.min = scales[0]
                    size.max = scales[-1] if len(scales) > 1 else scales[0]+.001
                    size.marks = tuple((s, f'{s:g}×') for s in scales)
                    size.disabled = len(scales) == 1
                    size.value = previous if previous in scales else 1. if 1. in scales else scales[0]
                if event.target is size:
                    scales = results.scales(method.value, hand.value, grasp.value, shape.value)
                    size.value = min(scales, key=lambda scale: abs(scale-float(size.value)))
                refresh(reset_view=event.target in (grasp, hand))
            except Exception as error:
                status.content = f"加载失败：{error}"
                raise
            finally:
                state["updating"] = False

    method.on_update(selection_changed)
    hand.on_update(selection_changed)
    grasp.on_update(selection_changed)
    shape.on_update(selection_changed)
    size.on_update(selection_changed)
    stage.on_update(selection_changed)

    @opacity.on_update
    def update_opacity(_):
        with lock, client.atomic():
            for handle, is_object in handles:
                if is_object:
                    handle.opacity = opacity.value

    @reset.on_click
    def on_reset(_):
        with lock:
            reset_camera()

    def show_frame(index):
        trajectory = state['playback']
        if trajectory is None:
            return
        index = min(max(index, 0), len(trajectory['time'])-1)
        if state['frame'] == index:
            return
        state['frame'] = index
        report = state['playback_report']
        labels = dict(retained='保持住（本次时长内）', dropped='掉落', escaped='滑移 / 逃逸',
                      simulation_failed='仿真失败，不能判为保持成功')
        outcome = labels[report['status']]
        first_drop = ('未达到掉落阈值' if report['first_drop_seconds'] is None
                      else f"{report['first_drop_seconds']:.2f} s")
        with client.atomic():
            for i, (handle, _) in enumerate(handles):
                handle.position = trajectory['geom_positions'][index, i]
                handle.wxyz = trajectory['geom_quaternions'][index, i]
            timeline.value = float(trajectory['time'][index])
            playback_status.content = (
                f"**{report['simulated_seconds']:.2f}秒仿真结果：{outcome}**\n\n"
                f"当前：**{trajectory['time'][index]:.2f} s**  \n"
                f"相对手的向下位移：**{trajectory['relative_fall_mm'][index]:.1f} mm**  \n"
                f"相对滑移距离：**{trajectory['relative_displacement_mm'][index]:.1f} mm**\n\n"
                f"判定阈值30 mm · 首次掉落：{first_drop}\n\n"
                + ('初始速度：保存的 qvel。' if report['saved_velocity'] else '初始速度：静止。')
                + '重力9.81，固定保存指令，无物体夹具 / MPC。\n\n'
                + (f"仿真错误：{report['simulation_error']}" if report['simulation_error'] else '')
            )

    def start_playing():
        if state['frame'] >= len(state['playback']['time'])-1:
            show_frame(0)
        state.update(playing=True, anchor_time=float(timeline.value), anchor_wall=time.perf_counter())
        play.label = '⏸ 暂停 / Pause'

    def ready(future, generation, model, meshes, center):
        try:
            path, report = future.result()
            with lock:
                if closed.is_set() or state['generation'] != generation:
                    return
            trajectory = playback_geometry(path, model, meshes, center)
            with lock:
                if closed.is_set() or state['generation'] != generation:
                    return
                state.update(playback=trajectory, playback_report=report, loading=False, frame=-1)
                timeline.max = max(.02, float(trajectory['time'][-1]))
                timeline.disabled, play.disabled = False, False
                show_frame(0)
                start_playing()
        except Exception as error:
            with lock:
                if closed.is_set() or state['generation'] != generation:
                    return
                state['loading'] = False
                play.disabled, play.label = False, '▶ 重试重力仿真'
                playback_status.content = f'**轨迹生成失败**\n\n{error}'

    @play.on_click
    def on_play(_):
        with lock:
            if state['stage'] != 'optimized' or state['row'].get('compact'):
                return
            if state['playback'] is not None:
                if state['playing']:
                    state['playing'] = False
                    play.label = '▶ 继续 / Play'
                else:
                    start_playing()
                return
            state['loading'] = True
            play.disabled = True
            playback_status.content = '**正在排队或生成真实 GPU 重力轨迹…** 首次需加载模型；完成后自动播放。'
            generation = state['generation']
            try:
                future = playback_manager.request(state['row']['case'], velocity.value)
            except Exception as error:
                play.disabled = False
                state['loading'] = False
                playback_status.content = f'**轨迹请求失败**\n\n{error}'
                return
            model, meshes, center = state['model'], state['meshes'], state['center']
            future.add_done_callback(lambda completed: ready(completed, generation, model, meshes, center))

    @rewind.on_click
    def on_rewind(_):
        with lock:
            state['playing'] = False
            if state['playback'] is not None:
                show_frame(0)
            play.label = '▶ 播放重力仿真'

    @timeline.on_update
    def on_seek(event):
        if event.client is None:
            return
        with lock:
            if state['playback'] is not None:
                index = int(np.argmin(np.abs(state['playback']['time']-timeline.value)))
                show_frame(index)
                state.update(anchor_time=float(timeline.value), anchor_wall=time.perf_counter())

    @speed.on_update
    def on_speed(_):
        with lock:
            state.update(anchor_time=float(timeline.value), anchor_wall=time.perf_counter())

    @velocity.on_update
    def on_velocity(_):
        with lock:
            refresh()

    def animate():
        while not closed.wait(.02):
            with lock:
                if not state['playing'] or state['playback'] is None:
                    continue
                elapsed = state['anchor_time']+(time.perf_counter()-state['anchor_wall'])*speed.value
                times = state['playback']['time']
                index = min(int(np.searchsorted(times, elapsed, side='right')-1), len(times)-1)
                show_frame(index)
                if elapsed >= times[-1]:
                    state['playing'] = False
                    play.label = '↻ 重播 / Replay'

    refresh(reset_view=True)
    threading.Thread(target=animate, daemon=True, name=f'viser-playback-{client.client_id}').start()
    return closed.set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='YAML category/method/hand filters and initial stage; result sources are automatic')
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--method", nargs=2, action="append", default=[], metavar=("LABEL", "SUMMARY"),
                        help="Additional saved method; new connections reload completed cases")
    parser.add_argument("--check-all", action="store_true", help="Check every saved mesh and pose, then exit")
    args = parser.parse_args()
    legacy_cli = args.gpu is not None or args.legacy is not None or bool(args.method)
    if args.config and legacy_cli:
        parser.error('--config cannot be combined with --gpu / --legacy / --method; YAML uses automatic result discovery')
    config_path = args.config or (None if legacy_cli else DEFAULT_CONFIG)
    try:
        config = load_config(config_path) if config_path else {}
        def result_loader():
            return (Results(config=config_path) if config_path else
                Results(args.gpu or DEFAULT_REPORT/'background_results/summary.json',
                        args.legacy or DEFAULT_REPORT/'matrix', args.method))
        results = result_loader()
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.check_all:
        for method, rows in results.rows.items():
            checked = 0
            for row in rows:
                for stage in available_stages(row['case']):
                    meshes, camera, _, _ = load_geometry(row["case"], stage)
                    assert len(meshes) > 1 and np.isfinite(camera).all()
                    checked += 1
            print(f"{method}: {len(rows)} cases, {checked} saved IK/output poses checked", flush=True)
        return
    host = args.host or config.get('server', {}).get('host', '127.0.0.1')
    port = args.port if args.port is not None else config.get('server', {}).get('port', 8080)
    server = viser.ViserServer(host=host, port=port, label="抓取 IK / 优化结果")
    server.gui.configure_theme(control_layout="fixed", control_width="medium", show_logo=False,
                               show_share_button=False, brand_color=(53, 143, 199))
    playback_manager = PlaybackManager()
    sessions = {}

    @server.on_client_connect
    def connected(client):
        sessions[client.client_id] = attach_client(client, result_loader(), playback_manager)

    @server.on_client_disconnect
    def disconnected(client):
        close = sessions.pop(client.client_id, None)
        if close is not None:
            close()

    print(f"READY http://localhost:{server.get_port()} | "
          + ", ".join(f"{method}: {len(rows)} cases" for method, rows in results.rows.items()), flush=True)
    try:
        while True:
            time.sleep(1.)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

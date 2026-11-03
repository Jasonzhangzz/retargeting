"""Manifest-based replay, video export and a filtered local 3-D browser."""

from pathlib import Path
import json, time, threading
import numpy as np
import mujoco
from retarget.config.run import ROOT
from retarget.viewer.dense_pose import load_dense_grid, dense_revision
from retarget.viewer.snapshot import get_snapshot, read_summary


def scan(root=None):
    rows = []
    for manifest in sorted(
        (Path(root) if root else ROOT / "output").glob("*/manifest.json")
    ):
        batch = manifest.parent
        try:
            for case in json.loads(manifest.read_text())["cases"]:
                if case.get("method") not in ("cem", "mppi", "scsp", "dexforge", "contactaware", "graspadmm", "graspadmm_mppi"):
                    continue
                rows.append(
                    dict(case, batch=batch, label=f"{batch.name} / {case['name']}")
                )
        except (OSError, ValueError, KeyError):
            continue
    return rows


def size_scale(row):
    return row.get("size_scale", row.get("primitive", {}).get("size_scale", row.get("radius_scale")))


def filter_rows(rows, method="全部", task="全部", hand="全部", template="全部", *, batch=None,
                stage="全部", size="全部"):
    return [
        r
        for r in rows
        if (method == "全部" or r["method"] == method)
        and (task == "全部" or r["task"] == task)
        and (hand == "全部" or r.get("hand") == hand)
        and (template == "全部" or r.get("template", r["name"]) == template)
        and (batch is None or r["batch"].name == batch)
        and (stage == "全部" or r.get("stage") == stage)
        and (size == "全部" or (size_scale(r) is not None and f"{size_scale(r):g}×" == size))
    ]


def load_case(batch, case, *, final_only=False):
    if 'cached_case' in case:
        snapshot = get_snapshot(batch)
        index = int(case['cached_case'])
        model, qs = snapshot.model(index), snapshot.poses['qpos'][index:index+1].copy()
        if qs.shape != (1,model.nq) or not np.isfinite(qs).all():
            raise ValueError('Cached pose does not match the shared scene')
        return model, qs
    path = Path(case["model"])
    path = path if path.is_absolute() else batch / path
    model = mujoco.MjModel.from_xml_path(str(path))
    with np.load(batch / case["trajectory"], allow_pickle=False) as values:
        qs = values["qpos"].copy()
    if final_only:
        if qs.ndim != 2 or not len(qs) or qs.shape[1] != model.nq or not np.isfinite(qs[-1]).all():
            raise ValueError("No finite final pose matching the saved model")
        qs = qs[-1:].copy()
    return model, qs


def final_result_rows(rows):
    """Keep one optimized result per case, excluding process/evaluation replays."""
    selected = {}
    priority = {"selected_pose": 3, "refinement": 2, "hold": 1}
    for row in rows:
        phase = row.get("phase", "")
        if phase in ("acquisition", "native_test", "pregrasp", "preview", "failure"):
            continue
        if row.get("has_hand_pose") is False or row.get("stage") in ("CA失败预览", "失败信息"):
            continue
        key = (str(row["batch"]), row.get("case_id", row["name"]), row.get("method"))
        previous = selected.get(key)
        if previous is None or priority.get(phase, 0) > priority.get(previous.get("phase"), 0):
            selected[key] = dict(row, stage="最终结果", initial_index=-1)
    return list(selected.values())


def final_summary(summary):
    diagnostics = summary.get("geometry_diagnostics", {})
    lines = ["**最终抓取结果**"]
    if "coverage" in diagnostics:
        if diagnostics.get("coverage_definition", '').startswith("anatomical_patch_interior_"):
            lines.append(f"patch 内部贴合率：{diagnostics['coverage']:.1%}"
                         f"（{diagnostics['covered_patch_count']}/{diagnostics['required_patch_count']}）")
            centered = diagnostics.get('contact_region_policy') == 'fixed_prismatic_pad_center_v1'
            label = '固定指腹锚点贴合率' if centered else '原模板点贴合率'
            lines.append(f"{label}：{diagnostics['template_point_coverage']:.1%}")
            if centered:
                lines.append('接触区域：指腹中段（末节长度 30%–70%）')
            if diagnostics.get('object_surface_regions'):
                if 'required_side_finger_count' in diagnostics:
                    lines.append(f"手指侧面贴合：{diagnostics['covered_side_finger_count']}/"
                                 f"{diagnostics['required_side_finger_count']}（每指至少一个指定接触）")
                else:
                    lines.append('指定接触区域仅计圆柱侧面贴合。')
        else:
            lines.append(f"模板点贴合率：{diagnostics['coverage']:.1%}")
    if "projected_closure" in diagnostics:
        lines.append("投影摩擦锥闭合：" + ("通过" if diagnostics["projected_closure"] else "未证实"))
    if diagnostics.get('articulation_policy'):
        error=diagnostics.get('max_finger_order_violation_m',0.)
        lines.append('手指顺序约束：' + ('通过' if error <= 1.e-5 else '未通过'))
    if "torsional_friction_m" in diagnostics:
        coefficient = diagnostics["torsional_friction_m"]
        lines.append(f"扭转摩擦：开启（ComFree 投影平衡，系数 {coefficient:g} m）" if coefficient > 0 else "扭转摩擦：关闭")
    if summary.get("success_criterion") in ("projected_soft_contact_closure_v1", "projected_soft_contact_closure_v2"):
        lines.append("软接触闭合验收：" + ("通过" if summary.get("task_success") else "未通过"))
        if "max_hand_object_penetration_m" in diagnostics:
            label = "软阈值" if summary["success_criterion"].endswith("v2") else "允许"
            lines.append(f"手物压入：{diagnostics['max_hand_object_penetration_m'] * 1000:.3f} mm"
                         f" / {label} {diagnostics['hand_object_allowance_m'] * 1000:.3f} mm")
        if "max_hand_hand_penetration_m" in diagnostics:
            lines.append(f"手部自穿透：{diagnostics['max_hand_hand_penetration_m'] * 1000:.3f} mm")
    elif summary.get("success_criterion") == "projected_template_geometry_v1":
        lines.append("几何验收：" + ("通过" if summary.get("task_success") else "未通过"))
    elif summary.get("native_evaluated") or "task_success" in summary:
        lines.append("原生物理评测：" + ("通过" if summary.get("task_success") else "未通过")
                     if summary.get("native_evaluated", True) else "原生物理评测：未运行")
    if summary.get("optimization_seconds") is not None:
        lines.append(f"优化耗时：{summary['optimization_seconds']:.2f} s")
    return "\n\n".join(lines)


def load_patch_coverage(row, template_file):
    """Metric-only batches may store sidecars beside a separate summary."""
    if 'cached_case' in row:
        return get_snapshot(row['batch']).patch_coverage(int(row['cached_case']))
    explicit = row.get("patch_coverage")
    path = row["batch"] / explicit if explicit else template_file.with_name("patch_contact_coverage.json")
    return json.loads(path.read_text()) if explicit or path.is_file() else None


def primary_primitive_rows(rows):
    """One catalog primary primitive per numbered grasp; retain custom tasks."""
    from retarget.env.grasp.primitives import CATALOG
    selected = []
    for row in rows:
        number = str(row.get("template", "")).partition("_")[0]
        primitive = row.get("primitive")
        if row.get("task") == "grasp" and number.isdigit() and int(number) in CATALOG and primitive:
            if primitive["kind"] != CATALOG[int(number)][0][0] and not row.get("historical_reevaluation"):
                continue
        selected.append(row)
    return selected


def preferred_batch_rows(rows, primary, fallback=None):
    """Replace a historic case as soon as its recomputation is recorded, even if failed."""
    selected = {}
    primary_ids={r.get('case_id',r['name']) for r in rows if r['batch'].name==primary}
    # A changed size grid replaces a whole type, even for a partial preview.
    # Otherwise historical type12 s04=1x can be mixed with new s04=1.5x.
    policies={}
    for row in rows:
        if row['batch'].name==primary:
            key=(row.get('task'),row.get('hand'),row.get('template'))
            policies.setdefault(key,set()).add(row.get('primitive',{}).get('size_policy_version',2))
    for batch in (fallback, primary):
        if batch is None:
            continue
        for row in rows:
            if row["batch"].name == batch:
                case_id=row.get('case_id',row['name'])
                if batch==fallback and case_id in primary_ids:continue
                key=(row.get('task'),row.get('hand'),row.get('template'))
                if batch==fallback and key in policies and row.get('primitive',{}).get('size_policy_version',2) not in policies[key]:continue
                selected[(case_id,row.get('phase'))] = row
    return list(selected.values())


def hand_body_id(model, name):
    for candidate in (name, "hand-" + name):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if body >= 0:
            return body
    return -1


def hand_front_camera(model, data, *, fov=np.deg2rad(45.), aspect=1.5, fill=.82):
    """Face the actual palmar normal and frame the hand, independent of object size."""
    palm = hand_body_id(model, "right_palm_link")
    if palm < 0:
        return None
    if not 0 < fov < np.pi or not aspect > 0 or not 0 < fill < 1:
        raise ValueError("Camera fov, aspect and frame fill must be valid")
    from retarget.env.grasp.hand import palmar_axis
    rotation = data.xmat[palm].reshape(3, 3)
    local_normal = np.asarray([0., 1., 0.]) if model.body(palm).name.startswith("hand-") else palmar_axis(model, palm, None)
    normal = rotation @ local_normal
    normal /= np.linalg.norm(normal)
    roots = [data.xpos[body] for finger in range(2, 6)
        if (body := hand_body_id(model, f"right_finger{finger}_link1")) >= 0]
    up = np.mean(roots, axis=0)-data.xpos[palm] if roots else rotation[:, 2].copy()
    up -= normal*np.dot(up, normal)
    if np.linalg.norm(up) < 1e-8:
        up = rotation[:, int(np.argmin(np.abs(rotation.T @ normal)))].copy()
        up -= normal*np.dot(up, normal)
    up /= np.linalg.norm(up)
    right = np.cross(up, normal)
    right /= np.linalg.norm(right)
    hand = np.zeros(model.nbody, bool)
    hand[palm] = True
    for body in range(palm+1, model.nbody):
        hand[body] = hand[int(model.body_parentid[body])]
    vertices = []
    for geom in range(model.ngeom):
        if not hand[int(model.geom_bodyid[geom])] or model.geom_rgba[geom, 3] == 0:
            continue
        typ = model.geom_type[geom]
        if typ == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom]); start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start:start+count]
        elif typ == mujoco.mjtGeom.mjGEOM_BOX:
            local = np.asarray([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*model.geom_size[geom]
        else:
            radius = np.max(model.geom_size[geom])
            local = np.asarray([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*radius
        vertices.append(local @ data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom])
    points = np.concatenate(vertices) if vertices else np.asarray([data.xpos[palm]])
    basis = np.column_stack((right, up, normal))
    chart = points @ basis
    low, high = chart.min(0), chart.max(0)
    center = basis @ ((low+high)*.5)
    half_width, half_height, half_depth = np.maximum((high-low)*.5, .005)
    tangent = np.tan(fov*.5)
    relative = chart-(low+high)*.5
    # Perspective bounds use each vertex's depth, so a curled hand still fills
    # the frame without the extra padding of a projected bounding box.
    clearance = np.maximum(np.abs(relative[:, 1])/(tangent*fill),
        np.abs(relative[:, 0])/(aspect*tangent*fill))
    distance = max(.08, half_depth+.01, float(np.max(relative[:, 2]+clearance)))
    return dict(position=center+normal*distance, look_at=center, up=up, normal=normal,
        distance=distance, width=2*half_width, height=2*half_height, depth=2*half_depth,
        points=points, right=right, fov=fov, aspect=aspect, fill=fill)


def camera_for(model, data):
    camera = mujoco.MjvCamera()
    name = (
        "torso_link"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link") >= 0
        else "right_palm_link" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm_link") >= 0
        else "hand-right_palm_link" if hand_body_id(model, "right_palm_link") >= 0
        else "grasp_object" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object") >= 0 else "world"
    )
    body = model.body(name).id
    camera.lookat[:] = data.xpos[body]
    camera.distance = 3.0 if name == "torso_link" else 0.4 if name.endswith("right_palm_link") else max(.15, model.stat.extent*2)
    camera.azimuth = 135.0
    camera.elevation = -20.0
    return camera


def video(batch, case):
    import imageio.v2 as imageio

    model, qs = load_case(batch, case)
    data = mujoco.MjData(model)
    destination = batch / Path(case["trajectory"]).parent / "video.mp4"
    dt = case.get("dt", 0.033333)
    stride = max(1, round(1 / (30 * dt)))
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    with (
        imageio.get_writer(destination, fps=1 / (stride * dt)) as writer,
        mujoco.Renderer(model, height=480, width=640) as renderer,
    ):
        for q in qs[::stride]:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera_for(model, data))
            writer.append_data(renderer.render())
    return destination


def replay(config):
    batch = config.destination
    manifest = json.loads((batch / "manifest.json").read_text())
    for case in manifest["cases"]:
        if config.visualization.get("save_video", True):
            video(batch, case)
        if config.visualization.get("live", False):
            import mujoco.viewer

            model, qs = load_case(batch, case)
            data = mujoco.MjData(model)
            with mujoco.viewer.launch_passive(model, data) as window:
                for q in qs:
                    if not window.is_running():
                        break
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    window.sync()
                    time.sleep(case.get("dt", 0.033333))


def nearest_size(scales, value):
    """Snap historical/discrete batches to a recorded size."""
    return min(scales, key=lambda s: abs(s-value)) if scales else None


def playback_index(times, elapsed, speed=1.):
    """Map wall time to the latest recorded state, including the exact end."""
    times = np.asarray(times, float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Playback needs finite increasing recorded state times")
    if not np.isfinite(speed) or speed <= 0 or not np.isfinite(elapsed):
        raise ValueError("Playback speed must be positive and elapsed time finite")
    return int(np.clip(np.searchsorted(times-times[0], max(elapsed, 0.)*speed, side="right")-1, 0, len(times)-1))


def object_rollout_path(model, qs):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
    if body < 0:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj-objectworld")
    if body < 0 or len(qs) < 2:
        return None
    data, points = mujoco.MjData(model), []
    for qpos in qs:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        points.append(data.xpos[body].copy())
    return np.asarray(points)


def web(config):
    import viser
    import trimesh
    from scipy.spatial.transform import Rotation

    final_only = config.visualization.get("final_only", False)
    server = viser.ViserServer(host=config.visualization.get("host", "127.0.0.1"),
                               port=int(config.visualization.get("port", 8081)))
    server.scene.world_axes.visible = False

    @server.on_client_connect
    def connect(client):
        client.scene.set_up_direction("+z")
        method = client.gui.add_dropdown("方法", ("全部", "cem", "mppi", "scsp", "dexforge", "contactaware", "graspadmm", "graspadmm_mppi"), visible=not final_only)
        task = client.gui.add_dropdown("任务", ("全部", "grasp", "dex", "humanoid"),
            initial_value=config.visualization.get("task", "全部"), visible=not final_only)
        hand = client.gui.add_dropdown("手型", ("全部", "wuji1", "wuji2"),
            initial_value=config.visualization.get("hand", "全部"), visible=not final_only)
        grasp = client.gui.add_dropdown("抓型", ("全部",))
        stage = client.gui.add_dropdown("阶段", ("全部", "接触建立", "静止保持", "MPPI轨迹", "CA失败预览", "失败信息", "GraspADMM评测"), visible=not final_only)
        size = client.gui.add_slider("尺寸倍率", min=.5, max=1.5, step=.25,
            initial_value=1., hint="拖动选择尺寸；提供加密数据时连续插值预览。")
        all_sizes = client.gui.add_checkbox("全部尺寸", initial_value=False)
        size_info = client.gui.add_markdown("")
        selection = client.gui.add_dropdown("结果", ("无结果",))
        refresh = client.gui.add_button("刷新 output")
        front = client.gui.add_button("正对手掌")
        play = client.gui.add_button("播放 rollout", disabled=True, visible=not final_only)
        show_palm=client.gui.add_checkbox('显示广义掌面候选',initial_value=False, visible=not final_only)
        show_contacts=client.gui.add_checkbox('显示接触 patch 与投影' if final_only else '显示接触点',initial_value=True)
        playback_status = client.gui.add_markdown("")
        details = client.gui.add_markdown("选择结果查看状态、cost 与耗时。")
        display_root = client.scene.add_frame("/display", show_axes=False)
        state = dict(dense_grid=None, dense_palm_anchor=None, dense_last_scale=None,
            dense_base_primitive=None, dense_dimensions=None, object_geom=None, projection_handle=None,
            dense_summary_cache={}, dense_revision=None, rows={}, handles=[], model=None, qs=None, data=None, busy=False, initialized=False,
            index=0, playing=False, times=None, scales=[], path=None, loaded=None, row=None,
            points=[],actual_contact_handle=None,contact_points=None,contact_forces=None,contact_counts=None)
        render_lock = threading.RLock()

        def serialized(callback):
            def protected(*args, **kwargs):
                with render_lock:
                    return callback(*args, **kwargs)
            return protected

        @serialized
        def move(index=None):
            if state["model"] is None:
                return
            model, data = state["model"], state["data"]
            if index is not None:
                state["index"] = min(int(index), len(state["qs"])-1)
            data.qpos[:] = state["qs"][state["index"]]
            mujoco.mj_kinematics(model, data)
            with client.atomic():
                palm = hand_body_id(model, "right_palm_link")
                if state["dense_grid"] is not None and palm >= 0:
                    rotation = data.xmat[palm].reshape(3,3)
                    origin = data.xpos[palm]
                    if state["dense_palm_anchor"] is None:
                        state["dense_palm_anchor"] = (origin.copy(), rotation.copy())
                    anchor_position, anchor_rotation = state["dense_palm_anchor"]
                    transform = anchor_rotation @ rotation.T
                    display_root.position = anchor_position-transform@origin
                    display_root.wxyz = Rotation.from_matrix(transform).as_quat(scalar_first=True)
                for geom, handle in state["handles"]:
                    if geom == state["object_geom"]:
                        body = model.geom_bodyid[geom]
                        handle.position = data.xpos[body]
                        handle.wxyz = Rotation.from_matrix(data.xmat[body].reshape(3,3)).as_quat(scalar_first=True)
                        primitive = state["dense_base_primitive"]
                        dimensions = state["dense_dimensions"]
                        handle.scale = (dimensions[0]/primitive['radius_m'], dimensions[0]/primitive['radius_m'],
                            dimensions[1]/primitive['height_m'])
                        continue
                    handle.position = data.geom_xpos[geom]
                    handle.wxyz = Rotation.from_matrix(
                        data.geom_xmat[geom].reshape(3, 3)
                    ).as_quat(scalar_first=True)
                for body,handle,kind in state['points']:
                    handle.position=data.xpos[body]
                    handle.wxyz=Rotation.from_matrix(data.xmat[body].reshape(3,3)).as_quat(scalar_first=True)
                    handle.visible=show_palm.value if kind=='palm' else show_contacts.value
                if state['actual_contact_handle'] is not None:
                    index=state['index'];count=int(state['contact_counts'][index])
                    points=state['contact_points'][index,:count]
                    if state['contact_forces'] is not None:points=points[state['contact_forces'][index,:count]>1e-6]
                    state['actual_contact_handle'].points=points.astype(np.float32)
                    state['actual_contact_handle'].colors=np.tile(np.asarray([255,120,30],np.uint8),(len(points),1))
                    state['actual_contact_handle'].visible=show_contacts.value and len(points)>0

        @serialized
        def show_dense_size(value):
            grid = state["dense_grid"]
            if grid is None or state["model"] is None:
                return
            value = float(np.clip(value, grid.scales[0], grid.scales[-1]))
            if state["dense_last_scale"] is not None and abs(value-state["dense_last_scale"]) < 1.e-12:
                return
            sample = grid.sample(value)
            state["dense_last_scale"] = value
            state["qs"] = sample["qpos"][None]
            state["index"] = 0
            state["dense_dimensions"] = (sample["radius_m"], sample["height_m"])
            p = state["dense_base_primitive"]
            axis = {"radius":"半径", "thickness":"厚度", "height":"高度"}.get(p.get("size_axis"), "尺寸")
            with client.atomic():
                if state["projection_handle"] is not None:
                    points = sample['projected_points']
                    if points is not None:
                        state["projection_handle"].points = points.astype(np.float32)
                        state["projection_handle"].visible = show_contacts.value
                    else:
                        state["projection_handle"].visible = False
                size_info.content = (f"**{p['kind']} · {axis} {value:.5g}×** · "
                    f"半径 {sample['radius_m']*1000:.3f} mm · 高度/厚度 {sample['height_m']*1000:.3f} mm")
                playback_status.content = "已保存求解点。" if sample['solved'] else "连续插值预览（未重新求解或评测）。"
                if sample['solved'] and getattr(grid, 'source_rows', None):
                    row = grid.source_rows[sample['source_index']]
                    key = (str(state['row']['batch']), row.get('cached_case', row['summary']))
                    if key not in state['dense_summary_cache']:
                        state['dense_summary_cache'][key] = final_summary(read_summary(state['row']['batch'], row))
                    details.content = state['dense_summary_cache'][key]
                else:
                    details.content = "物体尺寸与手姿插值；接触、闭合和碰撞指标未重新验证。"
                move(0)

        @serialized
        def focus_hand(_=None):
            if state["model"] is None:
                return
            view = hand_front_camera(state["model"], state["data"], fov=client.camera.fov,
                aspect=max(client.camera.image_width, 1)/max(client.camera.image_height, 1))
            with client.atomic():
                if view is not None:
                    client.camera.position = view["position"]
                    client.camera.look_at = view["look_at"]
                    client.camera.up_direction = view["up"]
                else:
                    camera = camera_for(state["model"], state["data"])
                    client.camera.look_at = camera.lookat
                    client.camera.position = camera.lookat+np.array([camera.distance, -camera.distance, camera.distance*.5])

        @serialized
        def show(_=None):
            if state["busy"]:
                return
            row = state["rows"].get(selection.value)
            if row is None:
                for _, handle in state["handles"]:
                    handle.remove()
                if state["path"] is not None:
                    state["path"].remove()
                for _,handle,_ in state['points']:handle.remove()
                if state['actual_contact_handle'] is not None:state['actual_contact_handle'].remove()
                state.update(points=[],actual_contact_handle=None)
                state.update(handles=[], path=None, model=None, qs=None, loaded=None, playing=False, dense_grid=None)
                play.disabled = True
                playback_status.content = "当前筛选条件下没有保存的结果。"
                details.content = "请选择其他抓型或尺寸。" if final_only else "请选择其他抓型、尺寸或阶段。"
                return
            if (state["dense_grid"] is not None and state["row"] is not None
                    and row.get("template") == state["row"].get("template")
                    and row.get("primitive",{}).get("kind") == state["row"].get("primitive",{}).get("kind")
                    and row["batch"] == state["row"]["batch"]
                    and state['dense_revision'] == dense_revision(row['batch'])):
                show_dense_size(size_scale(row))
                return
            try:
                key = (row["label"], (row["batch"]/row["trajectory"]).stat().st_mtime_ns, dense_revision(row["batch"]) if final_only else None)
                if state["loaded"] == key:
                    return
                model, qs = load_case(row["batch"], row, final_only=final_only)
                primitive = row.get('primitive')
                dense_grid = load_dense_grid(row['batch'], row.get('template'), primitive.get('kind')) if final_only and primitive else None
                if 'cached_case' in row:
                    times = np.array([0.])
                    contacts = dict(contact_points=None,contact_normal_force=None,contact_counts=None)
                else:
                    with np.load(row["batch"]/row["trajectory"], allow_pickle=False) as values:
                        times = values["time"].copy() if "time" in values else np.arange(len(qs))*row.get("dt", 1/30)
                        contacts={key:values[key].copy() if key in values else None for key in ('contact_points','contact_normal_force','contact_counts')}
                if final_only:
                    times = np.array([0.])
                    contacts = {key: value[-1:] if value is not None else None for key, value in contacts.items()}
                playback_index(times, 0.)
                for _,handle,_ in state['points']:handle.remove()
                if state['actual_contact_handle'] is not None:state['actual_contact_handle'].remove()
                # A size comparison keeps the user's camera relative to the
                # palm. Reframing each different finger configuration makes
                # a stationary object appear to jump even in a common frame.
                camera_in_palm = None
                previous = state["row"]
                if (previous is not None and previous.get("template")==row.get("template")
                        and previous.get("hand")==row.get("hand") and state["model"] is not None and state["dense_grid"] is None):
                    old_palm = hand_body_id(state["model"], "right_palm_link")
                    if old_palm >= 0:
                        old_data = state["data"]
                        inverse = old_data.xmat[old_palm].reshape(3,3).T
                        origin = old_data.xpos[old_palm]
                        camera_in_palm = (inverse@(np.asarray(client.camera.position)-origin),
                            inverse@(np.asarray(client.camera.look_at)-origin), inverse@np.asarray(client.camera.up_direction))
                for _, handle in state["handles"]:
                    handle.remove()
                if state["path"] is not None:
                    state["path"].remove()
                display_root.position = (0.,0.,0.)
                display_root.wxyz = (1.,0.,0.,0.)
                state.update(dense_grid=dense_grid, dense_palm_anchor=None, dense_last_scale=None,
                    dense_base_primitive=primitive, dense_dimensions=(primitive['radius_m'],primitive['height_m']) if primitive else None,
                    object_geom=None, projection_handle=None,
                    dense_revision=dense_revision(row['batch']) if dense_grid is not None else None)
                state.update(model=model, qs=qs, data=mujoco.MjData(model), handles=[],
                    index=0, times=times, playing=False, path=None, loaded=key, row=row)
                state.update(points=[],actual_contact_handle=None,contact_points=contacts['contact_points'],
                    contact_forces=contacts['contact_normal_force'],contact_counts=contacts['contact_counts'])
                selection.visible = not (final_only and dense_grid is not None and not all_sizes.value)
                state['data'].qpos[:] = qs[-1]
                mujoco.mj_kinematics(model,state['data'])
                if dense_grid is not None:
                    state['busy'] = True
                    size.min, size.max = float(dense_grid.scales[0]), float(dense_grid.scales[-1])
                    size.step = (size.max-size.min)/10000
                    size.value = float(np.clip(size_scale(row),size.min,size.max))
                    state['busy'] = False
                for g in range(model.ngeom):
                    if model.geom_rgba[g, 3] == 0:
                        continue
                    typ = model.geom_type[g]
                    geom_size = model.geom_size[g]
                    if typ == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh = model.geom_dataid[g]
                        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                        f, k = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                        vertices = model.mesh_vert[a : a + n]
                        faces = model.mesh_face[f : f + k]
                    elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                        m = trimesh.creation.box(2 * geom_size)
                        vertices, faces = m.vertices, m.faces
                    elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                        m = trimesh.creation.icosphere(subdivisions=1, radius=geom_size[0])
                        vertices, faces = m.vertices, m.faces
                    else:
                        continue
                    is_object = dense_grid is not None and model.body(int(model.geom_bodyid[g])).name == 'grasp_object'
                    if is_object:
                        # Store object vertices in its body frame. Per-axis
                        # scaling then changes dimensions without mesh uploads.
                        data = state['data']; body = model.geom_bodyid[g]
                        world = vertices @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
                        vertices = (world-data.xpos[body]) @ data.xmat[body].reshape(3,3)
                        state['object_geom'] = g
                    handle = client.scene.add_mesh_simple(
                        f"/display/model/{g}",
                        vertices=vertices,
                        faces=faces,
                        color=tuple((model.geom_rgba[g, :3] * 255).astype(int)),
                    )
                    state["handles"].append((g, handle))
                template_file = row["batch"] / Path(row["trajectory"]).parent / "template_contacts.json"
                patch_coverage = load_patch_coverage(row, template_file) if final_only and dense_grid is None else None
                if patch_coverage is not None:
                    projections = []
                    for i, patch in enumerate(patch_coverage["patches"]):
                        body = hand_body_id(model, patch["body"])
                        if body < 0:
                            raise ValueError(f"Missing contact patch body: {patch['body']}")
                        if patch["local_points"]:
                            handle = client.scene.add_point_cloud(f'/display/guidance/patch/{i}',
                                points=np.asarray(patch["local_points"], dtype=np.float32),
                                colors=(80,200,120) if patch["fitted"] else (240,150,40),
                                point_size=.0012, visible=show_contacts.value)
                            state['points'].append((body, handle, 'contact'))
                        witness = patch.get("witness")
                        if witness is not None and witness["fitted"]:
                            handle = client.scene.add_point_cloud(f'/display/guidance/patch_contact/{i}',
                                points=np.asarray([witness["local_point"]], dtype=np.float32),
                                colors=(30,255,80), point_size=.003, visible=show_contacts.value)
                            state['points'].append((body, handle, 'contact'))
                            projections.append(witness["projected_point"])
                    if projections:
                        handle = client.scene.add_point_cloud('/display/guidance/patch_projections',
                            points=np.asarray(projections, dtype=np.float32), colors=(40,160,255),
                            point_size=.002, visible=show_contacts.value)
                        state['points'].append((0, handle, 'contact'))
                elif final_only and (template_file.is_file() or 'cached_case' in row):
                    template = get_snapshot(row['batch']).template(int(row['cached_case'])) if 'cached_case' in row else json.loads(template_file.read_text())
                    names = np.asarray(template["body_names"])
                    local = np.asarray(template["local_points"], dtype=np.float32)
                    for name in dict.fromkeys(names):
                        body = hand_body_id(model, str(name))
                        if body >= 0:
                            handle = client.scene.add_point_cloud(f'/display/guidance/template/{body}',
                                points=local[names == name], colors=(255,120,30), point_size=.003,
                                visible=show_contacts.value)
                            state['points'].append((body, handle, 'contact'))
                    projection_file = template_file.with_name("projection_force_closure.json")
                    if projection_file.is_file() or 'cached_case' in row:
                        projection = get_snapshot(row['batch']).diagnostics(int(row['cached_case'])) if 'cached_case' in row else json.loads(projection_file.read_text())
                        handle = client.scene.add_point_cloud('/display/guidance/projections',
                            points=np.asarray(projection["projected_points"], dtype=np.float32),
                            colors=(40,160,255), point_size=.002, visible=show_contacts.value)
                        state['points'].append((0, handle, 'contact'))
                        if dense_grid is not None:state['projection_handle'] = handle
                if row.get('seed_source'):
                    with np.load(Path(row['seed_source'])/'spider_seed.npz',allow_pickle=False) as seed:
                        bodies=seed['palm_bodies'];local=seed['palm_local']
                        for body in np.unique(bodies):
                            handle=client.scene.add_point_cloud(f'/display/guidance/palm/{body}',points=local[bodies==body].astype(np.float32),
                                colors=(35,200,230),point_size=.0015,visible=show_palm.value)
                            state['points'].append((int(body),handle,'palm'))
                    if state['contact_points'] is not None:
                        state['actual_contact_handle']=client.scene.add_point_cloud('/display/guidance/actual_contacts',points=np.empty((0,3),np.float32),
                            colors=(255,120,30),point_size=.003,visible=show_contacts.value)
                    elif row.get('acquisition_report'):
                        path=Path(row['acquisition_report']).parent/'contacts.npz'
                        if path.is_file():
                            with np.load(path) as raw:
                                for body in np.unique(raw['bodies']):
                                    handle=client.scene.add_point_cloud(f'/display/guidance/contacts/{body}',points=raw['local'][raw['bodies']==body].astype(np.float32),
                                        colors=(255,120,30),point_size=.003,visible=show_contacts.value)
                                    state['points'].append((int(body),handle,'contact'))
                play.disabled = len(qs) < 2
                play.label = "播放 rollout" if len(qs) > 1 else "无 rollout 可播放"
                playback_status.content = (f"时长 {times[-1]-times[0]:.3f} s，点击播放完整轨迹（0.25 倍速）。"
                    if len(qs) > 1 else ("历史 CA 失败提议，仅展示已保存姿态。" if row.get('method')=='contactaware'
                                       else "求解异常，仅展示已保存的有限初值。") if row.get("has_hand_pose", True)
                    else "未保存手部姿态：仅展示该尺寸物体和错误信息。")
                p = row.get("primitive")
                if row.get('time_kind')=='optimization_iteration':
                    playback_status.content=f'固定物体的接触建立：{len(qs)} 个优化记录；橙色为原生几何接触点。'
                    play.label='播放接触建立'
                if final_only:
                    playback_status.content = ("绿色：已贴合 patch；亮绿大点：内部接触；橙色：未贴合 patch；蓝色：接触的物面投影。手部边界/尖锐棱边内缩 1 mm。"
                        if patch_coverage is not None else "橙色：手部模板接触；蓝色：物面投影。" if template_file.is_file() or 'cached_case' in row else "最终优化姿态。")
                axis = p.get("size_axis", "radius") if p else "radius"
                axis_label = {"thickness": "厚度", "height": "长度", "radius": "半径"}.get(axis, axis)
                scale_value = size_scale(row)
                if p:
                    radius_mm = p['radius_m']*1000
                    radius_text = f'{radius_mm:.3f}' if radius_mm < 1. else f'{radius_mm:.2f}'
                    dimensions = (f"板面 {p['dimensions_m'][0]*1000:g}×{p['dimensions_m'][1]*1000:g} mm · 厚度 {p['height_m']*1000:.2f} mm"
                        if p['kind'] in ('plate', 'cuboid') and p.get('dimensions_m') else
                        f"半径 {radius_text} mm · 长度/厚度 {p['height_m']*1000:.2f} mm")
                    size_info.content = f"**{p['kind']} · {axis_label} {scale_value if scale_value is not None else 1.:g}×** · {dimensions}"
                else:
                    size_info.content = ''
                if config.visualization.get("fallback_batch"):
                    size_info.content += ("\n\n当前约束重算结果" if row["batch"].name == config.visualization.get("batch")
                                          else "\n\n历史结果（等待该尺寸重算）")
                move(len(qs)-1 if row.get('initial_index',0)==-1 else row.get('initial_index',0))
                palm = hand_body_id(model, "right_palm_link")
                if camera_in_palm is not None and palm >= 0:
                    rotation = state["data"].xmat[palm].reshape(3,3)
                    origin = state["data"].xpos[palm]
                    with client.atomic():
                        client.camera.position = origin+rotation@camera_in_palm[0]
                        client.camera.look_at = origin+rotation@camera_in_palm[1]
                        client.camera.up_direction = rotation@camera_in_palm[2]
                else:
                    focus_hand()
                summary = read_summary(row['batch'], row)
                details.content = final_summary(summary) if final_only else "**"+row.get("stage", "结果")+"**\n\n```json\n" + json.dumps(summary,ensure_ascii=False,indent=2) + "\n```"
            except Exception as exc:
                details.content = f"加载失败：{exc}"

        @serialized
        def update(_=None):
            if state["busy"]:
                return
            rows = scan()
            selected_batch = config.visualization.get("batch")
            if selected_batch:
                rows = preferred_batch_rows(rows, selected_batch, config.visualization.get("fallback_batch"))
            if config.visualization.get("primary_shape_only", True):
                rows = primary_primitive_rows(rows)
            if final_only:
                rows = final_result_rows(rows)
                if selected_batch:
                    rows = [dict(row, label=row["name"]) for row in rows]
            types = sorted(
                {r.get("template", r["name"]) for r in rows if r["task"] == "grasp"},
                key=lambda name: (int(name.partition("_")[0]) if name.partition("_")[0].isdigit() else 10000, name)
            )
            state["busy"] = True
            if grasp.options != ("全部", *types):
                grasp.options = ("全部", *types)
            preferred = next((r for r in rows if r.get("case_id", r["name"]) == config.visualization.get("initial_case") and r.get('phase')!='acquisition'), None)
            if preferred is None:preferred=next((r for r in rows if r.get('case_id', r['name'])==config.visualization.get('initial_case')),None)
            if not state["initialized"] and preferred is not None:
                grasp.value = preferred.get("template", preferred["name"])
            rows = filter_rows(rows, method.value, task.value, hand.value, grasp.value)
            scales = sorted({size_scale(r) for r in rows if size_scale(r) is not None})
            state["scales"] = scales
            previous_size = size.value
            size.disabled = len(scales) < 2 or all_sizes.value
            if final_only:
                size.visible = len(scales) > 1
                all_sizes.visible = len(scales) > 1
            if scales:
                size.min = scales[0]
                size.max = scales[-1] if len(scales) > 1 else scales[0]+1.
                size.step = min(np.diff(scales)) if len(scales) > 1 else .25
                if not state["initialized"] and preferred is not None and size_scale(preferred) is not None:
                    previous_size = size_scale(preferred)
                size.value = nearest_size(scales, previous_size)
            rows = filter_rows(rows, stage=stage.value,
                size="全部" if all_sizes.value or not scales else f"{size.value:g}×")
            state["rows"] = {r["label"]: r for r in rows}
            previous = selection.value
            selection.options = tuple(state["rows"]) or ("无结果",)
            if previous in state["rows"]:
                selection.value = previous
            elif not state["initialized"]:
                preferred = config.visualization.get("initial_case")
                choice = next((label for label, row in state["rows"].items() if row.get("case_id", row["name"]) == preferred and row.get('phase')!='acquisition'), None)
                if choice is None:choice=next((label for label,row in state['rows'].items() if row.get('case_id', row['name'])==preferred),None)
                if choice:
                    selection.value = choice
                state["initialized"] = True
            state["busy"] = False
            selection.visible = not (final_only and state['dense_grid'] is not None and not all_sizes.value)
            show()

        @serialized
        def size_changed(_=None):
            if state["busy"]:
                return
            if state['dense_grid'] is not None and state['dense_revision'] != dense_revision(state['row']['batch']):
                state['dense_grid'] = None
                update()
                return
            if state['dense_grid'] is not None and grasp.value != '全部':
                show_dense_size(size.value)
                return
            # Historical batches keep discrete saved-size selection.
            update()

        @serialized
        def start_playback(_=None):
            if state["qs"] is None or len(state["qs"]) < 2:
                return
            move(0)
            points = object_rollout_path(state["model"], state["qs"])
            if state["path"] is not None:
                state["path"].remove()
            if points is not None:
                if np.max(np.linalg.norm(points-points[:1],axis=1))>1e-9:
                    segments = np.stack((points[:-1], points[1:]), axis=1).astype(np.float32)
                    state["path"] = client.scene.add_line_segments("/rollout/object_path", points=segments,
                        colors=(40, 160, 255), thickness=3., thickness_units="screen")
            state.update(playing=True, play_started=time.monotonic())
            play.disabled = True
            playback_status.content = "正在播放完整 rollout（0.25 倍速），蓝线为物体轨迹。"
            if state['row'].get('time_kind')=='optimization_iteration':
                playback_status.content='正在播放接触建立的优化记录（每秒 8 条）；物体保持固定。'

        for control in (method, task, hand, grasp, stage):
            control.on_update(update)
        size.on_update(size_changed)
        all_sizes.on_update(size_changed)
        refresh.on_click(update)
        front.on_click(focus_hand)
        selection.on_update(show)
        play.on_click(start_playback)
        show_palm.on_update(lambda _:move())
        show_contacts.on_update(lambda _:move())
        update()

        def animate():
            stamp = None
            next_scan = 0.
            while client.client_id in server.get_clients():
                now = time.monotonic()
                if now >= next_scan:
                    changed = tuple((p.stat().st_mtime_ns, str(p)) for p in (ROOT/"output").glob("*/manifest.json"))
                    if changed != stamp:
                        update(); stamp = changed
                    next_scan = now+1.
                with render_lock:
                    if state["playing"]:
                        iterative=state['row'].get('time_kind')=='optimization_iteration'
                        index = playback_index(state["times"], now-state["play_started"], 8. if iterative else .25)
                        if index != state["index"]:
                            move(index)
                        if index == len(state["qs"])-1:
                            state["playing"] = False
                            play.disabled = False
                            play.label = "重新播放 rollout"
                            playback_status.content = f"播放完成：{state['times'][-1]-state['times'][0]:.3f} s rollout。"
                            if iterative:playback_status.content='接触建立的优化记录播放完成。'
                time.sleep(1/60)

        threading.Thread(target=animate, daemon=True).start()

    print(f"Viewer: http://localhost:{server.get_port()}", flush=True)
    while True:
        time.sleep(1)


def run(config):
    return web(config) if config.mode == "web" else replay(config)

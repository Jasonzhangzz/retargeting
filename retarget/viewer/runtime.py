"""Manifest-based replay, video export and a filtered local 3-D browser."""

from pathlib import Path
import json, time, threading
import numpy as np
import mujoco
from retarget.config.run import ROOT


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


def load_case(batch, case):
    path = Path(case["model"])
    path = path if path.is_absolute() else batch / path
    model = mujoco.MjModel.from_xml_path(str(path))
    with np.load(batch / case["trajectory"], allow_pickle=False) as values:
        return model, values["qpos"].copy()


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
    for batch in (fallback, primary):
        if batch is None:
            continue
        for row in rows:
            if row["batch"].name == batch:
                case_id=row.get('case_id',row['name'])
                if batch==fallback and case_id in primary_ids:continue
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
    """Snap to a recorded size; never interpolate a hand pose across objects."""
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

    server = viser.ViserServer(port=int(config.visualization.get("port", 8081)))
    server.scene.world_axes.visible = False

    @server.on_client_connect
    def connect(client):
        client.scene.set_up_direction("+z")
        method = client.gui.add_dropdown("方法", ("全部", "cem", "mppi", "scsp", "dexforge", "contactaware", "graspadmm", "graspadmm_mppi"))
        task = client.gui.add_dropdown("任务", ("全部", "grasp", "dex", "humanoid"),
            initial_value=config.visualization.get("task", "全部"))
        hand = client.gui.add_dropdown("手型", ("全部", "wuji1", "wuji2"),
            initial_value=config.visualization.get("hand", "全部"))
        grasp = client.gui.add_dropdown("抓型", ("全部",))
        stage = client.gui.add_dropdown("阶段", ("全部", "接触建立", "静止保持", "MPPI轨迹", "CA失败预览", "失败信息", "GraspADMM评测"))
        size = client.gui.add_dropdown("尺寸倍率", ("全部",), initial_value="全部")
        size_info = client.gui.add_markdown("")
        selection = client.gui.add_dropdown("结果", ("无结果",))
        refresh = client.gui.add_button("刷新 output")
        front = client.gui.add_button("正对手掌")
        play = client.gui.add_button("播放 rollout", disabled=True)
        show_palm=client.gui.add_checkbox('显示广义掌面候选',initial_value=False)
        show_contacts=client.gui.add_checkbox('显示接触点',initial_value=True)
        playback_status = client.gui.add_markdown("")
        details = client.gui.add_markdown("选择结果查看状态、cost 与耗时。")
        state = dict(rows={}, handles=[], model=None, qs=None, data=None, busy=False, initialized=False,
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
                for geom, handle in state["handles"]:
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
                state.update(handles=[], path=None, model=None, qs=None, loaded=None, playing=False)
                play.disabled = True
                playback_status.content = "当前筛选条件下没有保存的结果。"
                details.content = "请选择其他抓型、尺寸或阶段。"
                return
            try:
                key = (row["label"], (row["batch"]/row["trajectory"]).stat().st_mtime_ns)
                if state["loaded"] == key:
                    return
                model, qs = load_case(row["batch"], row)
                with np.load(row["batch"]/row["trajectory"], allow_pickle=False) as values:
                    times = values["time"].copy() if "time" in values else np.arange(len(qs))*row.get("dt", 1/30)
                    contacts={key:values[key].copy() if key in values else None for key in ('contact_points','contact_normal_force','contact_counts')}
                playback_index(times, 0.)
                for _,handle,_ in state['points']:handle.remove()
                if state['actual_contact_handle'] is not None:state['actual_contact_handle'].remove()
                # A size comparison keeps the user's camera relative to the
                # palm. Reframing each different finger configuration makes
                # a stationary object appear to jump even in a common frame.
                camera_in_palm = None
                previous = state["row"]
                if (previous is not None and previous.get("template")==row.get("template")
                        and previous.get("hand")==row.get("hand") and state["model"] is not None):
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
                state.update(model=model, qs=qs, data=mujoco.MjData(model), handles=[],
                    index=0, times=times, playing=False, path=None, loaded=key, row=row)
                state.update(points=[],actual_contact_handle=None,contact_points=contacts['contact_points'],
                    contact_forces=contacts['contact_normal_force'],contact_counts=contacts['contact_counts'])
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
                    handle = client.scene.add_mesh_simple(
                        f"/model/{g}",
                        vertices=vertices,
                        faces=faces,
                        color=tuple((model.geom_rgba[g, :3] * 255).astype(int)),
                    )
                    state["handles"].append((g, handle))
                if row.get('seed_source'):
                    with np.load(Path(row['seed_source'])/'spider_seed.npz',allow_pickle=False) as seed:
                        bodies=seed['palm_bodies'];local=seed['palm_local']
                        for body in np.unique(bodies):
                            handle=client.scene.add_point_cloud(f'/guidance/palm/{body}',points=local[bodies==body].astype(np.float32),
                                colors=(35,200,230),point_size=.0015,visible=show_palm.value)
                            state['points'].append((int(body),handle,'palm'))
                    if state['contact_points'] is not None:
                        state['actual_contact_handle']=client.scene.add_point_cloud('/guidance/actual_contacts',points=np.empty((0,3),np.float32),
                            colors=(255,120,30),point_size=.003,visible=show_contacts.value)
                    elif row.get('acquisition_report'):
                        path=Path(row['acquisition_report']).parent/'contacts.npz'
                        if path.is_file():
                            with np.load(path) as raw:
                                for body in np.unique(raw['bodies']):
                                    handle=client.scene.add_point_cloud(f'/guidance/contacts/{body}',points=raw['local'][raw['bodies']==body].astype(np.float32),
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
                axis = p.get("size_axis", "radius") if p else "radius"
                axis_label = {"thickness": "厚度", "height": "长度", "radius": "半径"}.get(axis, axis)
                scale_value = size_scale(row)
                size_info.content = (f"**{p['kind']} · {axis_label} {scale_value if scale_value is not None else 1.:g}×** · 半径 {p['radius_m']*1000:.2f} mm · 长度/厚度 {p['height_m']*1000:.2f} mm" if p else "")
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
                summary = row["batch"] / row["summary"]
                details.content = "**"+row.get("stage", "结果")+"**\n\n```json\n" + summary.read_text() + "\n```"
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
            types = sorted(
                {r.get("template", r["name"]) for r in rows if r["task"] == "grasp"},
                key=lambda name: (int(name.partition("_")[0]) if name.partition("_")[0].isdigit() else 10000, name)
            )
            state["busy"] = True
            if grasp.options != ("全部", *types):
                grasp.options = ("全部", *types)
            preferred = next((r for r in rows if r.get("case_id") == config.visualization.get("initial_case") and r.get('phase')!='acquisition'), None)
            if preferred is None:preferred=next((r for r in rows if r.get('case_id')==config.visualization.get('initial_case')),None)
            if not state["initialized"] and preferred is not None:
                grasp.value = preferred.get("template", preferred["name"])
            rows = filter_rows(rows, method.value, task.value, hand.value, grasp.value)
            scales = sorted({size_scale(r) for r in rows if size_scale(r) is not None})
            state["scales"] = scales
            size.disabled = len(scales) < 2
            choices = ("全部", *(f"{s:g}×" for s in scales))
            previous_size = size.value
            if size.options != choices:
                size.options = choices
            if not state["initialized"] and preferred is not None and size_scale(preferred) is not None:
                size.value = f"{size_scale(preferred):g}×"
            elif previous_size in choices:
                size.value = previous_size
            else:
                size.value = "全部"
            rows = filter_rows(rows, stage=stage.value,
                size=size.value)
            state["rows"] = {r["label"]: r for r in rows}
            previous = selection.value
            selection.options = tuple(state["rows"]) or ("无结果",)
            if previous in state["rows"]:
                selection.value = previous
            elif not state["initialized"]:
                preferred = config.visualization.get("initial_case")
                choice = next((label for label, row in state["rows"].items() if row.get("case_id") == preferred and row.get('phase')!='acquisition'), None)
                if choice is None:choice=next((label for label,row in state['rows'].items() if row.get('case_id')==preferred),None)
                if choice:
                    selection.value = choice
                state["initialized"] = True
            state["busy"] = False
            show()

        @serialized
        def size_changed(_=None):
            if state["busy"]:
                return
            # Keep the same grasp/shape while selecting another sampled size.
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

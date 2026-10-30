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
                if case.get("method") not in ("cem", "mppi"):
                    continue
                rows.append(
                    dict(case, batch=batch, label=f"{batch.name} / {case['name']}")
                )
        except (OSError, ValueError, KeyError):
            continue
    return rows


def filter_rows(rows, method="全部", task="全部", hand="全部", template="全部"):
    return [
        r
        for r in rows
        if (method == "全部" or r["method"] == method)
        and (task == "全部" or r["task"] == task)
        and (hand == "全部" or r.get("hand") == hand)
        and (template == "全部" or r.get("template", r["name"]) == template)
    ]


def load_case(batch, case):
    path = Path(case["model"])
    path = path if path.is_absolute() else batch / path
    model = mujoco.MjModel.from_xml_path(str(path))
    values = np.load(batch / case["trajectory"], allow_pickle=False)
    return model, values["qpos"].copy()


def camera_for(model, data):
    camera = mujoco.MjvCamera()
    name = (
        "torso_link"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link") >= 0
        else "right_palm_link"
    )
    body = model.body(name).id
    camera.lookat[:] = data.xpos[body]
    camera.distance = 3.0 if name == "torso_link" else 0.4
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


def web(config):
    import viser
    import trimesh
    from scipy.spatial.transform import Rotation

    server = viser.ViserServer(port=int(config.visualization.get("port", 8081)))

    @server.on_client_connect
    def connect(client):
        client.scene.set_up_direction("+z")
        client.scene.world_axes.visible = False
        method = client.gui.add_dropdown("方法", ("全部", "cem", "mppi"))
        task = client.gui.add_dropdown("任务", ("全部", "grasp", "dex", "humanoid"))
        hand = client.gui.add_dropdown("手型", ("全部", "wuji1", "wuji2"))
        grasp = client.gui.add_dropdown("抓型", ("全部",))
        selection = client.gui.add_dropdown("结果", ("无结果",))
        refresh = client.gui.add_button("刷新 output")
        frame = client.gui.add_slider("帧", min=0, max=1, step=1, initial_value=0)
        play = client.gui.add_checkbox("播放", False)
        details = client.gui.add_markdown("选择结果查看状态、cost 与耗时。")
        state = dict(rows={}, handles=[], model=None, qs=None, data=None, busy=False)

        def move(_=None):
            if state["model"] is None:
                return
            model, data = state["model"], state["data"]
            data.qpos[:] = state["qs"][min(int(frame.value), len(state["qs"]) - 1)]
            mujoco.mj_forward(model, data)
            with client.atomic():
                for geom, handle in state["handles"]:
                    handle.position = data.geom_xpos[geom]
                    handle.wxyz = Rotation.from_matrix(
                        data.geom_xmat[geom].reshape(3, 3)
                    ).as_quat(scalar_first=True)

        def show(_=None):
            if state["busy"]:
                return
            row = state["rows"].get(selection.value)
            if row is None:
                return
            try:
                model, qs = load_case(row["batch"], row)
                for _, handle in state["handles"]:
                    handle.remove()
                state.update(model=model, qs=qs, data=mujoco.MjData(model), handles=[])
                for g in range(model.ngeom):
                    if model.geom_rgba[g, 3] == 0:
                        continue
                    typ = model.geom_type[g]
                    size = model.geom_size[g]
                    if typ == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh = model.geom_dataid[g]
                        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                        f, k = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                        vertices = model.mesh_vert[a : a + n]
                        faces = model.mesh_face[f : f + k]
                    elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                        m = trimesh.creation.box(2 * size)
                        vertices, faces = m.vertices, m.faces
                    elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                        m = trimesh.creation.icosphere(subdivisions=1, radius=size[0])
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
                frame.max = max(1, len(qs) - 1)
                frame.value = 0
                move()
                camera = camera_for(model, state["data"])
                client.camera.look_at = camera.lookat
                client.camera.position = camera.lookat + np.array(
                    [camera.distance, -camera.distance, camera.distance * 0.5]
                )
                summary = row["batch"] / row["summary"]
                details.content = "```json\n" + summary.read_text() + "\n```"
            except Exception as exc:
                details.content = f"加载失败：{exc}"

        def update(_=None):
            if state["busy"]:
                return
            rows = scan()
            types = sorted(
                {r.get("template", r["name"]) for r in rows if r["task"] == "grasp"}
            )
            state["busy"] = True
            grasp.options = ("全部", *types)
            rows = filter_rows(rows, method.value, task.value, hand.value, grasp.value)
            state["rows"] = {r["label"]: r for r in rows}
            selection.options = tuple(state["rows"]) or ("无结果",)
            state["busy"] = False
            show()

        for control in (method, task, hand, grasp):
            control.on_update(update)
        refresh.on_click(update)
        selection.on_update(show)
        frame.on_update(move)
        update()

        def animate():
            stamp = None
            while client.client_id in server.get_clients():
                now = tuple(
                    (p.stat().st_mtime_ns, str(p))
                    for p in (ROOT / "output").glob("*/manifest.json")
                )
                if now != stamp:
                    update()
                    stamp = now
                if play.value and state["qs"] is not None:
                    frame.value = (int(frame.value) + 1) % len(state["qs"])
                time.sleep(0.1)

        threading.Thread(target=animate, daemon=True).start()

    print(f"Viewer: http://localhost:{server.get_port()}", flush=True)
    while True:
        time.sleep(1)


def run(config):
    return web(config) if config.mode == "web" else replay(config)

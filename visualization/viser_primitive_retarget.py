#!/usr/bin/env python3
"""Browse paired primitive MANO/Moxian/Wuji2 meshes and RL endpoints in Viser.

Run with .venv-moxian-mano/bin/python; no simulation or GPU is required.
All panels preserve the original hand/object relative transform. RL objects
are deliberately NOT recentered, so their displacement remains visible.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.env.grasp.hand import hand_tree

PACKAGED_DATA = ROOT / 'data/datasets/dex_grasping_primitives_20261115'

PRESETS = {
    '圆盘环握 · RL 对照': ('10_Power_Disk', 27),
    '圆盘环握 · 小尺寸': ('10_Power_Disk', 4),
    '圆盘环握 · 大尺寸': ('10_Power_Disk', 64),
    '球体包握 · 小尺寸': ('11_Power_Sphere', 4),
    '球体包握 · 大尺寸': ('11_Power_Sphere', 28),
    '圆柱环握 · 中尺寸': ('3_Medium_Wrap', 16),
    '三指抓球 · 中尺寸': ('14_Tripod', 16),
    '指尖捏取 · 中尺寸': ('24_Tip_Pinch', 44),
}
RL_MODES = ('隐藏 RL', '虚拟扰动训练末帧', '真实扰动验证末帧')
EDGES = np.array([(a, b) for start in (1, 5, 9, 13, 17)
                  for a, b in zip([0, start, start + 1, start + 2],
                                  [start, start + 1, start + 2, start + 3])])
OFFSETS = np.array([[-.155, .15, 0], [.155, .15, 0],
                    [-.155, -.15, 0], [.155, -.15, 0]])


def matrix(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()


class Data:
    def __init__(self, root):
        self.root = root
        manifest = json.loads((root / 'source-manifest.json').read_text())
        self.entries = {entry['type']: entry for entry in manifest['entries']
                        if entry['kind'] != 'plate'}
        self.mano_faces = np.load(root / 'source-mano_geometry.npz')['faces']
        self.artifact = json.loads((root / 'glove/glove-model.json').read_text())
        runtime = root / 'viewer_runtime'
        if runtime.is_dir():
            name = 'primitive_viewer_runtime'
            spec = importlib.util.spec_from_file_location(name, runtime / '__init__.py',
                                                         submodule_search_locations=[str(runtime)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            surface = importlib.import_module(name + '.qnbot_surface').QnbotSurface
        else:
            from moxian_web.qnbot_surface import QnbotSurface as surface
        resources = root / 'viewer_resources'
        self.glove = surface('right', 'cpu', finger_twist=True,
                             root=resources if resources.is_dir() else None)
        self.glove.load_artifact(self.artifact)
        self.glove_faces = self.glove.faces.cpu().numpy()
        self.wuji_model = mujoco.MjModel.from_xml_string(
            ET.tostring(hand_tree('wuji2'), encoding='unicode'))
        self.wuji_data = mujoco.MjData(self.wuji_model)
        self.native_names = [self.wuji_model.joint(i).name for i in range(6, 26)]
        self.diagnostics = {}
        for name, filename in [('wuji', 'wuji2-initial-audit.csv'),
                               ('glove', 'glove-surface-audit.csv')]:
            with (root / 'physics-audit-final' / filename).open() as f:
                self.diagnostics[name] = {(r['type'], int(r['row'])): r for r in csv.DictReader(f)}
        self.rl = {}
        for mode, folder in [(RL_MODES[1], 'virtual-evaluation'),
                             (RL_MODES[2], 'physical-validation')]:
            path = root / 'rl-staged-gravity' / folder / 'learned-evaluation.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    self.rl[mode] = {key: z[key] for key in z.files}

    @lru_cache(maxsize=62)
    def grid(self, target, kind):
        with np.load(self.root / target / self.entries[kind]['file'], allow_pickle=False) as z:
            return {key: z[key] for key in z.files}

    @lru_cache(maxsize=128)
    def glove_mesh(self, kind, row):
        z = self.grid('glove', kind)
        if list(z['joint_names']) != self.glove.joint_names:
            raise ValueError('Frozen glove joint order does not match the loaded 29-DOF model')
        with torch.no_grad():
            vertices, _, _ = self.glove(torch.tensor(z['target_qpos'][row:row+1], dtype=torch.float32))
        base = z['target_base_pose'][row]
        return vertices[0].numpy() @ matrix(base).T + base[:3]

    def wuji_mesh(self, q, base):
        model, data = self.wuji_model, self.wuji_data
        data.qpos[:6] = np.r_[base[:3], Rotation.from_matrix(matrix(base)).as_euler('XYZ')]
        data.qpos[6:26] = q
        mujoco.mj_forward(model, data)
        vertices, faces, offset = [], [], 0
        for g in range(model.ngeom):
            if (model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                    or not (model.geom_contype[g] or model.geom_conaffinity[g])):
                continue
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            v = (model.mesh_vert[start:start+count] @ data.geom_xmat[g].reshape(3, 3).T
                 + data.geom_xpos[g])
            face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            vertices.append(v)
            faces.append(model.mesh_face[face_start:face_start+face_count] + offset)
            offset += count
        return np.concatenate(vertices), np.concatenate(faces)

    @lru_cache(maxsize=128)
    def initial_mesh(self, kind, row):
        z = self.grid('wuji2', kind)
        order = [list(z['joint_names']).index(name) for name in self.native_names]
        return self.wuji_mesh(z['target_qpos'][row, order], z['target_base_pose'][row])

    def object_mesh(self, kind, row, pose=None):
        z = self.grid('wuji2', kind)
        radius, height = float(z['radius_m'][row]), float(z['height_m'][row])
        if self.entries[kind]['kind'] == 'sphere':
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        else:
            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        if pose is None:
            pose = z['object_pose_world'][row]
        return mesh.vertices @ matrix(pose).T + pose[:3], mesh.faces

    def panels(self, kind, row, mode, env_index):
        z = self.grid('wuji2', kind)
        g = self.grid('glove', kind)
        obj = z['object_pose_world'][row]
        panels = [dict(title='MANO · 原始姿态', mesh=(z['mano_vertices'][row], self.mano_faces),
                       points=z['mano_joints'][row], color=(227, 176, 150), object_pose=obj),
                  dict(title='Moxian glove · 29 DOF', mesh=(self.glove_mesh(kind, row), self.glove_faces),
                       points=g['target_keypoints_world'][row], color=(206, 184, 232), object_pose=obj),
                  dict(title='Wuji2 · 运动学映射', mesh=self.initial_mesh(kind, row),
                       points=z['target_keypoints_world'][row], color=(145, 188, 225), object_pose=obj)]
        if kind == '10_Power_Disk' and row == 27 and mode in self.rl:
            rl = self.rl[mode]
            task_names = [f'right_finger{f}_joint{j}' for f in range(1, 6) for j in range(1, 5)]
            order = [list(rl['joint_names']).index(name) for name in task_names]
            q = rl['target_qpos'][env_index, order]
            panels.append(dict(title=f'Wuji2 · RL 末帧 / env {env_index}',
                mesh=self.wuji_mesh(q, rl['target_base_pose'][env_index]), points=None,
                color=(130, 215, 178), object_pose=rl['object_pose_world'][env_index],
                contacts=rl['contact_pos'][env_index][rl['contact_found'][env_index] > 0]))
        return panels


def check(data):
    for label, (kind, row) in PRESETS.items():
        panels = data.panels(kind, row, RL_MODES[2], 0)
        for panel in panels:
            v, f = panel['mesh']
            assert np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v)
            ov, of = data.object_mesh(kind, row, panel['object_pose'])
            assert np.isfinite(ov).all() and of.max() < len(ov)
        print(f'CHECK {label}: {len(panels)} valid panels', flush=True)
    # Verify the frozen model provenance used to generate this dataset.
    kind, row = PRESETS['圆盘环握 · RL 对照']
    vertices = data.glove_mesh(kind, row)
    assert data.artifact['model_hash'] == 'f0dd73b5a88b6ae0b6f2c0cc68fccfd69d5195ab39fec52db6a00fd36c463c3c'
    print(f'Model verified: {data.artifact["model_hash"]}; {len(vertices)} glove vertices')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--poses', type=Path, default=PACKAGED_DATA if PACKAGED_DATA.is_dir()
                    else ROOT / 'output/primitives_robokit_20261115')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8767)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--augmentation', type=Path, help='Browse an independent primitive augmentation campaign')
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.augmentation:
        from visualization.viser_primitive_augmentation import run
        run(args.augmentation, args.host, args.port, args.check)
        return
    data = Data(args.poses.resolve())
    if args.check:
        check(data)
        return
    server = viser.ViserServer(host=args.host, port=args.port, label='Primitive grasp · MANO / Moxian / Wuji2')
    server.gui.configure_theme(dark_mode=True, control_layout='collapsible', control_width='medium',
                               brand_color=(126, 186, 215), show_share_button=False)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/grid', width=1.2, height=1.2, cell_size=.025,
                           plane='xy', position=(0, 0, -.13))
    server.gui.add_markdown('## Primitive 抓姿对照\n实际手部网格 · 相同物体坐标系\n\n'
                            '左键旋转，右键平移，滚轮缩放。')
    preset = server.gui.add_dropdown('快捷样例', tuple(PRESETS), initial_value=next(iter(PRESETS)))
    kind_gui = server.gui.add_dropdown('抓型', tuple(data.entries), initial_value='10_Power_Disk')
    row_gui = server.gui.add_slider('尺寸行号', 0, 80, 1, 27)
    prev = server.gui.add_button('上一尺寸')
    nxt = server.gui.add_button('下一尺寸')
    with server.gui.add_folder('显示设置'):
        mesh_gui = server.gui.add_checkbox('手部网格', True)
        skeleton_gui = server.gui.add_checkbox('关节骨架', False)
        object_gui = server.gui.add_checkbox('显示物体', True)
        opacity = server.gui.add_slider('手部透明度', .15, 1., .05, .9)
        object_opacity = server.gui.add_slider('物体透明度', .15, 1., .05, .8)
        contacts_gui = server.gui.add_checkbox('RL 接触位置', True)
        reset = server.gui.add_button('重置视角')
    with server.gui.add_folder('RL 单样本对照'):
        mode_gui = server.gui.add_dropdown('RL 结果', RL_MODES, initial_value=RL_MODES[2])
        env_gui = server.gui.add_slider('并行环境编号', 0, 127, 1, 0)
        jump = server.gui.add_button('跳到下一条扰动通过样本')
    info = server.gui.add_markdown('加载中…')
    diagnostics = server.gui.add_markdown('')
    server.gui.add_markdown('Glove 使用新上传的 Moxian 模型 `f0dd73b5a88b…`。\n\n'
        '未显示尺寸信息不完整的 106 条薄板数据。几何接触、穿透与动力学成功是不同指标。')
    lock = threading.RLock()
    state = {'updating': False}
    handles = []

    def camera(client):
        client.camera.position = (0.0, -.80, .92)
        client.camera.look_at = (0.0, .01, .015)
        client.camera.up_direction = (0, 0, 1)
        client.camera.fov = .65

    @server.on_client_connect
    def _(client):
        camera(client)

    def render():
        with lock:
            kind, row = kind_gui.value, int(row_gui.value)
            z = data.grid('wuji2', kind)
            obj = z['object_pose_world'][row]
            rotation = matrix(obj)
            def local(points):
                return ((points - obj[:3]) @ rotation).astype(np.float32)
            panels = data.panels(kind, row, mode_gui.value, int(env_gui.value))
            with server.atomic():
                for handle in reversed(handles):
                    handle.remove()
                handles.clear()
                for i, panel in enumerate(panels):
                    offset = OFFSETS[i]
                    name = f'/panels/{i}'
                    handles.append(server.scene.add_frame(name, position=offset, show_axes=False))
                    v, f = panel['mesh']
                    handles.append(server.scene.add_mesh_simple(name+'/hand', local(v), f,
                        color=panel['color'], opacity=float(opacity.value), material='standard',
                        side='double', visible=bool(mesh_gui.value)))
                    ov, of = data.object_mesh(kind, row, panel['object_pose'])
                    handles.append(server.scene.add_mesh_simple(name+'/object', local(ov), of,
                        color=(246, 185, 67), opacity=float(object_opacity.value), side='double',
                        visible=bool(object_gui.value)))
                    handles.append(server.scene.add_label(name+'/label', panel['title'],
                        position=(-.105, .105, .12), font_screen_scale=.85, anchor='center-left'))
                    if panel['points'] is not None:
                        p = local(panel['points'])
                        handles.append(server.scene.add_line_segments(name+'/skeleton', p[EDGES],
                            (255, 255, 255), thickness=.0015, visible=bool(skeleton_gui.value)))
                    if 'contacts' in panel:
                        handles.append(server.scene.add_point_cloud(name+'/contacts',
                            local(panel['contacts']), np.array([255, 90, 100], np.uint8),
                            point_size=.003, visible=bool(contacts_gui.value)))
            radius, height = z['radius_m'][row], z['height_m'][row]
            sphere = data.entries[kind]['kind'] == 'sphere'
            dimensions = f'直径 **{radius*2000:.1f} mm**'
            if not sphere:
                dimensions += f' · 高/厚 **{height*1000:.1f} mm**'
            info.content = (f'### {kind}\n行号 **{row} / {len(z["target_qpos"])-1}** · {dimensions}\n\n'
                f'本查看器覆盖 **{len(data.entries)} 种抓型 / 1,141 组尺寸**；完整映射为 1,247 组。')
            w = data.diagnostics['wuji'][(kind, row)]
            g = data.diagnostics['glove'][(kind, row)]
            text = (f'**当前初态诊断**\n\n'
                f'- Wuji2 物体最大穿透：**{float(w["object_penetration_mm"]):.2f} mm**\n'
                f'- Glove 顶点最大穿透：**{float(g["max_vertex_penetration_mm"]):.2f} mm**\n'
                f'- Wuji2 接触手指数：**{w["contact_finger_count"]}**\n\n')
            enabled = kind == '10_Power_Disk' and row == 27
            env_gui.disabled = not enabled
            jump.disabled = not enabled
            if enabled and mode_gui.value in data.rl:
                rl, e = data.rl[mode_gui.value], int(env_gui.value)
                passed = bool(rl['hold_pass'][e])
                text += f'**RL / env {e}**\n\n'
                if mode_gui.value == RL_MODES[1]:
                    text += '虚拟多方向扳手检查：' + ('通过' if passed else '**未通过**') + '。\n\n'
                    text += f'两指接触保持比例：{100*float(rl["contact_retention_fraction"][e]):.1f}%。\n\n'
                else:
                    text += '该方向真实扰动：' + ('**通过**' if passed else '**未通过**') + '。\n\n'
                    axis = e % 12
                    text += f'扰动：{"力" if axis < 6 else "力矩"} · {"+" if axis % 2 == 0 else "−"}{"XYZ"[(axis//2)%3]}。\n\n'
                text += (f'最大位移：{1000*float(rl["max_displacement_m"][e]):.2f} mm。\n\n'
                         '这是同一个圆盘初态的并行试验，**不代表全方向力闭合已验证**。')
            elif not enabled:
                text += 'RL 仅训练过 **10_Power_Disk / 行 27**。选“圆盘环握 · RL 对照”查看第四个面板。'
            diagnostics.content = text

    def refresh(_event=None):
        if not state['updating']:
            render()

    @preset.on_update
    def _(_event):
        with lock:
            state['updating'] = True
            kind, row = PRESETS[preset.value]
            kind_gui.value = kind
            row_gui.max = data.entries[kind]['samples'] - 1
            row_gui.value = row
            state['updating'] = False
            render()

    @kind_gui.on_update
    def _(_event):
        if state['updating']:
            return
        with lock:
            state['updating'] = True
            count = data.entries[kind_gui.value]['samples']
            row_gui.max = count - 1
            row_gui.value = count // 2
            state['updating'] = False
            render()

    for widget in (row_gui, mesh_gui, skeleton_gui, object_gui, opacity, object_opacity,
                   contacts_gui, mode_gui, env_gui):
        widget.on_update(refresh)

    @prev.on_click
    def _(_event):
        row_gui.value = max(0, int(row_gui.value)-1)

    @nxt.on_click
    def _(_event):
        row_gui.value = min(data.entries[kind_gui.value]['samples']-1, int(row_gui.value)+1)

    @jump.on_click
    def _(_event):
        mode_gui.value = RL_MODES[2]
        ids = np.flatnonzero(data.rl[RL_MODES[2]]['hold_pass'])
        later = ids[ids > int(env_gui.value)]
        if len(ids):
            env_gui.value = int(later[0] if len(later) else ids[0])

    @reset.on_click
    def _(_event):
        for client in server.get_clients().values():
            camera(client)

    render()
    print(f'VISER_READY http://127.0.0.1:{server.get_port()}', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()

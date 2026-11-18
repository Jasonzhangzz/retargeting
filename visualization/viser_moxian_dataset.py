#!/usr/bin/env python3
"""Browse paired Moxian/Wuji2 grasps and object meshes on CPU with Viser."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading
import time
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moxian_web.qnbot_surface import read_skin
from retarget.env.grasp.hand import hand_tree, asset_root

DEFAULT_DATA = ROOT / 'data/datasets/moxian_grasp'
PALM_NODES = [0, 6, 11, 16, 21]  # wrist, index/middle/ring/little MCP
MANO_PALM = [0, 5, 9, 13, 17]


def register_wrist(local_palm, world_palm):
    """Rigid palm registration with the anatomical wrist fixed exactly.

    The export copied MANO's model translation/rotation into moxian_qpos[:6].
    MANO's model origin is not its wrist, and its axes differ from QnBot's.
    MCP origins are independent of the finger flexion, so they define the
    wrist frame without moving an object or fitting a grasp to it.
    """
    a = np.asarray(local_palm, np.float64)
    b = np.asarray(world_palm, np.float64)
    da, db = a[..., 1:, :] - a[..., :1, :], b[..., 1:, :] - b[..., :1, :]
    u, _, vt = np.linalg.svd(np.swapaxes(da, -1, -2) @ db)
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
    correction[..., 2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt  # row-vector rotation
    translation = b[..., 0, :] - np.einsum('...i,...ij->...j', a[..., 0, :], rotation)
    return rotation, translation


def panel_layout(moxian, wuji, obj):
    """Center the two object bounding boxes symmetrically using rigid offsets."""
    lo = np.minimum.reduce([moxian.min(0), wuji.min(0), obj.min(0)])
    hi = np.maximum.reduce([moxian.max(0), wuji.max(0), obj.max(0)])
    object_center = (obj.min(0) + obj.max(0)) / 2
    gap = max(.32, float(hi[0] - lo[0]) + .12)
    offsets = np.array([[-gap/2, 0., 0.], [gap/2, 0., 0.]]) - object_center
    bounds = np.concatenate([np.array([lo, hi]) + offset for offset in offsets])
    radius = max(.12, float(np.max(np.linalg.norm(bounds, axis=-1))))
    return offsets, radius


def center_camera(camera, radius, aspect=1.):
    """Set position before look_at: Viser position also translates look_at."""
    fov = np.deg2rad(50.)
    half_angle = min(fov/2, np.arctan(np.tan(fov/2) * max(.25, aspect)))
    distance = radius / np.sin(half_angle) * 1.12
    if hasattr(camera, 'up_direction'):
        camera.up_direction = (0., 0., 1.)
    else:
        camera.up = (0., 0., 1.)  # Viser's initial camera configuration
    camera.fov = fov
    camera.position = np.array([0., -1., .65]) / np.linalg.norm([0., -1., .65]) * distance
    camera.look_at = np.zeros(3)


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('schema') != 'moxian_grasp_v1' or not self.manifest.get('complete'):
            raise ValueError('需要完整的 moxian_grasp_v1 数据集')
        self.types = sorted(self.manifest['types'], key=lambda name: int(name.split('_')[0]))
        self.nodes = self.manifest['node_order']
        self.skin = read_skin(self.root / 'hand_assets/moxian/resources/meshes/right/visual/right_hand_hd.skn')
        self.bone_indices = [self.nodes.index(name.removeprefix('right_')) for name in self.skin['bone_names']]
        self.local_skin = [(self.skin['vertices'] - position) @ rotation
                           for position, rotation in zip(self.skin['bind_position'], self.skin['bind_rotation'])]
        node_index = {name: i for i, name in enumerate(self.nodes)}
        edges = []
        for finger in ('thumb', 'index', 'middle', 'ring', 'little'):
            names = ['wrist'] + [name for name in self.nodes if name.startswith(finger + '_')]
            edges.extend((node_index[a], node_index[b]) for a, b in zip(names[:-1], names[1:]))
        self.edges = np.asarray(edges)
        self.assets = self.root / 'viewer_assets'
        self.objects = json.loads((self.assets / 'objects.json').read_text())
        self.pairs = self.root / 'wuji2_pairs'
        self.pair_manifest = json.loads((self.pairs / 'manifest.json').read_text())
        if self.pair_manifest['source_moxian_manifest_sha256'] != sha256(self.root / 'manifest.json'):
            raise ValueError('Wuji2 配对数据与 Moxian 数据集不匹配')
        self.wuji_model = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'), encoding='unicode'))
        self.wuji_data = mujoco.MjData(self.wuji_model)
        native_names = [self.wuji_model.joint(i).name for i in range(self.wuji_model.njnt)]
        if native_names != self.pair_manifest['joint_names']:
            raise ValueError('Wuji2 原始关节顺序与当前模型不一致')
        for name, expected in self.pair_manifest['model_files'].items():
            if sha256(asset_root('wuji2') / name) != expected:
                raise ValueError('Wuji2 模型与源数据不一致：' + name)
        self.wuji_geoms = [g for g in range(self.wuji_model.ngeom)
                           if self.wuji_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                           and self.wuji_model.geom_group[g] == 1]
        self.wuji_bodies = [self.wuji_model.body(name).id for name in
                           ['right_palm_link'] + [f'right_finger{f}_link{j}'
                                                 for f in range(1, 6) for j in range(1, 5)]]
        self.wuji_edges = np.array([(a, b) for start in (1, 5, 9, 13, 17)
                                  for a, b in zip([0, start, start+1, start+2],
                                                  [start, start+1, start+2, start+3])])
        self.contact_root = self.root / 'contact_retarget_v1'
        self._contact_config = None
        self.moxian_model = mujoco.MjModel.from_xml_path(str(self.root / 'hand_assets/moxian/resources/mujoco/right.xml'))
        self.moxian_data = mujoco.MjData(self.moxian_model)
        self.moxian_nodes = [self.moxian_model.body('right_'+name).id for name in self.nodes]

    def contact_path(self, grasp_type, row):
        config = self.contact_root / 'config.json'
        if self._contact_config is None and config.is_file():
            self._contact_config = json.loads(config.read_text())
            if self._contact_config['source_manifest_sha256'] != sha256(self.root/'manifest.json'):
                raise ValueError('接触重算结果与源数据不一致')
        if self._contact_config is None:
            return None
        size = self._contact_config['batch_size']
        return self.contact_root / grasp_type / f'{row//size*size:07d}.npz'

    @lru_cache(maxsize=2)
    def contact_shard(self, path, stamp):
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}

    def retargeted_pose(self, grasp_type, row):
        path = self.contact_path(grasp_type, row)
        if path is None or not path.is_file():
            return None
        z = self.contact_shard(path, path.stat().st_mtime_ns)
        i = int(np.searchsorted(z['sample_indices'], row))
        if i >= len(z['sample_indices']) or int(z['sample_indices'][i]) != row:
            raise ValueError('接触重算样本编号不匹配')
        grid = self.grid(grasp_type)
        for key in ('source_case_name', 'wuji_pose_id', 'object_scene', 'object_pose_world', 'object_scale'):
            np.testing.assert_array_equal(z[key][i], grid[key][row], err_msg='接触结果配对不匹配：'+key)
        record = {key: value[i] for key, value in z.items()}
        self.moxian_data.qpos[:] = record['target_qpos']
        mujoco.mj_kinematics(self.moxian_model, self.moxian_data)
        p = self.moxian_data.xpos[self.moxian_nodes]
        r = self.moxian_data.xmat[self.moxian_nodes].reshape(-1,3,3)
        vertices = np.zeros_like(self.skin['vertices'])
        for bone,node in enumerate(self.bone_indices):
            vertices += (self.local_skin[bone]@r[node].T+p[node])*self.skin['weights'][:,bone,None]
        rotation,translation = record['target_base_rotation'],record['target_base_position']
        return grid,(vertices@rotation.T+translation).astype(np.float32),(p@rotation.T+translation).astype(np.float32),record

    def accepted_rows(self, grasp_type):
        rows=[]
        for path in sorted((self.contact_root/grasp_type).glob('*.npz')):
            with np.load(path,allow_pickle=False) as z:
                rows.extend(z['sample_indices'][z['contact_correspondence_valid']])
        return np.asarray(rows,dtype=int)

    def visual_penetration(self, grid, row, meshes):
        """Check rendered surfaces, including edge midpoints; not self-collision."""
        import torch
        from retarget.env.grasp.moxian_contact_retarget import Objects
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        objects = Objects(self, device)
        objects.batch(grid, np.array([row]))
        result = {}
        for name, vertices, faces in meshes:
            tri = vertices[faces]
            points = np.concatenate((vertices, tri.mean(1),
                ((tri + np.roll(tri, 1, axis=1)) / 2).reshape(-1, 3)))
            depth, valid = objects.penetration(torch.as_tensor(points[None], device=device))
            if not bool(valid.item()):
                raise ValueError(name + '：穿透查询失败')
            result[name] = float(depth.item()) * 1000
        return result

    @lru_cache(maxsize=2)
    def grid(self, grasp_type):
        keys = ('moxian_qpos', 'moxian_skeleton_positions', 'moxian_skeleton_quaternions_xyzw',
                'object_pose_world', 'object_scale', 'scene_scale', 'source_case_name', 'object_scene',
                'mano_joints', 'source_rows', 'valid', 'rollout_valid', 'rollout_stable')
        with np.load(self.root / 'dexonomy' / grasp_type / 'poses.npz', allow_pickle=False) as data:
            grid = {key: data[key] for key in keys}
        with np.load(self.pairs / (grasp_type + '.npz'), allow_pickle=False) as paired:
            for key in ('source_rows', 'source_case_name', 'object_scene', 'object_pose_world', 'object_scale'):
                np.testing.assert_array_equal(grid[key], paired[key], err_msg=f'{grasp_type}: {key}')
            np.testing.assert_array_equal(paired['joint_names'], self.pair_manifest['joint_names'])
            grid['wuji_qpos'] = paired['wuji_qpos']
            grid['wuji_pose_id'] = paired['wuji_pose_id']
        grid['wrist_rotation'], grid['wrist_translation'] = register_wrist(
            grid['moxian_skeleton_positions'][:, PALM_NODES], grid['mano_joints'][:, MANO_PALM])
        return grid

    def wuji_pose(self, data, row):
        model, state = self.wuji_model, self.wuji_data
        state.qpos[:] = data['wuji_qpos'][row]
        mujoco.mj_kinematics(model, state)
        vertices, faces, offset = [], [], 0
        for geom in self.wuji_geoms:
            mesh = model.geom_dataid[geom]
            start, size = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            v = model.mesh_vert[start:start+size] @ state.geom_xmat[geom].reshape(3, 3).T + state.geom_xpos[geom]
            start, size = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            vertices.append(v)
            faces.append(model.mesh_face[start:start+size] + offset)
            offset += len(v)
        return np.concatenate(vertices).astype(np.float32), np.concatenate(faces), state.xpos[self.wuji_bodies].copy()

    def pose(self, grasp_type, row, registered=True):
        data = self.grid(grasp_type)
        qpos = data['moxian_qpos'][row]
        local_positions = data['moxian_skeleton_positions'][row]
        local_rotations = Rotation.from_quat(data['moxian_skeleton_quaternions_xyzw'][row]).as_matrix()
        vertices = np.zeros_like(self.skin['vertices'])
        for bone, node in enumerate(self.bone_indices):
            posed = self.local_skin[bone] @ local_rotations[node].T + local_positions[node]
            vertices += posed * self.skin['weights'][:, bone, None]
        if registered:
            rotation, translation = data['wrist_rotation'][row], data['wrist_translation'][row]
        else:
            rotation, translation = Rotation.from_rotvec(qpos[3:6]).as_matrix().T, qpos[:3]
        positions = local_positions @ rotation + translation
        vertices = vertices @ rotation + translation
        return data, vertices.astype(np.float32), positions.astype(np.float32)

    @lru_cache(maxsize=32)
    def object_mesh(self, relative):
        path = (self.assets / 'meshes' / relative).resolve()
        if not path.is_relative_to((self.assets / 'meshes').resolve()):
            raise ValueError('物体路径超出数据目录')
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh = trimesh.load(path, force='mesh', process=False, skip_materials=True)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('物体网格为空：' + relative)
        return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)

    def object(self, data, row):
        name = text(data['object_scene'][row])
        record = self.objects['scenes'][name]
        vertices, faces = self.object_mesh(record['mesh'])
        pose = data['object_pose_world'][row]
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        vertices = (vertices * data['object_scale'][row]) @ rotation.T + pose[:3]
        return vertices.astype(np.float32), faces, record

    def check(self):
        count = 0
        for grasp_type in self.types:
            info = self.manifest['types'][grasp_type]
            for name, expected in info['files'].items():
                assert sha256(self.root / 'dexonomy' / grasp_type / name) == expected, (grasp_type, name)
            data = self.grid(grasp_type)
            n = len(data['moxian_qpos'])
            assert n == info['samples'] == 4000
            assert data['moxian_qpos'].shape == (n, 31)
            assert data['wuji_qpos'].shape == (n, 26) and np.isfinite(data['wuji_qpos']).all()
            entry = self.pair_manifest['files'][grasp_type + '.npz']
            assert entry['samples'] == n and sha256(self.pairs / (grasp_type + '.npz')) == entry['sha256']
            assert data['moxian_skeleton_positions'].shape == (n, 25, 3)
            for name in ('moxian_qpos', 'moxian_skeleton_positions', 'object_scale', 'object_pose_world'):
                assert np.isfinite(data[name]).all(), (grasp_type, name)
            np.testing.assert_allclose(np.linalg.norm(data['moxian_skeleton_quaternions_xyzw'], axis=-1), 1., atol=2e-4)
            assert len(np.unique(data['source_case_name'])) == n
            for name in np.unique(data['object_scene']):
                assert text(name) in self.objects['scenes'], (grasp_type, name)
            for row in (0, n // 2, n - 1):
                _, vertices, joints = self.pose(grasp_type, row)
                np.testing.assert_allclose(joints[0], data['mano_joints'][row, 0], atol=2e-7)
                obj, faces, record = self.object(data, row)
                wv, wf, wj = self.wuji_pose(data, row)
                assert np.isfinite(wv).all() and np.isfinite(wj).all()
                assert wf.min() >= 0 and wf.max() < len(wv)
                assert np.isfinite(vertices).all() and np.isfinite(obj).all()
                assert faces.min() >= 0 and faces.max() < len(obj)
                offsets, _ = panel_layout(vertices, wv, obj)
                midpoint = (obj.min(0) + obj.max(0)) / 2 + offsets.mean(0)
                np.testing.assert_allclose(midpoint, 0., atol=1e-7)
                np.testing.assert_allclose(data['object_scale'][row],
                    np.asarray(record['scene_scale']) * data['scene_scale'][row], rtol=1e-5, atol=1e-7)
            count += n
            print(f'CHECK {grasp_type}: {n:,} exact pairs, wrist registration and object-centered layout OK', flush=True)
        for name, expected in self.manifest['mesh_files'].items():
            assert sha256(self.root / 'hand_assets/moxian' / name) == expected, name
        for name, entry in self.objects['files'].items():
            assert sha256(self.assets / 'meshes' / name) == entry['sha256'], name
        print(f'Verified {len(self.types)} types, {count:,} poses, {len(self.objects["files"])} object meshes.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--grasp-type', default='24_Tip_Pinch')
    parser.add_argument('--sample', type=int, default=0)
    args = parser.parse_args()
    data = Dataset(args.data)
    if args.check:
        return data.check()
    server = viser.ViserServer(host=args.host, port=args.port, label='Moxian / Wuji2 · Paired Grasps')
    server.scene.set_up_direction('+z')
    server.gui.configure_theme(titlebar_content=None, control_layout='floating')
    total = sum(info['samples'] for info in data.manifest['types'].values())
    server.gui.add_markdown(f'## Moxian / Wuji2 抓握对照\n**{len(data.types)} 类 · {total:,} 对样本**\n\n左：Moxian 原生25自由度＋物体。右：配对的 Wuji2＋物体。切换类别或编号同步更新两侧。')
    grasp_type = server.gui.add_dropdown('Grasp type', tuple(data.types), initial_value=args.grasp_type)
    sample = server.gui.add_slider('样本编号（从0开始）', min=0, max=3999, step=1, initial_value=args.sample)
    pose_mode = server.gui.add_dropdown('Moxian 姿态', ('接触对应重算', '原始关节映射'))
    with server.gui.add_folder('样本切换'):
        accepted_navigation = server.gui.add_checkbox('翻页／随机跳过未通过接触阈值的样本', True)
        previous = server.gui.add_button('上一个')
        following = server.gui.add_button('下一个')
        random_sample = server.gui.add_button('随机样本')
        next_accepted = server.gui.add_button('下一个通过接触阈值的样本')
        server.gui.add_markdown('编号滑块仍可直接检查任意样本，包括失败样本。通过阈值仍允许最多 **2 mm 采样穿透**。')
    with server.gui.add_folder('显示设置'):
        show_hand = server.gui.add_checkbox('Moxian 蒙皮网格', True)
        show_wuji = server.gui.add_checkbox('Wuji2 网格', True)
        show_object = server.gui.add_checkbox('两侧物体网格', True)
        show_skeleton = server.gui.add_checkbox('两侧手部骨架', False)
        show_mano = server.gui.add_checkbox('原始 MANO 骨架', False)
        show_contacts = server.gui.add_checkbox('接触锚点与误差连线', True)
        registered = server.gui.add_checkbox('校正 MANO→Moxian 腕部坐标', True)
        hand_opacity = server.gui.add_slider('手部透明度', min=.1, max=1., step=.05, initial_value=1.)
        object_opacity = server.gui.add_slider('物体透明度', min=.1, max=1., step=.05, initial_value=.8)
        reset_camera = server.gui.add_button('物体居中／重置视角')
        refresh = server.gui.add_button('刷新当前样本')
        diagnose = server.gui.add_button('检查当前两侧手–物体穿透（加密采样）')
    info = server.gui.add_markdown('正在加载…')
    collision_info = server.gui.add_markdown('')
    progress = server.gui.add_markdown('')
    server.gui.add_markdown('紫色：Moxian；蓝色：Wuji2 源姿态；橙色：物体。绿色点：物体接触锚点；红／青色点：对应手部接触点。\n\n**接触阈值通过不代表无穿模。** 左侧重算允许最多 2 mm 采样穿透，未检查自碰撞；右侧保留源 Wuji2 姿态，其视觉网格也可能穿透物体。源 MANO 的 valid／rollout 不作为 Moxian 验收结论。')
    lock = threading.RLock()
    radius = .3
    last_selection = None
    last_contact_stamp = None
    rng = np.random.default_rng()

    def camera(client):
        center_camera(client.camera, radius, client.camera.aspect)

    @server.on_client_connect
    def connect(client):
        camera(client)

    def update(_event=None):
        nonlocal radius, last_selection, last_contact_stamp
        with lock:
            kind, row = grasp_type.value, int(sample.value)
            solved = data.retargeted_pose(kind, row) if pose_mode.value == '接触对应重算' else None
            record = None
            if solved is None:
                grid, vertices, joints = data.pose(kind, row, registered=registered.value)
            else:
                grid, vertices, joints, record = solved
            path = data.contact_path(kind,row)
            last_contact_stamp = path.stat().st_mtime_ns if path is not None and path.is_file() else None
            registered.disabled = pose_mode.value == '接触对应重算'
            wv, wf, wj = data.wuji_pose(grid, row)
            obj, faces, _ = data.object(grid, row)
            # One common object-centered translation per hand/object pair.
            offsets, radius = panel_layout(vertices, wv, obj)
            hi = np.maximum.reduce([vertices.max(0), wv.max(0), obj.max(0)])
            object_center = (obj.min(0) + obj.max(0)) / 2
            mano = grid['mano_joints'][row]
            chains = [[0, *range(start, start + 4)] for start in (1, 5, 9, 13, 17)]
            mano_edges = np.array([[mano[a], mano[b]] for chain in chains for a, b in zip(chain[:-1], chain[1:])])
            with server.atomic():
                for name, title, v, f, points, edges, color, visible, offset in (
                    ('moxian', 'Moxian · 25 DOF', vertices, data.skin['faces'], joints, data.edges,
                     (190, 169, 232), show_hand.value, offsets[0]),
                    ('wuji2', 'Wuji2 · source pose (unchanged)', wv, wf, wj, data.wuji_edges,
                     (135, 185, 225), show_wuji.value, offsets[1])):
                    path = '/panels/' + name
                    server.scene.add_frame(path, position=offset, show_axes=False)
                    server.scene.add_mesh_simple(path+'/hand', v, f, color=color,
                        opacity=hand_opacity.value, side='double', visible=visible)
                    server.scene.add_mesh_simple(path+'/object', obj, faces, color=(245, 167, 74),
                        opacity=object_opacity.value, side='double', visible=show_object.value)
                    server.scene.add_label(path+'/label', title,
                        position=(object_center[0], object_center[1], hi[2]+.035))
                    lines = points[edges]
                    server.scene.add_line_segments(path+'/skeleton/edges', lines,
                        np.broadcast_to(np.array([80, 220, 235], np.uint8), lines.shape),
                        thickness=.0015, thickness_units='world', visible=show_skeleton.value)
                    server.scene.add_point_cloud(path+'/skeleton/joints', points,
                        np.array([190, 250, 255], np.uint8), point_size=.003,
                        point_shape='circle', visible=show_skeleton.value)
                    server.scene.add_line_segments(path+'/mano', mano_edges,
                        np.broadcast_to(np.array([245, 150, 95], np.uint8), mano_edges.shape),
                        thickness=.001, thickness_units='world', visible=show_mano.value)
                    if record is not None and record['contact_active'].any():
                        active = record['contact_active']
                        anchor = record['object_contact_points'][active]
                        contact = record['glove_contact_points' if name=='moxian' else 'wuji_contact_points'][active]
                        server.scene.add_point_cloud(path+'/contacts/object', anchor, np.array([60,245,100],np.uint8),
                            point_size=.0045, visible=show_contacts.value)
                        server.scene.add_point_cloud(path+'/contacts/hand', contact,
                            np.array([255,70,80] if name=='moxian' else [60,220,255],np.uint8),
                            point_size=.0035, visible=show_contacts.value)
                        lines = np.stack((anchor,contact),1)
                        server.scene.add_line_segments(path+'/contacts/error',lines,
                            np.broadcast_to(np.array([255,240,100],np.uint8),lines.shape),
                            thickness=.001,thickness_units='world',visible=show_contacts.value)
                    else:
                        for leaf in ('object','hand','error'):
                            server.scene.remove_by_name(path+'/contacts/'+leaf)
                case = text(grid['source_case_name'][row])
                scene = text(grid['object_scene'][row])
                info.content = (f'**{kind} · {row + 1:,} / {len(grid["moxian_qpos"]):,}**\n\n'
                    f'配对 case：`{case}`\n\n物体：`{scene.split("/scene_cfg/")[-1]}`\n\n'
                    f'物体缩放：`{grid["object_scale"][row].round(5).tolist()}`\n\n'
                    f'源 valid：**{bool(grid["valid"][row])}** · rollout valid：**{bool(grid["rollout_valid"][row])}** · stable：**{bool(grid["rollout_stable"][row])}**')
                if record is not None:
                    active=record['contact_active'];count=int(active.sum())
                    error=float(record['anchor_error_m'][active].max())*1000 if count else 0.
                    accepted=bool(record['contact_correspondence_valid'])
                    info.content = ('**接触阈值通过（非无穿模认证）**\n\n' if accepted else
                        '**⚠ 当前为重算失败样本，不能作为有效抓握使用。**\n\n') + info.content
                    info.content += (f'\n\n**接触重算：{"通过接触阈值" if accepted else "未通过接触阈值"}**'
                        f'\n\n接触区域：{count} · 最大锚点误差：**{error:.2f} mm**'
                        f'\n\n全网格顶点／面中心的最大采样穿透：**{float(record["sampled_penetration_m"])*1000:.2f} mm**')
                    if not accepted:
                        reasons=[]
                        if not count:reasons.append('未检测到源接触')
                        if error>5:reasons.append('接触位置误差')
                        if record['sampled_penetration_m']>.002:reasons.append('穿透超限')
                        if count and record['normal_dot'][active].min()<.5:reasons.append('法向不匹配')
                        if count and record['material_side_dot'][active].min()<.5:reasons.append('接触侧面不匹配')
                        if record['unexpected_contact']:reasons.append('额外接触区域')
                        info.content += '\n\n未通过原因：'+'、'.join(reasons)
                elif pose_mode.value == '接触对应重算':
                    info.content += '\n\n**当前样本待重算；暂显示原始映射。**'
                selection = (kind, row, registered.value, pose_mode.value, record is not None)
                if last_selection != selection:
                    collision_info.content = ''
                    center_camera(server.initial_camera, radius)
                    for client in server.get_clients().values():
                        camera(client)
                    last_selection = selection

    @grasp_type.on_update
    def change_type(event):
        rows = data.accepted_rows(grasp_type.value) if accepted_navigation.value else np.arange(4000)
        sample.value = int(rows[0]) if len(rows) else 0
        update(event)

    for control in (sample, pose_mode, show_hand, show_wuji, show_object, show_skeleton, show_mano, show_contacts, registered, hand_opacity, object_opacity):
        control.on_update(update)
    def navigate(direction):
        with lock:
            rows = data.accepted_rows(grasp_type.value) if accepted_navigation.value else np.arange(4000)
            if not len(rows):
                info.content += '\n\n当前类别没有通过接触阈值的样本；可关闭跳过选项检查失败样本。'
                return
            if direction == 0:
                row = rng.choice(rows)
            elif direction > 0:
                row = rows[np.searchsorted(rows, int(sample.value), side='right') % len(rows)]
            else:
                row = rows[(np.searchsorted(rows, int(sample.value)) - 1) % len(rows)]
            sample.value = int(row)
    previous.on_click(lambda _: navigate(-1))
    following.on_click(lambda _: navigate(1))
    random_sample.on_click(lambda _: navigate(0))
    refresh.on_click(update)

    @diagnose.on_click
    def diagnose_collision(_event):
        with lock:
            diagnose.disabled = True
            collision_info.content = '正在检查两侧实际显示网格…'
            try:
                kind, row = grasp_type.value, int(sample.value)
                solved = data.retargeted_pose(kind, row) if pose_mode.value == '接触对应重算' else None
                if solved is None:
                    grid, vertices, _ = data.pose(kind, row, registered=registered.value)
                else:
                    grid, vertices, _, _ = solved
                wv, wf, _ = data.wuji_pose(grid, row)
                depths = data.visual_penetration(grid, row, [
                    ('Moxian', vertices, data.skin['faces']), ('Wuji2', wv, wf)])
                collision_info.content = (f'**{kind} · 原始编号 {row}：视觉网格加密采样穿透**\n\n'
                    f'左 Moxian：**{depths["Moxian"]:.3f} mm**\n\n'
                    f'右 Wuji2 源姿态：**{depths["Wuji2"]:.3f} mm**\n\n'
                    '检查顶点、面中心、边中点；不含自碰撞，也不保证连续三角面无交叠。')
            except Exception as error:
                collision_info.content = '穿透诊断失败：' + str(error)
            finally:
                diagnose.disabled = False

    @next_accepted.on_click
    def advance_accepted(_event):
        rows=data.accepted_rows(grasp_type.value)
        if not len(rows):
            info.content += '\n\n当前类别尚无通过验收的已完成样本。'
            return
        pose_mode.value='接触对应重算'
        sample.value=int(rows[np.searchsorted(rows,int(sample.value)+1)%len(rows)])

    @reset_camera.on_click
    def reset(_event):
        for client in server.get_clients().values():
            camera(client)

    update()
    print(f'Viser URL: http://127.0.0.1:{args.port}', flush=True)
    while True:
        time.sleep(2.)
        status_path = data.contact_root/'status.json'
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            progress.content = f'接触重算进度：**{status["computed"]:,} / {status["total"]:,}** · {status["state"]}'
        with lock:
            path = data.contact_path(grasp_type.value,int(sample.value))
            stamp = path.stat().st_mtime_ns if path is not None and path.is_file() else None
            if stamp != last_contact_stamp:
                update()


if __name__ == '__main__':
    main()

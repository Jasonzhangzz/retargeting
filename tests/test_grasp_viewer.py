"""YAML selection and IK/output switching without GPU or report dependencies."""
from contextlib import nullcontext
from concurrent.futures import Future
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from scripts.view_grasp_results import Results, available_stages, attach_client, load_geometry
from retarget.grasp_results import register_results
import retarget.grasp_results as result_store


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(result_store, 'RESULTS_ROOT', tmp_path/'store')
    sources = []
    for method, hand, category in [('cem', 'wuji1', 2), ('cem', 'wuji2', 2), ('adamw', 'wuji2', 13)]:
        folder = tmp_path/f'{method}_{hand}'
        folder.mkdir()
        case = folder/'case'
        case.mkdir()
        xml = '''<mujoco><asset><mesh name="tetra"
            vertex="0 0 0 .01 0 0 0 .01 0 0 0 .01"/></asset><worldbody>
            <body name="right_palm_link"><joint type="slide" axis="1 0 0"/>
            <geom type="mesh" mesh="tetra" group="1"/></body>
            <body name="grasp_object" pos=".05 0 0"><freejoint name="obj_joint"/>
            <geom type="mesh" mesh="tetra" group="1"/></body></worldbody></mujoco>'''
        (case/'scene.xml').write_text(xml)
        model = mujoco.MjModel.from_xml_string(xml)
        initial = model.qpos0.copy()
        optimized = initial.copy(); optimized[0] += .02
        np.savez(case/'grasp.npz', initial_qpos=initial, qpos=optimized)
        row = dict(case=str(case), template=f'{category}_Template', hand_model=hand,
            primitive=dict(kind='sphere', radial_scale=1., radius_m=.02, height_m=.04),
            initialization=dict(initial_penetration_mm=.5),
            force_closure=dict(force_closure=False), feasible=False, required_groups_satisfied=False,
            active_contacts=0, max_object_penetration_mm=2., max_self_penetration_mm=0.)
        (folder/'summary.json').write_text(json.dumps([row]))
        register_results(folder, method, hand)
        sources.append(dict(method=method, hand=hand, path=f'{folder.name}/summary.json'))
    return tmp_path, sources


def config(sources, **settings):
    path, _ = sources
    config = path/'viewer.yaml'
    config.write_text(yaml.safe_dump(settings))
    return config


def test_yaml_filters_automatically_discovered_categories_methods_and_hands(sources, monkeypatch):
    file = config(sources, categories=[2], methods=['cem'], hands=['wuji2'], stage='ik')
    monkeypatch.chdir('/tmp')
    results = Results(config=file)
    assert results.initial_stage == 'ik'
    assert results.hands('cem') == ['wuji2']
    assert results.types('cem', 'wuji2') == ['2_Template']
    assert len(results.index) == 1


def test_same_method_can_contain_both_hands_and_invalid_selection_is_explicit(sources):
    results = Results(config=config(sources, methods=['cem']))
    assert results.hands('cem') == ['wuji1', 'wuji2']
    assert len(results.index) == 2
    with pytest.raises(ValueError, match='No saved results match'):
        Results(config=config(sources, categories=[99]))
    with pytest.raises(ValueError, match='Invalid methods'):
        Results(config=config(sources, methods=['missing']))
    source = sources[0]/sources[1][0]['path']
    rows = json.loads(source.read_text())
    rows[0]['hand_model'] = 'wuji2'
    source.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='Hand mismatch'):
        Results(config=config(sources))


def test_new_run_is_automatically_selected_when_nonempty_without_mixing_batches(sources):
    path, entries = sources
    file = config(sources, methods=['cem'], hands=['wuji2'])
    archived = path/entries[1]['path']
    original = archived.read_bytes()
    assert len(Results(config=file).index) == 1
    new_run = path/'new_run'
    register_results(new_run, 'cem', 'wuji2')
    # An empty background run must not hide the previously saved result.
    assert Results(config=file).rows['cem'][0]['template'] == '2_Template'
    new_run.mkdir()
    rows = json.loads(original)
    rows[0]['template'] = '17_New'
    (new_run/'summary.json').write_text(json.dumps(rows))
    selected = Results(config=file)
    assert selected.types('cem', 'wuji2') == ['17_New']
    assert selected.rows['cem'][0]['result_batch'] == 'new_run'
    assert archived.read_bytes() == original
    # Do not fill a missing category with older poses from another batch.
    with pytest.raises(ValueError, match='No saved results match'):
        Results(config=config(sources, methods=['cem'], hands=['wuji2'], categories=[2]))


def test_directly_named_batch_directory_is_discovered_and_can_be_registered(sources):
    path, entries = sources
    folder = result_store.RESULTS_ROOT/'cem'/'wuji2'/'direct_batch'
    folder.mkdir()
    (folder/'summary.json').write_bytes((path/entries[1]['path']).read_bytes())
    register_results(folder, 'cem', 'wuji2')
    assert not folder.is_symlink()
    selected = Results(config=config(sources, methods=['cem'], hands=['wuji2']))
    assert selected.rows['cem'][0]['result_batch'] == 'direct_batch'
    assert (folder.parent/'latest').resolve() == folder


@pytest.mark.parametrize('optimizer,explicit,method', [
    ('cem', True, 'cem_explicit'), ('cem', False, 'cem'), ('adamw', True, 'adamw')])
def test_solver_entry_automatically_registers_results(sources, monkeypatch, optimizer, explicit, method):
    from run_grasp import main
    import retarget.grasp as grasp
    path, _ = sources
    annotations = path/'dexonomy/assets/hand/shadow/raw_anno'
    annotations.mkdir(parents=True)
    (annotations/'2_Template.yaml').write_text('{}')
    monkeypatch.setattr(grasp, 'run', lambda args: args.output/args.name)
    output = main(['--archive-results', '--primitives', '--templates', '2', '--dexonomy-root', str(path/'dexonomy'),
        '--output', str(path/'output'), '--name', 'automatic', '--hand-model', 'wuji2',
        '--grasp-optimizer', optimizer, '--explicit-pose' if explicit else '--no-explicit-pose',
        '--no-viewer', '--no-render'])
    assert (result_store.RESULTS_ROOT/method/'wuji2/latest').resolve() == output
    assert (result_store.RESULTS_ROOT/method/'wuji2/automatic').is_symlink()


def test_ik_and_output_render_saved_fk_in_same_comparison_frame(sources):
    results = Results(config=config(sources, methods=['cem'], hands=['wuji2']))
    case = results.rows['cem'][0]['case']
    outputs = [load_geometry(case, stage) for stage in ('ik', 'optimized')]
    assert available_stages(case) == ['ik', 'optimized']
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
    for stage, (meshes, _, center, model) in zip(('initial_qpos', 'qpos'), outputs):
        data = mujoco.MjData(model)
        with np.load(f'{case}/grasp.npz') as saved:
            data.qpos[:] = saved[stage]
        mujoco.mj_kinematics(model, data)
        for vertices, _, _, _, position, quat, geom in meshes:
            displayed = Rotation.from_quat(quat, scalar_first=True).apply(vertices[0])+position
            expected = data.geom_xmat[geom].reshape(3, 3)@vertices[0]+data.geom_xpos[geom]-center
            np.testing.assert_allclose(displayed, expected, atol=1.e-8)
    with np.load(f'{case}/grasp.npz') as saved:
        q = saved['qpos'].copy()
    np.savez(f'{case}/grasp.npz', qpos=q)
    load_geometry.cache_clear()
    assert available_stages(case) == ['optimized']
    with pytest.raises(ValueError, match='No saved (initial_qpos|ik pose)'):
        load_geometry(case, 'ik')


class Widget:
    def __init__(self, client, label, options=(), initial_value=None, **attributes):
        self.client, self.label = client, label
        self.options = list(options)
        self.value = initial_value if initial_value is not None else next(iter(options), None)
        self.disabled = False
        self.content = ''
        self.callback = None
        self.__dict__.update(attributes)

    def on_update(self, callback):
        self.callback = callback
        return callback

    on_click = on_update

    def emit(self, value=None):
        if value is not None:
            self.value = value
        self.callback(SimpleNamespace(target=self, client=self.client))


class Client:
    def __init__(self):
        self.client_id = 1
        self.widgets = {}
        self.gui = SimpleNamespace(add_folder=lambda *a, **kw: nullcontext())
        for kind in ('dropdown', 'button', 'slider', 'checkbox', 'markdown'):
            setattr(self.gui, f'add_{kind}', self.add_widget)
        self.scene = SimpleNamespace(set_up_direction=lambda *a: None,
            world_axes=SimpleNamespace(visible=True), add_mesh_simple=self.add_mesh)
        self.camera = SimpleNamespace()

    def add_widget(self, label, options=(), **kw):
        widget = Widget(self, label, options, **kw)
        self.widgets[label] = widget
        return widget

    def add_mesh(self, *a, **kw):
        return SimpleNamespace(remove=lambda: None, **kw)

    def atomic(self):
        return nullcontext()


def test_sidebar_switches_hand_method_and_stage_and_discards_stale_playback(sources):
    results = Results(config=config(sources, stage='ik'))
    client = Client()
    pending = Future()
    requests = []
    def request(*args):
        requests.append(args)
        return pending
    close = attach_client(client, results, SimpleNamespace(request=request))
    try:
        widgets = client.widgets
        stage = widgets['结果阶段 / Stage']
        play = widgets['▶ 播放重力仿真']
        status = widgets['']
        assert play.disabled
        assert '初始化 IK' in status.content
        assert '| 综合合格' not in status.content
        stage.emit('优化后结果')
        assert not play.disabled
        assert '| 综合合格' in status.content
        camera = client.camera.position.copy()
        play.emit()
        assert len(requests) == 1
        stage.emit('初始化 IK')
        pending.set_result(('/not/a/trajectory.npz', {}))
        assert play.disabled
        np.testing.assert_array_equal(camera, client.camera.position)
        play.emit()
        assert len(requests) == 1
        widgets['手型 / Hand'].emit('wuji2')
        widgets['方法 / Method'].emit('adamw')
        assert widgets['抓取类别 / Grasp type'].value == '13_Template'
        assert widgets['手型 / Hand'].value == 'wuji2'
        assert '初始化 IK' in status.content
    finally:
        close()


def test_size_bar_snaps_to_saved_scales_and_disables_for_one_size(sources):
    path, entries = sources
    source = path/entries[0]['path']
    row = json.loads(source.read_text())[0]
    rows = []
    for scale in (.75, 1., 1.25):
        copied = dict(row, primitive=dict(row['primitive'], radial_scale=scale))
        rows.append(copied)
    source.write_text(json.dumps(rows))
    client = Client()
    close = attach_client(client, Results(config=config(sources)), SimpleNamespace())
    try:
        size = client.widgets['尺寸 / Size ×']
        assert size.min == .75 and size.max == 1.25 and not size.disabled
        assert [value for value, _ in size.marks] == [.75, 1., 1.25]
        assert not size.options  # A slider, not a dropdown with size options.
        size.emit(.91)
        assert size.value == 1.
        size.emit(1.14)
        assert size.value == 1.25
        client.widgets['方法 / Method'].emit('adamw')
        assert size.value == 1. and size.disabled
    finally:
        close()

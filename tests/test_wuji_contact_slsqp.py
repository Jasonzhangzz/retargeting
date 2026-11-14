"""Actual Moxian material binding, semantic graph and native analytic solver."""
import copy
from dataclasses import replace
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from moxian_web.contact import default_mapping
from moxian_web.qnbot_surface import QnbotSurface
from moxian_web.wuji_contact_source import MoxianContactSource, contact_edges, semantic_link, closest_triangle
from moxian_web.wuji_contact_cost import ContactObjective, ContactSLSQPSettings
from moxian_web.wuji_contact_slsqp import ContactSLSQPRetarget, NativeContactKinematics, palm_basis
from moxian_web.wuji_surface import WujiSurface

CONTACT_LINKS = ['thumb_distal', 'index_distal', 'palm']


@pytest.fixture(scope='module', params=['left', 'right'])
def source(request):
    torch.set_num_threads(1)
    m = QnbotSurface(request.param, 'cpu', finger_twist=True)
    m.bind_mapping(dict(assignments=default_mapping(m.side), confirmed=[]))
    return m


def frame(m, active=(), stamp=1000):
    with torch.no_grad(): v, j, _ = m(m.q0[None])
    mask = np.zeros(460, bool); mask[list(active)] = True
    return dict(vertices_m=v[0].tolist(), joints_m=j[0].tolist(), timestamp_ms=stamp,
        hand_side=m.side, geometry_version=m.version(), binding_version=m.binding_hash,
        tactile_timestamp_ms=stamp, source_contact=dict(ready=True, timestamp_ms=stamp,
            normalized=mask.astype(float).tolist(), cell_contact=mask.tolist()))


def test_all_taxels_follow_original_moxian_mesh_with_offsets_and_articulation(source):
    m = source; topo = copy.deepcopy(m.topology()); topo['sid_normal_offset_m'][0] = .001
    material = MoxianContactSource(topo)
    q = m.q0.clone(); q[7] += .2; q[-1] += .15
    with torch.no_grad(): v, j, _ = m(q[None])
    vertices = v[0].numpy().astype(np.float64); tri = vertices[np.asarray(topo['faces'])[topo['sid_face_ids']]]
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]); normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    expected = (tri*np.asarray(topo['sid_barycentric'])[..., None]).sum(1)
    expected[0] += .001*normals[0]
    points, actual_normals = material.taxels(vertices)
    np.testing.assert_allclose(points, expected @ material.mirror.T, atol=1e-10)
    np.testing.assert_allclose(actual_normals, normals @ material.mirror.T, atol=1e-10)
    assert len(points) == 460 and len(vertices) == 11598
    f = frame(m, [0]); f['vertices_m'] = vertices.tolist(); f['joints_m'] = j[0].tolist()
    a = material.read(f)['anchors'][0]
    np.testing.assert_allclose(a['position_m'], points[0], atol=1e-10)
    assert a['source_face'] == topo['sid_face_ids'][0]


def test_patch_representatives_keep_palm_tiles_but_exclude_same_semantic_link_edges(source):
    m = source; material = MoxianContactSource(m.topology())
    patches = np.asarray(m.mapping['assignments'])
    ids = [int(np.flatnonzero(patches == p)[0]) for p in (0, 2, 14, 15, 18)]
    result = material.read(frame(m, ids)); a = result['anchors']
    assert len(a) == 5 and len(result['edges']) == 7  # 10 pairs minus 3 palm-palm pairs
    assert all(a[i]['semantic_link'] != a[j]['semantic_link'] for i, j in result['edges'])
    assert semantic_link(0) == 'thumb_distal' and semantic_link(2) == 'index_distal'
    assert semantic_link(14) == semantic_link(19) == 'palm'
    repeated = [dict(semantic_link='index_distal'), dict(semantic_link='index_distal'), dict(semantic_link='palm')]
    assert contact_edges(repeated).tolist() == [[0, 2], [1, 2]]
    for anchor in a:
        face = np.asarray(m.topology()['faces'])[anchor['source_face']]
        expected = np.asarray(anchor['barycentric']) @ np.asarray(result['vertices'])[face]
        np.testing.assert_allclose(anchor['position_m'], expected, atol=1e-9)


def test_unknown_stale_single_patch_and_model_mismatch(source):
    material = MoxianContactSource(source.topology()); f = frame(source, [0])
    result = material.read(f); assert len(result['anchors']) == 1 and result['edges'].shape == (0, 2)
    for change in ({'tactile_timestamp_ms': 1026}, {'source_contact': {'ready': False, 'normalized': None}}, {'source_contact': {'ready': True, 'timestamp_ms': 974}}):
        assert not material.read({**f, **change})['anchors']
    assert not material.read(f, False)['anchors']
    with pytest.raises(ValueError, match='binding'): material.read({**f, 'binding_version': 'different'})
    with pytest.raises(ValueError, match='timestamps'): material.read({**f, 'skeleton_timestamp_ms': 970})
    with pytest.raises(ValueError, match='Moxian glove'): MoxianContactSource({**source.topology(), 'surface_backend': 'mano'})
    topo = {**source.topology(), 'model_hash': 'model-a'}
    with pytest.raises(ValueError, match='model hashes'): MoxianContactSource(topo).read(f)


@pytest.fixture(scope='module')
def kinematics():
    s = WujiSurface(); fk = NativeContactKinematics(s)
    contacts = []
    for owner in (4, 8, 0):
        mesh = s.meshes[owner]; face = int(np.argmax(mesh.area_faces))
        contacts.append(dict(robot_owner=owner, robot_local_m=mesh.triangles[face].mean(0), robot_local_normal=mesh.face_normals[face]))
    fk.set_contacts(contacts)
    return fk


def test_native_point_and_normal_jacobians_and_each_cost_term(kinematics):
    fk = kinematics; q = np.clip(np.full(20, .23), fk.lower+.1, fk.upper-.1)
    geometry = fk.forward(q); eps = 1e-6
    source_q = q+np.linspace(-.04, .03, 20)
    targets = fk.forward(source_q)
    objective = ContactObjective(targets[0], targets[2], targets[3], [[0, 1], [0, 2], [1, 2]], q+.03,
        contact_links=CONTACT_LINKS)
    for stage in (1, 2):
        value, gradient, terms = objective.evaluate(q, geometry, stage)
        assert np.isfinite(value) and all(v >= -1e-12 for v in terms.values())
        for k in range(20):
            d = np.eye(20)[k]*eps
            plus = fk.forward(q+d); minus = fk.forward(q-d)
            np.testing.assert_allclose((plus[0]-minus[0])/(2*eps), geometry[1][..., k], atol=1e-9)
            np.testing.assert_allclose((plus[2]-minus[2])/(2*eps), geometry[4][..., k], atol=1e-9)
            np.testing.assert_allclose((plus[3]-minus[3])/(2*eps), geometry[5][..., k], atol=1e-9)
            fd = (objective.evaluate(q+d, plus, stage)[0]-objective.evaluate(q-d, minus, stage)[0])/(2*eps)
            assert gradient[k] == pytest.approx(fd, abs=2e-7, rel=2e-6)
    # Direct baseline parity at identical prepared targets, not different model/frames.
    from dataclasses import asdict
    from third_party.emf_wuji_reference.targets import prepare_targets
    from third_party.emf_wuji_reference.objectives import FrameObjective
    t = prepare_targets(targets[0], thumb_free=True, thumb_rel_skip_proximal=2)
    reference = FrameObjective(t['S'], t['edges'], directions=t['directions'], laplacian=t['L'],
        differential=t['D'], weights=t['row_weights'], thumb_triplets=t['thumb_triplets'],
        params={**asdict(ContactSLSQPSettings()), 'thumb_rel_full_rad': .35, 'thumb_rel_off_rad': .5},
        q_ref=q+.03, antirev_indices=[i for i in range(20) if i % 4 in (2, 3)])
    _, base_g, base_terms = reference.evaluate(q, geometry[0], geometry[1])
    off = ContactObjective(targets[0], [], [], [], q+.03)
    _, g, t = off.evaluate(q, (geometry[0], geometry[1], *[np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3, 20)), np.empty((0, 3, 20))]))
    np.testing.assert_allclose(g, base_g, atol=1e-12)
    assert all(t[k] == base_terms[k] for k in base_terms)
    weak = ContactObjective(targets[0], targets[2], targets[3], objective.edges, q+.03,
        contact_links=CONTACT_LINKS, settings=replace(ContactSLSQPSettings(), lambda_contact_edge=150.))
    np.testing.assert_allclose(objective.adjacency[21:, 21:], 10*weak.adjacency[21:, 21:])
    assert objective.evaluate(q, geometry)[2]['laplacian'] > weak.evaluate(q, geometry)[2]['laplacian']
    assert 'contact_edge' not in objective.evaluate(q, geometry)[2]


def test_single_contact_and_normal_changes_add_no_tracking_cost_without_edges(kinematics):
    fk = kinematics; q = np.clip(np.full(20, .2), fk.lower+.1, fk.upper-.1)
    p, jp, c, n, jc, jn = fk.forward(q)
    objective = ContactObjective(p, c[:1]+.001, -n[:1], [], q)
    _, _, terms = objective.evaluate(q, (p, jp, c[:1], n[:1], jc[:1], jn[:1]))
    assert not {'contact_position', 'contact_normal', 'contact_edge'} & terms.keys()
    same = ContactObjective(p, c[:1], n[:1], [], q)
    baseline = same.evaluate(q, (p, jp, c[:1], n[:1], jc[:1], jn[:1]))
    changed = objective.evaluate(q, (p, jp, c[:1], n[:1], jc[:1], jn[:1]))
    assert baseline[0] == changed[0]
    np.testing.assert_array_equal(baseline[1], changed[1])


@pytest.mark.parametrize('with_contacts',[False,True])
def test_vectorized_realtime_objective_matches_all_reference_terms_and_gradients(kinematics,with_contacts):
    fk=kinematics;q=np.full(20,.2);g=fk.forward(q);target=fk.forward(q+np.linspace(-.03,.04,20))
    edges=[[0,1],[1,2]] if with_contacts else []
    kwargs=dict(contact_links=CONTACT_LINKS)
    reference=ContactObjective(target[0],target[2],target[3],edges,q+.02,**kwargs)
    fast=ContactObjective(target[0],target[2],target[3],edges,q+.02,fast=True,**kwargs)
    for stage in (1,2):
        a=reference.evaluate(q,g,stage);b=fast.evaluate(q,g,stage)
        assert a[0]==pytest.approx(b[0],abs=1e-12)
        np.testing.assert_allclose(a[1],b[1],atol=1e-11)
        for name in a[2]:assert a[2][name]==pytest.approx(b[2][name],abs=1e-12)
        for j in range(20):
            d=np.eye(20)[j]*1e-6
            fd=(fast.evaluate(q+d,fk.forward(q+d),stage)[0]-fast.evaluate(q-d,fk.forward(q-d),stage)[0])/2e-6
            assert b[1][j]==pytest.approx(fd,abs=2e-7)


def test_unified_graph_couples_contact_and_skeleton_but_ignores_normals_and_common_translation(kinematics):
    fk = kinematics; q = np.full(20, .2)
    geometry = fk.forward(q); p, jp, c, n, jc, jn = geometry
    offset = np.array([.02, -.03, .04])
    objective = ContactObjective(p, c+.002, n, [[0, 1], [1, 2]], q, contact_links=CONTACT_LINKS)
    shifted = ContactObjective(p+offset, c+.002+offset, -n, [[0, 1], [1, 2]], q, contact_links=CONTACT_LINKS)
    a = objective.evaluate(q, geometry); b = shifted.evaluate(q, geometry)
    assert a[0] == pytest.approx(b[0], abs=1e-12)
    np.testing.assert_allclose(a[1], b[1], atol=1e-12)
    assert a[2]['laplacian'] > 0  # Attachment edges couple the two node sets.
    normal_change = ContactObjective(p, c+.002, -n, [[0, 1], [1, 2]], q, contact_links=CONTACT_LINKS)
    np.testing.assert_allclose(a[1], normal_change.evaluate(q, geometry)[1], atol=1e-12)
    assert a[0] == normal_change.evaluate(q, geometry)[0]


def test_single_augmented_laplacian_has_shared_rows_and_hand_computed_loss(kinematics):
    fk = kinematics; q = np.full(20, .2)
    geometry = fk.forward(q); p, jp, c, n, jc, jn = geometry
    offset = np.array([[0., 0., 0.], [.001, 0., 0.], [.003, 0., 0.]])
    objective = ContactObjective(p, c+offset, n, [[0, 1], [1, 2]], q, contact_links=CONTACT_LINKS)
    assert objective.laplacian.shape == (24, 24)
    assert np.any(objective.laplacian[:21, 21:]) and np.any(objective.laplacian[21:, :21])
    assert objective.adjacency[21, 22] == 10 and objective.adjacency[22, 21] == 5
    assert objective.adjacency[21, 3] == objective.adjacency[3, 21] == .5
    assert objective.adjacency[23, 0] == .2
    np.testing.assert_allclose(objective.laplacian.sum(axis=1), 0, atol=1e-14)
    # Skeleton rows: two index endpoint residuals of .5 mm; five palm
    # attachment residuals of .6 mm. Shared material rows mix attachment and
    # boosted contact edges: +10, +4 and -23 mm (including cross terms).
    expected = .5*150*(2*.0005**2+5*.0006**2+(.01**2+.004**2+.023**2)/3)
    value, _, terms = objective.evaluate(q, geometry)
    assert terms['laplacian'] == pytest.approx(expected, abs=1e-12)
    assert value == pytest.approx(expected, abs=1e-12)
    assert set(terms) == {'laplacian', 'direction', 'smooth', 'thumb_relative', 'antireverse'}


def test_unified_graph_requires_semantics_and_has_no_same_link_contact_edges(kinematics):
    fk = kinematics; q = np.full(20, .2); p, _, c, n, _, _ = fk.forward(q)
    with pytest.raises(ValueError, match='Semantic link required'):
        ContactObjective(p, c, n, [[0, 1]], q)
    with pytest.raises(ValueError, match='different semantic links'):
        ContactObjective(p, c, n, [[0, 1]], q, contact_links=['palm', 'palm', 'thumb_distal'])
    disabled = ContactObjective(p, c+.01, n, [[0, 1]], q,
        settings=replace(ContactSLSQPSettings(), lambda_contact_edge=0))
    assert not disabled.graph_enabled and disabled.laplacian.shape == (21, 21)
    assert disabled.evaluate(q, fk.forward(q))[2]['laplacian'] == pytest.approx(0, abs=1e-20)


def test_wrist_alignment_uses_actual_native_asset_axes(kinematics):
    from scipy.spatial.transform import Rotation
    native = kinematics.forward(np.zeros(20))[0]
    expected = Rotation.from_rotvec([.7, -.2, 1.3]).as_matrix(); origin = np.array([.02, -.04, .08])
    observed = native @ expected.T+origin
    rotation = palm_basis(observed) @ kinematics.palm_basis.T
    np.testing.assert_allclose(rotation, expected, atol=1e-12)
    np.testing.assert_allclose((observed-observed[0]) @ rotation, native-native[0], atol=1e-12)


def test_separation_gradients_include_both_moving_links_and_normal(kinematics):
    from moxian_web.wuji_contact_collision import SeparationConstraints
    fk = kinematics; q = np.full(20, .2)
    constraints = SeparationConstraints(fk, [4, 8], [[.003, .001, -.01], [.002, .001, -.006]],
        [8, 4], [[.001, -.003, -.009], [.001, .002, -.01]], [[0., 1., 0.], [1., 0., 0.]])
    gap, jac = constraints.evaluate(q)
    for k in range(20):
        d = np.eye(20)[k]*1e-6
        fd = (constraints.evaluate(q+d)[0]-constraints.evaluate(q-d)[0])/2e-6
        np.testing.assert_allclose(jac[:, k], fd, atol=1e-9)


def test_slsqp_moves_negative_object_gap_to_zero(kinematics):
    from moxian_web.wuji_contact_collision import SeparationConstraints
    fk = kinematics; q = np.full(20, .2); fk.forward(q)
    owner = 8; local = np.array([0., 0., -.02]); body = fk.body_ids[owner]
    r = fk.data.xmat[body].reshape(3, 3); point = fk.data.xpos[body]+r @ local
    normal = np.array([0., 1., 0.]); target = point+.002*normal
    constraints = SeparationConstraints(fk, [owner], [local], [-1], [[0., 0., 0.]], [normal], [target], [normal])
    class Projection:
        def evaluate(self, x, geometry, stage):
            delta = x-q
            return float(.5*delta @ delta), delta, {}
    engine = ContactSLSQPRetarget.__new__(ContactSLSQPRetarget)
    engine.fk = fk; engine.settings = ContactSLSQPSettings()
    assert constraints.evaluate(q)[0][0] < -.0019
    result, trace = engine._solve(Projection(), q, 2, 80, constraints)
    assert not trace['fallback']
    assert constraints.evaluate(result)[0][0] >= -1e-7
    assert np.linalg.norm(result-q) > 0


def test_actual_known_object_witnesses_use_the_fixed_source_wrist_transform(kinematics):
    import trimesh
    from scipy.spatial.transform import Rotation
    from moxian_web.card_geometry import TrackedMesh
    from moxian_web.wuji_collision import WujiMeshCollision
    from moxian_web.wuji_contact_collision import NativeSeparation
    fk = kinematics; q = np.full(20, .2); fk.forward(q)
    owner = 8; local = np.asarray(fk.surface.meshes[owner].vertices)[0]
    body = fk.body_ids[owner]
    point = fk.data.xpos[body]+fk.data.xmat[body].reshape(3, 3) @ local
    rotation = Rotation.from_rotvec([.4, -.2, .3]).as_matrix(); origin = np.array([.1, -.2, .05])
    transform = np.eye(4); transform[:3, 3] = rotation @ point+origin
    box = trimesh.creation.box(extents=[.006]*3)
    obj = TrackedMesh(box.vertices, box.faces, 'cpu', transform)
    auditor = WujiMeshCollision(fk.surface, 'cpu')
    separation=NativeSeparation(fk,auditor,obj)
    constraints = separation.refresh(q, rotation, origin, per_pair=1)
    assert constraints is not None and np.any(constraints.b == -1)
    gap, jac = constraints.evaluate(q)
    assert np.any(gap[constraints.b == -1] < -.001)
    dense=separation.refresh(q,rotation,origin,per_pair=2)
    dense_gap=dense.evaluate(q)[0]
    def deepest(c,g):
        values={}
        for a,b,v in zip(c.a,c.b,g):values[(int(a),int(b))]=min(values.get((int(a),int(b)),1e9),v)
        return values
    reduced_values=deepest(constraints,gap);dense_values=deepest(dense,dense_gap)
    assert reduced_values.keys()==dense_values.keys()
    for pair in dense_values:assert reduced_values[pair]==pytest.approx(dense_values[pair],abs=1e-7)
    for k in (4, 5, 6, 7):
        d = np.eye(20)[k]*1e-6
        fd = (constraints.evaluate(q+d)[0]-constraints.evaluate(q-d)[0])/2e-6
        np.testing.assert_allclose(jac[:, k], fd, atol=1e-9)


def test_exact_triangle_projection():
    tri = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    for query, target in (([.2, .3, 2], [.2, .3, 0]), ([2, 2, 0], [.5, .5, 0]), ([-1, -1, 1], [0, 0, 0])):
        p, face, bc = closest_triangle(tri, query)
        np.testing.assert_allclose(p, target); np.testing.assert_allclose(bc @ tri[face], target)


def test_solver_contract_failure_reset_and_offline(source, tmp_path):
    m = source; engine = ContactSLSQPRetarget(m.topology(), tmp_path/'mapping', 'cpu')
    ids = [int(np.flatnonzero((np.asarray(m.mapping['assignments']) == p)&engine.valid_sids)[0]) for p in (0, 2, 14, 15)]
    f = frame(m, ids); output = engine.step(f)
    assert len(output['qpos']) == 26 and output['solver']['backend'] == 'nlopt.LD_SLSQP'
    assert output['postprocessing'] == 'none' and len(output['contact_edges']) == 5
    graph = output['retarget_graph']
    assert graph['type'] == 'unified_weighted_laplacian' and graph['material_node_count'] == 4
    assert len(graph['source_nodes_m']) == len(graph['robot_nodes_m']) == 25
    assert {'skeleton', 'contact', 'material_attachment'} == {edge['kind'] for edge in graph['edges']}
    assert 'contact_edge' not in output['quality']['cost_terms'] and 'laplacian' in output['quality']['cost_terms']
    # Recompute the sole Laplacian cost from the exported graph and positions.
    a = np.zeros((25, 25))
    for edge in graph['edges']: a[tuple(edge['nodes'])] = edge['weight']
    lap = np.diag(a.sum(axis=1))-a
    residual = lap @ (np.asarray(graph['robot_nodes_m'])-np.asarray(graph['source_nodes_m']))
    exported_cost = .5*graph['coefficient']*np.sum(np.asarray(graph['row_weights'])[:, None]*residual**2)
    assert output['quality']['cost_terms']['laplacian'] == pytest.approx(exported_cost, abs=1e-11)
    assert output['collision']['self_hard_limit_mm'] == .5
    assert not output['solver']['used_history_fallback']
    assert [s['maxeval'] for s in output['solver']['stages']] == [50, 80]
    assert all(np.isfinite(output['qpos']))
    json.dumps(output, allow_nan=False)
    session = output['solver_session_id']
    f['timestamp_ms'] += 500; f['tactile_timestamp_ms'] += 500; f['source_contact']['timestamp_ms'] += 500
    assert engine.step(f)['solver_session_id'] != session
    # A failed optimizer stage cannot publish its restored stage start as feasible.
    original = engine._solve
    engine._solve = lambda objective, start, stage, budget, constraints=None: (start, dict(stage=stage, status=-4, fallback=True, converged=False))
    f['timestamp_ms'] += 10
    failed = engine.step(f)
    assert not failed['ready'] and 'optimizer_failed' in failed['solver']['failure_reasons']
    engine._solve = original
    from scripts.retarget_moxian_wuji2 import replay
    recording = tmp_path/'input.jsonl'
    recording.write_text(json.dumps(dict(hand_model=frame(m, ids), hand_model_topology=m.topology()))+'\n')
    destination = tmp_path/'output.jsonl'
    result = replay(recording, destination, cache=tmp_path/'mapping')
    assert result['solver'] == 'slsqp' and result['postprocessing'] == 'none'
    row = json.loads(destination.read_text()); assert row['wuji2']['source_surface_backend'] == 'moxian'
    with pytest.raises(FileExistsError): replay(recording, destination)


def test_worker_switches_actual_independent_backends(source, tmp_path):
    if source.side != 'right': pytest.skip('One process integration exercises both backend selections')
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    events = tmp_path/'events.jsonl'; log = tmp_path/'worker.log'
    fd = os.open(events, os.O_WRONLY | os.O_CREAT, 0o600)
    with log.open('w') as stderr:
        process = subprocess.Popen([sys.executable, '-I', str(Path.cwd()/'moxian_web/wuji_worker.py'),
            '--event-fd', str(fd), '--home', str(tmp_path/'worker'), '--solver', 'slsqp'],
            pass_fds=(fd,), stdin=subprocess.PIPE, stdout=stderr, stderr=stderr, text=True)
    os.close(fd)
    def send(message):
        process.stdin.write(json.dumps(message)+'\n'); process.stdin.flush()
    def wait_for(predicate, timeout=45):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            rows = []
            for line in events.read_text().splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
            matches = [r for r in rows if predicate(r)]
            if matches: return matches[-1]
            assert process.poll() is None, log.read_text()
            errors = [r for r in rows if r.get('status') == 'error']
            assert not errors, str(errors)+log.read_text()
            time.sleep(.05)
        pytest.fail('Worker timeout: '+log.read_text())
    try:
        send(dict(type='topology', model=source.topology()))
        f = frame(source, [], time.time()*1000); send(dict(type='frame', frame=f))
        result = wait_for(lambda r: r.get('type') == 'frame' and r.get('solver_backend') == 'slsqp')
        assert result['solver']['backend'] == 'nlopt.LD_SLSQP'
        send(dict(type='control', action='solver_set', data=dict(solver='cem'), id='switch-cem'))
        wait_for(lambda r: r.get('type') == 'ack' and r.get('id') == 'switch-cem')
        f = frame(source, [], time.time()*1000); send(dict(type='frame', frame=f))
        result = wait_for(lambda r: r.get('type') == 'frame' and r.get('algorithm', '').startswith('card_'))
        assert len(result['qpos']) == 26
        send(dict(type='control', action='solver_set', data=dict(solver='slsqp'), id='switch-back'))
        wait_for(lambda r: r.get('type') == 'ack' and r.get('id') == 'switch-back')
        f = frame(source, [], time.time()*1000); send(dict(type='frame', frame=f))
        new = wait_for(lambda r: r.get('type') == 'frame' and r.get('solver_backend') == 'slsqp'
                       and r.get('timestamp_ms') == f['timestamp_ms'])
        assert new['settings']['wrist_policy'] == 'fixed_source_wrist'
        assert new['active_solver_session_id']==new['solver_session_id']
        assert new['geometry_f32']['format']=='wuji-f32-v1' and 'source_vertices_m' not in new
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        process.stdin.close()


def test_warm_start_uses_last_finite_candidate_and_does_not_reset_for_processed_frame_skips(source,tmp_path,monkeypatch):
    engine=ContactSLSQPRetarget(source.topology(),tmp_path/'mapping','cpu')
    real_audit=engine.audit
    def reject(*args):
        ok,report,p,r=real_audit(*args)
        report['max_penetration_mm']=.1;report['self_mesh']['max_penetration_mm']=.1
        return ok,report,p,r
    monkeypatch.setattr(engine,'audit',reject)
    monkeypatch.setattr(engine.separation,'refresh',lambda *args,**kwargs:None)
    a=engine.step({**frame(source,stamp=1000.),'input_stream_epoch':0})
    assert not a['solution_feasible']
    b=engine.step({**frame(source,stamp=1600.),'input_stream_epoch':0})
    np.testing.assert_array_equal(b['initial_qpos'][6:],a['raw_qpos'][6:])
    assert a['solver_session_id']==b['solver_session_id'] and b['solver']['initial_guess']=='previous_raw_solution'
    assert [s['maxeval'] for s in b['solver']['stages']]==[1,10]
    c=engine.step({**frame(source,stamp=1610.),'input_stream_epoch':1})
    assert c['solver_session_id']!=b['solver_session_id'] and c['solver']['reset_reason']=='input_stream_interrupted'


def test_source_gap_resets_solver_without_midrange_pose_jump(source,tmp_path,monkeypatch):
    engine=ContactSLSQPRetarget(source.topology(),tmp_path/'gap-continuity','cpu',realtime=True)
    monkeypatch.setattr(engine.separation,'refresh',lambda *args,**kwargs:None)
    a=engine.step({**frame(source,stamp=1000.),'input_stream_epoch':0})
    b=engine.step({**frame(source,stamp=1016.),'input_stream_epoch':0})
    c=engine.step({**frame(source,stamp=1400.),'input_stream_epoch':0})
    np.testing.assert_array_equal(c['initial_qpos'][6:],b['raw_qpos'][6:])
    assert c['solver']['reset_reason']=='source_gap_continuity_seed'
    assert abs(np.asarray(c['qpos'][6:])-b['raw_qpos'][6:]).max()<=2*np.pi/30+1e-10


def test_realtime_slsqp_constrains_joint_step_inside_optimization(source,tmp_path):
    engine=ContactSLSQPRetarget(source.topology(),tmp_path/'mapping','cpu',realtime=True)
    a=engine.step({**frame(source,stamp=1000.),'input_stream_epoch':0})
    patches=np.asarray(source.mapping['assignments']);ids=[int(np.flatnonzero(patches==p)[0]) for p in (0,2,14)]
    b=engine.step({**frame(source,ids,stamp=1016.),'input_stream_epoch':0})
    np.testing.assert_array_equal(b['initial_qpos'][6:],a['raw_qpos'][6:])
    assert abs(np.asarray(b['qpos'][6:])-a['qpos'][6:]).max()<=2*np.pi*.016+1e-10
    assert b['solver']['warm_start_source_timestamp_ms']==1000.
    assert b['settings']['execution_profile']=='realtime_stream' and b['postprocessing']=='none'
    assert b['solver']['separation_recovery']==[] or len(b['solver']['separation_recovery'])==1


def test_normal_compatible_mapping_excludes_bad_sids_but_preserves_observations(source,tmp_path):
    engine=ContactSLSQPRetarget(source.topology(),tmp_path/'normal-mapping','cpu')
    records=engine.mapping['records'];valid=engine.valid_sids
    assert len(records)==460 and valid.any()
    assert all(r['normal_error_deg']<=60.+1e-8 for r in records if r['normal_valid'])
    assert engine.mapping['normal_rejected_sids']==np.flatnonzero(~valid).tolist()
    observed=engine.source.read(frame(source,range(460)))
    assert observed['weights']['observed_mask'].all()
    np.testing.assert_array_equal(observed['weights']['mask'],valid)
    for a in engine.material_contacts(observed):
        assert a['projection_normal_error_deg']<=60.+1e-8
        assert valid[np.asarray(a['sids'])].all()
    # The CEM mapping remains its independent legacy policy/cache identity.
    from moxian_web.wuji_contact_source import robot_mapping
    legacy=robot_mapping(MoxianContactSource(source.topology()),engine.surface,tmp_path/'legacy')
    assert legacy['version']!=engine.mapping['version'] and 'normal_rejected_sids' not in legacy


def test_restoration_keeps_making_progress_when_zero_gap_is_outside_motion_box():
    from moxian_web.wuji_contact_cost import SeparationRestorationObjective
    class LinearGap:
        a=np.array([0])
        def evaluate(self,q):
            jac=np.zeros((1,20));jac[0,0]=.04  # 40 mm link lever arm.
            return np.array([.04*q[0]-.016]),jac
    engine=object.__new__(ContactSLSQPRetarget)
    engine.fk=SimpleNamespace(lower=-np.ones(20),upper=np.ones(20),forward=lambda q:None)
    engine.settings=ContactSLSQPSettings();engine.realtime=True
    q=np.zeros(20);objective=SeparationRestorationObjective(LinearGap(),q)
    value,gradient,_=objective.evaluate(q,None)
    for j in (0,1):
        d=np.eye(20)[j]*1e-6
        difference=(objective.evaluate(q+d,None)[0]-objective.evaluate(q-d,None)[0])/2e-6
        assert gradient[j]==pytest.approx(difference,abs=1e-5,rel=1e-9)
    for _ in range(5):
        engine.solve_lower=q-.1;engine.solve_upper=q+.1
        engine.deadline=time.perf_counter()-1.  # Expired tracking budget must not starve recovery.
        updated,trace=engine._solve(SeparationRestorationObjective(LinearGap(),q),q,2,60)
        assert np.max(abs(updated-q))<=.1+1e-9
        if q[0]<.3999:assert updated[0]>q[0]+.01
        q=updated
    assert q[0]>=.3999


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Dense native GPU recovery regression')
def test_realtime_escapes_a_deeply_penetrating_warm_start(source,tmp_path):
    engine=ContactSLSQPRetarget(source.topology(),tmp_path/'recovery','cuda:0',realtime=True)
    q=np.array([1.291,-.29950217,.63081916,.12025834,.54494104,.50084703,.25564316,.03984771,
        .48350511,.33397265,-.02039607,.08483385,.222141,.35959966,1.07520912,.00172664,
        -.02712961,-.51986308,.94930865,1.31123959])
    engine.previous=q.copy();engine.previous_timestamp=1000.-1000/60
    reached=False;saw_infeasible=False;saw_recovery=False
    for i in range(45):
        r=engine.step({**frame(source,stamp=1000.+i*1000/60),'input_stream_epoch':0})
        np.testing.assert_allclose(r['initial_qpos'][6:],q,atol=1e-12)
        updated=np.asarray(r['qpos'][6:]);assert abs(updated-q).max()<=2*np.pi/60+1e-8
        saw_infeasible|=not r['solution_feasible'];saw_recovery|=bool(r['solver']['separation_recovery'])
        q=updated
        if r['solution_feasible']:
            assert r['collision']['max_penetration_mm']<=engine.settings.separation_tolerance_mm
            reached=True;break
    assert reached and saw_infeasible and saw_recovery

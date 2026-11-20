import time
import json

import numpy as np
import pytest
import torch

from moxian_web.wuji_sdf_comparison import CollisionComparison, visual_witnesses
from moxian_web.wuji_sdf_service import CollisionComparisonService, comparison_packet


def frame():
    return dict(ready=True, qpos=[.12] * 6 + [.25] * 20, source_timestamp_ms=12345.,
                joint_names=[f'j{i}' for i in range(20)], solver_backend='cart',
                solver_session_id='session1', mapping_version='mapping', playback={'active': True})


def test_packet_uses_one_published_pose_and_tracks_context():
    w = frame()
    packet = comparison_packet(w)
    assert packet['frame']['qpos'] == w['qpos'][6:]
    assert packet['timestamp_ms'] == w['source_timestamp_ms']
    assert packet['input_kind'] == 'replay'
    old = packet['context_key']
    w['solver_session_id'] = 'session2'
    assert comparison_packet(w)['context_key'] != old
    w['ready'] = False
    assert comparison_packet(w) is None
    w['display_available'] = True
    assert comparison_packet(w) is not None  # Can inspect an infeasible displayed solve.
    w['qpos'][8] = float('nan')
    assert comparison_packet(w) is None
    w['qpos'] = [0.] * 20
    assert comparison_packet(w) is None


def test_unavailable_or_expired_results_never_display_old_collision(tmp_path):
    service = CollisionComparisonService(lambda: None, tmp_path)
    service.closed.set()
    service.enabled = True
    service.result = dict(ready=True, timestamp_ms=time.time() * 1000 - 3000, input_kind='live',
                          dense_mesh={'collision': True}, neural={'collision': False}, pairs=[{}])
    service.received = time.monotonic() - 3
    state = service.snapshot()
    assert state['status'] == 'stale' and not state['ready']
    assert state['dense_mesh'] is None and state['neural'] is None and state['pairs'] == []
    assert state['visualization'] is None
    service.result['input_kind'] = 'replay'
    assert service.snapshot()['ready']  # Paused playback intentionally retains its result.
    w = frame()
    service.result['context_key'] = comparison_packet(w)['context_key']
    assert service.snapshot(w)['ready']
    w['solver_session_id'] = 'another'
    switched = service.snapshot(w)
    assert not switched['ready'] and switched['neural'] is None and not switched['pairs']
    service.reason = 'waiting'
    state = service.snapshot()
    assert not state['ready'] and state['neural'] is None
    assert 'Wuji2' in state['error']
    service.close()


def test_same_sample_oracle_agreement_domain_and_missing_model(tmp_path):
    torch.set_num_threads(1)
    c = CollisionComparison(tmp_path / 'missing.pt', 'cpu', per_link=54)
    q = np.full(20, .2, np.float32)
    first = c.query(q, c.geometry.joint_names, 'one')
    assert first['ready'] and first['dense_mesh']['query_valid']
    assert first['neural'] is None and first['model_error']
    assert first['samples_per_link'] == 54 and first['query_points'] > 1000

    class ExactTeacher:
        def __call__(self, angles, x, links):
            values = c.geometry.label(x[None], links[None], c.geometry.frames(angles[None]))[0]
            return torch.tensor(values, device=x.device)

    c.model = ExactTeacher()
    exact = c.query(q, c.geometry.joint_names, 'one')
    assert exact['comparison']['matched_agrees']
    assert exact['comparison']['point_mae_mm'] == 0
    assert exact['matched_mesh']['max_penetration_mm'] == exact['neural']['max_penetration_mm']
    assert exact['comparison']['frames'] == 2
    colliding = np.random.default_rng(104).uniform(c.geometry.domain['lower'], c.geometry.domain['upper']).astype(np.float32)
    collision = c.query(colliding, c.geometry.joint_names, 'one')
    assert collision['dense_mesh']['collision'] and collision['pairs']
    v = collision['visualization']
    assert len(v['link_positions_m']) == 21 and len(v['link_rotations']) == 21
    assert v['witnesses']['displayed_queries'] > 0
    assert v['witnesses']['mesh_sdf_mm'] == v['witnesses']['neural_sdf_mm']
    json.dumps(collision, allow_nan=False)  # Include colliding per-link records on the JSONL transport.
    # Physical rigid root offsets are irrelevant to self collision; no root is
    # passed to the conditioned field and no out-of-domain q is silently clipped.
    q[6] = -.15
    outside = c.query(q, c.geometry.joint_names, 'new-session')
    assert not outside['in_training_domain'] and len(outside['out_of_domain_joints']) == 1
    assert outside['qpos'][6] == pytest.approx(-.15)
    assert outside['comparison']['frames'] == 1
    with pytest.raises(ValueError, match='关节顺序'):
        c.query(q, list(reversed(c.geometry.joint_names)))


def test_visual_witnesses_preserve_missed_and_false_positive_points():
    x = torch.arange(18).reshape(6, 3).float()
    source, target = torch.tensor([1]*6), torch.tensor([8]*6)
    real = torch.tensor([-.003, -.002, .001, .002, -.004, .001])
    predicted = torch.tensor([-.003, .001, -.002, .003, -.005, -.004])
    v = visual_witnesses(x, source, target, real, predicted, per_group=1)
    # One strongest example from each category: both, mesh-only, neural-only.
    assert v['displayed_queries'] == 3 and v['total_collision_queries'] == 5
    selected = {tuple(p) for p in v['positions_m']}
    assert selected == {tuple(x[i].tolist()) for i in [4, 1, 5]}
    assert len(v['positions_m']) == len(v['mesh_sdf_mm']) == len(v['neural_sdf_mm'])


def test_visual_assets_export_and_strict_paths(tmp_path):
    torch.set_num_threads(1)
    c = CollisionComparison(tmp_path / 'missing.pt', 'cpu', per_link=54)
    c.export_visual_model(tmp_path)
    model = json.loads((tmp_path / 'visual_model.json').read_text())
    assert model['ready'] and model['asset_sha256'] == c.geometry.surface.asset_hash
    assert len(model['meshes']) == 21 and all(m['vertices_m'] and m['faces'] for m in model['meshes'])
    service = CollisionComparisonService(lambda: None, tmp_path)
    service.close()
    assert json.loads(service.visual_asset('model'))['asset_sha256'] == model['asset_sha256']
    assert not service.visual_asset('example', 0)['ready']
    with pytest.raises(ValueError):
        service.visual_asset('example', -1)
    with pytest.raises(ValueError):
        service.visual_asset('../model')

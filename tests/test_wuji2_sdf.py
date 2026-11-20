"""Articulated labels, pose isolation, and gradients of the direct q-conditioned field."""
from types import SimpleNamespace
from unittest.mock import patch
import os
import signal

import mujoco
import numpy as np
import pytest
import torch
import trimesh

from retarget.learning.wuji2_sdf.data import batches, read_manifest, sample
from retarget.learning.wuji2_sdf.geometry import Geometry
from retarget.learning.wuji2_sdf.model import ConfigurationSDF, load_model
from retarget.learning.wuji2_sdf.train import train
from retarget.learning.wuji2_sdf.benchmark import distance_metrics, pose_keys


def test_fresh_benchmark_metrics_and_pose_overlap_keys():
    truth = np.array([-.002, -.001, .001, .003], np.float32)
    prediction = np.array([-.001, .001, -.001, .004], np.float32)
    report = distance_metrics(truth, prediction)
    assert report['queries'] == 4
    assert report['mae_mm'] == pytest.approx(1.5)
    assert report['rmse_mm'] == pytest.approx(np.sqrt(2.5))
    assert report['bias_mm'] == pytest.approx(.5)
    assert report['penetration_false_negative_rate'] == .5
    assert report['outside_false_positive_rate'] == .5
    assert report['sign_accuracy_away_0p1mm'] == .5
    assert report['near_1mm']['queries'] == 2
    q = np.zeros((1, 20), np.float32)
    other = q.copy()
    other[:, 0] = -0.
    assert pose_keys(q) == pose_keys(other)
    other[:, 0] = .00001
    assert not pose_keys(q) & pose_keys(other)


@pytest.fixture(scope='module')
def geometry():
    torch.set_num_threads(1)
    return Geometry('cpu')


def test_frames_labels_move_with_joint_angles(geometry):
    g = geometry
    q = np.zeros((2, 20), np.float32)
    q[1, 4] = 1.
    frames = g.frames(q)
    # Check the actual compiled model frames against independent MuJoCo FK.
    data = mujoco.MjData(g.model)
    for i in range(2):
        data.qpos[6:] = q[i]
        mujoco.mj_kinematics(g.model, data)
        np.testing.assert_allclose(frames[0][i].numpy(), data.xpos, atol=1e-7)
        np.testing.assert_allclose(frames[1][i].numpy(), data.xmat.reshape(-1, 3, 3), atol=2e-7)
    owner = 8
    mesh = g.surface.meshes[owner]
    local = np.asarray(mesh.vertices)[[0, len(mesh.vertices) // 2]]
    p, r = (x[:, g.surface.body_ids].numpy() for x in frames)
    x = np.einsum('bij,nj->bni', r[:, owner], local) + p[:, owner, None]
    links = np.full((2, len(local)), owner, np.int32)
    # Moving material points remain on the SAME link surface at both poses.
    np.testing.assert_allclose(g.label(x, links, frames), 0, atol=2e-7)
    # Fixed wrist-space points acquire different distances when q changes.
    fixed = np.broadcast_to(x[0], x.shape).copy()
    distance = g.label(fixed, links, frames)
    assert abs(distance[1]).max() > .005
    # Unsigned triangle distance agrees with independent trimesh closest-point.
    transformed = (fixed[1] - p[1, owner]) @ r[1, owner]
    _, expected, _ = trimesh.proximity.closest_point(mesh, transformed)
    np.testing.assert_allclose(abs(distance[1]), expected, atol=2e-7)
    # Independent sign check on one closed native mesh (trimesh uses inside+).
    closed = g.surface.meshes[3]
    local = np.random.default_rng(913).uniform(*closed.bounds, size=(64, 3)).astype(np.float32)
    world = (local @ r[0, 3].T + p[0, 3])[None]
    signed = g.label(world, np.full((1, 64), 3, np.int32), tuple(f[:1] for f in frames))[0]
    expected = -trimesh.proximity.signed_distance(closed, local)
    assert (signed < 0).any() and (signed > 0).any()
    np.testing.assert_allclose(signed, expected, atol=1e-5)


def test_direct_model_broadcast_dependency_and_q_gradient(geometry):
    torch.manual_seed(2)
    g = geometry
    model = ConfigurationSDF(g.domain['lower'], g.domain['upper'], g.mask.tolist(), width=32, depth=2)
    q = torch.rand(2, 20, requires_grad=True)
    x = (torch.rand(2, 5, 3) * .1).requires_grad_()
    link = torch.full((2, 5), 8)
    values = model(q[:, None], x, link)
    assert values.shape == (2, 5)
    gradq, gradx = torch.autograd.grad(values.sum(), (q, x))
    assert gradq[:, 4:8].abs().sum() > 0 and gradx.abs().sum() > 0
    assert torch.count_nonzero(gradq[:, :4]) == 0
    assert torch.count_nonzero(gradq[:, 8:]) == 0
    changed = q.detach().clone()
    changed[:, 8:] += 10
    torch.testing.assert_close(values, model(changed[:, None], x, link), atol=0, rtol=0)
    assert model.per_link(q, x).shape == (2, 5, 21)
    torch.testing.assert_close(model.union(q, x), model.per_link(q, x).amin(-1))
    palm = torch.zeros((2, 5), dtype=torch.long)
    torch.testing.assert_close(model(q[:, None], x, palm), model(changed[:, None], x, palm))


def test_shards_resume_split_integrity_and_training(tmp_path):
    args = SimpleNamespace(output=str(tmp_path / 'data'), poses=40, per_link=8, shard_poses=20,
                           pose_batch=4, collision_threshold_mm=.5, pose_filter='all', seed=714,
                           resume=False, device='cpu')
    manifest = sample(args)
    assert manifest['queries'] == 40 * 21 * 8
    args.resume = True
    assert sample(args) == manifest
    m = read_manifest(args.output, verify=True)
    ids = [set(), set(), set()]
    for s in m['shards']:
        with np.load(tmp_path / 'data' / s['file']) as a:
            for split in range(3):
                ids[split].update(a['pose_id'][a['split'] == split].tolist())
            assert np.all(a['collision'] == (a['max_penetration_mm'] > .5))
            assert np.isfinite(a['sdf']).all()
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert len(set.union(*ids)) == 40
    count = sum(len(b['q']) for b in batches(args.output, m, 0, 53))
    assert count == len(ids[0]) * 21 * 8
    t = SimpleNamespace(data=args.output, output=str(tmp_path / 'model'), threads=1, epochs=2,
                        batch_size=256, lr=.001, eikonal_weight=.01, sign_weight=.02,
                        width=32, depth=2, seed=714, val_queries=512, steps_per_epoch=2,
                        device='cpu', resume=False, checkpoint_steps=1)
    report = train(t)
    assert report['training_steps'] == 4 and np.isfinite(report['mae_mm'])
    assert report['configuration_response']['pose_pairs'] >= 2
    t.resume = True
    assert train(t) == report
    model, info = load_model(tmp_path / 'model' / 'best.pt')
    assert info['dataset_fingerprint'] == m['fingerprint']
    assert model(torch.zeros(1, 20), torch.zeros(1, 3), torch.tensor([1])).isfinite().all()
    # Interrupt midway through an epoch, then resume to the exact CPU result.
    reference = torch.load(tmp_path / 'model' / 'last.pt', weights_only=False)
    t.output = str(tmp_path / 'interrupted')
    t.resume = False
    original_step = torch.optim.AdamW.step
    calls = 0
    def interrupt_step(opt, *positional, **keywords):
        nonlocal calls
        result = original_step(opt, *positional, **keywords)
        calls += 1
        if calls == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return result
    with patch.object(torch.optim.AdamW, 'step', interrupt_step):
        assert train(t)['state'] == 'paused'
    t.resume = True
    resumed = train(t)
    assert resumed['training_steps'] == report['training_steps']
    recovered = torch.load(tmp_path / 'interrupted' / 'last.pt', weights_only=False)
    for name in reference['model']:
        torch.testing.assert_close(reference['model'][name], recovered['model'][name], atol=0, rtol=0)
    # Corrupt shards must never be accepted as an unchanged dataset.
    target = tmp_path / 'data' / m['shards'][0]['file']
    target.write_bytes(target.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='corrupt'):
        read_manifest(args.output, verify=True)

"""Complete geometry->force->motion derivatives and GPU replay for the new model."""
from pathlib import Path
import json

import mujoco
import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from tests.test_grasp_device import make_runtime
from tests.test_grasp_explicit import contact_reference
from retarget.grasp_cem import CardCemNode
from retarget.grasp_device import CudaNodeGraph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_node(path, *, horizon=2, refinement=1):
    runtime = make_runtime(path, .025, primitive='cuboid', search='cem', hand_model='wuji2')
    runtime.args.explicit_pose = True
    runtime.args.explicit_horizon = horizon
    runtime.args.explicit_refinement_steps = refinement
    runtime.args.explicit_pose_weight = 1.
    return CardCemNode(runtime, capture_standalone=False)


def reference(node, x, selected):
    """MuJoCo Jacobians + independent double-precision mesh/projection equations."""
    e, r = node.explicit, node.runtime
    p, model = e.parameters, r.model
    obj = p.object_body
    from retarget.contact_exploration import ObjectMeshSurface
    surface = ObjectMeshSurface.from_model(model, obj)
    triangles = surface.triangles
    mesh = trimesh.Trimesh(triangles.reshape(-1, 3)*1000,
        np.arange(len(triangles)*3).reshape(-1, 3), process=False)
    hull = ConvexHull(triangles.reshape(-1, 3)).equations
    q = r.qpos.astype(float).copy()
    hand = x[:26].copy()
    root = np.array(node.state.root_rotation).reshape(3, 3)
    hand[3:6] = (Rotation.from_matrix(root)*Rotation.from_rotvec(x[3:6])).as_euler('XYZ')
    q[r.index.hand_qpos] = hand
    command = hand.copy(); command[6:] += x[26:]
    initial_data = mujoco.MjData(model); initial_data.qpos[:] = q; mujoco.mj_forward(model, initial_data)
    command += initial_data.qfrc_bias[model.jnt_dofadr[p.joints.numpy()]]/p.stiffness.numpy()
    hold = np.clip(command, p.command_low.numpy(), p.command_high.numpy())
    bodies = p.bodies.numpy()
    ids = p.hand_qpos.numpy()
    joints = p.joints.numpy()
    columns = np.r_[np.arange(model.jnt_dofadr[model.joint('obj_joint').id],
                               model.jnt_dofadr[model.joint('obj_joint').id]+6), model.jnt_dofadr[joints]]
    inverse, kp = p.inverse.numpy().astype(float), p.stiffness.numpy().astype(float)
    motion_costs = []
    for step in range(e.horizon):
        indices = e.stages[step].selected.numpy()[0]
        samples = p.samples.numpy()[np.maximum(indices, 0)]
        data = mujoco.MjData(model); data.qpos[:] = q
        mujoco.mj_forward(model, data)
        rotation = data.xmat[obj].reshape(3, 3)
        world = data.xpos[bodies]+np.einsum('nij,nj->ni', data.xmat[bodies].reshape(-1, 3, 3), samples)
        local = (world-data.xpos[obj])@rotation
        nearest, distance, faces = trimesh.proximity.closest_point_naive(mesh, local*1000)
        nearest, distance = nearest/1000, distance/1000
        inside = (local@hull[:, :3].T+hull[:, 3]).max(axis=1) <= 1.e-10
        distance = np.where(inside, -distance, distance)
        jac, gaps = np.zeros((4*len(bodies), 32)), np.ones(4*len(bodies))
        for c, body in enumerate(bodies):
            if indices[c] < 0 or distance[c] > p.cutoff or p.forbidden.numpy()[indices[c]]:
                continue
            outward = mesh.face_normals[faces[c]]
            displacement = local[c]-nearest[c]
            if np.linalg.norm(displacement) > 1.e-7:
                outward = (-1 if inside[c] else 1)*displacement/np.linalg.norm(displacement)
            normal = -(rotation@outward)
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            t1 = np.cross(normal, axis); t1 /= np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            rays = normal+p.friction*np.array([t1, t2, -t1, -t2])
            midpoint = (world[c]+data.xpos[obj]+rotation@nearest[c])/2
            jp, jo = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, None, midpoint, int(body))
            mujoco.mj_jac(model, data, jo, None, midpoint, obj)
            jac[4*c:4*c+4] = rays@(jo-jp)[:, columns]
            gaps[4*c:4*c+4] = distance[c]
        b = np.r_[np.array(list(p.gravity_force)), np.zeros(3), np.clip(kp*(hold-q[ids]),p.force_low.numpy(),p.force_high.numpy())-data.qfrc_bias[model.jnt_dofadr[joints]]]
        free = inverse*b
        force = contact_reference(jac, free, gaps, inverse, p.friction, p.sigma, e.settings['contact_iterations'])
        velocity = (free+inverse*(jac.T@force))/p.dt
        a = p.object_qpos
        q[a:a+3] += p.dt*velocity[:3]
        old = q[a+3:a+7].copy(); omega = velocity[3:6]
        delta = np.r_[-old[1:]@omega, old[0]*omega+np.cross(old[1:], omega)]
        quat = old+.5*p.dt*delta
        q[a+3:a+7] = quat/np.linalg.norm(quat)
        q[ids] += p.dt*velocity[6:]
        goal = r.qpos[a+3:a+7].astype(float); goal /= np.linalg.norm(goal)
        motion_costs.append(p.weight*(5000*np.sum((q[a:a+3]-r.qpos[a:a+3])**2)+50*(1-np.dot(q[a+3:a+7], goal)**2)))
    a = p.object_qpos; seed = r.qpos.astype(float)
    quat, goal = q[a+3:a+7], seed[a+3:a+7]/np.linalg.norm(seed[a+3:a+7])
    motion = p.weight*(5000*np.sum((q[a:a+3]-seed[a:a+3])**2)+50*(1-np.dot(quat, goal)**2))
    return float(np.mean(motion_costs))+geometry_reference(node, x)



def geometry_reference(node, x):
    """Independent double precision implementation of the CARD geometry costs."""
    r, d = node.runtime, node.cost
    model = r.model
    q = r.qpos.astype(float).copy()
    hand = x[:26].copy()
    root = Rotation.from_matrix(np.array(node.state.root_rotation).reshape(3, 3))
    hand[3:6] = (root*Rotation.from_rotvec(x[3:6])).as_euler('XYZ')
    q[r.index.hand_qpos] = hand
    data = mujoco.MjData(model); data.qpos[:] = q; mujoco.mj_kinematics(model, data)
    positions, rotations = data.xpos, data.xmat.reshape(-1, 3, 3)
    value = 0.
    obj = d.object_body
    sample = d.sample_local.numpy().astype(float)
    normals = d.sample_normal.numpy().astype(float)
    pad = d.sample_pad.numpy().astype(bool)
    for g, body in enumerate(d.group_body.numpy()):
        ids = np.arange(d.group_start.numpy()[g], d.group_end.numpy()[g])
        ids = ids[pad[ids]]
        points = sample[ids]@rotations[body].T+positions[body]
        anchor = rotations[obj]@d.anchor.numpy()[g]+positions[obj]
        pref = rotations[body]@d.preference.numpy()[g]+positions[body]
        scores = ((points-anchor)**2).sum(1)+d.alpha.numpy()[g]*((points-pref)**2).sum(1)
        index = ids[np.argmin(scores)]
        point = rotations[body]@sample[index]+positions[body]
        previous = rotations[body]@d.previous_local.numpy()[g]+positions[body]
        normal = rotations[body]@normals[index]
        push = rotations[obj]@d.push.numpy()[g]
        value += d.rho.numpy()[g]**2*(1600*(np.sum(((point-anchor)*1000)**2)+200*(1-np.clip(normal@push,-1,1)))+6.25*np.sum(((point-previous)*1000)**2))
    for i, body in enumerate(d.style_body.numpy()):
        axis = d.style_axis.numpy()[i]
        previous_rotation = d.seed_rotation.numpy()[body]
        value += 400*(1-np.clip((rotations[body]@axis)@(previous_rotation@axis),-1,1))
        if i % 3 == 2:
            apex = d.style_apex.numpy()[i]
            delta = (rotations[body]@apex+positions[body]-previous_rotation@apex-d.seed_position.numpy()[body])*1000
            value += 9*np.sum(delta**2)
            normal = d.style_normal.numpy()[i]
            value += 3600*(1-np.clip((rotations[body]@normal)@(previous_rotation@normal),-1,1))
    delta = x-node.state.seed.numpy()
    value += 125e6*np.sum(delta[:3]**2)+100*np.sum(delta[3:6]**2)+.01*np.sum(delta[6:]**2)
    return value

def test_complete_pose_gradient_matches_independent_geometry_reference(tmp_path):
    node = make_node(tmp_path)
    e, s = node.explicit, node.state
    with wp.ScopedStream(node.stream):
        e.backward()
    wp.synchronize_stream(node.stream)
    x, gradient = s.x.numpy().copy(), s.x.grad.numpy()[0].copy()
    selected = e.selected.numpy()[0].copy()
    assert np.isfinite(gradient).all()
    base = reference(node, x[0].astype(float), selected)
    assert e.objective.numpy()[0] == pytest.approx(base, rel=2.e-4, abs=2.e-2)
    for start, stop, delta in [(0, 3, 1.e-6), (3, 6, 1.e-5), (6, 26, 1.e-5), (26, 46, 1.e-5)]:
        column = start+int(np.argmax(np.abs(gradient[start:stop])))
        assert abs(gradient[column]) > 1.e-5
        plus, minus = x[0].astype(float).copy(), x[0].astype(float).copy()
        plus[column] += delta; minus[column] -= delta
        fd = (reference(node, plus, selected)-reference(node, minus, selected))/(2*delta)
        assert gradient[column] == pytest.approx(fd, rel=.02, abs=2.e-3)


def test_refinement_scores_geometry_at_each_proposed_configuration(tmp_path, monkeypatch):
    node = make_node(tmp_path, horizon=1)
    original_score = node._score
    calls = []

    def checked_score(worlds=None):
        # Read only in this diagnostic test, outside CUDA graph replay.
        wp.synchronize_stream(node.stream)
        count = node.state.population if worlds is None else worlds
        x = node.state.x.numpy()[:count]
        q = node.state.qpos.numpy()[:count, node.runtime.index.hand_qpos]
        columns = np.r_[0:3, 6:26]
        np.testing.assert_allclose(q[:, columns], x[:, columns], atol=1.e-7,
            err_msg='Geometric cost evaluated a different configuration than the pose predictor')
        calls.append(x[0].copy())
        original_score(worlds)

    monkeypatch.setattr(node, '_score', checked_score)
    with wp.ScopedStream(node.stream):
        node._compute()
    wp.synchronize_stream(node.stream)
    assert len(calls) >= node.iterations+node.explicit.settings['line_search_steps']+2
    proposals = np.array(calls[node.iterations+1:-1])
    assert np.max(np.abs(proposals-calls[node.iterations])) > 1.e-7


def test_explicit_nodes_are_isolated_and_capture_gradient_without_host_reads(tmp_path, monkeypatch):
    nodes = [make_node(tmp_path/f'n{i}') for i in range(2)]
    assert all(node.explicit.qpos[0].shape == (1, node.runtime.model.nq) for node in nodes)
    assert all(node.explicit.cost.shape == (1,) for node in nodes)
    assert nodes[0].explicit.qpos[1].ptr != nodes[1].explicit.qpos[1].ptr
    dag = CudaNodeGraph(nodes)
    def forbidden(*a, **kw):
        raise AssertionError('Host access during explicit-model graph replay')
    with monkeypatch.context() as patch:
        patch.setattr(wp.array, 'numpy', forbidden)
        patch.setattr(torch.Tensor, 'cpu', forbidden)
        patch.setattr(torch.Tensor, 'item', forbidden)
        patch.setattr(wp, 'synchronize_device', forbidden)
        patch.setattr(wp, 'synchronize_stream', forbidden)
        dag.launch()
    wp.synchronize_stream(dag.stream)
    first = [node.state.qpos.numpy().copy() for node in nodes]
    for node in nodes:
        plan, snapshot, output = node.export()
        report = node.runtime.optimizer.last_solver_diag
        prediction = report['explicit_prediction']
        assert report['gradient_source'] == 'explicit_model_reverse_AD'
        assert prediction['pose_gradient_finite']
        assert report['prediction_horizon_seconds'] == pytest.approx(.2)
        assert 'object_pose' in report['cost_groups']
        assert report['optimizer_success'] == bool(np.isfinite(node.state.best_score.numpy()[0])
            and np.isfinite(node.state.scores.numpy()[0]))
        assert plan.loss == pytest.approx(sum(report['cost_groups'].values()))
        assert snapshot['explicit_predicted_qpos'].shape == (3, node.runtime.model.nq)
        assert snapshot['explicit_pose_gradient_valid']
        assert snapshot['explicit_pose_gradient'].shape == (46,)
        assert snapshot['explicit_preload_offsets'].shape == (20,)
        np.testing.assert_allclose(output[node.runtime.index.object_qpos:], node.runtime.qpos[node.runtime.index.object_qpos:])
        assert prediction['predicted_displacement_mm'] > 0
        history = [v for v in prediction['refinement_best_cost_history'] if v is not None]
        assert np.all(np.diff(history) <= 1.e-5)
    dag.launch(); wp.synchronize_stream(dag.stream)
    for node, initial in zip(nodes, first):
        np.testing.assert_allclose(node.state.qpos.numpy(), initial, atol=2.e-6)
    dag.close()


def test_export_rejects_invalid_final_geometry_even_with_finite_historical_score(tmp_path):
    node = make_node(tmp_path, horizon=1, refinement=0)
    dag = CudaNodeGraph([node])
    dag.launch()
    wp.synchronize_stream(dag.stream)
    node.state.best_score.fill_(1.)
    node.state.scores.fill_(float('inf'))
    wp.synchronize_device()
    node.export()
    report = node.runtime.optimizer.last_solver_diag
    assert not report['optimizer_success']
    assert not report['hard_constraints_satisfied']
    assert report['termination'] == 'final_configuration_failed_hard_constraints'
    dag.close()

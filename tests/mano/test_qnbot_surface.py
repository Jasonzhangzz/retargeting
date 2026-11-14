"""Independent native MuJoCo parity, geometry gradients and material binding."""
import copy
import numpy as np
import pytest
import torch
import mujoco
from moxian_web.qnbot_surface import QnbotSurface
from moxian_web.contact import default_mapping


@pytest.fixture(params=['left','right'])
def model(request):
    torch.set_num_threads(1)
    return QnbotSurface(request.param)


def test_native_forward_matches_independent_mujoco_random_poses(model):
    m=model;generator=np.random.default_rng(87);q=generator.uniform(m.limits[:,0].numpy(),m.limits[:,1].numpy(),(4,25)).astype(np.float32);q[0]=0
    with torch.no_grad():v,j,r=m(torch.tensor(q))
    data=mujoco.MjData(m.mj_model);scene=mujoco.MjvScene(m.mj_model,maxgeom=3000);opt=mujoco.MjvOption();cam=mujoco.MjvCamera()
    for k,angles in enumerate(q):
        data.qpos[:]=angles;mujoco.mj_forward(m.mj_model,data);mujoco.mjv_updateScene(m.mj_model,data,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL,scene)
        wr=data.xmat[m.wrist_body].reshape(3,3);wp=data.xpos[m.wrist_body]
        np.testing.assert_allclose(v[k],(scene.skinvert.reshape(-1,3)-wp)@wr,atol=2e-7,rtol=1e-5)
        np.testing.assert_allclose(j[k],(data.xpos-wp)@wr,atol=2e-7,rtol=1e-5)
    assert len(m.vertices0)==11598 and len(m.faces)==23184


def test_updated_inverse_binds_keep_skin_continuous_and_gradients_match_fd(model):
    m=model;q=torch.zeros(1,25);q[0,7]=.3
    with torch.no_grad():m.geometry['length'][5]=.08;m.geometry['root'][1,0]=.1
    ids=torch.tensor([100,500,5000]);v=m(q,vertex_ids=ids)[0];loss=v.square().sum();loss.backward()
    analytic=float(m.geometry['length'].grad[5]);eps=2e-3
    with torch.no_grad():
        m.geometry['length'][5]+=eps;plus=float(m(q,vertex_ids=ids)[0].square().sum());m.geometry['length'][5]-=2*eps;minus=float(m(q,vertex_ids=ids)[0].square().sum());m.geometry['length'][5]+=eps
    assert analytic==pytest.approx((plus-minus)/(2*eps),rel=.04,abs=2e-6)
    assert torch.isfinite(m.geometry['residual'].grad).all()
    with torch.no_grad():
        full=m(q)[0];subset=m(q,vertex_ids=ids)[0]
    torch.testing.assert_close(full[:,ids],subset)


def test_material_regions_keep_all_sids_and_reject_old_weights(model):
    m=model;m.bind_mapping(dict(assignments=default_mapping(m.side),confirmed=[]));topo=m.topology()
    assert len(m.regions)==24 and len(topo['sid_face_ids'])==460
    assert len(set(topo['sid_patch_indices']))==20
    for sid,face in enumerate(m.sid_faces):
        from moxian_web.contact import PATCH_IDS
        assert face in m.regions[PATCH_IDS[topo['sid_patch_indices'][sid]]]
    np.testing.assert_allclose(m.sid_bc.sum(1),1,atol=1e-6)
    artifact=m.artifact();other=QnbotSurface(m.side);other.load_artifact(artifact)
    assert other.version()==m.version()
    bad=copy.deepcopy(artifact);bad['surface_backend']='mano'
    with pytest.raises(ValueError,match='不兼容'):other.load_artifact(bad)


def test_frozen_rest_cache_invalidates_on_artifact_load(model):
    m=model;m.fixed=True
    with torch.no_grad():before=m(m.q0[None])[0].clone()
    artifact=m.artifact();artifact['geometry']['residual'][0][0]=.001;m.load_artifact(artifact)
    with torch.no_grad():after=m(m.q0[None])[0]
    assert float((after-before).abs().max())>.0005


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_original_triangle_collision_query_and_differentiable_witnesses():
    from moxian_web.qnbot_collision import QnbotCollision
    m=QnbotSurface('right','cuda:0');c=QnbotCollision(m,dense=True)
    v=m(m.q0[None])[0][0];w=c.query(v.detach());res=c.residual(v,w)
    assert w['samples']==11598+23184 and w['max_penetration_mm']<.1
    assert torch.isfinite(res).all()
    if len(res):res.square().sum().backward();assert torch.isfinite(m.geometry['residual'].grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_batched_dense_collision_matches_single_world_and_graph_replay():
    from moxian_web.qnbot_collision import QnbotCollision,QnbotBatchCollision
    m=QnbotSurface('right','cuda:0');torch.manual_seed(34)
    with torch.no_grad():
        q=torch.rand(4,25,device=m.device)*.6;vertices=m(q)[0];single=QnbotCollision(m,dense=True)
        expected=torch.tensor([single.query(v)['max_penetration_mm'] for v in vertices],device=m.device)
        batch=QnbotBatchCollision(m,batch=4);actual=batch.query(vertices)
        torch.testing.assert_close(expected,actual,atol=.005,rtol=.005)
        torch.testing.assert_close(actual,batch.query(vertices),atol=.001,rtol=.001)

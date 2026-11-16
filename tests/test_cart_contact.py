import json
import numpy as np
import pytest
import trimesh
from moxian_web.cart_protocol import STAGES,ACTION_COUNT,NEUTRAL_INDEX,PROTOCOL,MotionTracker,features
from moxian_web.cart_geometry import hull_data,FreeSpace,transform_hull,triangle_intersects_interior

def test_protocol_has_five_categories_and_eight_actions():
    assert [sum(s['category']==i for s in STAGES) for i in range(1,6)]==[4,1,1,1,1]
    assert len({s['id'] for s in STAGES})==8
    assert ACTION_COUNT==9 and NEUTRAL_INDEX==8
    assert all(s['cycles']==2 for s in STAGES)
    assert STAGES[-2]['pairs']==[[f+'_distal',f+'_proximal'] for f in ('index','middle','ring','pinky')]
    assert len(STAGES[-1]['patches'])==8


def test_cycles_require_real_close_release_and_breaks_do_not_manufacture_cycles():
    t=MotionTracker()
    for i,gap in enumerate([.02,.015,.006,.001,.001,.008,.02,.015,.006,.001,.008,.02]):
        result=t.update(i*20,gap)
    assert result['cycles']==2 and not result['closed']
    t.reset();t.update(0,.02);t.update(20,.001);t.update(1000,.02)
    assert t.cycles==0 and not t.closed

def test_current_features_and_sequence_split():
    from moxian_web.cart_data import sequence_indices
    p=np.random.default_rng(9).normal(size=(21,3))*.03
    assert features(p,.01).shape==(84,)
    rows=[dict(timestamp_ms=i*10) for i in range(18)]+[dict(timestamp_ms=1000+i*10) for i in range(18)]
    windows=sequence_indices(rows)
    assert len(windows)==2
    assert all(max(ids)-min(ids)==15 for ids in windows)
    assert not any(min(ids)<18<=max(ids) for ids in windows)

def test_rigid_object_has_no_scale_and_planes_transform_with_vertices():
    from scipy.spatial.transform import Rotation
    p=np.random.default_rng(82).normal(size=(20,3))*.02
    original=hull_data(p);rot=Rotation.from_rotvec([.5,.4,.2]).as_matrix();t=np.array([.1,.2,.3])
    result=transform_hull(original,rot,t)
    assert result['volume_m3']==original['volume_m3']
    v=np.asarray(result['vertices_m']);planes=np.asarray(result['planes'])
    assert (v@planes[:,:3].T+planes[:,3]).max()<1e-10
    assert np.linalg.norm(v[1]-v[0])==pytest.approx(np.linalg.norm(np.asarray(original['vertices_m'])[1]-original['vertices_m'][0]))

def test_collision_checks_triangle_interiors_not_only_hull_vertices():
    cube=trimesh.creation.box(extents=[.1,.1,.1]);h=hull_data(cube.vertices)
    tri=np.array([[-.2,0,0],[.2,0,0],[0,.2,0]])
    assert not ((tri@np.asarray(h['planes'])[:,:3].T+np.asarray(h['planes'])[:,3])<=0).all(1).any()
    assert triangle_intersects_interior(tri,np.asarray(h['planes']))
    assert not triangle_intersects_interior(tri+[0,0,.2],np.asarray(h['planes']))
    obstacle=trimesh.creation.box(extents=[.2,.005,.005])
    field=FreeSpace(obstacle.vertices,obstacle.faces)
    assert not field.clear(h,[0,.03,0])
    outside=hull_data(cube.vertices+[0,.2,0]);assert field.clear(outside,[0,.2,0])

def test_old_unlabelled_recording_is_not_cart_training(tmp_path):
    from moxian_web.cart_data import manifest
    folder=tmp_path/'recordings/old';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps(dict(complete=True,frames=100)))
    with pytest.raises(ValueError,match='分类协议'):manifest(tmp_path,'old')

def test_network_is_causal_bounded_and_has_rigid_rotation():
    import torch
    from moxian_web.cart_network import CartNetwork
    torch.manual_seed(714);model=CartNetwork().eval();x=torch.randn(2,16,84);h=torch.zeros(2,16,30)
    with torch.no_grad():
        a=model(x,h);changed=x.clone();changed[:,8:]+=100;b=model(changed,h)
    torch.testing.assert_close(a['q'][:,:8],b['q'][:,:8])
    assert a['action'].shape==(2,16,ACTION_COUNT)
    assert a['q'].abs().max()<=1
    r=a['rotation'];torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=1e-5,rtol=1e-5)
    assert torch.linalg.det(r).min()>.999

@pytest.fixture(scope='module')
def native():
    from moxian_web.cart_losses import NativeHand
    return NativeHand('cpu')

def test_native_fk_matches_mujoco_and_gradient(native):
    import torch
    h=native;rng=np.random.default_rng(17);q=rng.uniform(h.robot.lower,h.robot.upper,(12,20))
    x=h.tensor(q).requires_grad_();p,r=h.fk.frames(torch.cat((x.new_zeros((12,6)),x),1))
    tips=[]
    for name in ('thumb','index','middle','ring','pinky'):
        s=h.robot.model.site('right_'+name+'_tip');b=int(s.bodyid[0]);tips.append(p[:,b]+r[:,b]@h.tensor(s.pos))
    tip=(torch.stack(tips,1)-p[:,h.ids[0],None])@h.basis
    expected=np.stack([h.robot.fk(a) for a in q])
    np.testing.assert_allclose(tip.detach().numpy(),expected,atol=1e-7)
    tip.square().sum().backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_surface_loss_retains_contact_and_rejects_wrong_patch(native):
    import torch
    h=native;q=((h.low+h.high)/2)[None].requires_grad_();p,r=h.forward(q)
    distance,normal,*_=h.pair(p[0],r[0],'thumb_distal','index_distal')
    assert distance>0
    loss=distance/.002**2+normal;loss.backward();assert torch.isfinite(q.grad).all()
    with pytest.raises(KeyError):h.pair(p[0],r[0],'thumb_distal','unlabelled_target')

def test_all_mesh_penetration_loss_has_gradients(native):
    import torch
    h=native;q=(h.low+(h.high-h.low)*.9)[None].requires_grad_();p,r=h.forward(q)
    loss,depth=h.self_penetration(p,r);loss.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(depth).all()

def test_object_penetration_and_every_required_patch_are_penalized(native):
    import torch
    from moxian_web.cart_losses import contact_losses
    h=native;q=((h.low+h.high)/2)[None].requires_grad_()
    target=hull_data(trimesh.creation.box(extents=[.3,.3,.3]).vertices+[0,0,.05])
    target.update(contact_mode='hull',contacts=[dict(patch=p,object_point_m=[0,0,.2]) for p in ('thumb_distal','index_distal')])
    joints=np.random.default_rng(41).normal(size=(21,3))*.03
    row=dict(stage='precision_grasp',phase='hold',joints_m=joints.tolist(),target=target)
    t=torch.zeros(1,3,requires_grad=True);r=torch.eye(3)[None].requires_grad_()
    parts,quality=contact_losses(h,q,t,r,[row]);sum(parts.values()).backward()
    assert parts['penetration']>0 and quality[0]['required']==2 and quality[0]['object_penetration_mm']>0
    assert torch.isfinite(q.grad).all() and torch.isfinite(t.grad).all() and torch.isfinite(r.grad).all()

def test_prismatic_opposition_loss_differentiates_hand_patch_normals(native):
    import torch
    from moxian_web.cart_losses import contact_losses
    from moxian_web.cart_protocol import stage
    h=native;q=((h.low+h.high)/2)[None].requires_grad_()
    target=hull_data(trimesh.creation.box(extents=[.03,.08,.04]).vertices+[.03,0,.07])
    target.update(contact_mode='hull',contacts=[dict(patch=p,object_point_m=[.03,0,.07]) for p in stage('prismatic')['patches']])
    joints=np.random.default_rng(84).normal(size=(21,3))*.03
    row=dict(stage='prismatic',phase='hold',joints_m=joints.tolist(),target=target)
    parts,quality=contact_losses(h,q,torch.zeros(1,3),torch.eye(3)[None],[row])
    parts['wrap'].backward()
    assert quality[0]['required']==8 and torch.isfinite(q.grad).all() and q.grad.abs().sum()>0

@pytest.mark.parametrize('use_hull',[False,True])
def test_smoke_training_saves_checkpoints_and_cannot_be_activated(tmp_path,monkeypatch,use_hull):
    import hashlib,torch
    from moxian_web.cart_data import input_digest,load_sessions
    from moxian_web.cart_protocol import PROTOCOL
    from moxian_web.cart_training import train,Predictor
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    action='precision_grasp' if use_hull else 'pinch_index'
    index=4 if use_hull else 0
    category=2 if use_hull else 1
    cached={};rebuilt=[]
    for split in ('train','validation'):
        folder=tmp_path/'recordings'/split;folder.mkdir(parents=True)
        np.save(folder/'points.npy',np.zeros((16,5,3),np.float32));np.save(folder/'timestamps.npy',np.arange(16)*16.)
        np.savez(folder/'motion.npz',qpos=np.zeros((16,29)),joints=np.zeros((16,35,3)))
        for f in ('profile.json','topology.json'):(folder/f).write_text('{}')
        m=dict(id=split,protocol=PROTOCOL,source={'synthetic_test_only':True},stage=action,category=category,split=split,
               session_id=split,complete=True,frames=16,points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest())
        (folder/'manifest.json').write_text(json.dumps(m))
        rows=[]
        j=np.random.default_rng(10).normal(size=(21,3))*.04
        target=dict(ready=False,contact_mode='hand',pairs=[dict(regions=['thumb_distal','index_distal'],gap_m=.02)])
        if use_hull:
            target=hull_data(trimesh.creation.box(extents=[.018,.018,.12]).vertices+[.02,.02,.06])
            target.update(contact_mode='hull',initialization=dict(kind='box',center_m=[.02,.02,.06],rotation=np.eye(3).tolist()))
            target['contacts']=[dict(patch=p,object_point_m=point) for p,point in
                zip((f+'_distal' for f in ('thumb','index','middle','ring','pinky')),
                    ([.011,.02,.06],[.029,.02,.06],[.02,.029,.06],[.02,.011,.06],[.02,.02,.12]))]
        for i in range(16):
            rows.append(dict(stage=action,stage_index=index,category=category,phase='hold' if use_hull else 'open',timestamp_ms=i*16.,
                features=features(j,.016).tolist(),joints_m=j.tolist(),target=target))
        path=folder/'cart_targets.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        info=dict(profile='smoke_test',complete_cycles=False,
            fingerprint=input_digest(folder,m,False),targets_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        cached[split]=info
        (folder/'cart_preprocess.json').write_text(json.dumps({**info,'fingerprint':'old_convex_targets'} if use_hull else info))
    if use_hull:
        # Synthetic raw joints cannot be reconstructed as a real source hand;
        # stand in only for reconstruction while exercising stale-cache routing.
        def rebuild(root,key,**kwargs):
            rebuilt.append(key)
            (root/'recordings'/key/'cart_preprocess.json').write_text(json.dumps(cached[key]))
        monkeypatch.setattr('moxian_web.cart_training.preprocess',rebuild)
    folder=train(tmp_path,['train'],['validation'],'smoke',epochs=1,batch_size=2,smoke=True)
    assert (folder/'epoch_001.pth').is_file()
    assert json.loads((folder/'status.json').read_text())['status']=='smoke_test'
    assert rebuilt==(['train','validation'] if use_hull else [])
    checkpoint=torch.load(folder/'last.pth',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
    moments=[value['exp_avg'] for value in checkpoint['optimizer']['state'].values() if 'exp_avg' in value]
    assert moments and all(torch.isfinite(value).all() for value in moments)
    assert any(torch.count_nonzero(value) for value in moments)
    with pytest.raises(ValueError,match='完整训练'):Predictor(folder)
    (tmp_path/'recordings/train/cart_targets.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='指纹'):load_sessions(tmp_path,['train'],'train',require_complete=False)

def test_five_distal_pads_produce_a_fitted_primitive_with_contact_delta():
    from moxian_web.cart_geometry import estimate
    from tests.test_cart_primitives import pads
    names=[f+'_distal' for f in ('thumb','index','middle','ring','pinky')]
    mesh,regions=pads(names,[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',full=True)
    assert result['ready'] and result['dimension']==3 and result['volume_m3']>0
    assert result['contact_mode']=='hull' and not result['free_convex_shape']
    assert result['primitive']['reference_type']==13
    assert result['coverage']==1
    assert FreeSpace(mesh.vertices,mesh.faces).clear(result,result['center_m'])


def test_live_source_preview_keeps_unbatched_original_mesh(monkeypatch):
    import torch
    from types import SimpleNamespace
    from moxian_web.cart_data import SourceGeometry
    import moxian_web.qnbot_link_contact as module
    class Query:
        def __init__(self,*args):pass
        def witnesses(self,v):
            return dict(triangles=torch.tensor([[[0,1,2],[3,4,5]]]),
                        barycentric=torch.tensor([[[1/3]*3,[1/3]*3]]))
    monkeypatch.setattr(module,'LinkContactQuery',Query)
    source=SourceGeometry.__new__(SourceGeometry)
    source.adapter=SimpleNamespace(indices=list(range(21)),transform=np.eye(3))
    source.topology={'geometry_version':'test'};source.faces=np.array([[0,1,2],[3,4,5]])
    source.regions={'thumb_distal':[0],'index_distal':[1]}
    v=np.random.default_rng(73).normal(size=(6,3))*.03;j=np.random.default_rng(74).normal(size=(21,3))*.03
    _,preview,preview_joints=source.frame(None,'pinch_index',1000,MotionTracker(),raw=(v,j))
    np.testing.assert_array_equal(preview,v);np.testing.assert_array_equal(preview_joints,j)
    assert preview.shape==(6,3)

def test_cart_runtime_uses_direct_network_output_and_matching_fk(native,monkeypatch):
    import torch
    from types import SimpleNamespace
    import moxian_web.cart_runtime as runtime
    from retarget.baseline.geort.adapter import SourceAdapter
    h=native
    topology=dict(surface_backend='qnbot',coordinate_frame='wrist_local',hand_side='right',geometry_version='g',
        binding_version='b',model_hash='model',palm_view=dict(normal=[1,0,0],forward=[0,0,1]),
        joint_chains=[[0,1+4*f,2+4*f,3+4*f,4+4*f] for f in range(5)],faces=[[0,1,2]],regions={})
    frame=dict(calibrated=True,preview=False,hand_side='right',geometry_version='g',binding_version='b',model_hash='model',
        calibration_fingerprint='calibration',skeleton_runtime_revision='source',serial='test',timestamp_ms=1000.)
    identity=SourceAdapter(topology).identity('test',frame);angle=(h.low+h.high)/2
    class FakePredictor:
        def __init__(self,*args):self.hand=h;self.config={'source':identity};self.last_stamp=None
        def reset(self):self.last_stamp=None
        def encode(self,row):
            self.last_stamp=row['timestamp_ms'];logits=torch.zeros(1,1,ACTION_COUNT);logits[0,0,0]=10
            return None,logits,None
        def decode(self,*args):return angle,torch.zeros(3),torch.eye(3)
    monkeypatch.setattr(runtime,'Predictor',FakePredictor)
    j=np.random.default_rng(97).normal(size=(21,3))*.04
    target=dict(ready=False,contact_mode='hand',pairs=[dict(regions=['thumb_distal','index_distal'],gap_m=.04,points_m=[[0,0,0],[.04,0,0]])])
    row=dict(stage='pinch_index',category=1,phase='open',joints_m=j.tolist(),target=target)
    monkeypatch.setattr(runtime.SourceGeometry,'frame',lambda self,*a,**k:(row,k['raw'][0],k['raw'][1]))
    solver=runtime.CartRetarget(topology,model_path='test_fixture_only')
    result=solver.step(dict(frame,joints_m=j.tolist(),vertices_m=[[0,0,0],[.01,0,0],[0,.01,0]]))
    np.testing.assert_allclose(result['joint_positions_rad'],angle.numpy())
    assert result['algorithm']==PROTOCOL and result['virtual_only']
    assert result['timestamp_ms']==result['cart_hull']['timestamp_ms']==1000
    assert result['cart_robot_hull']['contact_points_m']

def test_orphan_capture_restores_only_complete_raw_frames(tmp_path):
    from moxian_web.cart_protocol import PROTOCOL
    from moxian_web.cart_service import CartService
    folder=tmp_path/'recordings/orphan';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,complete=False)))
    rows=[dict(points=np.zeros((5,3)).tolist(),timestamp_ms=i*10.,qpos=[0.]*29,joints_m=np.zeros((21,3)).tolist()) for i in range(2)]
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"torn":')
    service=CartService(tmp_path)
    try:
        meta=json.loads((folder/'manifest.json').read_text());assert meta['complete'] and meta['frames']==2 and meta['recovered']
        assert len((folder/'frames.jsonl').read_text().splitlines())==2
        assert np.load(folder/'motion.npz')['qpos'].shape==(2,29)
        assert not service.recorder.recording['active']
    finally:service.close()

def test_cli_and_web_cannot_preprocess_same_session_concurrently(tmp_path):
    import fcntl
    from moxian_web.cart_protocol import PROTOCOL
    from moxian_web.cart_data import preprocess
    folder=tmp_path/'recordings/busy';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,complete=True,frames=2,stage='pinch_index')))
    with (folder/'cart_preprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='已有预处理任务'):preprocess(tmp_path,'busy')


def test_preview_transforms_hand_witnesses_and_invalid_hull_candidate():
    from scipy.spatial.transform import Rotation
    rot=Rotation.from_rotvec([.2,.3,.4]).as_matrix();shift=np.array([.1,.2,.3])
    candidate=hull_data(trimesh.creation.box(extents=[.01,.02,.03]).vertices)
    target=dict(ready=False,contact_mode='unavailable',diagnostic_candidate=candidate,
        pairs=[dict(points_m=[[0,0,0],[.01,0,0]],normals=[[1,0,0],[-1,0,0]])])
    result=transform_hull(target,rot,shift)
    assert not result['ready']
    np.testing.assert_allclose(result['pairs'][0]['points_m'],np.asarray(target['pairs'][0]['points_m'])@rot.T+shift)
    np.testing.assert_allclose(result['pairs'][0]['normals'],np.asarray(target['pairs'][0]['normals'])@rot.T)
    np.testing.assert_allclose(result['diagnostic_candidate']['vertices_m'],np.asarray(candidate['vertices_m'])@rot.T+shift)
    assert result['diagnostic_candidate']['volume_m3']==candidate['volume_m3']
    assert target['pairs'][0]['points_m'][0]==[0,0,0]

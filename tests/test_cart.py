"""Behavioral checks for contact evidence, causal inference and offline tubes."""
import json
import numpy as np
import pytest
import torch

from moxian_web.cart_contacts import taxel_evidence,transform_contacts,link_owner,overlay
from moxian_web.cart_network import LegacyHistoryNetwork as CartNetwork,FiniteScalarQuantizer,CausalBlock,JointHistory
from moxian_web.cart_tube import TubeBuilder,TubeSettings,path_search,interpolate_row,signature
from moxian_web.cart_data import sequence_indices
from retarget.baseline.card.math import CardSettings,cem,joint_elite_cem


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def empty_row(stamp=0.):
    return dict(timestamp_ms=stamp,joints_m=np.zeros((21,3)).tolist(),
        contacts=dict(tactile_status='unknown',tactile_ready=False,self_pairs=[],object_contacts=[]))


def test_sensor_unknown_and_palm_exclusion():
    patches=np.zeros(460,int);patches[0]=14
    frame=dict(timestamp_ms=100.,tactile_timestamp_ms=100.,source_contact=dict(ready=True,timestamp_ms=100.,
        cell_contact=[True]*460,normalized=[.1]*460))
    valid,mask,values=taxel_evidence(frame,patches)
    assert valid and mask.sum()==459 and not mask[0]
    frame['tactile_timestamp_ms']=74.
    valid,mask,_=taxel_evidence(frame,patches)
    assert not valid and not mask.any()
    frame['tactile_timestamp_ms']=100.;frame['source_contact']['timestamp_ms']=74.
    assert not taxel_evidence(frame,patches)[0]
    frame['source_contact']['timestamp_ms']=100.;frame['source_contact']['cell_contact']=[False]*460
    valid,mask,_=taxel_evidence(frame,patches)
    assert valid and not mask.any()


def test_material_reflection_round_trip_and_visual_only_cloud():
    c=empty_row()['contacts'];c['object_contacts']=[dict(region='thumb_distal',position_m=[1.,2.,3.],normal=[0.,1.,0.],force=.1)]
    c['self_pairs']=[dict(regions=['index_distal','thumb_distal'],points_m=[[1.,0.,0.],[2.,0.,0.]],normals=[[1.,0.,0.],[-1.,0.,0.]])]
    transform=np.diag([-1.,1.,1.]);root=np.array([.1,.2,.3])
    changed=transform_contacts(c,transform,root);back=transform_contacts(changed,transform,root,True)
    np.testing.assert_allclose(back['object_contacts'][0]['position_m'],c['object_contacts'][0]['position_m'])
    np.testing.assert_allclose(back['self_pairs'][0]['normals'],c['self_pairs'][0]['normals'])
    shown=overlay(changed);assert len(shown['contact_points_m'])==3 and len(shown['patch_points_m'])==72
    assert 'patch_points_m' not in changed
    assert link_owner('thumb_metacarpal')==2 and link_owner('thumb_proximal')==3 and link_owner('index_distal')==8


def test_fsq_round_trip_extremes_and_gradient():
    fsq=FiniteScalarQuantizer();z=(torch.randn(128,8)*3).requires_grad_()
    codes,indices=fsq(z);assert indices.min()>=0 and indices.max()<2560000
    torch.testing.assert_close(fsq.decode_indices(indices),codes.detach())
    codes.sum().backward();assert z.grad.isfinite().all() and z.grad.abs().sum()>0
    extremes,_=fsq(torch.tensor([[-100.]*8,[100.]*8]))
    torch.testing.assert_close(extremes[0],torch.tensor([-1.]*8))
    assert extremes[1,0]==.75 and extremes[1,-1]==1


def test_tcn_is_causal_and_network_has_fsq_gradients():
    layer=CausalBlock(22,64,2).eval();x=torch.randn(2,22,10);changed=x.clone();changed[:,:,7:]+=100
    torch.testing.assert_close(layer(x)[:,:,:7],layer(changed)[:,:,:7])
    model=CartNetwork(np.zeros(20),np.ones(20));points=torch.randn(2,5,3);history=torch.rand(2,10,20)
    out=model(points,history,return_codes=True)
    out['q'].sum().backward()
    assert model.encoder[-1].weight.grad.isfinite().all() and model.encoder[-1].weight.grad.abs().sum()>0
    assert out['q'].shape==(2,20) and (out['q']>=0).all() and (out['q']<=1).all()
    mask=torch.zeros(2,10,dtype=torch.bool)
    torch.testing.assert_close(model(points,history,mask),model(points,history+1000,mask))
    with pytest.raises(ValueError):model(points,torch.zeros(2,11,20))


def test_history_precedes_current_and_resets_boundaries():
    h=JointHistory();assert not h.input(0)[1].any()
    h.append(0,np.ones(20));q,mask,dt=h.input(10,0)
    assert mask.sum()==1 and dt[-1]==pytest.approx(.01)
    assert not h.input(300,0)[1].any()
    h.append(300,np.ones(20));assert not h.input(301,1)[1].any()
    h.append(301,np.ones(20));assert not h.input(301,1)[1].any()
    h.append(301,np.ones(20));h.reset();assert not h.input(302)[1].any()


def test_card_optional_archive_preserves_rng_output_and_no_nonfinite_elites():
    low=torch.full((20,),-1.);high=-low
    initial=(torch.zeros(3),torch.eye(3),torch.zeros(20))
    settings=CardSettings(population=16,iterations=2)
    def score(t,r,q):return q.square().sum(-1)+t.square().sum(-1)
    generator=torch.Generator().manual_seed(42)
    best,diagnostic=cem(initial,None,low,high,score,settings,generator)
    after=generator.get_state();archive={};generator.manual_seed(42)
    archived,other=cem(initial,None,low,high,score,settings,generator,elite_archive=archive)
    for a,b in zip(best,archived):torch.testing.assert_close(a,b)
    assert diagnostic==other and torch.equal(after,generator.get_state())
    assert torch.isfinite(archive['costs']).all() and archive['values'].shape[1]==26


def test_local_elites_stay_local_and_never_relabel_infeasible():
    settings=CardSettings(population=32,iterations=2,joint_step_rad=.1)
    centers=torch.tensor([[.5,.5],[-.5,-.5]])
    result=joint_elite_cem(centers,torch.full((2,),-1.),torch.ones(2),
        lambda q:(q.square().sum(-1),q[:,0]>0),settings,torch.Generator().manual_seed(1))
    assert len(result['values']) and (result['values'][:,0]>0).all()
    assert ((result['values'][:,None]-centers).abs().amax(-1).min(-1).values<=.100001).all()
    result=joint_elite_cem(centers,torch.full((2,),-1.),torch.ones(2),
        lambda q:(q.square().sum(-1),torch.zeros(len(q),dtype=torch.bool)),settings,torch.Generator().manual_seed(1))
    assert not len(result['values'])


def test_second_order_path_prefers_smooth_branch_and_reports_gaps():
    nodes=[np.array([[0.],[1.]]),np.array([[0.],[1.]]),np.array([[0.],[1.]])]
    edges=[[],[[0,0],[0,1],[1,0],[1,1]],[[0,0],[0,1],[1,0],[1,1]]]
    paths,gaps=path_search(nodes,[[0.,.1],[.1,0.],[0.,.1]],edges,[0.,20.,40.])
    assert len(paths)==1 and len({k for _,k in paths[0]})==1 and not gaps
    edges[2]=[];paths,gaps=path_search(nodes,[[0.,1.]]*3,edges,[0.,20.,40.])
    assert len(paths)==2 and gaps[0]['frame']==2


def test_interpolation_detects_collision_between_feasible_endpoints():
    class Hand:
        device='cpu'
        def tensor(self,x):return torch.as_tensor(x,dtype=torch.float32)
    class Objective:
        hand=Hand()
        def __call__(self,q,row,dense=False):
            assert dense
            return None,None,dict(feasible=q[:,0].abs()>.02)
    builder=TubeBuilder(Objective(),TubeSettings(neighbors=1,interpolation_step_rad=.01))
    a=np.full((1,20),-.05);b=np.full((1,20),.05)
    assert not builder.connections(a,b,empty_row(0),empty_row(20))
    assert builder.connections(a,a,empty_row(0),empty_row(20))==[[0,0]]
    changed=empty_row(20);changed['contacts']['tactile_status']='valid'
    assert not builder.connections(a,a,empty_row(0),changed)
    with pytest.raises(ValueError):interpolate_row(empty_row(),changed,.5)


def test_windows_never_bridge_sensor_segments_or_require_tube():
    rows=[dict(timestamp_ms=i*20,stream_epoch=i//30) for i in range(60)]
    windows=sequence_indices(rows,length=3)
    assert windows and all(len({rows[i]['stream_epoch'] for i in w})==1 for w in windows)
    assert all(len(w)>=13 for w in windows)
    assert not sequence_indices(rows[:12],length=3)


@pytest.fixture(scope='module')
def hand_objective():
    from moxian_web.cart_objective import CartHand,CartObjective
    hand=CartHand('cpu');return hand,CartObjective(hand)


def test_native_fk_full_hand_cost_gradient_and_zero_contact_batch(hand_objective):
    hand,objective=hand_objective;q=((hand.low+hand.high)/2)[None].requires_grad_()
    _,_,j,*_=hand.frames(q);row=empty_row();row['joints_m']=j[0].detach().tolist()
    loss,parts,quality=objective(q,row)
    assert parts['contact'].item()==0 and parts['normal'].item()==0 and quality['feasible'].item()
    loss.sum().backward();assert q.grad.isfinite().all()
    robot=hand.robot;robot.fk(q[0].detach().numpy())
    tips=torch.tensor(robot.fk(q[0].detach().numpy()))
    torch.testing.assert_close(j[0,[4,8,12,16,20]],tips,atol=1e-6,rtol=1e-5)


def test_object_contact_uses_frozen_observation_and_has_gradient(hand_objective):
    hand,objective=hand_objective
    q=((hand.low+hand.high)/2)[None];p,r,j,*_=hand.frames(q)
    mesh=hand.surface.contact_surface(8);tri=torch.tensor(mesh['triangles'][0],dtype=torch.float32)
    local=tri.mean(0);ln=torch.tensor(mesh['normals'][0],dtype=torch.float32)
    point=(r[0,8]@local+p[0,8]).tolist();normal=(r[0,8]@ln).tolist()
    row=empty_row();row['joints_m']=j[0].tolist()
    row['contacts']['object_contacts']=[dict(region='index_distal',position_m=point,normal=normal,force=1.)]
    _,parts,quality=objective(q,row);assert parts['contact'].item()<1e-7
    moved=q.clone();moved[0,6]+=.05;moved.requires_grad_()
    loss,parts,quality=objective(moved,row);assert parts['contact'].item()>0
    loss.sum().backward();assert moved.grad.isfinite().all() and moved.grad.abs().sum()>0


def test_full_hand_wuji_cost_and_gradient_match_reference(hand_objective):
    from retarget_reference.preprocess import prepare_targets
    from retarget_reference.objectives import FrameObjective
    hand,objective=hand_objective
    reference=(hand.low+hand.high)/2;q=(reference+.03).detach().requires_grad_()
    source=hand.frames(reference[None])[2][0].detach().numpy()
    j=hand.frames(q[None])[2][0]
    jac=torch.autograd.functional.jacobian(lambda x:hand.frames(x[None])[2][0],q)
    targets=prepare_targets(source,thumb_free=False)
    original=FrameObjective(source,targets['edges'],directions=targets['directions'],laplacian=targets['L'],
        differential=targets['D'],weights=targets['row_weights'],thumb_triplets=np.array([[1,2,3],[2,3,4]]),
        q_ref=reference.numpy(),antirev_indices=np.array([2,3,6,7,10,11,14,15,18,19]))
    row=empty_row();row['joints_m']=source.tolist()
    for stage in (1,2):
        _,parts,_=objective(q[None],row,reference[None],stage=stage)
        cost=sum(parts[k].sum() for k in ('direction','laplacian','smooth','antireverse','thumb_relative'))
        gradient=torch.autograd.grad(cost,q,retain_graph=True)[0]
        expected,expected_gradient,_=original.evaluate(q.detach().numpy(),j.detach().numpy(),jac.detach().numpy(),stage)
        assert cost.item()==pytest.approx(expected,abs=2e-6,rel=2e-5)
        np.testing.assert_allclose(gradient.detach().numpy(),expected_gradient,atol=3e-5,rtol=3e-4)


def test_recorder_keeps_same_frame_tactile_before_preview():
    import queue,threading
    from moxian_web.geort_service import GeoRTService
    from retarget.baseline.geort.adapter import SourceAdapter
    topology=dict(surface_backend='qnbot',coordinate_frame='wrist_local',hand_side='right',geometry_version='g',
        binding_version='b',model_hash='m',joint_chains=[[0]+list(range(1+f*4,5+f*4)) for f in range(5)],
        palm_view=dict(normal=[1,0,0],forward=[0,0,1]))
    frame=dict(ready=True,calibrated=True,timestamp_ms=100.,skeleton_timestamp_ms=100.,geometry_version='g',hand_side='right',
        model_hash='m',source_joint_positions_rad=[0.]*29,calibration_fingerprint='c',cart_label=dict(protocol='cart_contact_tube_v4'),
        source_contact=dict(ready=True,cell_contact=[True]*460,normalized=[.2]*460),tactile_timestamp_ms=99.)
    service=GeoRTService.__new__(GeoRTService);service.lock=threading.RLock();service.recording=dict(invalid_frames=0)
    service.capture=dict(source=SourceAdapter(topology).identity('serial',frame),last_stamp=0.,
        meta=dict(capture_mode='cart_v4'),queue=queue.Queue())
    service.observe(frame,topology,np.zeros((21,3)),'serial')
    item=service.capture['queue'].get_nowait();frame['source_contact']['normalized'][0]=99.
    assert item['source_contact']['normalized'][0]==.2 and item['tactile_timestamp_ms']==99.


def test_preview_mailbox_carries_synchronized_taxels_without_latest_frame_fusion(tmp_path):
    from moxian_web.cart_service import CartService
    service=CartService(tmp_path)
    try:
        service.preview_proc=object()
        service.preview_key=('g','b','m','c')
        cells=[False]*460;cells[400]=True
        frame=dict(ready=True,calibrated=True,timestamp_ms=100.,tactile_timestamp_ms=99.,
            geometry_version='g',binding_version='b',model_hash='m',calibration_fingerprint='c',
            source_joint_positions_rad=[0.]*29,
            source_contact=dict(ready=True,timestamp_ms=99.,cell_contact=cells,normalized=[.2]*460))
        service.observe(frame,{},np.zeros((21,3)),'fixture')
        queued=service.preview_queue.get_nowait()
        valid,mask,_=taxel_evidence(queued,np.zeros(460,int))
        assert valid and mask.sum()==1 and mask[400]
        assert queued['timestamp_ms']==100. and queued['tactile_timestamp_ms']==99.
        service.last_sent=0.
        frame=dict(frame,timestamp_ms=130.)
        service.observe(frame,{},np.zeros((21,3)),'fixture')
        assert not taxel_evidence(service.preview_queue.get_nowait(),np.zeros(460,int))[0]
    finally:
        service.preview_proc=None
        service.close()


def test_cart_v4_recording_writes_tactile_and_stops_on_binding_change(tmp_path,monkeypatch):
    import hashlib,time
    import moxian_web.geort_service as module
    import moxian_web.source_ik_initialization as initialization
    from moxian_web.cart_protocol import PROTOCOL
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(initialization,'validate_recorded_source',lambda profile,source:None)
    serial='fixture';key=hashlib.sha256(serial.encode()).hexdigest()[:16]
    profile=tmp_path/'output/moxian/hand_model'/key/'qnbot/current.json';profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps(dict(model_hash='m')))
    topology=dict(surface_backend='qnbot',coordinate_frame='wrist_local',hand_side='right',geometry_version='g',
        binding_version='b',model_hash='m',joint_chains=[[0]+list(range(1+f*4,5+f*4)) for f in range(5)],
        palm_view=dict(normal=[1,0,0],forward=[0,0,1]))
    def frame(stamp):
        return dict(ready=True,calibrated=True,timestamp_ms=stamp,skeleton_timestamp_ms=stamp,geometry_version='g',hand_side='right',
            model_hash='m',source_joint_positions_rad=[0.]*29,calibration_fingerprint='c',cart_label=dict(protocol=PROTOCOL),
            source_contact=dict(ready=True,timestamp_ms=stamp-1,cell_contact=[True]*460,normalized=[.2]*460),tactile_timestamp_ms=stamp-1)
    service=module.GeoRTService(tmp_path/'cart')
    try:
        service.observe(frame(100.),topology,np.zeros((21,3)),serial)
        record=service.start_recording('fixture',capture_mode='cart_v4',metadata=dict(protocol=PROTOCOL,split='train'))
        service.observe(frame(120.),topology,np.zeros((21,3)),serial)
        service.observe(frame(140.),topology,np.zeros((21,3)),serial)
        changed=dict(topology,binding_version='new')
        service.observe(frame(160.),changed,np.zeros((21,3)),serial)
        deadline=time.monotonic()+5
        while service.recording['saving'] and time.monotonic()<deadline:time.sleep(.01)
        folder=tmp_path/'cart/recordings'/record['id'];meta=json.loads((folder/'manifest.json').read_text())
        rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        assert meta['complete'] and meta['frames']==2 and '绑定' in meta['stop_reason']
        assert [r['tactile_timestamp_ms'] for r in rows]==[119.,139.]
        assert rows[0]['source_contact']['cell_contact']==[True]*460
        assert np.load(folder/'motion.npz')['qpos'].shape==(2,29)
    finally:service.close()


def test_motion_summary_uses_actual_intervals_and_session_boundaries():
    from moxian_web.cart_training import motion_summary
    metrics=motion_summary([(0.,[0.]*20,'a'),(20.,[1.]*20,'a'),(60.,[3.]*20,'a'),(80.,[100.]*20,'b')])
    assert metrics['joint_velocity_rms_rad_s']==pytest.approx(50.)
    assert metrics['joint_acceleration_rms_rad_s2']==pytest.approx(0.)


def test_mesh_penetration_and_tactile_are_independent_material_evidence():
    import trimesh
    from moxian_web.cart_contacts import SourceContacts
    meshes=[]
    for x in (-.1,0.,.015):
        mesh=trimesh.creation.box(extents=(.02,.02,.02));mesh.apply_translation((x,0,0));meshes.append(mesh)
    vertices=np.vstack([m.vertices for m in meshes]);faces=np.vstack([m.faces+8*i for i,m in enumerate(meshes)])
    class Model:
        bone_names=['right_wrist','right_thumb_distal','right_index_distal']
        weights=torch.nn.functional.one_hot(torch.arange(3).repeat_interleave(8),3).float()
        q0=torch.zeros(1)
        def __call__(self,q):return torch.tensor(vertices,dtype=torch.float32)[None],None,None
    sid_faces=[12]*460;sid_faces[1]=24;sid_faces[-1]=0
    patches=[0]*460;patches[1]=2;patches[-1]=14
    topology=dict(faces=faces.tolist(),surface_policy='qnbot_original_triangles',sid_face_ids=sid_faces,
                  sid_barycentric=[[1/3]*3]*460,sid_patch_indices=patches)
    detector=SourceContacts(topology,Model());joints=np.zeros((21,3))
    separate=vertices.copy();separate[16:,0]+=.1
    assert not detector.detect(separate,joints,dict(timestamp_ms=100.))['self_pairs']
    evidence=detector.detect(vertices,joints,dict(timestamp_ms=100.))
    assert len(evidence['self_pairs'])==1 and not evidence['object_contacts'] and not evidence['tactile_ready']
    cells=[False]*460
    for i in (0,1,459):cells[i]=True
    frame=dict(timestamp_ms=100.,tactile_timestamp_ms=100.,source_contact=dict(ready=True,timestamp_ms=100.,cell_contact=cells,normalized=[.2]*460))
    evidence=detector.detect(vertices,joints,frame)
    assert len(evidence['self_pairs'])==1 and len(evidence['object_contacts'])==2
    assert all(not c['region'].startswith('palm') for c in evidence['object_contacts'])
    for c in evidence['object_contacts']:
        np.testing.assert_allclose(c['position_m'],np.asarray(c['barycentric'])@vertices[faces[c['source_face']]],atol=1e-8)
    frame['tactile_timestamp_ms']=70.
    stale=detector.detect(vertices,joints,frame)
    assert stale['self_pairs'] and not stale['object_contacts'] and stale['tactile_status']=='unknown'


def test_old_and_smoke_weights_cannot_replace_active_model(tmp_path):
    from moxian_web.cart_service import CartService
    (tmp_path/'active.json').write_text(json.dumps(dict(model='keep')))
    for name,status in (('old','completed'),('smoke','smoke_test')):
        folder=tmp_path/'models'/name;folder.mkdir(parents=True)
        (folder/'status.json').write_text(json.dumps(dict(status=status)))
        (folder/'config.json').write_text(json.dumps(dict(protocol='cart_contact_v3')))
    service=CartService(tmp_path)
    try:
        for name in ('old','smoke'):
            with pytest.raises(ValueError):service.activate(name)
        assert service.selected=='keep' and json.loads((tmp_path/'active.json').read_text())['model']=='keep'
    finally:service.close()

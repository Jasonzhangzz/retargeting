"""Synthetic relabelled fixtures exercise math, never claim new-action accuracy."""
import copy,json,threading,os
from pathlib import Path
import numpy as np
import pytest,torch
from moxian_web.contact_protocol import PROTOCOL,STAGES
from moxian_web.calibration_targets import target
from moxian_web.contact_templates import FrozenContactReference,prepare_templates,nearest_anchors,region_geometry_report


def test_exact_skin_distance_includes_edge_edge_witness():
    # Closest points lie inside long skew edges; vertex-only fails this case.
    v=np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,-1,.01],[0,1,.01],[1,0,1.]])
    faces=np.array([[0,1,2],[3,4,5]])
    gap,anchors=nearest_anchors(v,faces,{'a':[0],'b':[1]},'a','b')
    assert gap==pytest.approx(.01,abs=1e-8)
    points=[np.asarray(a['barycentric'])@v[faces[a['face']]] for a in anchors]
    assert np.linalg.norm(points[0]-points[1])==pytest.approx(gap,abs=1e-8)


def test_intersecting_triangle_interiors_have_exact_zero_gap():
    v=np.array([[-1,-1,0],[1,-1,0],[0,1,0],[0,-.2,-1],[0,-.2,1],[0,.2,.5]])
    faces=np.array([[0,1,2],[3,4,5]])
    gap,anchors=nearest_anchors(v,faces,{'a':[0],'b':[1]},'a','b')
    assert gap==0
    points=[np.asarray(a['barycentric'])@v[faces[a['face']]] for a in anchors]
    np.testing.assert_allclose(points[0],points[1],atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_spread_reference_and_joint_sensor_loss_have_no_distal_contact_equality():
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_joint_delta import JointGPUManoObjective
    from moxian_web.mano_objective import SolverSettings
    from scripts.calibrate_wuji_mano import CHAINS
    path=Path('output/moxian/hand_model/348640923d9e2f3f/gpu-cache/0068fe4460bc85b9ba441601c268a744ca5ae728c53b509a93c66a27c9441bc8/prepared.json')
    if not path.exists():pytest.skip('Retained SDK math fixture required')
    initial=json.loads(path.read_text());mapping=initial['rows'][0]['mapping']
    ref=FrozenContactReference(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),'right',mapping,initial)
    row=copy.deepcopy(next(r for r in initial['rows'] if r.get('skeleton')))
    row.update(stage='motion_spread_together',capture_protocol=PROTOCOL,partition='train',initialization=False,
               contact_confirmed=False,guided_contact=False,motion_event_confirmed=True,contact_edges=[])
    row.update(ref(row));assert np.isfinite(row['reference_gap_m']) and not row['reference_edges']
    model=ref.model;pose,root,*_=model.official_fk.parameters(model,row,CHAINS)
    objective=JointGPUManoObjective(model,[row],np.asarray([pose]),np.asarray([root]),training=True,shared=True,tactile=True,fixed_forward=True,settings=SolverSettings(adam_steps=2,lm_steps=0))
    assert not objective.contact_records and objective.mask_np.tolist()==[[1.,1.,1.,1.,1.]]
    values=objective.initial.detach().clone().requires_grad_(True)
    loss=objective.residual(values).square().sum();assert torch.isfinite(loss)
    loss.backward();assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize('side',['left','right'])
def test_virtual_palm_regions_are_palmar_disjoint_and_keep_taxel_abi(side):
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.contact_templates import region_faces
    model=ManoSurface(DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl'),side=side,device='cpu')
    mapping=json.loads(Path('output/moxian/hand_model/348640923d9e2f3f/motion-capture.json').read_text())['rows'][0]['mapping']
    model.build_binding(mapping);before=model.topology();regions=region_faces(model)
    groups=[set(regions['palm_mcp_'+f]) for f in ('index','middle','ring','pinky')]
    assert all(len(g)>0 for g in groups)
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    assert len(before['sid_face_ids'])==460 and len(regions)==24
    for group in groups:
        tri=model.rest_np[model.faces[list(group)]];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        assert np.all(normal@model.anatomy.palmar_normal>0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_all_new_contacts_reach_joint_loss_gradients_and_holdout_does_not_choose_templates(monkeypatch,tmp_path):
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_joint_delta import JointGPUManoObjective
    from moxian_web.mano_objective import SolverSettings
    from moxian_web.mano_mixed_training import solve_mixed
    from moxian_web.mano_gpu_validation import geometry_report
    path=Path('output/moxian/hand_model/348640923d9e2f3f/gpu-cache/0068fe4460bc85b9ba441601c268a744ca5ae728c53b509a93c66a27c9441bc8/prepared.json')
    if not path.exists():pytest.skip('Retained prepared SDK fixture required')
    initial=json.loads(path.read_text());mapping=initial['rows'][0]['mapping']
    ref=FrozenContactReference(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),'right',mapping,initial)
    m=ref.model
    base=next(r for r in initial['rows'] if r['stage']=='pinch_index' and r.get('contact_confirmed'))
    rows=[]
    for partition in ('train','validation'):
        for s in STAGES:
            for rep in range(2):
                r=copy.deepcopy(base);r.update(stage=s['id'],capture_protocol=PROTOCOL,initialization=False,partition=partition,capture_pass=partition,
                    timestamp_ms=base['timestamp_ms']+20*len(rows),contact_confirmed=True,contact_event_id=f'{partition}:{s["id"]}:{rep}',
                    contact_constraint_source='trajectory_assumed_contact',reference_edges=copy.deepcopy(target(s['id'])['edges']),contact_reference_id='fixture',
                    reference_alignment=ref.spec['raw_rotation'],reference_mount_rotation=ref.spec['mount_rot'])
                r['skeleton']['timestamp_ms']=r['timestamp_ms']
                rows.append(r)
    templates=prepare_templates(m,rows,steps=2)
    assert len(templates['actions'])==16 and not templates['validation_used']
    source_rows=copy.deepcopy(rows)
    for r in source_rows:r['reference_gap_m']=.01
    snapshots=copy.deepcopy(templates)
    train=rows[:32];valid=rows[32:]
    # No validation coordinates/poses are read to derive any anchor.
    for r in valid:r['fingers']['thumb']['position'][0]+=10
    assert templates==snapshots
    values=[m.official_fk.parameters(m,r, __import__('scripts.calibrate_wuji_mano',fromlist=['CHAINS']).CHAINS) for r in train]
    poses=np.asarray([v[0] for v in values]);roots=np.asarray([v[1] for v in values])
    o=JointGPUManoObjective(m,train,poses,roots,training=True,shared=True,tactile=True,fixed_forward=True,settings=SolverSettings(adam_steps=2,lm_steps=0))
    assert {r['stage'] for r in o.contact_records}=={s['id'] for s in STAGES}
    o.rawp.requires_grad_(True);o.rawr.requires_grad_(True)
    x=o.initial.detach().clone().requires_grad_(True);residual=o.residual(x,record=True)
    total=residual.square().sum();total.backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    inactive=torch.as_tensor(o.mask_np==0,device=m.device)
    assert torch.count_nonzero(o.rawp.grad[inactive])==0
    assert torch.count_nonzero(o.rawr.grad[inactive])==0
    o.rawp=o.rawp.detach();o.rawr=o.rawr.detach()
    assert o.joint_policy.net[-1].weight.grad.norm()>0
    assert o.mask_np[-1].tolist()==[0,1,1,1,1]
    assert o.last_training_metrics['joint_limit_saturation_fraction']>=0
    from moxian_web.mano_mixed_training import geometry_gradient_mask
    mask=geometry_gradient_mask(train)
    # pair_index cannot update unrelated ring geometry; full-hand sample can beta.
    assert not mask[0,:10].any() and mask[18,:10].all()
    # Single tiny uncompiled update is a functional test, not a retraining run.
    progress=[];report=solve_mixed(o,threading.Event(),progress.append,max_seconds=120,compile_loss=os.getenv('MOXIAN_CONTACT_COMPILE_TEST')=='1')
    assert report['joint_geometry_policy'] and report['phases'][0]['steps']==2
    point=progress[-1]['training_sample']['raw'] if 'training_sample' in progress[-1] else report['training_trace']['points'][-1]['raw']
    assert all(k in point for k in ('geometry_gradient_norm','policy_gradient_norm','geometry_learning_rate','joint_limit_saturation_fraction'))
    with torch.no_grad():vertices=o.surface(o.initial)[1]
    geo=region_geometry_report(m,train,vertices)
    assert len(geo['contact_stages'])==16 and geo['contact_clusters']
    assert all(r['query']=='original_region_triangles' for r in geo['contact_material_records'])

    # Exercise the public fit/report wiring as well as the objective. MJWarp's
    # independent backend is tested separately; avoid compiling it in this test.
    from moxian_web.mano_fit import fit_capture
    from moxian_web.mano_runtime import DEFAULT_MATERIAL
    import moxian_web.mano_mjwarp_check as cross
    monkeypatch.setattr(cross,'cross_check',lambda *a,**k:dict(query_valid=True,compilation_s=0.))
    init=copy.deepcopy(initial['rows'][:240]);origin=max(r['timestamp_ms'] for r in init)+10000
    for r in init:r.update(capture_protocol=PROTOCOL,partition='initialization',initialization=True,contact_confirmed=False,guided_contact=False)
    for i,r in enumerate(source_rows):
        r['timestamp_ms']=origin+i*20;r['skeleton']['timestamp_ms']=r['timestamp_ms']
    data=init+source_rows
    profile=json.loads(Path('output/moxian/hand_model/348640923d9e2f3f/candidate.json').read_text())
    profile.update(shape=[0.]*41,glove_parameters=[0.]*43,rest_vertex_residual_m=None,flat_reference=None,sensor_joint_position=None,
                   binding_overrides={},contact_registration=None,artifact_home=str(tmp_path))
    # Fix registration from SDK initialization inside fit, as the real entry does.
    for method in ('parameters','flashsac','delta'):
        result,report=fit_capture(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),profile,mapping,data,lambda _:None,threading.Event(),
                                  gpu_method=method,compile_loss=False,solver_settings=SolverSettings(adam_steps=2,lm_steps=0),max_seconds=120)
        assert report['capture_protocol']==PROTOCOL and report['validation_scope']=='separate_complete_contact_pass'
        assert report['selected_training_frames']==32 and report['selected_validation_frames']==32
        assert len(report['contact_stages'])==16 and report['independent_cycles_valid']
        assert report['auxiliary_initialization_checks'] is not None
        assert result['contact_templates']['validation_used'] is False
        assert bool(result.get('delta_policy'))==(method=='delta')
        if method=='delta':assert result['delta_policy']['joint_geometry_training']
        assert not (tmp_path/'current.json').exists()


def test_release_guard_catches_an_always_closed_model_on_original_triangles():
    from types import SimpleNamespace
    stage='contact_pair_index';faces=np.array([[0,1,2],[3,5,4]])
    anchor=[dict(face=0,barycentric=[1,0,0]),dict(face=1,barycentric=[1,0,0])]
    edges=[dict(regions=['thumb_distal','index_distal'],required=True,anchors=anchor)]
    model=SimpleNamespace(faces=faces,face_t=torch.tensor(faces),contact_template_artifact=dict(regions={'thumb_distal':[0],'index_distal':[1]},actions=[dict(stage=stage,edges=edges)]))
    rows=[];vertices=[]
    base=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,0],[.01,0,0],[0,.01,0]])
    for i,gap in enumerate([.01,.0005,.01,.0005,.01]):
        v=base.copy();v[3:,2]=gap;vertices.append(v)
        rows.append(dict(stage=stage,timestamp_ms=i*20,window_id='test',reference_gap_m=gap,contact_confirmed=i in (1,3),contact_event_id=str(i),contact_edges=edges if i in (1,3) else []))
    good=region_geometry_report(model,rows,torch.tensor(np.array(vertices),dtype=torch.float32))
    assert good['approach_release_valid']
    collapsed=np.array(vertices);collapsed[:,3:,2]=.0005
    bad=region_geometry_report(model,rows,torch.tensor(collapsed,dtype=torch.float32))
    assert not bad['approach_release_valid']
    assert all(s['query']=='original_region_triangles' for s in bad['contact_material_records'])

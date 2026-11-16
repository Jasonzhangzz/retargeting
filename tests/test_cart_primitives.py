import hashlib
import json
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from moxian_web.cart_geometry import estimate, transform_hull, FreeSpace
from moxian_web.cart_primitives import templates, surface, primitive_target
from moxian_web.cart_convex import geometry_identity
from moxian_web.cart_protocol import STAGES, stage, MotionTracker


def pads(names, directions, radius, center=(0,0,0)):
    meshes=[];regions={};offset=0
    for name,direction in zip(names,directions):
        direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
        box=trimesh.creation.box(extents=[.009,.009,.003])
        rotation=Rotation.align_vectors([direction],[[0,0,1]])[0].as_matrix()
        box.vertices=box.vertices@rotation.T+np.asarray(center)+direction*(radius+.0015)
        regions[name]=(offset+np.flatnonzero(box.face_normals@(-direction)>.9)).tolist()
        meshes.append(box);offset+=len(box.faces)
    return trimesh.util.concatenate(meshes),regions


def test_dataset_mapping_dimensions_and_wxyz_pose_conversion():
    assert all(not s['initial_shapes'] for s in STAGES[:4])
    assert all(set(s['initial_shapes'])<=set(('cylinder','sphere')) for s in STAGES[4:])
    rows=templates();assert len(rows)==297
    assert {r['type_id'] for r in rows}=={3,5,6,7,8,11,13,26,28}
    assert all(r['kind']=='sphere' for r in rows if r['type_id'] in (11,13,26,28))
    for type_id,height in ((3,.18),(6,.18),(7,.12),(8,.12)):
        assert all(r['height_m']==pytest.approx(height) for r in rows if r['type_id']==type_id)
    from moxian_web.cart_primitives import DATASET
    from retarget.baseline.geort.adapter import anatomical_basis
    ref=next(r for r in rows if r['type_id']==8 and r['row']==16)
    with np.load(DATASET/ref['template_file'],allow_pickle=False) as z:
        p=z['target_keypoints_world'][16];pose=z['object_pose_world'][16]
        basis=anatomical_basis(p)
        np.testing.assert_allclose(ref['center']@basis.T+p[0],pose[:3],atol=1e-9)
        np.testing.assert_allclose(basis@ref['rotation'],Rotation.from_quat(pose[[4,5,6,3]]).as_matrix(),atol=1e-9)
    assert len(geometry_identity()['templates_sha256'])==64


@pytest.mark.parametrize('kind',['cylinder','sphere'])
def test_analytic_surface_has_signed_distances_unit_normals_and_rigid_invariance(kind):
    p=np.array([[0,0,0],[.01,0,0],[.03,.02,.08],[0,0,.06]])
    d,q,n=surface(p,np.zeros(3),np.eye(3),.01,.12,kind)
    assert d[0]==pytest.approx(-.01) and d[1]==pytest.approx(0)
    np.testing.assert_allclose(np.linalg.norm(n,axis=-1),1,atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(p-q,axis=-1),np.abs(d),atol=1e-9)
    r=Rotation.from_rotvec([.4,.8,.1]).as_matrix();t=np.array([.1,.2,.3])
    actual=surface(p@r.T+t,t,r,.01,.12,kind)
    np.testing.assert_allclose(actual[0],d,atol=1e-9)
    np.testing.assert_allclose(actual[1],q@r.T+t,atol=1e-9)
    np.testing.assert_allclose(actual[2],n@r.T,atol=1e-9)


def test_key_fingers_come_from_annotations_including_thumb():
    rows=templates()
    for type_id,fingers in ((6,{'thumb','index','middle','ring','pinky'}),
                           (7,{'thumb','index','middle','ring'}),(8,{'thumb','index','middle'}),
                           (28,{'thumb','index','middle','ring'})):
        ref=next(r for r in rows if r['type_id']==type_id)
        assert {p.split('_')[0] for p in ref['required_patches']}==fingers


def test_size_changes_with_current_hand_instead_of_the_template_grid():
    from moxian_web.cart_primitive_fit import fit_primitive
    from moxian_web.cart_convex import _patches
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    for scale in (.7,1.8):
        p=_patches(mesh.vertices*scale,mesh.faces,regions,list(regions),True)
        fit=fit_primitive('sphere',p,np.zeros(3),np.eye(3))
        assert fit['radius_m']==pytest.approx(.012*scale,abs=1e-5)
        assert fit['confidence']>.95


def test_low_confidence_one_sided_hand_abandons_fit_without_delta_or_history():
    mesh,regions=pads(stage('precision_grasp')['patches'],[[1,0,0]]*5,.012)
    previous=dict(ready=True,vertices_m=[[0,0,0]],primitive={'kind':'sphere'})
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',full=True,previous=previous)
    assert result['status']=='low_confidence' and not result['ready']
    assert result['contact_mode']=='unavailable' and not result['supervision_valid']
    assert not result.get('diagnostic_candidate') and 'primitive' not in result
    assert all(a['confidence']==0 for a in result['fit_attempts'])


def test_similarity_alignment_uses_current_hand_scale():
    from moxian_web.cart_primitives import _proposals
    ref=next(r for r in templates() if r['type_id']==8 and r['row']==16)
    shift=np.array([.01,.02,.03]);scale=1.4
    proposals=_proposals({**stage('precision_grasp'),'primitive_types':[8]},ref['joints']*scale+shift,ref['joints'])
    p=next(p for p in proposals if p[1]['row']==16)
    assert p[0]<1e-20
    np.testing.assert_allclose(p[2],ref['center']*scale+shift,atol=1e-10)


def test_a_missing_key_finger_cannot_be_silently_omitted(monkeypatch):
    import moxian_web.cart_primitive_fit as fitting
    ref=next(r for r in templates() if r['type_id']==7)
    names=['index_distal','middle_distal','ring_distal','pinky_distal']
    mesh,regions=pads(names,[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0]],.012)
    # Supply the action's envelope, but deliberately omit the annotated thumb.
    for finger in ('index','middle','ring','pinky'):
        for part in ('proximal','intermediate'):regions[finger+'_'+part]=regions[finger+'_distal']
    monkeypatch.setattr(fitting,'_proposals',lambda *a:[(0.,ref,np.zeros(3),np.eye(3))])
    result=estimate(mesh.vertices,mesh.faces,regions,'prismatic',full=True,witnesses=[])
    assert not result['ready'] and result['contact_mode']=='unavailable'
    assert result['fit_attempts']==[{'type_id':7,'status':'missing_key_finger_surface'}]


def test_template_shape_error_is_not_geometry_confidence():
    from moxian_web.cart_primitive_fit import fit_primitive
    from moxian_web.cart_convex import _patches
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    patches=_patches(mesh.vertices,mesh.faces,regions,list(regions),True)
    a=fit_primitive('sphere',patches,np.zeros(3),np.eye(3),hand_error_m=0)
    b=fit_primitive('sphere',patches,np.zeros(3),np.eye(3),hand_error_m=.1)
    assert a['confidence']==b['confidence'] and b['confidence']>.95
    assert b['hand_alignment_rms_m']==.1


def test_parallel_opposing_prismatic_contacts_are_not_zero_confidence():
    from moxian_web.cart_primitive_fit import fit_primitive
    from moxian_web.cart_convex import _patches
    mesh,regions=pads(['thumb_distal','index_distal','middle_distal'],[[-1,0,0],[1,0,0],[1,0,0]],.012)
    fit=fit_primitive('cylinder',_patches(mesh.vertices,mesh.faces,regions,list(regions),True),np.zeros(3),np.eye(3))
    assert fit['opposing_contacts'] and fit['dimensions_observable']
    assert fit['confidence']>=.55 and fit['radius_m']==pytest.approx(.012,abs=1e-5)


def test_each_key_finger_needs_one_contact_not_every_annotated_phalanx(monkeypatch):
    import moxian_web.cart_primitive_fit as fitting
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    ref=next(r for r in templates() if r['type_id']==11)
    monkeypatch.setattr(fitting,'_proposals',lambda *a:[(0.,ref,np.zeros(3),np.eye(3))])
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',full=True,witnesses=[])
    assert result['ready'] and len(result['contacts'])==5
    assert {c['patch'].split('_')[0] for c in result['contacts']}=={'thumb','index','middle','ring','pinky'}
    assert 'thumb_proximal' in result['allowed_contact_patches']


def test_admitted_candidate_finishes_validation_after_search_budget(monkeypatch):
    import moxian_web.cart_primitive_fit as fitting
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    ref=next(r for r in templates() if r['type_id']==13);clock=[0.]
    monkeypatch.setattr(fitting.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(fitting,'_proposals',lambda *a:[(0.,ref,np.zeros(3),np.eye(3))])
    original=fitting.fit_primitive
    def slow(*a,**k):
        result=original(*a,**k);clock[0]=1.;return result
    monkeypatch.setattr(fitting,'fit_primitive',slow)
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',budget_ms=120,witnesses=[])
    assert result['ready'] and result['budget_exhausted'] and result['collision_free']
    assert all(c['gap_m']<=.002 and c['normal_error_deg']<=30 for c in result['contacts'])


def test_failed_realtime_candidates_do_not_starve_later_types(monkeypatch):
    import moxian_web.cart_primitive_fit as fitting
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    refs=[next(r for r in templates() if r['type_id']==i) for i in (6,7,8)]
    monkeypatch.setattr(fitting,'_proposals',lambda *a:[(0.,r,np.zeros(3),np.eye(3)) for r in refs])
    clock=[0.];monkeypatch.setattr(fitting.time,'perf_counter',lambda:clock[0])
    def failed(*a,**k):
        clock[0]+=1.
        return dict(confidence=0.,gap_rms_m=.02,normal_rms_deg=90.,hand_alignment_rms_m=0.)
    monkeypatch.setattr(fitting,'fit_primitive',failed)
    previous=None;seen=[]
    for _ in range(3):
        previous=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',budget_ms=120,witnesses=[],previous=previous)
        seen.extend(a['type_id'] for a in previous['fit_attempts'])
    assert seen==[6,7,8]


def test_precision_fits_current_size_and_all_key_contacts_with_bounded_delta():
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',full=True)
    assert result['ready'] and result['contact_mode']=='hull'
    assert set(result['initial_shapes_tried'])<=set(('cylinder','sphere'))
    assert not result['free_convex_shape']
    assert result['volume_m3']>=result['initial_volume_m3']
    assert result['primitive']['kind']=='sphere'
    assert result['primitive']['radius_m']==pytest.approx(.012,abs=1e-5)
    assert result['delta_geometry']['max_offset_m']<=result['delta_geometry']['limit_m']
    assert result['delta_geometry']['core_preserved']
    delta=result['delta_geometry']
    assert len(delta['vertices_m'])==len(delta['offsets_m'])
    if delta['offsets_m']:np.testing.assert_allclose(np.linalg.norm(delta['offsets_m'],axis=1).max(),delta['max_offset_m'])
    moved=transform_hull(result,np.eye(3),[.01,.02,.03])
    np.testing.assert_allclose(np.asarray(moved['delta_geometry']['vertices_m']).reshape(-1,3),np.asarray(delta['vertices_m']).reshape(-1,3)+[.01,.02,.03])
    np.testing.assert_allclose(moved['delta_geometry']['offsets_m'],delta['offsets_m'])
    assert {c['patch'] for c in result['contacts']}==set(stage('precision_grasp')['patches'])
    assert result['collision_free'] and result['coverage']==1
    assert FreeSpace(mesh.vertices,mesh.faces).clear(result,result['center_m'])
    warm=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',budget_ms=0,previous=result)
    assert not warm['ready'] and warm['status']=='budget_exhausted'
    blocked=trimesh.util.concatenate([mesh,trimesh.creation.box(extents=[.04,.04,.04])])
    invalid=estimate(blocked.vertices,blocked.faces,regions,'precision_grasp',budget_ms=0,previous=result)
    assert not invalid['ready']


def test_power_uses_every_phalanx_and_wrap_prismatic_have_different_focus():
    power=stage('power_grasp');wrap=stage('wrap');prismatic=stage('prismatic')
    assert len(power['patches'])==14 and 'thumb_intermediate' not in power['patches']
    assert all(any(p.endswith('_'+part) for p in power['patches']) for part in ('proximal','intermediate','distal'))
    assert len(wrap['patches'])==12 and len(prismatic['patches'])==8
    assert all(not p.startswith('thumb') for p in wrap['patches']+prismatic['patches'])
    assert wrap['patch_weights']['index_distal']==2*wrap['patch_weights']['index_intermediate']
    assert prismatic['opposition_weight']>wrap['opposition_weight']
    directions=[]
    for i in range(len(power['patches'])):
        z=1-2*(i+.5)/len(power['patches']);angle=i*np.pi*(3-np.sqrt(5))
        directions.append([np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z])
    mesh,regions=pads(power['patches'],directions,.03)
    target=estimate(mesh.vertices,mesh.faces,regions,'power_grasp',full=True,witnesses=[])
    candidate=target if target['ready'] else target['diagnostic_candidate']
    if candidate:
        assert {c['patch'] for c in candidate['contacts']}==set(candidate['required_patches'])
        assert candidate['initialization']['kind'] in ('cylinder','sphere')
    if target['ready']:
        assert all(c['gap_m']<=.002 and c['normal_error_deg']<=30 for c in target['contacts'])
    else:
        assert target['fit_attempts'] and not target['supervision_valid']


def test_bounded_delta_reaches_fingers_outside_the_primitive():
    meshes=[];regions={};offset=0
    directions=[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]]
    for i,(name,direction) in enumerate(zip(stage('precision_grasp')['patches'],directions)):
        mesh,patch=pads([name],[direction],.017 if i<2 else .012)
        regions[name]=[j+offset for j in patch[name]];offset+=len(mesh.faces);meshes.append(mesh)
    mesh=trimesh.util.concatenate(meshes)
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',full=True,witnesses=[])
    assert result['ready'] and result['collision_free']
    delta=result['delta_geometry'];assert delta['added_facets']>=2
    assert .002<delta['max_offset_m']<=delta['limit_m']
    assert all(c['gap_m']<=.002 and c['normal_error_deg']<=30 for c in result['contacts'])
    primitive=result['primitive']
    d,_,_=surface(np.array([c['point_m'] for c in result['contacts']]),np.array(primitive['center_m']),
                   np.array(primitive['rotation']),primitive['radius_m'],primitive['height_m'],primitive['kind'])
    assert d.max()>.002  # The plain primitive alone cannot meet all contacts.


def test_mirror_and_robot_rigid_transform_preserve_primitive_dimensions():
    ref=next(r for r in templates() if r['type_id']==8)
    original=primitive_target('cylinder',.0123,.12,[.01,.02,.03],np.eye(3),ref)
    r=Rotation.from_rotvec([.3,.6,.2]).as_matrix()@np.diag([-1,1,1]);t=np.array([.1,.2,.3])
    target=transform_hull(original,r,t)
    assert target['primitive']['radius_m']==original['primitive']['radius_m']
    assert target['primitive']['height_m']==original['primitive']['height_m']
    assert target['volume_m3']==original['volume_m3']
    np.testing.assert_allclose(target['primitive']['center_m'],r@np.array(original['center_m'])+t)
    assert np.linalg.det(target['primitive']['rotation'])==pytest.approx(1)
    mesh=trimesh.Trimesh(target['vertices_m'],target['faces'],process=False)
    assert mesh.is_winding_consistent and mesh.volume>0
    assert original['primitive']['rotation']==np.eye(3).tolist()


def test_timeout_and_missing_region_cannot_be_valid_supervision():
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',budget_ms=0)
    assert not result['ready'] and result['status']=='budget_exhausted'
    del regions['middle_distal']
    with pytest.raises(ValueError,match='缺少语义曲面'):
        estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',full=True)


def test_collision_candidate_is_never_hand_contact_or_supervision():
    mesh,regions=pads(stage('precision_grasp')['patches'],[[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],.012)
    obstacle=trimesh.creation.box(extents=[.04,.04,.14])
    mesh=trimesh.util.concatenate([mesh,obstacle])
    result=estimate(mesh.vertices,mesh.faces,regions,'precision_grasp',full=True)
    assert not result['ready'] and result['contact_mode']=='unavailable'
    if result.get('diagnostic_candidate'):assert result['diagnostic_candidate']['supervision_valid'] is False


def test_only_actual_closure_switches_to_hand_contact(monkeypatch):
    from types import SimpleNamespace
    from moxian_web.cart_data import SourceGeometry
    import moxian_web.cart_geometry as geometry
    source=SourceGeometry.__new__(SourceGeometry)
    source.adapter=SimpleNamespace(indices=list(range(21)),transform=np.eye(3))
    source.topology={'geometry_version':'test'};source.faces=np.array([[0,1,2]]);source.regions={}
    gap=[.01];calls=[]
    source.witnesses=lambda cv,pairs,full:[dict(regions=p,gap_m=gap[0],normals=[[1,0,0],[-1,0,0]]) for p in pairs]
    def unavailable(*a,**k):
        calls.append(1)
        return dict(ready=False,contact_mode='unavailable',reference_gap_m=.01)
    monkeypatch.setattr(geometry,'estimate',unavailable)
    raw=(np.zeros((3,3)),np.random.default_rng(3).normal(size=(21,3))*.03)
    tracker=MotionTracker();row,_,_=source.frame(None,'wrap',0,tracker,raw=raw)
    assert row['target']['contact_mode']=='unavailable' and len(calls)==1
    gap[0]=.001;row,_,_=source.frame(None,'wrap',20,tracker,raw=raw)
    assert row['target']['contact_mode']=='hand' and len(calls)==1
    assert row['target']['timestamp_ms']==row['timestamp_ms']==20

def test_live_warm_start_resets_on_time_gaps_and_action_changes(monkeypatch):
    from types import SimpleNamespace
    from moxian_web.cart_data import SourceGeometry
    from moxian_web.cart_geometry import hull_data
    import moxian_web.cart_geometry as geometry
    source=SourceGeometry.__new__(SourceGeometry)
    source.adapter=SimpleNamespace(indices=list(range(21)),transform=np.eye(3))
    source.topology={'geometry_version':'test'};source.faces=np.array([[0,1,2]]);source.regions={}
    source.witnesses=lambda cv,pairs,full:[dict(regions=p,gap_m=.01) for p in pairs]
    seen=[]
    def estimate(*a,**k):
        seen.append(k['previous'])
        return dict(hull_data(trimesh.creation.box(extents=[.01]*3).vertices),contact_mode='hull')
    monkeypatch.setattr(geometry,'estimate',estimate)
    raw=(np.zeros((3,3)),np.random.default_rng(3).normal(size=(21,3))*.03)
    tracker=MotionTracker()
    for action,stamp in [('precision_grasp',0),('precision_grasp',20),('precision_grasp',1000),('prismatic',1020)]:
        source.frame(None,action,stamp,tracker,raw=raw)
    assert seen[0] is None and seen[1] is not None
    assert seen[2] is None and seen[3] is None


def test_target_cache_and_model_geometry_invalidate_with_templates(tmp_path,monkeypatch):
    from moxian_web.cart_data import input_digest,preprocessing_current
    from moxian_web.cart_training import validate_model_geometry
    import moxian_web.cart_convex as primitives
    for name in ('motion.npz','timestamps.npy','profile.json','topology.json'):
        (tmp_path/name).write_bytes(b'original')
    metadata=dict(source={},stage='wrap',points_sha256='raw')
    target=tmp_path/'cart_targets.jsonl';target.write_text('{}\n')
    info=dict(profile='full',fingerprint=input_digest(tmp_path,metadata,True),
        targets_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (tmp_path/'cart_preprocess.json').write_text(json.dumps(info))
    config={'protocol':'cart_contact_v3','geometry':geometry_identity()}
    assert preprocessing_current(tmp_path,metadata)
    validate_model_geometry(config)
    with pytest.raises(ValueError,match='旧模型'):validate_model_geometry({})
    rows=[dict(r) for r in templates()];rows[0]['template_sha256']='changed'
    monkeypatch.setattr(primitives,'templates',lambda:rows)
    assert not preprocessing_current(tmp_path,metadata)
    with pytest.raises(ValueError,match='模板版本'):validate_model_geometry(config)


def test_preprocess_keeps_every_raw_timestamp_and_counts_only_valid_convex_targets(tmp_path,monkeypatch):
    from moxian_web.cart_protocol import PROTOCOL
    from moxian_web.cart_data import preprocess
    folder=tmp_path/'recordings/session';folder.mkdir(parents=True)
    stamps=np.arange(6)*16.
    np.save(folder/'points.npy',np.zeros((6,5,3),np.float32))
    np.save(folder/'timestamps.npy',stamps)
    np.savez(folder/'motion.npz',qpos=np.arange(6)[:,None])
    for name in ('topology.json','profile.json'):(folder/name).write_text('{}')
    metadata=dict(id='session',protocol=PROTOCOL,source={'synthetic_test_only':True},
        stage='precision_grasp',category=2,split='train',session_id='session',complete=True,frames=6,
        points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(metadata))
    class Geometry:
        def __init__(self,*a):pass
        def frame(self,q,stage_id,stamp,tracker,full=False):
            tracker.cycles=2
            valid=int(q[0])%2==0
            target=dict(ready=valid,contact_mode='hull' if valid else 'unavailable',timestamp_ms=stamp)
            if not valid:target['diagnostic_candidate']=dict(ready=True,supervision_valid=False)
            return dict(timestamp_ms=stamp,phase='hold',target=target),None,None
    monkeypatch.setattr('moxian_web.cart_data.SourceGeometry',Geometry)
    info=preprocess(tmp_path,'session')
    rows=[json.loads(line) for line in (folder/'cart_targets.jsonl').read_text().splitlines()]
    assert [r['timestamp_ms'] for r in rows]==stamps.tolist()
    assert all(r['timestamp_ms']==r['target']['timestamp_ms'] for r in rows)
    assert info['hull_frames']==info['supervised_contact_frames']==3
    assert info['contact_modes']==dict(hull=3,hand=0,unavailable=3)
    assert info['contact_targets_valid'] and info['complete_cycles']
    assert info['geometry']==geometry_identity()

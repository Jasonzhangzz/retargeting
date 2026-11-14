"""Tactile evidence changes official hand_pose, never the MANO model surface."""
import copy,json,threading
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.contact import PATCH_IDS
from moxian_web.tactile_pose import frame_constraints,correct_pose,associate_self_contacts
from scripts.calibrate_wuji_mano import CHAINS
from moxian_web.emf_slsqp import SkeletonKinematics
from moxian_web.skeleton import JOINT_NAMES


@pytest.fixture
def pinch():
    path=Path('output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json')
    asset=DEFAULT_MANO.with_name('MANO_RIGHT.pkl')
    if not path.exists() or not asset.exists():pytest.skip('Retained official right calibration and MANO required')
    rows=json.loads(path.read_text())['rows'];row=copy.deepcopy(next(r for r in rows if r['stage']=='pinch_middle'))
    torch.set_num_threads(1);m=ManoSurface(asset,device='cpu',side='right');m.enable_official_fk();m.build_binding(row['mapping'])
    # Runtime replay files are replaced by calibration. Use a fixed synthetic
    # articulated skeleton so this regression does not depend on yesterday's IK.
    urdf=m.official_fk.path.with_name('right.urdf')
    names=[e.get('name') for e in ET.parse(urdf).getroot().findall('joint') if e.get('type')!='fixed']
    kin=SkeletonKinematics(urdf,'right',names,JOINT_NAMES);angles=kin.straight.copy()
    angles[names.index('right_middle_mcp_flexion')]=.9
    angles[names.index('right_middle_pip_flexion')]+=.9
    angles[names.index('right_middle_dip_flexion')]=.1
    angles[names.index('right_thumb_cmc_abduction')]=.25
    angles[names.index('right_thumb_cmc_opposition')]=.3
    nodes=kin.nodes(angles)
    row['skeleton']={'timestamp_ms':row['timestamp_ms'],'hand_side':'right','coordinate_frame':'wrist_local','names':JOINT_NAMES,
                     'thumb_kinematics':kin.thumb_model,'joint_names':names,'joint_positions_rad':angles.tolist(),
                     'positions_local':nodes[:,:3,3].tolist(),'orientations_local_xyzw':Rotation.from_matrix(nodes[:,:3,:3]).as_quat().tolist()}
    ids=np.isin(m.mapping['assignments'],[0,PATCH_IDS.index('middle_distal')])
    row['contact']=dict(ready=True,timestamp_ms=row['timestamp_ms'],cell_contact=ids.tolist(),normalized=ids.astype(float).tolist())
    row['tactile_timestamp_ms']=row['timestamp_ms']
    q,r,_,_,_=m.official_fk.parameters(m,row,CHAINS)
    with torch.no_grad():v,_,_,j=m.decode(torch.tensor(q[None]),torch.tensor(r[None]))
    return m,row,q,r,v[0].numpy(),j[0].numpy()


def test_distant_bilateral_activity_does_not_override_pose_control(pinch):
    m,row,q,r,v,j=pinch;rest=m.rest_np.copy();shape=m.shape.copy()
    corrected,report=correct_pose(m,row,q,r,v,j)
    assert not report['applied'] and not report['pairs']
    np.testing.assert_array_equal(corrected,q)
    np.testing.assert_array_equal(m.rest_np,rest);np.testing.assert_array_equal(m.shape,shape)


def test_unknown_stale_or_unilateral_tactile_does_not_invent_a_partner(pinch):
    m,row,q,r,v,j=pinch
    for kind in ('unknown','stale','unilateral'):
        frame=copy.deepcopy(row)
        if kind=='unknown':frame['contact']['ready']=False
        if kind=='stale':frame['tactile_timestamp_ms']-=26
        if kind=='unilateral':frame['contact']['cell_contact']=(np.asarray(m.mapping['assignments'])==0).tolist()
        corrected,report=correct_pose(m,frame,q,r,v,j)
        assert not report['applied'] and not report['pairs'];np.testing.assert_array_equal(corrected,q)


def test_active_sid_location_changes_material_constraint(pinch):
    m,row,q,r,v,j=pinch;patches=np.asarray(m.mapping['assignments']);ids=np.flatnonzero(patches==0)
    points=[]
    for sid in (ids[0],ids[-1]):
        frame=copy.deepcopy(row);frame['contact']['cell_contact']=[k==int(sid) for k in range(460)]
        anchors,_=frame_constraints(m,frame,v,j);a=anchors[0]
        expected=np.asarray(a['barycentric'])@v[m.faces[a['source_face']]]
        np.testing.assert_allclose(a['position_m'],expected,atol=1e-8);assert a['sids']==[int(sid)]
        points.append(a['position_m'])
    assert np.linalg.norm(np.array(points[0])-points[1])>.001


def test_live_official_tracking_reports_and_publishes_corrected_upstream_mesh(pinch):
    m,row,q,r,v,j=pinch
    actual=m.track(row)
    from moxian_web.hand_model import MANO_SOLVER_REVISION
    assert not m.contact_correction['applied'] and m.solver_report['revision']==MANO_SOLVER_REVISION
    assert not m.retarget_contacts
    assert m.solver_report['constraints_satisfied']==(m.solver_report['collision']['constraints_satisfied'] and all(p['satisfied'] for p in m.retarget_contacts))
    with torch.no_grad():expected=m.decode(torch.tensor(m.last_pose[None]),torch.tensor(m.last_root[None]))
    np.testing.assert_allclose(actual[0],expected[0][0].numpy(),atol=1e-7)
    np.testing.assert_array_equal(m.raw_official_pose,q)


def test_fist_thumb_observation_cannot_change_shared_pose_fit(pinch):
    from moxian_web.mano_fit import fit_pose_zeros
    m,row,q,r,v,j=pinch;fist=copy.deepcopy(row);fist['stage']='fist';fist['contact']['ready']=False
    rows=[row,fist];fit_pose_zeros(m,rows,threading.Event(),steps=2);first=m.calibration_pose_offsets.copy()
    m.calibration_pose_offsets[:]=0
    changed=copy.deepcopy(rows);changed[1]['fingers']['thumb']['position']=[10000.,-10000.,10000.]
    fit_pose_zeros(m,changed,threading.Event(),steps=2)
    np.testing.assert_array_equal(m.calibration_pose_offsets,first)


def test_unmatched_activity_remains_external_evidence_without_invented_self_closure(pinch):
    from moxian_web.wuji_retarget import TactileRetarget
    if not torch.cuda.is_available():pytest.skip('CUDA needed')
    m,row,q,r,v,j=pinch;v,_,_,ordered,_=m.track(row)
    solver=TactileRetarget(m.topology(),Path('output/moxian/debug/contact-fix/mappings'))
    results=[]
    for i in range(6):
        stamp=row['timestamp_ms']+i*100
        f=dict(timestamp_ms=stamp,tactile_timestamp_ms=stamp,vertices_m=v.tolist(),joints_m=ordered.tolist(),
               geometry_version=m.shape_version,binding_version=m.binding_version,self_contact_pairs=m.retarget_contacts,
               source_contact={**row['contact'],'timestamp_ms':stamp})
        results.append(solver.step(f))
    assert all(r['solution_feasible'] and r['solver']['sampled_candidates']==16384 for r in results)
    first,last=results[0],results[-1]
    assert not first['self_contact_pairs'] and not last['self_contact_pairs']
    assert last['quality']['cost_subterms'].get('contact_self_closure',0)==0
    assert last['taxel_projection']['mapped_sids']==460
    assert last['sdf']['vertices']==2 and not last['sdf']['ready']


def test_even_dominant_pressure_cannot_identify_a_distant_or_same_facing_partner():
    def anchor(p,f,x,force):
        return dict(patch=p,finger=f,segment=2,region=3+3*f,position_m=[x,0,0],direction=[0,0,1],force=force,sids=[2*p,2*p+1])
    thumb=anchor(0,0,0,2);index=anchor(2,1,.03,10);ring=anchor(8,3,.06,2)
    assert not associate_self_contacts([thumb,index,ring])
    index['position_m']=[.005,0,0];assert not associate_self_contacts([thumb,index,ring])
    thumb['direction']=[1,0,0];index['direction']=[-1,0,0]
    pairs=associate_self_contacts([thumb,index,ring]);assert len(pairs)==1 and pairs[0]['patches']==[0,2]
    assert pairs[0]['evidence']=='bilateral_tactile_geometry_association'


def test_actual_retest_same_facing_activity_is_not_a_verified_contact():
    folder=Path('output/moxian/debug/contact-fix');source=folder/'live-retest.jsonl';topology=folder/'retest-topology.json'
    if not source.exists() or not topology.exists():pytest.skip('Retained live regression frame required')
    row=next((r for r in map(json.loads,source.open()) if r['elapsed_s']>25 and 'thumb_distal' in r['patches']),None)
    if row is None:pytest.skip('No thumb-index regression sample')
    top=json.loads(topology.read_text());h=row['hand'];torch.set_num_threads(1)
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),device='cpu',side='right');m.set_shape(h['betas'])
    m.build_binding(dict(assignments=top['sid_patch_indices'],version='live-regression'));m.sid_faces=np.asarray(top['sid_face_ids']);m.sid_bary=np.asarray(top['sid_barycentric'])
    f=dict(timestamp_ms=row['timestamp_ms'],tactile_timestamp_ms=h['tactile_timestamp_ms'],contact=h['source_contact'])
    _,result=correct_pose(m,f,np.asarray(h['pose_axis_angle']),np.asarray(h['root_rotation']),np.asarray(h['vertices_m']),np.asarray(h['joints_m']))
    assert not result['pairs'] and not result['applied']

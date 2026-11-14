"""Sampling and independent evaluation contracts for glove calibration."""
import copy
from moxian_web.mano_fit import choose_rows,holdout_rows,trajectory_summary,candidate_validation

def sequence():
    rows=[]
    for stage in ['open','mcp90','hook','pinch_index','pinch_middle','pinch_ring','pinch_pinky','open_repeat']:
        for i in range(20):
            touching=i%10>=5
            rows.append({'stage':stage,'timestamp_ms':1000+len(rows)*50,'cycle':i//10,
                'fingers':{f:{'position':[.01*(10-i%10),.01*(f!='thumb'),0]} for f in ('thumb','index','middle','ring','pinky')},
                'contact':{'ready':True,'patches':[{'id':f+'_distal','contact':touching,'response':(i%10)/10} for f in ('thumb','index','middle','ring','pinky')]}})
    return rows

def test_sampling_keeps_near_misses_and_separates_whole_repetitions():
    rows=sequence();original=copy.deepcopy(rows);train,valid=choose_rows(rows)
    assert rows==original
    assert not {r['timestamp_ms'] for r in train}&{r['timestamp_ms'] for r in valid}
    assert {r['timestamp_ms'] for r in valid}<={r['timestamp_ms'] for r in holdout_rows(rows)}
    for stage in ('pinch_index','pinch_middle','pinch_ring','pinch_pinky'):
        for part,cycle in ((train,0),(valid,1)):
            group=[r for r in part if r['stage']==stage]
            assert {r['load_role'] for r in group}=={'onset','peak','approach'}
            assert all(r['cycle']==cycle for r in group)
            assert min(r['timestamp_ms'] for r in group if r['load_role']=='approach')==min(r['timestamp_ms'] for r in rows if r['stage']==stage and r['cycle']==cycle)
    assert all(r['stage']!='open_repeat' for r in train)

def test_penetration_is_a_two_mm_hard_gate_even_when_sensor_fit_improves():
    geo={'noncontact_penetration_m':.012,'contact_gap_m':.003,'contact_stages':['pinch_'+f for f in ('index','middle','ring','pinky')]}
    assert candidate_validation(20,10,geo,.005)==(False,True)
    assert candidate_validation(20,10,{**geo,'noncontact_penetration_m':.002},.005)==(True,True)
    assert candidate_validation(20,10,{**geo,'noncontact_penetration_m':.00201},.01)==(False,True)
    assert not candidate_validation(20,30,geo,.005)[0]

def test_trajectory_report_distinguishes_near_miss_from_contact():
    summary=trajectory_summary([{'stage':'pinch_index','rmse_mm':3.,'bilateral_contact':False},{'stage':'pinch_index','rmse_mm':4.,'bilateral_contact':False},{'stage':'pinch_index','rmse_mm':1.,'bilateral_contact':True}])['pinch_index']
    assert summary['frames']==3 and summary['approach_frames']==2
    assert abs(summary['approach_sensor_rmse_mm']-(12.5**.5))<1e-10


def test_together_score_does_not_hide_one_separated_pair_in_mean():
    from moxian_web.mano_fit import geometry_score
    balanced=dict(contact_gap_m=0,noncontact_penetration_m=0,together_gap_m=.003,together_pair_gaps_mm=dict(index_middle=3,middle_ring=3,ring_pinky=3))
    separated={**balanced,'together_pair_gaps_mm':dict(index_middle=0,middle_ring=0,ring_pinky=9)}
    assert geometry_score(0,separated,.005)>2*geometry_score(0,balanced,.005)


def test_together_validation_requires_each_pair_and_rejects_hidden_regression():
    from moxian_web.mano_fit import together_validation
    baseline={'together_pair_gaps_mm':dict(index_middle=5,middle_ring=5,ring_pinky=5)}
    after={'together_observations':9,'together_pair_gaps_mm':dict(index_middle=.2,middle_ring=.2,ring_pinky=6)}
    assert not together_validation(after,baseline)
    after['together_pair_gaps_mm']['ring_pinky']=2
    assert together_validation(after,baseline)
    del after['together_pair_gaps_mm']['middle_ring']
    assert not together_validation(after,baseline)


def test_large_pinch_gap_cannot_pass_merely_because_baseline_was_also_bad():
    from moxian_web.mano_fit import candidate_validation,together_validation
    geo={'contact_gap_m':.0322,'baseline_contact_gap_m':.0313,'noncontact_penetration_m':.0058,'contact_stages':['pinch_'+f for f in ('index','middle','ring','pinky')],'together_observations':9,'together_pair_gaps_mm':dict(index_middle=1.48,middle_ring=.53,ring_pinky=3.41)}
    assert candidate_validation(.00235,.00227,geo,.005)==(False,False)
    assert not together_validation(geo,{})
    assert not candidate_validation(.00235,.0024,geo,.005)[0]
    assert not candidate_validation(.00235,.00227,{**geo,'contact_gap_m':.04},.005)[0]


def test_missing_cross_sections_fail_validation_without_aborting_shape_search():
    from moxian_web.mano_fit import geometry_score,together_validation
    broken={'contact_gap_m':0.,'noncontact_penetration_m':0.,'together_observations':0,
            'together_missing_sections':[{'timestamp_ms':1,'reason':'no common section'}]}
    assert geometry_score(0.,broken,.005)>=1e6
    assert not together_validation(broken,{})
    valid={'together_pair_gaps_mm':dict(index_middle=1.,middle_ring=1.,ring_pinky=1.),
           'together_observations':9,'together_surface_distances_mm':[[1.,1.,1.]]}
    assert together_validation(valid,broken)


def test_shared_geometry_search_includes_all_four_pinches_without_load_role():
    from moxian_web.mano_fit import calibration_geometry_indices
    stages=['together','open','mcp90','fist','pinch_index','pinch_middle','pinch_ring','pinch_pinky']
    rows=[{'stage':s,'capture_protocol':'official_eight_v1'} for s in stages for _ in range(3)]
    selected=calibration_geometry_indices(rows)
    assert {rows[i]['stage'] for i in selected}==set(stages)
    assert len(selected)==8


def test_official_betas_validate_each_held_out_cross_section_against_contact_tolerance():
    from moxian_web.mano_fit import together_validation
    pairs=dict(index_middle=5.,middle_ring=5.,ring_pinky=5.)
    baseline={'together_surface_distances_mm':[[5.,5.,5.]]*3,'together_pair_gaps_mm':pairs}
    after={'together_surface_distances_mm':[[2.,2.,2.]]*3,'together_pair_gaps_mm':dict(index_middle=2.,middle_ring=2.,ring_pinky=2.),'together_observations':3}
    assert together_validation(after,baseline)
    after['together_surface_distances_mm']=[[4.5,4.5,7.]]*3
    assert not together_validation(after,baseline)


def test_fist_is_last_and_thumb_is_excluded_from_every_sensor_loss():
    from moxian_web.mano_fit import calibration_phases,sensor_sample_weights
    rows=[{'stage':s} for s in ['fist','pinch_index','together','open','mcp90']]
    phases=calibration_phases(rows)
    assert [p for p,_ in phases]==['flat_and_together','tactile_pinches','fist_refinement']
    assert all(r['stage']!='fist' for _,group in phases[:2] for r in group)
    assert phases[-1][1][-1]['stage']=='fist'
    weights=sensor_sample_weights(rows)
    assert weights[0,0]==0 and (weights[0,1:]==1).all()
    assert weights[1].tolist()==[1,1,0,0,0]
    assert (weights[2:,0]==0).all() and (weights[2:,1:]==1).all()


def test_flat_physical_bone_loss_is_rigidly_invariant_and_detects_bend():
    from types import SimpleNamespace
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from moxian_web.mano_fit import flat_finger_terms
    from scripts.calibrate_wuji_mano import CHAINS,TIPS
    model=SimpleNamespace(anatomy=SimpleNamespace(palmar_normal=np.array([0.,0.,1.])))
    vertices=torch.zeros(1,778,3,dtype=torch.float64);joints=torch.zeros(1,16,3,dtype=torch.float64)
    rotations=torch.eye(3,dtype=torch.float64).expand(1,16,3,3).clone()
    for f in range(1,5):
        for k,j in enumerate(CHAINS[f]):joints[0,j]=torch.tensor([f*.02,k*.025,0.])
        vertices[0,TIPS[f]]=torch.tensor([f*.02,.075,0.])
    straight,plane,_,_=flat_finger_terms(model,vertices,joints,rotations)
    assert straight<1e-10 and plane<1e-10
    vertices[0,TIPS[1],2]=.02
    bend,plane,_,_=flat_finger_terms(model,vertices,joints,rotations)
    turn=torch.tensor(Rotation.from_euler('xyz',[.3,-.7,.2]).as_matrix())
    translated=torch.tensor([.05,-.1,.03])
    other,p2,_,_=flat_finger_terms(model,vertices@turn.T+translated,joints@turn.T+translated,turn@rotations)
    assert bend>1 and plane>1
    torch.testing.assert_close(other,bend);torch.testing.assert_close(p2,plane)

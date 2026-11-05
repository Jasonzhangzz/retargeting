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

def test_penetration_above_five_mm_is_soft_and_does_not_veto_improved_candidate():
    geo={'noncontact_penetration_m':.012,'contact_gap_m':.003,'contact_stages':['pinch_'+f for f in ('index','middle','ring','pinky')]}
    assert candidate_validation(20,10,geo,.005)==(True,True)
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
    after['together_pair_gaps_mm']['ring_pinky']=3
    assert together_validation(after,baseline)
    del after['together_pair_gaps_mm']['middle_ring']
    assert not together_validation(after,baseline)


def test_primary_sensor_error_and_geometry_guards_are_separate_from_soft_score():
    from moxian_web.mano_fit import candidate_validation,together_validation
    geo={'contact_gap_m':.0322,'baseline_contact_gap_m':.0313,'noncontact_penetration_m':.0058,'contact_stages':['pinch_'+f for f in ('index','middle','ring','pinky')],'together_observations':9,'together_pair_gaps_mm':dict(index_middle=1.48,middle_ring=.53,ring_pinky=3.41)}
    assert candidate_validation(.00235,.00227,geo,.005)==(True,True)
    assert together_validation(geo,{})
    assert not candidate_validation(.00235,.0024,geo,.005)[0]
    assert not candidate_validation(.00235,.00227,{**geo,'contact_gap_m':.04},.005)[0]

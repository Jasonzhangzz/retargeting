"""Pinch recovery before geometric closure, with strict output collision gates."""
import numpy as np
import pytest
import torch

from moxian_web.card_math import CardSettings,cem
from moxian_web.wuji_tactile import pinch_hypothesis,pinch_seed


@pytest.mark.parametrize('target',[6,9,12,15])
def test_bilateral_pressure_and_robokit_can_propose_a_30mm_pinch(target):
    anchors=[dict(region=3,position_m=[0,0,0]),dict(region=target,position_m=[.030,0,0])]
    native={3:(np.zeros(3),np.array([1.,0,0])),target:(np.array([.033,0,0]),np.array([-1.,0,0]))}
    pair,report=pinch_hypothesis(anchors,native)
    assert pair==(3,target) and report['reason']=='bilateral_tactile_robokit_hypothesis'
    assert report['candidates'][0]['source_gap_mm']==pytest.approx(30)
    assert pinch_hypothesis(anchors,native,external_object=True)[0] is None
    assert pinch_hypothesis(anchors[:1],native)[0] is None
    native[target]=(np.array([.033,0,0]),np.array([1.,0,0]))
    assert pinch_hypothesis(anchors,native)[0] is None
    native[target]=(np.array([.080,0,0]),np.array([-1.,0,0]))
    assert pinch_hypothesis(anchors,native)[0] is None


def test_ambiguous_tactile_partners_do_not_force_a_pinch():
    anchors=[dict(region=r,position_m=[x,0,0]) for r,x in ((3,0),(6,.03),(9,.031))]
    native={r:(np.array([x,0,0]),np.array([1. if r==3 else -1.,0,0])) for r,x in ((3,0),(6,.03),(9,.031))}
    pair,report=pinch_hypothesis(anchors,native)
    assert pair is None and report['reason']=='ambiguous_partner'


def test_cem_can_escape_an_entirely_infeasible_initial_population():
    settings=CardSettings(population=256,iterations=16)
    initial=(torch.zeros(3),torch.eye(3),torch.full((20,),.3))
    calls=[]
    def evaluate(t,r,j):
        violation=(j[:,0]-.09).clamp_min(0).square()
        calls.append(bool((violation==0).any()))
        return (j[:,0]-.06).square(),violation
    best,report=cem(initial,None,-torch.ones(20),torch.ones(20),evaluate,settings,torch.Generator().manual_seed(7))
    assert not calls[0] and report['feasibility_recovery_iterations']>0
    assert report['feasible'] and best[2][0]<=.09
    assert report['sampled_candidates']==4096
    def impossible(t,r,j):return j.square().sum(-1),torch.ones(len(j))
    _,failed=cem(initial,None,-torch.ones(20),torch.ones(20),impossible,settings,torch.Generator().manual_seed(7))
    assert not failed['feasible'] and failed['best_cost'] is None


def test_cem_returns_improved_current_finite_candidate_when_no_pose_passes():
    settings=CardSettings(population=128,iterations=8)
    initial=(torch.zeros(3),torch.eye(3),torch.full((20,),.3))
    def evaluate(t,r,j):return (j[:,0]-.2).square(),(j[:,0]-.1).square()+.01
    best,report=cem(initial,None,-torch.ones(20),torch.ones(20),evaluate,settings,torch.Generator().manual_seed(7),verify=lambda t,r,j:torch.zeros(len(j),dtype=torch.bool))
    assert not report['feasible'] and report['finite_candidate_found']
    assert report['output_selection']=='least_violation_current_candidate'
    assert report['fallback_violation']<.02 and best[2][0]<.2
    assert torch.isfinite(best[2]).all()


def test_current_endpoint_can_pass_even_when_path_from_old_pose_intersects_obstacle():
    settings=CardSettings(population=32,iterations=2)
    previous=(torch.zeros(3),torch.eye(3),torch.zeros(20));initial=(previous[0],previous[1],torch.full((20,),.8))
    # The swept path 0 -> .8 crosses [.2,.6]. Only candidate endpoints matter.
    def valid(t,r,j):return (j[:,0]<.2)|(j[:,0]>.6)
    def evaluate(t,r,j):return (j[:,0]-.8).square(),(~valid(t,r,j)).float()
    best,report=cem(initial,previous,-torch.ones(20),torch.ones(20),evaluate,settings,torch.Generator().manual_seed(7),verify=valid)
    assert report['feasible'] and best[2][0]>.6
    assert report['collision_scope']=='candidate_endpoint_only_no_swept_path_check'


def test_native_tactile_seed_closes_pinch_without_moving_other_branches():
    from moxian_web.wuji_surface import WujiSurface
    from retarget.env.grasp.pose_gpu import ModelFK,WujiPoseAdapter
    surface=WujiSurface();device=torch.device('cpu');fk=ModelFK(surface.model,device,torch.float32)
    adapter=WujiPoseAdapter(surface.model,device)
    q=torch.tensor([.5699903,-.5983614,.2214672,.0391892,1.4341613,-.2499747,.3348071,.4446326,
        1.3211882,-.3268131,.0926974,.0588825,1.2567488,-.3131828,.0504129,-.0362178,
        1.1641736,-.3064929,.1786014,-.1831942])
    local={3:torch.tensor([-.002775062,.007835438,-.020570922]),6:torch.tensor([.000842185,.008141250,-.012646377])}
    normals={3:torch.tensor([-.2063977,.8458588,-.4918571]),6:torch.tensor([.1629621,.9510565,-.2625552])}
    bodies=[surface.body_ids[i] for i in [0]+[2+4*f+k for f in range(5) for k in range(3)]]
    initial=(torch.zeros(3),torch.eye(3),q)
    proposal,report=pinch_seed(fk,initial,bodies,local,normals,(3,6),adapter.low[6:].float(),adapter.high[6:].float())
    assert report['applied'] and report['gap_before_mm']>30 and report['gap_after_mm']<3
    torch.testing.assert_close(proposal[2][8:],q[8:],rtol=0,atol=0)
    torch.testing.assert_close(proposal[0],initial[0]);torch.testing.assert_close(proposal[1],initial[1])
    assert torch.all(proposal[2]>=adapter.low[6:]) and torch.all(proposal[2]<=adapter.high[6:])


def test_default_card_preserves_source_contacts_instead_of_inventing_pinch_closure(tmp_path):
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.contact import default_mapping
    from moxian_web.wuji_retarget import TactileRetarget
    asset=DEFAULT_MANO.with_name('MANO_RIGHT.pkl')
    if not asset.exists() or not torch.cuda.is_available():pytest.skip('Local MANO and CUDA required')
    torch.set_num_threads(1);m=ManoSurface(asset,'cpu',side='right');m.enable_official_fk()
    m.build_binding(dict(assignments=default_mapping('right'),version='pinch-regression'))
    # A finite source pose with a roughly 30 mm thumb/index material gap.
    pose=torch.tensor([[.1752506,-.1107622,1.8511391],[-.0034721,.0858432,-.0188167],[.0006075,.0305272,.3174581],
        [-.2487103,.2022830,1.5232008],[.0154081,-.0052470,-.0880643],[-.0395337,-.0080462,.2336584],
        [-.7375149,.5369210,.9943999],[.0169309,-.0555751,-.0200023],[-.0367797,-.0372683,.0521598],
        [-.2364290,.2790353,1.3760868],[.0245852,.1363470,-.1263052],[-.0389552,-.0252916,.1722028],
        [1.2062163,.2266542,.2214631],[-.5231077,-.0558528,.1265721],[.0391709,-.1656356,.0631481]])
    with torch.no_grad():v,_,_,j=m.decode(pose[None],torch.eye(3)[None])
    top=m.topology();mask=np.isin(top['sid_patch_indices'],[0,2])
    frame=dict(timestamp_ms=1000.,tactile_timestamp_ms=1000.,vertices_m=v[0].tolist(),joints_m=j[0].tolist(),
        geometry_version=m.shape_version,binding_version=m.binding_version,self_contact_pairs=[],
        source_contact=dict(ready=True,timestamp_ms=1000.,cell_contact=mask.tolist(),normalized=mask.astype(float).tolist()))
    solver=TactileRetarget(top,tmp_path);off=solver.step(frame,contact_enabled=False)
    solver.reset();on=solver.step(frame)
    assert off['solution_feasible'] and not off['solver']['tactile_seed']['applied']
    assert on['display_available']
    assert not on['settings']['experimental_self_contact'] and not on['solver']['tactile_seed']['applied']
    assert on['solver']['contact_association']['selected'] is None and on['self_contact_pairs']==[]
    assert on['contact_constraints_satisfied'] is None
    assert not on['taxel_projection']['physical_registration_verified']
    assert on['solver']['contact_objective_policy']=='CARD_source_CoP_direction_force_and_semantic_pad_search'
    (_,_,_,_),_,anchors=solver.extractor.extract(frame)
    assert len(anchors)==len(on['contact_anchors'])
    for source,output in zip(anchors,on['contact_anchors']):
        np.testing.assert_array_equal(source['position_m'],output['position_m'])
        np.testing.assert_array_equal(source['direction'],output['direction'])
        owner=output['robot_owner'];face=output['robot_face_id'];bc=np.asarray(output['robot_barycentric'])
        local=bc@solver.surface.local[solver.surface.faces[face]]
        point=local@np.asarray(on['link_rotations'][owner]).T+on['link_positions_m'][owner]
        # Native FP64 reconstruction versus independent FP32 Warp/Torch FK;
        # allow 0.1 micrometre, while source targets above remain bit-exact.
        np.testing.assert_allclose(point,output['robot_position_m'],rtol=0,atol=1e-7)
        assert np.min(bc)>=0 and np.sum(bc)==pytest.approx(1.)
    assert not on['source_contact_geometry']['contact_geometry_ready']
    assert len(on['source_contact_geometry']['active_unverified_sids'])==int(mask.sum())
    assert on['quality']['cost_subterms']['contact_self_closure']==0
    assert on['solver']['sampled_candidates']==16384 and not on['solver']['previous_used_as_search_center']
    np.testing.assert_array_equal(on['initial_qpos'],on['robokit_qpos'])

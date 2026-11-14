"""Fixed geometry/axes caching preserves learned outputs and invalidation."""
import copy,json
from pathlib import Path
import numpy as np
import pytest,torch
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface


@pytest.mark.parametrize('side',['left','right'])
def test_cached_fixed_rest_decode_matches_original_pose_correctives_and_lbs(side):
    model=ManoSurface(DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl'),device='cpu',side=side)
    model.set_shape(np.r_[np.full(10,.1),np.zeros(31)])
    model.set_glove_geometry(np.r_[np.full(25,.01),np.zeros(18)])
    model.set_rest_vertex_residual(np.full((778,3),.0001,dtype=np.float32))
    pose=torch.randn(3,15,3,generator=torch.Generator().manual_seed(32))*.3
    root=torch.eye(3)[None].expand(3,-1,-1)
    with torch.enable_grad():expected=model.decode(pose,root)
    with torch.no_grad():actual=model.decode(pose,root)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cached_delta_matches_reference_as_angles_and_touch_change():
    from moxian_web.mano_calibrate_delta import load_policy
    from moxian_web.mano_joint_delta import infer
    from scripts.calibrate_wuji_mano import CHAINS
    base=Path('output/moxian/hand_model/348640923d9e2f3f')
    if not (base/'candidate.json').exists():pytest.skip('Trained checkpoint required')
    p=json.loads((base/'candidate.json').read_text())
    if not p.get('delta_policy'):pytest.skip('Delta checkpoint required')
    rows=json.loads((base/'gpu-cache/78dc1f475a1919e1b9910f8c077f6ee554ee44a7aca194922600718284c79dcc/prepared.json').read_text())['rows']
    chosen=[next(r for r in rows if r.get('partition')=='validation' and r['stage']==s) for s in ('contact_pair_index','contact_five','contact_fist')]
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),side='right');m.enable_official_fk();m.build_binding(p['mapping'])
    m.set_rest_vertex_residual(p['rest_vertex_residual_m']);m.set_glove_geometry(p['glove_parameters']);m.set_shape(p['shape'])
    m.raw_rotation=np.asarray(p['raw_rotation']);m.raw_translation=np.asarray(p['raw_translation']);m.flat_reference=p['flat_reference'];m.sensor_joint_position=np.asarray(p['sensor_joint_position']);m.mount_rot=np.asarray(p['mount_rot'])
    m.delta_artifact=p['delta_policy'];m.delta_policy=load_policy(p['delta_policy'],m.device)
    for r in chosen:
        pose,root,*_=m.official_fk.parameters(m,r,CHAINS)
        reference=infer(m,[r],pose[None],root[None],fast=False)
        actual=infer(m,[r],pose[None],root[None])
        assert not getattr(m,'_delta_disabled_reason',None)
        torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
        row=copy.deepcopy(r);row.update(contact={'ready':False},tactile_timestamp_ms=None)
        torch.testing.assert_close(infer(m,[row],pose[None],root[None]),infer(m,[row],pose[None],root[None],fast=False),atol=2e-6,rtol=2e-5)
    previous=m._delta_runtime_chart
    m.delta_artifact=copy.deepcopy(m.delta_artifact)
    infer(m,[chosen[-1]],pose[None],root[None]);assert m._delta_runtime_chart is not previous

"""Numerical parity with upstream MANO, including the old override paths."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from smplx import MANO
from smplx.utils import Struct
from moxian_web.mano_runtime import ManoSurface
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.tactile_surface import CurvedTaxels, surface_points
from scripts.calibrate_wuji_mano import _ManoReader, as_array

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_surface_matches_upstream_for_nonzero_betas_and_articulation(device):
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO weights required')
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=ManoSurface(DEFAULT_MANO,device=device)
    with DEFAULT_MANO.open('rb') as stream:data=_ManoReader(stream,encoding='latin1').load()
    fields={k:as_array(data[k]) for k in ('v_template','shapedirs','posedirs','J_regressor','weights','hands_components','hands_mean','kintree_table','f')}
    reference=MANO(str(DEFAULT_MANO),data_struct=Struct(**fields),is_rhand=False,
                   use_pca=False,flat_hand_mean=True,create_transl=False).to(device)
    rng=np.random.default_rng(42)
    pose=torch.tensor(rng.normal(0,.3,(4,15,3)),dtype=torch.float32,device=device)
    pose[0]=0
    root=torch.tensor(Rotation.from_rotvec(rng.normal(0,.3,(4,3))).as_matrix(),dtype=torch.float32,device=device)
    for beta in (np.zeros(10),rng.normal(0,.7,10)):
        m.bone_reference=np.full((5,3),.1) # A glove bone prior must not change shape.
        m.set_shape(beta)
        b=torch.tensor(beta,dtype=torch.float32,device=device)[None].expand(4,-1)
        with torch.no_grad():
            expected=reference(betas=b,hand_pose=pose.reshape(4,45),global_orient=torch.zeros(4,3,device=device))
            vertices,joints,_,_=m.decode(pose,root,torch.ones(4,5,3,device=device),torch.ones(4,16,3,device=device))
        shaped=reference.v_template+torch.einsum('vck,k->vc',reference.shapedirs,b[0])
        wrist=(reference.J_regressor@shaped)[0]
        torch.testing.assert_close(vertices,(expected.vertices-wrist)@root.transpose(-1,-2),atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(joints,(expected.joints-wrist)@root.transpose(-1,-2),atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(m.rest,shaped)
        # Nonzero articulation must keep the upstream pose corrective contribution.
        assert vertices.shape==(4,778,3) and m.faces.shape==(1538,3)
    with pytest.raises(ValueError):m.set_dimensions(np.ones(20)*.01)
    shape=np.zeros(41);shape[21]=.001
    with pytest.raises(ValueError):m.set_shape(shape)


def test_official_taxels_use_original_triangle_surface():
    vertices=np.array([[0.,0,0],[1,0,0],[0,1,1],[0,0,1]])
    topology={'faces':[[0,1,2],[0,3,1]],'sid_face_ids':[0,1],
              'sid_barycentric':[[.2,.3,.5],[.1,.6,.3]],'surface_policy':'official_MANO_triangles'}
    points,normals=CurvedTaxels(topology).evaluate(vertices)
    np.testing.assert_allclose(points,surface_points(vertices,np.array(topology['faces']),topology['sid_face_ids'],np.array(topology['sid_barycentric'])))
    assert np.isfinite(normals).all()


def test_official_reference_uses_fixed_recorded_pose_without_extra_ik():
    from unittest.mock import patch
    from moxian_web.official_mano import reference_from_frame
    from scripts.calibrate_wuji_mano import CHAINS
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO weights required')
    m=ManoSurface(DEFAULT_MANO,device='cpu');m.enable_official_fk()
    skeleton=dict(m.official_fk.reference);skeleton['timestamp_ms']=1000.
    frame={'timestamp_ms':1000.,'skeleton':skeleton}
    pose,root,_,_,_=m.official_fk.parameters(m,frame,CHAINS)
    with torch.no_grad():_,j,r,_=m.decode(torch.tensor(pose[None]),torch.tensor(root[None]))
    offsets=m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]]
    mount=offsets-.008*offsets/np.linalg.norm(offsets,axis=1,keepdims=True)
    distal=r[0,[c[-1] for c in CHAINS]].numpy()
    predicted=j[0,[c[-1] for c in CHAINS]].numpy()+(distal@mount[:,:,None]).squeeze(-1)
    rotate=Rotation.from_euler('xyz',[10,-5,20],degrees=True).as_matrix();translation=np.array([.01,.02,.03])
    observed=predicted@rotate.T+translation
    names=('thumb','index','middle','ring','pinky')
    frame['fingers']={f:{'position':observed[i].tolist(),'quaternion_xyzw':Rotation.from_matrix(rotate@distal[i]).as_quat().tolist()} for i,f in enumerate(names)}
    reference=reference_from_frame(frame,m.weights_hash,'test','recorded_open')
    before=m.rest_np.copy()
    with patch.object(m,'fit_thumb_directions',side_effect=AssertionError('extra thumb IK')),patch('moxian_web.mano_runtime.least_squares',side_effect=AssertionError('extra pose fitting')):
        report=m.install_flat_reference(reference)
    np.testing.assert_allclose(m.raw_rotation,rotate.T,atol=1e-6)
    np.testing.assert_allclose(m.raw_translation,-rotate.T@translation,atol=1e-6)
    np.testing.assert_array_equal(m.rest_np,before)
    np.testing.assert_allclose(m.flat_seed,pose)
    assert not report['changes_shape'] and not report['changes_pose']

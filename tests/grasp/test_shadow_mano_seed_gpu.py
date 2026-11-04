import os
import pytest
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from retarget.env.grasp.shadow_mano_seed import ShadowManoTransfer
from retarget.env.grasp.shadow_mano_seed_gpu import CudaShadowManoSeed

ROOT=Path(__file__).resolve().parents[2]
SHADOW=ROOT/'output/graspadmm_native_20261104/upstream/assets/hand/shadow/right.xml'
MANO=ROOT/'output/graspadmm_mano_native_20261104/hand/mano/right.xml'


SHADOW=Path(os.environ.get('RETARGET_SHADOW_XML',str(SHADOW)))
MANO=Path(os.environ.get('RETARGET_RIGID_MANO_XML',str(MANO)))
pytestmark=pytest.mark.skipif(not SHADOW.is_file() or not MANO.is_file(),reason='Requires external original Shadow and repaired rigid MANO assets')

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_gpu_rigid_initialization_retains_correspondences_bounds_and_position_cost():
    reference=ShadowManoTransfer(SHADOW,MANO)
    bodies=['rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_thdistal']
    local=np.array([reference.shadow.mesh(b).vertices[0] for b in bodies])
    adapter=CudaShadowManoSeed(SHADOW,MANO,bodies,local)
    q=np.r_[[.1,.0,.1],Rotation.from_rotvec([.2,-.1,.3]).as_quat(scalar_first=True),np.zeros(22)]
    gpu=torch.tensor(q,device='cuda',dtype=torch.float64)[None]
    points,p,r=adapter.points(gpu[:,7:],source=True)
    reference.shadow.set(q[7:])
    np.testing.assert_allclose(points[0].cpu(),reference.shadow.points_local(),atol=1e-10)
    result=adapter.fit(gpu)
    assert result['finite'].all() and result['grasp_qpos'].is_cuda
    assert adapter.bodies==['index3','middle3','ring3','thumb3']
    assert (result['contact_cost']<=result['initial_contact_cost']).all()
    actual=result['grasp_qpos'][0].cpu().numpy()
    assert np.all(actual[7:]>=reference.mano.low) and np.all(actual[7:]<=reference.mano.high)
    world=reference.mano.positions(actual,adapter.bodies,adapter.target_local.cpu().numpy())
    np.testing.assert_allclose(world,result['hand_cpn_w'][0,:,:3].cpu(),atol=1e-10)
    target=reference.shadow.positions(q,bodies,local)
    np.testing.assert_allclose(target,result['source_hand_points'][0].cpu(),atol=1e-10)
    expected=np.square(world-target).sum()
    np.testing.assert_allclose(result['contact_cost'][0].cpu(),expected,atol=1e-12,rtol=1e-8)

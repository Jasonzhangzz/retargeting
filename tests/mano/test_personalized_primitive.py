"""Reflection and actual personalized link geometry, without an optimizer run."""
from pathlib import Path
import numpy as np
import torch
import pytest
from scipy.spatial.transform import Rotation
from retarget.env.grasp.personalized_primitive import FrozenPersonalizedDecoder
from retarget.env.grasp.mano_solver import ManoDecoder
SNAPSHOT=Path('output/mano_wuji_primitives_20261105/saved_left')
@pytest.mark.skipif(not SNAPSHOT.exists(),reason='Saved local personal hand required')
def test_reflection_preserves_dimensions_winding_and_articulation():
    m=FrozenPersonalizedDecoder(SNAPSHOT,'cpu');a=np.load(SNAPSHOT/'geometry.npz');mirror=np.array([-1,1,1]);axial=np.array([1,-1,-1])
    np.testing.assert_allclose(m.rest.numpy(),a['rest_vertices']*mirror,atol=1e-8)
    np.testing.assert_allclose(m.joints_rest.numpy(),a['rest_joints']*mirror,atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(np.diff(m.rest.numpy(),axis=0),axis=-1),np.linalg.norm(np.diff(a['rest_vertices'],axis=0),axis=-1),atol=1e-8)
    np.testing.assert_array_equal(m.base.faces,a['faces'][:,[0,2,1]])
    assert torch.count_nonzero(m.base.posedirs)==0
    original=ManoDecoder(SNAPSHOT/'MANO_LEFT.pkl','cpu');original.rest=torch.tensor(a['rest_vertices']);original.joints_rest=torch.tensor(a['rest_joints']);original.relative_joints=torch.stack([original.joints_rest[j] if p<0 else original.joints_rest[j]-original.joints_rest[p] for j,p in enumerate(original.base.parents)]);original.base.posedirs.zero_()
    q=torch.linspace(-.15,.15,30).reshape(1,30);q[:,:6]=0
    with torch.no_grad():v=original(q,full=True)[0].numpy();reflected=m(q,full=True)[0].numpy()
    np.testing.assert_allclose(reflected,v*mirror,atol=1e-7)
    r=np.array([.5,-.2,.3]);M=np.diag(mirror)
    np.testing.assert_allclose(Rotation.from_rotvec(r*axial).as_matrix(),M@Rotation.from_rotvec(r).as_matrix()@M,atol=1e-12)

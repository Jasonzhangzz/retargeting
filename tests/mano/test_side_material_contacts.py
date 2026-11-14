import numpy as np
import pytest
import torch
from moxian_web.side_contact import closest_segments,side_material_pairs
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.proximal_shape import proximal_pair_gaps
from scripts.calibrate_wuji_mano import CHAINS


def test_coplanar_closest_points_find_interior_contact_and_reject_connected_pairs():
    a=np.array([[[-1.,0,0],[1.,0,0]]]);b=np.array([[[0,-1.,0],[0,1.,0]],[[3,-1.,0],[3,1.,0]]])
    i,j,t,s=closest_segments(a,b,np.array([0,0,1.]),np.array([1.,0,0]),np.zeros((1,2),bool))
    assert (i,j)==(0,0) and np.isclose(t,.5) and np.isclose(s,.5)
    i,j,t,s=closest_segments(a,b,np.array([0,0,1.]),np.array([1.,0,0]),np.array([[True,False]]))
    assert j==1 and np.isclose(t,1) and np.isclose(s,.5)


def test_side_witnesses_and_validation_use_identical_three_proximal_sections():
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(DEFAULT_MANO,device='cpu');pairs=side_material_pairs(m,m.rest_np,m.rest_j_np)
    sections=proximal_pair_gaps(m.rest_np,m.rest_j_np,m.faces,m.model.weights.cpu().numpy(),CHAINS,True)
    assert len(pairs)==len(sections)==9
    for a,b in zip(pairs,sections):
        assert a['pair']==b['pair'] and a['level']==b['level']
        assert abs(a['gap_m']-b['surface_distance_m'])<1e-4
        assert a['normal_dot']<-.1
        for key in ('a','b'):
            bc=np.array(a['barycentric_'+key]);assert np.all(bc>=-1e-6) and np.isclose(bc.sum(),1)

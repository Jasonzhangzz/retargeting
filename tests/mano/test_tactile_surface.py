import numpy as np
import pytest
from moxian_web.tactile_surface import bind_grid,surface_points
from moxian_web.hand_dimensions import scale_proximal_width


def test_grid_material_binding_is_continuous_and_follows_a_deforming_triangle():
    vertices=np.array([[0.,0,0],[1.,0,0],[1.,1,0],[0.,1,0]])
    faces=np.array([[0,1,2],[0,2,3]])
    sids=np.array([0,1,2,3,20,21,22,23])
    ids,bc=bind_grid(vertices[faces],sids,[0,1,0],[1,0,0])
    points=surface_points(vertices,faces,ids,bc)
    assert len(np.unique(points,axis=0))==8
    assert np.allclose(points[:4,0],[.125,.375,.625,.875])
    assert np.allclose(points[:4,1],.25)
    deformed=vertices.copy();deformed[2,2]=.4
    moved=surface_points(deformed,faces,ids,bc)
    assert np.allclose(moved[:,:2],points[:,:2])
    assert np.max(moved[:,2])>0
    assert np.allclose(bc.sum(1),1) and np.min(bc)>=0


def test_real_mano_460_distinct_fixed_material_points_and_shape_changes():
    from moxian_web.hand_model import DEFAULT_MANO
    if not DEFAULT_MANO.exists():pytest.skip('local licensed MANO missing')
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.contact import default_mapping
    m=ManoSurface(DEFAULT_MANO,'cpu');m.set_dimensions(np.zeros(20))
    m.build_binding({'assignments':default_mapping('left'),'version':'test'})
    face=m.sid_faces.copy();bc=m.sid_bary.copy()
    p=surface_points(m.rest_np,m.faces,face,bc)
    assert len(np.unique(np.round(p,8),axis=0))==460
    assert all(m.face_patches[face[i]]==m.mapping['assignments'][i] for i in range(460))
    m.proximal_lateral_scales=np.log([1.1,1.2,1.,1.1]);oldj=m.rest_j_np.copy();m.set_shape(m.shape)
    assert np.array_equal(face,m.sid_faces) and np.array_equal(bc,m.sid_bary)
    assert np.array_equal(oldj,m.rest_j_np)
    assert not np.allclose(p,surface_points(m.rest_np,m.faces,face,bc))


def test_zero_proximal_scale_is_identity():
    vertices=np.random.default_rng(1).normal(size=(20,3));joints=np.arange(48,dtype=float).reshape(16,3)+np.random.default_rng(2).normal(size=(16,3))
    chains=((13,14,15),(1,2,3),(4,5,6),(10,11,12),(7,8,9))
    weights=np.eye(16)[np.arange(20)%16]
    assert np.array_equal(scale_proximal_width(vertices,joints,weights,chains,np.array([0.,0,1]),np.zeros(4)),vertices)

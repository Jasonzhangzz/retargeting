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
    oldj=m.rest_j_np.copy();beta=np.zeros(10);beta[0]=1.;m.set_shape(beta)
    assert np.array_equal(face,m.sid_faces) and np.array_equal(bc,m.sid_bary)
    assert not np.array_equal(oldj,m.rest_j_np)
    assert not np.allclose(p,surface_points(m.rest_np,m.faces,face,bc))


def test_zero_proximal_scale_is_identity():
    vertices=np.random.default_rng(1).normal(size=(20,3));joints=np.arange(48,dtype=float).reshape(16,3)+np.random.default_rng(2).normal(size=(16,3))
    chains=((13,14,15),(1,2,3),(4,5,6),(10,11,12),(7,8,9))
    weights=np.eye(16)[np.arange(20)%16]
    assert np.array_equal(scale_proximal_width(vertices,joints,weights,chains,np.array([0.,0,1]),np.zeros(4)),vertices)

def test_cross_section_contact_measures_true_distance_not_only_projection():
    from moxian_web.proximal_shape import section_distance
    a=np.array([[[0.,0.],[1.,0.]]]);b=np.array([[[0.,.005],[1.,.005]]])
    assert section_distance(a,b)==pytest.approx(.005)
    assert section_distance(a,np.array([[[.5,-1.],[.5,1.]]]))==0


def test_overlapping_chart_faces_preserve_measured_depth_layer():
    from moxian_web.card_geometry import chart_point
    tri=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,.02],[1,0,.02],[0,1,.02]]],float)
    point,face,bc=chart_point(tri,np.zeros(3),np.array([1.,0,0]),np.array([0.,0,1]),np.array([.2,.3]),np.array([.2,.3,.02]))
    assert face==1
    np.testing.assert_allclose(point,[.2,.3,.02]);np.testing.assert_allclose(bc@tri[face],point)


@pytest.mark.parametrize('side',['left','right'])
def test_isolated_taxel_contact_does_not_jump_to_another_material_face(side):
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.contact import default_mapping
    from moxian_web.card_geometry import AnchorExtractor
    from moxian_web.wuji_surface import mirror_matrix,CHAINS
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('licensed MANO missing')
    m=ManoSurface(path,'cpu',side);m.build_binding({'assignments':default_mapping(side),'version':'isolated'})
    topology=m.topology();ex=AnchorExtractor(topology,mirror_matrix(topology))
    j=m.rest_j_np;tips=m.tip_anchors
    frame=dict(vertices_m=m.rest_np.tolist(),joints_m=np.vstack((j[0],*[np.vstack((j[list(c)],tips[f])) for f,c in enumerate(CHAINS)])).tolist(),
               timestamp_ms=1000.,tactile_timestamp_ms=1000.)
    for sid in range(460):
        mask=[False]*460;mask[sid]=True
        frame['source_contact']=dict(ready=True,timestamp_ms=1000.,cell_contact=mask,normalized=[.5]*460)
        (_,_,points,_),_,anchors=ex.extract(frame)
        assert len(anchors)==1
        np.testing.assert_allclose(anchors[0]['position_m'],points[sid],atol=1e-12)
        assert anchors[0]['source_face']==m.sid_faces[sid]
        if sid%20==0:
            from moxian_web.tactile_pose import material_anchors
            local=material_anchors(topology,m.rest_np,np.asarray(frame['joints_m']),frame['source_contact'],1000.,1000.)
            np.testing.assert_allclose(local[0]['position_m'],surface_points(m.rest_np,m.faces,m.sid_faces,m.sid_bary)[sid],atol=1e-9)


def test_registered_material_points_survive_beta_edits_but_validation_is_stale():
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.contact import default_mapping
    from moxian_web.contact_registration import context,fit_registration
    if not DEFAULT_MANO.exists():pytest.skip('licensed MANO missing')
    m=ManoSurface(DEFAULT_MANO,'cpu','left');m.build_binding({'assignments':default_mapping('left'),'version':'registration'})
    top=m.topology();sid=354;face=int(m.sid_faces[sid]);bc=np.array([.2,.3,.5]);frames=[]
    for i in range(4):
        v=m.rest_np.copy();v[:,2]+=i*.001;tri=v[m.faces[face]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
        frames.append(dict(id=str(i),split='train' if i<2 else 'validation',vertices_m=v.tolist(),observations=[dict(sid=sid,position_m=(bc@tri).tolist(),normal=normal.tolist())]))
    artifact=fit_registration(top,'device',dict(context=context(top,'device'),coordinate_frame='mano_wrist_local',units='m',measurement_source='synthetic test reference',frames=frames))
    m.install_contact_registration(artifact,'device')
    assert m.contact_registration_status['verified_sids']==[sid]
    oldface=m.sid_faces.copy();oldbary=m.sid_bary.copy();before=m.binding_version
    with pytest.raises(ValueError):m.install_contact_registration(artifact,'different device')
    assert m.binding_version==before
    beta=np.zeros(10);beta[0]=.1;m.set_shape(beta)
    assert m.sid_faces[sid]==oldface[sid]
    np.testing.assert_array_equal(m.sid_bary[sid],oldbary[sid])
    assert m.contact_registration_status['status']=='stale_or_invalid'
    assert not m.contact_registration_status['verified_sids']

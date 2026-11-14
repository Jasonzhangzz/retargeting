import copy
import numpy as np
import pytest
from moxian_web.contact_registration import context,fit_registration,validate_registration,registration_status,digest


def fixture():
    vertices=np.array([[0.,0,0],[.02,0,0],[0,.02,0],[0,0,-.001],[.02,0,-.001],[0,.02,-.001]])
    top=dict(hand_side='right',weights_hash='asset',geometry_version='shape',faces=[[0,1,2],[3,5,4]],
             face_patches=[0,0],sid_patch_indices=[0]*460,rest_vertices_m=vertices.tolist())
    data=dict(context=context(top,'device'),coordinate_frame='mano_wrist_local',units='m',
              measurement_source='independent measured probe and surface normal',frames=[])
    bc=np.array([.2,.3,.5])
    for i in range(4):
        v=vertices.copy();v[2,2]=i*.001
        tri=v[[0,1,2]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
        data['frames'].append(dict(id=str(i),split='train' if i<2 else 'validation',vertices_m=v.tolist(),
                                  observations=[dict(sid=3,position_m=(bc@tri).tolist(),normal=normal.tolist())]))
    return top,data,bc


def test_measured_registration_selects_material_face_and_tracks_deformation():
    top,data,bc=fixture();result=fit_registration(top,'device',data)
    assert result['accepted'] and not result['complete']
    record=validate_registration(result,top,'device')['3']
    assert record['face']==0
    np.testing.assert_allclose(record['barycentric'],bc,atol=1e-12)
    faces=np.zeros(460,int);bary=np.zeros((460,3));bary[3]=bc
    status=registration_status(result,top,'device',faces,bary)
    assert status['verified_sids']==[3] and status['status']=='partial' and not status['physical_registration_verified']
    bary[3]=[0,0,1] # Manual edits cannot inherit measured validation.
    assert not registration_status(result,top,'device',faces,bary)['verified_sids']


def test_holdout_does_not_update_material_coordinates_or_accept_bad_fit():
    top,data,_=fixture();first=fit_registration(top,'device',data)
    data['frames'][3]['observations'][0]['position_m'][0]+=.01
    rejected=fit_registration(top,'device',data)
    assert not rejected['accepted'] and '3' in rejected['failures']
    assert first['records']['3']['metrics']['validation']['max_error_mm']<1e-8


@pytest.mark.parametrize('field',['geometry_version','weights_hash','hand_side'])
def test_stale_model_cannot_inherit_measured_registration(field):
    top,data,_=fixture();result=fit_registration(top,'device',data);top[field]='changed'
    with pytest.raises(ValueError):validate_registration(result,top,'device')


def test_rejects_wrong_units_duplicates_nonfinite_and_claim_without_evidence():
    top,data,_=fixture()
    for mutate in (lambda x:x.update(units='mm'),lambda x:x['frames'][3].update(id='0'),
                   lambda x:x['frames'][0]['observations'][0].update(position_m=[np.nan,0,0]),
                   lambda x:x.update(measurement_source='')):
        bad=copy.deepcopy(data);mutate(bad)
        with pytest.raises(ValueError):fit_registration(top,'device',bad)
    result=fit_registration(top,'device',data);result['records']['3']['metrics']['validation']['frames']=0
    result['version']=digest({k:v for k,v in result.items() if k!='version'})
    with pytest.raises(ValueError):validate_registration(result,top,'device')


def test_normal_disambiguation_and_validation_are_independent():
    top,data,_=fixture()
    for f in data['frames'][2:]:f['observations'][0]['normal']=[0,0,-1]
    result=fit_registration(top,'device',data)
    assert not result['accepted'] and result['failures']['3']['validation']['max_normal_error_deg']>170

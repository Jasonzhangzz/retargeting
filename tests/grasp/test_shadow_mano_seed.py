import os
import pytest
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from retarget.env.grasp.shadow_mano_seed import ShadowManoTransfer

ROOT=Path(__file__).resolve().parents[2]
SHADOW=ROOT/'output/graspadmm_native_20261104/upstream/assets/hand/shadow/right.xml'
MANO=ROOT/'output/graspadmm_mano_native_20261104/hand/mano/right.xml'


SHADOW=Path(os.environ.get('RETARGET_SHADOW_XML',str(SHADOW)))
MANO=Path(os.environ.get('RETARGET_RIGID_MANO_XML',str(MANO)))
pytestmark=pytest.mark.skipif(not SHADOW.is_file() or not MANO.is_file(),reason='Requires external original Shadow and repaired rigid MANO assets')

def test_transfer_uses_correct_fingers_and_reduces_correspondence_error():
    fit=ShadowManoTransfer(SHADOW,MANO)
    hand=fit.shadow;q=np.r_[[.01,-.02,.03],Rotation.from_rotvec([.2,-.1,.3]).as_quat(scalar_first=True),np.zeros(hand.model.nq)]
    bodies=['rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_thdistal']
    local=np.array([hand.mesh(b).vertices[np.argmax(hand.mesh(b).vertices[:,0])] for b in bodies])
    mapped,owners,contacts,info=fit.fit(q,bodies,local)
    assert owners==['index3','middle3','ring3','thumb3']
    assert len(mapped)==27 and np.isfinite(mapped).all()
    assert np.all(mapped[7:]>=fit.mano.low) and np.all(mapped[7:]<=fit.mano.high)
    assert info['contact_after_rms_m']<=info['contact_before_rms_m']+1e-9
    np.testing.assert_allclose(np.linalg.norm(contacts[:,3:],axis=1),1.,atol=1e-8)
    reconstructed=fit.mano.contacts(mapped,owners,np.array(info['mapped_hand_cpn_b'])[:,:3])
    np.testing.assert_allclose(reconstructed,contacts,atol=1e-10)


def test_external_wrist_transform_matches_mujoco_attached_free_hand():
    fit=ShadowManoTransfer(SHADOW,MANO);hand=fit.mano
    spec=mujoco.MjSpec();child=mujoco.MjSpec.from_file(str(MANO))
    for mesh in child.meshes:mesh.file=str(MANO.parent/child.meshdir/mesh.file)
    child.meshdir='';body=spec.worldbody.add_frame().attach_body(child.worldbody,'hand-','');body.add_freejoint()
    model=spec.compile();data=mujoco.MjData(model)
    q=np.r_[[.12,-.07,.2],Rotation.from_rotvec([.4,-.2,.1]).as_quat(scalar_first=True),np.clip(np.ones(20)*.15,hand.low,hand.high)]
    data.qpos[:]=q;mujoco.mj_kinematics(model,data)
    bodies=['index3','middle3','ring3','thumb3'];local=np.array([hand.mesh(b).vertices[0] for b in bodies])
    expected=[]
    for name,p in zip(bodies,local):
        bid=model.body('hand-'+name).id;expected.append(data.xpos[bid]+data.xmat[bid].reshape(3,3)@p)
    np.testing.assert_allclose(hand.positions(q,bodies,local),expected,atol=1e-12)


def test_surface_normals_follow_the_selected_surface_point():
    fit=ShadowManoTransfer(SHADOW,MANO);hand=fit.mano;mesh=hand.mesh('index3')
    positive=mesh.vertices[np.argmax(mesh.vertices[:,0])];negative=mesh.vertices[np.argmin(mesh.vertices[:,0])]
    _,_,normal=hand.project('index3',np.stack((positive,negative)))
    assert normal[0,0]>0 and normal[1,0]<0
    assert not np.allclose(normal[0],normal[1])


def test_full_dataset_contact_regions_stay_on_corresponding_segments():
    fit=ShadowManoTransfer(SHADOW,MANO)
    bodies=['rh_ffproximal','rh_ffmiddle','rh_thmiddle','rh_palm','rh_lfmetacarpal']
    local=np.array([fit.shadow.mesh(b).vertices[0] for b in bodies])
    owners,points=fit.map_contacts(bodies,local)
    assert owners==['index1x','index2','thumb2','rh_palm','rh_palm']
    assert np.isfinite(points).all()
    for body,point in zip(owners,points):
        _,gap,_=fit.mano.project(body,point[None])
        assert gap[0]<1e-10

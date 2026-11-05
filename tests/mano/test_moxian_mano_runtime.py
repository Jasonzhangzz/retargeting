"""Real local MANO/GPU checks; no hardware commands or user calibration writes."""
import copy,time
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from moxian_web.mano_runtime import ManoSurface,CollisionSurface,deform_surface,fuse_contacts,vertex_normals,blend_rotations
from moxian_web.skeleton import JOINT_NAMES
from scipy.spatial.transform import Rotation
from moxian_web.contact import PATCH_IDS,default_mapping,PATCHES
from moxian_web.hand_model import DEFAULT_MANO
from scripts.calibrate_wuji_mano import CHAINS

@pytest.fixture(scope='module')
def model():
    if not DEFAULT_MANO.exists() or not torch.cuda.is_available():pytest.skip('Local left MANO and CUDA required')
    m=ManoSurface(DEFAULT_MANO);m.build_binding({'assignments':default_mapping('left'),'version':'test','confirmed':PATCH_IDS.copy()});return m

def test_778_vertices_460_unique_sids_twenty_anatomical_patches(model):
    m=model
    assert m.rest_np.shape==(778,3)
    assert len(m.sid_faces)==460 and len(m.sid_bary)==460
    assert all(len(p)>0 for p in m.patch_faces)
    assert np.allclose(m.sid_bary.sum(-1),1) and np.min(m.sid_bary)>=0
    assert set(m.mapping['assignments'])==set(range(20))
    assert all(m.face_patches[m.sid_faces[sid]]==m.mapping['assignments'][sid] for sid in range(460))

def test_thickness_keeps_joint_lengths_and_small_length_corrections_are_distinct(model):
    m=model;m.bone_reference=np.array([[.035,.025,.02]]*5);m.set_shape(np.zeros(41));basej=m.rest_j_np.copy();basev=m.rest_np.copy();tips=m.tip_anchors.copy()
    shape=m.shape.copy();shape[21:]=.001;m.set_shape(shape)
    assert np.allclose(m.rest_j_np,basej) and np.allclose(m.tip_anchors,tips)
    assert not np.allclose(m.rest_np,basev)
    shape[11]=np.log(1.02);shape[16]=.001;m.set_shape(shape)
    # Bone segments are information about total size, not exact surface lengths.
    assert not np.isclose(np.linalg.norm(basej[14]-basej[13]),.035,atol=1e-6)
    assert np.linalg.norm(m.rest_j_np[14]-m.rest_j_np[13])>np.linalg.norm(basej[14]-basej[13])
    assert np.linalg.norm(m.tip_anchors[0]-m.rest_j_np[15])>np.linalg.norm(tips[0]-basej[15])
    m.bone_reference=None;m.set_shape(np.zeros(41))

def test_native_collision_version_and_triangle_verification(model):
    m=model;m.set_shape(np.zeros(41));q=torch.zeros((1,15,3),device=m.device);root=torch.eye(3,device=m.device)[None]
    with torch.no_grad():v,j,r,o=m.decode(q,root)
    g=CollisionSurface(m);geom=g.query(v[0].cpu().numpy(),j[0].cpu().numpy(),r[0].cpu().numpy())
    assert not geom['overflow'] and geom['gap'].shape==(1238,20)
    assert np.isfinite(geom['gap']).all()
    # Deliberate convex-hull false positive: actual opposing surfaces remain
    # 15 mm apart. ADC must not turn it into a confirmed geometric contact.
    fake=copy.deepcopy(geom);fake['gap'][:]=.015;fake['facing'][:]=-.8
    fake['native_pairs']={tuple(sorted((a,b))) for a in m.patch_owners for b in m.patch_owners if a!=b}
    frame={'contact':{'ready':True,'patches':[{**p,'contact':True,'response':.3,'active_cells':2} for p in PATCHES]}}
    result=fuse_contacts(m,frame,fake)
    assert result['count']==0 and all(p['state']=='conflict' for p in result['patches'])
    change=m.shape.copy();change[21]=.001;m.set_shape(change)
    with pytest.raises(ValueError,match='旧尺寸'):g.query(v[0].cpu().numpy(),j[0].cpu().numpy(),r[0].cpu().numpy())
    fresh=CollisionSurface(m);assert fresh.version==m.shape_version
    m.set_shape(np.zeros(41))

def test_compression_does_not_mutate_rest_and_releases(model):
    m=model;v=m.rest_np.copy();saved=m.rest_np.copy();p=m.rest_np[m.faces[m.patch_faces[0][0]]].mean(0)
    f={'ready':True,'pairs':[{'state':'confirmed','patch_a':PATCH_IDS[0],'patch_b':PATCH_IDS[2],
        'point_a_m':p.tolist(),'point_b_m':p.tolist(),'normal_a':[0,0,1],'normal_b':[0,0,-1],
        'compression_a_m':.001,'compression_b_m':.0005}]}
    out,report=deform_surface(m,v,f,1000)
    assert report['max_m']<=.002 and np.array_equal(m.rest_np,saved)
    assert np.allclose(out[m.wrist_mask],v[m.wrist_mask])
    for t in range(1040,2000,40):out,report=deform_surface(m,v,{'ready':True,'pairs':[]},t)
    assert report['max_m']<1e-6 and np.array_equal(m.rest_np,saved)

def test_bounded_powell_recovers_observed_thickness_with_noisy_surface_samples(model):
    m=model;m.set_shape(np.zeros(41));known=.0012;shape=m.shape.copy();shape[21]=known;m.set_shape(shape)
    rng=np.random.default_rng(7);target=m.rest_np.copy()+rng.normal(0,.00002,(778,3))
    ids=np.flatnonzero(m.vertex_patches==0)
    def objective(x):
        value=np.zeros(41);value[21]=x[0];m.set_shape(value)
        return float(np.mean((m.rest_np[ids]-target[ids])**2))
    result=minimize(objective,[0.],method='Powell',bounds=[(-.002,.002)],options={'xtol':1e-7})
    assert abs(result.x[0]-known)<.0001
    # No information was supplied for other patches, and none was invented.
    assert np.allclose(m.shape[22:],0)
    m.set_shape(np.zeros(41))

def test_rotation_smoothing_uses_short_path_across_pi():
    a=Rotation.from_rotvec([0,0,np.deg2rad(179)]).as_matrix()
    b=Rotation.from_rotvec([0,0,np.deg2rad(-179)]).as_matrix()
    middle=blend_rotations(a,b,.5)
    assert np.isclose(Rotation.from_matrix(a.T@middle).magnitude(),np.deg2rad(1))
    assert np.isclose(Rotation.from_matrix(middle).magnitude(),np.pi)

def test_anatomical_hinges_reject_roll_splay_and_overfolding(model):
    m=model;m.bone_reference=None;m.set_shape(np.zeros(41))
    rng=np.random.default_rng(21);wild=rng.normal(0,4,(12,15,3));projected=m.anatomy.project(wild)
    a=m.anatomy.angles(projected)
    assert np.all(a>=m.anatomy.lower-1e-5) and np.all(a<=m.anatomy.upper+1e-5)
    hinges=[c[k]-1 for f,c in enumerate(CHAINS) for k in range(1 if f else 2,3)]
    assert np.abs(a[:,hinges,1:]).max()<1e-5
    # Learned relaxed-hand flexion establishes the left hand's palmar sign.
    mean=m.anatomy.angles(m.model.pose_mean.cpu().numpy().reshape(15,3))
    assert np.median([mean[c[1]-1,0] for c in CHAINS[1:]])>0

def test_thumb_opposition_solves_three_link_directions_together(model):
    m=model;m.bone_reference=None;m.set_shape(np.zeros(41))
    for cmc in ([20,-20,0],[50,10,20],[-15,25,-15]):
        angles=np.zeros((15,3));angles[12:15]=np.deg2rad([cmc,[40,10,0],[30,0,0]])
        b=m.anatomy.basis
        pose=Rotation.from_matrix(b@Rotation.from_euler('XYZ',angles).as_matrix()@b.swapaxes(-2,-1)).as_rotvec()
        with torch.no_grad():
            _,_,_,known=m.decode(torch.tensor(pose[None],dtype=torch.float32,device=m.device),torch.eye(3,device=m.device)[None])
        target=known[0].cpu().numpy();seed,root=m.pose_from_bones(target)
        with torch.no_grad():_,_,_,solved=m.decode(torch.tensor(seed[None],device=m.device),torch.tensor(root[None],device=m.device))
        actual=np.diff(solved[0,1:5].cpu().numpy(),axis=0);expected=np.diff(target[1:5],axis=0)
        cosine=(actual*expected).sum(1)/np.linalg.norm(actual,axis=1)/np.linalg.norm(expected,axis=1)
        errors=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
        assert errors.max()<4.,(cmc,errors)

def test_unknown_mount_cannot_absorb_large_errors_as_thumb_opposition(model):
    m=model;m.bone_reference=None;m.set_shape(np.zeros(41));m.mounting_calibrated=False
    seed=np.tile(np.eye(3),(15,1,1))
    angles=np.zeros((15,3));angles[12:15]=np.deg2rad([[60,-40,40],[70,10,5],[60,0,0]])
    b=m.anatomy.basis;wild=Rotation.from_matrix(b@Rotation.from_euler('XYZ',angles).as_matrix()@b.swapaxes(-2,-1)).as_rotvec()
    q=torch.tensor(wild[None],device=m.device,dtype=torch.float32)
    m.constrain_thumb_reference_(q,seed)
    correction=Rotation.from_rotvec(q[0,12:15].cpu().numpy()).magnitude()
    assert np.rad2deg(correction).max()<=8.01
    m.mounting_calibrated=True;q=torch.tensor(wild[None],device=m.device,dtype=torch.float32);m.constrain_thumb_reference_(q,seed)
    assert np.rad2deg(Rotation.from_rotvec(q[0,12:15].cpu().numpy()).magnitude()).max()<=20.01
    m.mounting_calibrated=False

def test_neutral_geometry_has_no_false_remote_patch_penetration(model):
    from moxian_web.mano_fit import geometry_report
    m=model;m.bone_reference=None;m.set_shape(np.zeros(41))
    row={'stage':'open','contact':{'ready':True,'patches':[{**p,'contact':False,'response':0} for p in PATCHES]}}
    report=geometry_report(m,[row],torch.zeros((1,15,3),device=m.device),torch.eye(3,device=m.device)[None])
    assert report['noncontact_penetration_m']<1e-5

def test_closed_volume_sign_not_backside_of_open_patch(model):
    import warp as wp
    import trimesh
    from moxian_web.mano_runtime import inside_queries,triangle_queries
    box=trimesh.creation.box(extents=[.02,.02,.02]);device=str(model.device)
    with wp.ScopedDevice(device):
        vertices=wp.array(box.vertices,dtype=wp.vec3,device=device)
        solid=wp.Mesh(points=vertices,indices=wp.array(box.faces.ravel(),dtype=int,device=device),support_winding_number=True)
        positive=box.faces[box.face_normals[:,0]>.9];negative=box.faces[box.face_normals[:,0]<-.9]
        patches=[wp.Mesh(points=vertices,indices=wp.array(f.ravel(),dtype=int,device=device)) for f in (positive,negative)]
        points=wp.array([[.025,0,0],[.009,0,0],[0,0,0]],dtype=wp.vec3,device=device)
        inside=wp.zeros((3,1),dtype=float,device=device);distance=wp.zeros((3,1),dtype=float,device=device)
        gap=wp.zeros((3,2),dtype=float,device=device);facing=wp.zeros((3,2),dtype=float,device=device);nearest=wp.zeros((3,2),dtype=wp.vec3,device=device)
        wp.launch(inside_queries,(3,1),inputs=[wp.array([solid.id],dtype=wp.uint64,device=device),points,inside,distance],device=device)
        wp.launch(triangle_queries,(3,2),inputs=[wp.array([p.id for p in patches],dtype=wp.uint64,device=device),points,
            wp.array([0,0],dtype=int,device=device),inside,distance,wp.zeros(3,dtype=wp.vec3,device=device),gap,facing,nearest],device=device)
        g=gap.numpy()
    assert np.all(g[0]>0)  # Outside, including behind a far open patch normal.
    assert np.isclose(g[1,0],-.001,atol=1e-6)  # True local penetration.
    assert g[1,1]>0  # A far patch cannot claim 19 mm penetration.

def test_compression_can_be_disabled_and_never_changes_pose_or_bones(model):
    m=model;pose=m.last_pose.copy();joints=m.rest_j_np.copy();base=m.rest_np.copy()
    m.compression[:]=.001
    displayed,report=deform_surface(m,base,{'ready':False,'pairs':[]},1000,enabled=False)
    assert np.array_equal(displayed,base) and report['max_m']==0 and not report['affects_joints']
    assert np.array_equal(pose,m.last_pose) and np.array_equal(joints,m.rest_j_np)

def test_unmodified_mesh_matches_original_mano_skinning(model):
    m=model;m.bone_reference=None;m.set_shape(np.zeros(41))
    pose=torch.tensor(m.anatomy.project(m.model.pose_mean.cpu().numpy().reshape(15,3))[None],device=m.device)
    zeros=torch.zeros(10,device=m.device);scale=torch.tensor(0.,device=m.device);finger=torch.zeros(5,device=m.device)
    with torch.no_grad():
        expected,j,r=m.model(zeros,scale,finger,finger,pose)
        actual,jj,rr,o=m.decode(pose,torch.eye(3,device=m.device)[None])
    assert torch.allclose(actual,expected-m.rest_j[0],atol=1e-7)
    assert torch.allclose(jj,j-m.rest_j[0],atol=1e-7)

def test_tracking_continues_through_missing_skeleton_without_seed_reset_jumps(model):
    m=ManoSurface(DEFAULT_MANO)
    with torch.no_grad():v,j,rr,ordered=m.decode(torch.zeros((1,15,3),device=m.device),torch.eye(3,device=m.device)[None])
    points=ordered[0].cpu().numpy();bone_positions=[points[0],*points[1:5]]
    for f in range(1,5):bone_positions.extend([points[1+4*f]*.5,*points[1+4*f:5+4*f]])
    rng=np.random.default_rng(11);outputs=[]
    for i in range(115):
        noisy=np.array(bone_positions)+rng.normal(0,.00004,(25,3));noisy[0]=0
        bone={'hand_side':'left','coordinate_frame':'wrist_local','names':JOINT_NAMES,
              'positions_local':noisy.tolist(),'timestamp_ms':1000+i*33}
        fingers={f:{'position':(points[4+4*k]+rng.normal(0,.00005,3)).tolist(),'quaternion_xyzw':[0,0,0,1]}
                 for k,f in enumerate(('thumb','index','middle','ring','pinky'))}
        frame={'timestamp_ms':1000+i*33,'fingers':fingers,'skeleton':bone if i%4!=1 else None}
        outputs.append(m.track(frame,steps=1)[0])
    steps=np.linalg.norm(np.diff(outputs[60:],axis=0),axis=-1)
    assert np.percentile(steps,95)<.0005
    assert steps.max()<.0015
    # Smoothing must still follow a deliberate bend, not freeze the surface.
    held=outputs[-1].copy()
    for i in range(30):
        pose=torch.zeros((1,15,3),device=m.device);pose[0,:,2]=min(i/15,1)*.3
        with torch.no_grad():v,j,rr,ordered=m.decode(pose,torch.eye(3,device=m.device)[None])
        bent=ordered[0].cpu().numpy();bp=[bent[0],*bent[1:5]]
        for f in range(1,5):bp.extend([bent[1+4*f]*.5,*bent[1+4*f:5+4*f]])
        frame={'timestamp_ms':5000+i*33,'skeleton':{**bone,'positions_local':np.array(bp).tolist()},
               'fingers':{f:{'position':bent[4+4*k].tolist(),'quaternion_xyzw':[0,0,0,1]}
                          for k,f in enumerate(('thumb','index','middle','ring','pinky'))}}
        moved=m.track(frame,steps=1)[0]
    assert np.max(np.linalg.norm(moved-held,axis=1))>.005
    old_optimizer=m.tracking_optimizer
    # A genuine interruption must reinitialize instead of reusing stale moments.
    frame['timestamp_ms']+=1000;frame['skeleton']=bone;m.track(frame,steps=1)
    assert m.tracking_optimizer is not old_optimizer


def test_wrist_targets_bypass_rigid_guess_and_validate_time_side_and_rotation():
    m=ManoSurface(DEFAULT_MANO)
    base={'timestamp_ms':1000.,'fingers':{f:{'position':[.5,.4,.3],'quaternion_xyzw':[0,0,0,1]} for f in ('thumb','index','middle','ring','pinky')}}
    target={'timestamp_ms':1000.,'hand_side':'left','coordinate_frame':'wrist_local','positions_local':[[.01*k,.05,.02] for k in range(5)],'rotations_local':[np.eye(3).ravel().tolist()]*5}
    frame={**base,'skeleton_targets':target}
    p,r,b=m.observations(frame)
    assert np.allclose(p,target['positions_local']) and np.allclose(m.raw_rotation,np.eye(3)) and np.allclose(m.raw_translation,0)
    for change in ({'timestamp_ms':1026},{'hand_side':'right'},{'coordinate_frame':'emf_local'},{'rotations_local':[[0]*9]*5}):
        with pytest.raises(ValueError):m.observations({**frame,'skeleton_targets':{**target,**change}})
    with pytest.raises(ValueError):m.observations(base)


def test_mapped_targets_allow_global_root_rotation_without_thumb_contortion():
    m=ManoSurface(DEFAULT_MANO)
    q=np.zeros((15,3),np.float32);root=Rotation.from_euler('xyz',[25,30,-15],degrees=True).as_matrix().astype(np.float32)
    with torch.no_grad():v,j,rr,ordered=m.decode(torch.tensor(q[None],device=m.device),torch.tensor(root[None],device=m.device))
    points=ordered[0].cpu().numpy();positions=v[0,[744,320,443,554,671]].cpu().numpy();rotations=rr[0,[c[-1] for c in CHAINS]].cpu().numpy()
    bp=[points[0],*points[1:5]]
    for f in range(1,5):bp.extend([points[1+4*f]*.5,*points[1+4*f:5+4*f]])
    # Keep true lengths fixed to isolate coordinate recovery, and deliberately
    # provide raw EMF values unrelated to wrist coordinates.
    m.bone_reference=np.array([[.03,.02,.02]]*5)
    errors=[]
    for i in range(30):
        stamp=1000+i*33;frame={'timestamp_ms':stamp,'calibration_fingerprint':'test',
          'skeleton':{'hand_side':'left','coordinate_frame':'wrist_local','names':JOINT_NAMES,'positions_local':np.array(bp).tolist(),'timestamp_ms':stamp},
          'fingers':{f:{'position':[.6,.7,.8],'quaternion_xyzw':[0,0,0,1]} for f in ('thumb','index','middle','ring','pinky')},
          'skeleton_targets':{'timestamp_ms':stamp,'hand_side':'left','coordinate_frame':'wrist_local','positions_local':positions.tolist(),'rotations_local':rotations.reshape(5,9).tolist()}}
        errors.append(m.track(frame,steps=2)[-1])
    assert errors[-1]<2.,errors[-1]
    assert Rotation.from_matrix(m.last_root@root.T).magnitude()<np.deg2rad(3)
    assert np.max(np.linalg.norm(m.last_pose,axis=1))<np.deg2rad(6)
    assert m.thumb_quality['source']=='sdk_wrist_target' and m.thumb_quality['max_joint_correction_deg'] is None


def _motion_frame(m,pose,root,stamp,frozen_bones=False):
    with torch.no_grad():
        v,_,rr,ordered=m.decode(torch.tensor(pose[None],device=m.device),torch.tensor(root[None],device=m.device))
        if frozen_bones:_,_,_,ordered=m.decode(torch.zeros((1,15,3),device=m.device),torch.tensor(root[None],device=m.device))
    points=ordered[0].cpu().numpy();p=v[0,[744,320,443,554,671]].cpu().numpy();r=rr[0,[c[-1] for c in CHAINS]].cpu().numpy()
    bp=[points[0],*points[1:5]]
    for f in range(1,5):bp.extend([points[1+4*f]*.5,*points[1+4*f:5+4*f]])
    return {'timestamp_ms':stamp,'skeleton':{'timestamp_ms':stamp,'hand_side':'left','coordinate_frame':'wrist_local','names':JOINT_NAMES,'positions_local':np.array(bp).tolist()},
      'fingers':{f:{'position':p[i].tolist(),'quaternion_xyzw':Rotation.from_matrix(r[i]).as_quat().tolist()} for i,f in enumerate(('thumb','index','middle','ring','pinky'))},
      'skeleton_targets':{'timestamp_ms':stamp,'hand_side':'left','coordinate_frame':'wrist_local','positions_local':p.tolist(),'rotations_local':r.reshape(5,9).tolist()}},p


def test_current_frame_ik_follows_step_and_release_despite_unresponsive_bone_seed():
    m=ManoSurface(DEFAULT_MANO);m.bone_reference=np.array([[.03,.02,.02]]*5);identity=np.eye(3,dtype=np.float32)
    a=np.zeros((15,3));a[:12:3,0]=.65;a[1:12:3,0]=.9;a[2:12:3,0]=.5;b=m.anatomy.basis
    bent=Rotation.from_matrix(b@Rotation.from_euler('XYZ',a).as_matrix()@b.swapaxes(-2,-1)).as_rotvec().astype(np.float32)
    shape=m.shape.copy();i=0
    for q in (np.zeros((15,3),np.float32),bent,np.zeros((15,3),np.float32)):
        errors=[]
        for k in range(8):
            frame,p=_motion_frame(m,q,identity,1000+i*22.2,True);v,*_=m.track(frame,steps=1);i+=1
            errors.append(np.linalg.norm(v[[744,320,443,554,671]]-p,axis=1)*1000)
        # Three output frames, about 67 ms at the synthetic 45 Hz input.
        assert np.max(errors[2])<2.,errors
        assert np.max(errors[-1])<.5,errors
    assert np.array_equal(shape,m.shape) and np.all(m.compression==0)


def test_responsive_tracking_retains_real_palm_rotation_and_static_stability():
    m=ManoSurface(DEFAULT_MANO);m.bone_reference=np.array([[.03,.02,.02]]*5);q=np.zeros((15,3),np.float32);outputs=[]
    for i in range(45):
        degrees=min(i,12)*2;root=Rotation.from_euler('y',degrees,degrees=True).as_matrix().astype(np.float32)
        frame,p=_motion_frame(m,q,root,1000+i*22.2)
        if i>15:
            noisy=p+np.random.default_rng(i).normal(0,.00004,p.shape);frame['skeleton_targets']['positions_local']=noisy.tolist()
        v,*_=m.track(frame,steps=1)
        if i==14:assert np.rad2deg(Rotation.from_matrix(m.last_root@root.T).magnitude())<2
        if i>20:outputs.append(v.copy())
    steps=np.linalg.norm(np.diff(outputs,axis=0),axis=-1)*1000
    assert np.percentile(steps,95)<.4 and steps.max()<1.,(np.percentile(steps,95),steps.max())


def _emf_observation(m,q,stamp,emf_rotation,emf_translation,mount_rotation):
    root=np.eye(3,dtype=np.float32)
    frame,_=_motion_frame(m,np.zeros((15,3),np.float32),root,stamp,True)
    with torch.no_grad():v,j,r,o=m.decode(torch.tensor(q[None],device=m.device),torch.eye(3,device=m.device)[None])
    distal=r[0,[c[-1] for c in CHAINS]].cpu().numpy()
    positions=j[0,[c[-1] for c in CHAINS]].cpu().numpy()+(distal@m.sensor_offsets()[:,:,None]).squeeze(-1)
    raw=positions@emf_rotation.T+emf_translation;rotations=emf_rotation@distal@mount_rotation
    frame['fingers']={f:{'position':raw[k].tolist(),'quaternion_xyzw':Rotation.from_matrix(rotations[k]).as_quat().tolist()} for k,f in enumerate(('thumb','index','middle','ring','pinky'))}
    return frame,raw,rotations


def test_raw_emf_sensor_is_not_mesh_tip_and_flat_reference_does_not_change_size():
    m=ManoSurface(DEFAULT_MANO);m.bone_reference=np.array([[.03,.02,.02]]*5)
    axis=m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]];axis/=np.linalg.norm(axis,axis=1,keepdims=True);m.mount_pos=-axis*.008
    q=np.zeros((15,3),np.float32);emfr=Rotation.from_euler('xyz',[15,-25,35],degrees=True).as_matrix();emft=np.array([.035,-.02,.015]);mr=Rotation.from_euler('xyz',np.array([[7,3,-8]]*5),degrees=True).as_matrix()
    frame,p,r=_emf_observation(m,q,1000,emfr,emft,mr);m.last_frame=frame
    reference={'pose':'flat_open','hand_side':'left','observation_space':'raw_emf_sensor','positions_local':p.tolist(),'rotations_local':r.reshape(5,9).tolist(),'reference_id':'test-sensor-flat'}
    shape=m.shape.copy();m.install_flat_reference(reference)
    for i in range(25):
        frame['timestamp_ms']=1000+i*25;v,j,rr,o,error=m.track(frame,steps=1)
    assert np.array_equal(m.shape,shape)
    assert np.max(np.abs(np.rad2deg(m.anatomy.angles(m.last_pose)[:12,0])))<4
    assert np.max(np.linalg.norm(m.sensor_predictions[1:]-v[[320,443,554,671]],axis=1))>.006
    assert m.alignment_quality['observation_space']=='raw_emf_sensor'
    assert m.alignment_quality['sensor_mounting_status']=='orientation_reference_position_prior'
    # Matching an EMF origin must not pull the skin tip onto the sensor.
    assert np.mean(list(m.alignment_quality['tip_errors_mm'].values())[1:])<1


def test_sensor_mount_projection_keeps_emf_behind_tip_without_changing_shape():
    m=ManoSurface(DEFAULT_MANO);shape=m.shape.copy();mount=torch.full((5,3),.1,device=m.device)
    m.project_sensor_mount_(mount)
    vector=m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]];length=np.linalg.norm(vector,axis=1);axis=vector/length[:,None]
    value=mount.cpu().numpy();along=(value*axis).sum(-1);radial=value-along[:,None]*axis
    assert np.all(along<=-.000999) and np.all(along>=-np.minimum(.035,length-.002)-1e-6)
    assert np.max(np.linalg.norm(radial,axis=1))<=.015001 and np.array_equal(shape,m.shape)


def test_calibrated_emf_direction_follows_bending_and_returns_to_flat():
    m=ManoSurface(DEFAULT_MANO);m.bone_reference=np.array([[.03,.02,.02]]*5)
    axis=m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]];axis/=np.linalg.norm(axis,axis=1,keepdims=True);m.mount_pos=-axis*.008
    emfr=Rotation.from_euler('xyz',[15,-25,35],degrees=True).as_matrix();emft=np.array([.035,-.02,.015]);mr=Rotation.from_euler('xyz',np.array([[7,3,-8]]*5),degrees=True).as_matrix()
    frame,p,r=_emf_observation(m,np.zeros((15,3),np.float32),1000,emfr,emft,mr);m.last_frame=frame
    m.install_flat_reference({'pose':'flat_open','hand_side':'left','observation_space':'raw_emf_sensor','positions_local':p.tolist(),'rotations_local':r.reshape(5,9).tolist(),'reference_id':'motion-reference'})
    levels=[0]*6+list(np.linspace(0,1,8))+[1]*6+list(np.linspace(1,0,8))+[0]*10
    errors=[];direction=[]
    for i,level in enumerate(levels):
        a=np.zeros((15,3));a[:12:3,0]=.6*level;a[1:12:3,0]=.8*level;a[2:12:3,0]=.5*level
        b=m.anatomy.basis;q=Rotation.from_matrix(b@Rotation.from_euler('XYZ',a).as_matrix()@b.swapaxes(-2,-1)).as_rotvec().astype(np.float32)
        frame,p,r=_emf_observation(m,q,1000+i*25,emfr,emft,mr);v,j,rr,o,error=m.track(frame,steps=1);errors.append(error)
        predicted=rr[[c[-1] for c in CHAINS]]@m.mount_rot
        expected=m.raw_rotation@r
        direction.append(np.rad2deg(Rotation.from_matrix(predicted.swapaxes(-2,-1)@expected).magnitude()))
    assert np.mean(errors[8:26])<3.,errors
    assert np.max(direction[-1])<3.,direction[-1]
    assert np.max(np.abs(np.rad2deg(m.anatomy.angles(m.last_pose)[:12,0])))<6.


def test_anatomical_tip_extension_is_independent_of_distal_to_emf_mount():
    m=ManoSurface(DEFAULT_MANO);m.flat_reference={'reference_id':'test'}
    m.sensor_joint_position=(m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]])*.7
    sensor=m.sensor_offsets().copy();tip=m.tip_anchors.copy();joints=m.rest_j_np.copy()
    changed=m.shape.copy();changed[16:21]=.0015;m.set_shape(changed)
    assert np.array_equal(m.sensor_offsets(),sensor)
    assert np.allclose(m.rest_j_np,joints) and np.min(np.linalg.norm(m.tip_anchors-tip,axis=1))>.001


def test_fixed_segment_dimensions_change_only_segment_scale_and_finger_radius():
    m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20));base_v=m.rest_np.copy();base_j=m.rest_j_np.copy()
    x=np.zeros(20);x[3]=np.log(1.12);x[16]=np.log(1.2);m.set_dimensions(x)
    c=CHAINS[1]
    assert np.isclose(np.linalg.norm(m.rest_j_np[c[1]]-m.rest_j_np[c[0]])/np.linalg.norm(base_j[c[1]]-base_j[c[0]]),1.12,atol=1e-5)
    assert np.allclose(m.rest_j_np[[CHAINS[0][0],CHAINS[2][0]]],base_j[[CHAINS[0][0],CHAINS[2][0]]])
    fixed=m.rest_np.copy();dimensions=m.dimension_parameters.copy();geometry_version=m.shape_version
    weights=m.model.weights.cpu().numpy();ids=np.flatnonzero(weights[:,c[-1]]>.9999)
    assert len(ids)>2
    for angle in (0,.2,.5,.8):
        q=np.zeros((15,3),np.float32);q[c[0]-1,2]=angle;q[c[1]-1,2]=angle
        with torch.no_grad():v,j,r,o=m.decode(torch.tensor(q[None],device=m.device),torch.eye(3,device=m.device)[None])
        expected=j[0,c[-1]].cpu().numpy()+(fixed[ids]-m.rest_j_np[c[-1]])@r[0,c[-1]].cpu().numpy().T
        assert np.max(np.linalg.norm(v[0,ids].cpu().numpy()-expected,axis=1))<1e-5
        assert np.array_equal(m.rest_np,fixed) and np.array_equal(m.dimension_parameters,dimensions) and m.shape_version==geometry_version
    # The retired patch and fingertip displacement entries cannot change fixed geometry.
    shape=m.shape.copy();shape[16:]=.001;m.set_shape(shape)
    assert np.allclose(m.rest_np,fixed)


def test_sensor_supported_pinch_is_pose_only_and_requires_bilateral_tactile():
    m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20))
    m.build_binding({'assignments':default_mapping('left'),'version':'test','confirmed':PATCH_IDS.copy()})
    q=np.zeros((15,3),np.float32);root=np.eye(3,dtype=np.float32)
    with torch.no_grad():_,j,r,_=m.decode(torch.tensor(q[None],device=m.device),torch.eye(3,device=m.device)[None])
    rotations=r[0,[c[-1] for c in CHAINS]].cpu().numpy()
    observed=j[0,[c[-1] for c in CHAINS]].cpu().numpy()+(rotations@m.sensor_offsets()[:,:,None]).squeeze(-1)
    rest=m.rest_np.copy();joints=m.rest_j_np.copy();dims=m.dimension_parameters.copy();version=m.shape_version
    for finger in range(1,5):
        near=observed.copy();near[finger]=near[0]+np.array([.02,0,0])
        frame={'contact':{'ready':False,'patches':[{'id':'thumb_distal','contact':True},{'id':('index','middle','ring','pinky')[finger-1]+'_distal','contact':True}]}}
        assert np.array_equal(m.fit_sensor_supported_pinch(q,root,near,rotations,frame),q)
        assert not m.retarget_contacts
        frame['contact']['ready']=True;frame['contact']['patches'][1]['contact']=False
        assert np.array_equal(m.fit_sensor_supported_pinch(q,root,near,rotations,frame),q)
        assert not m.retarget_contacts
        frame['contact']['patches'][1]['contact']=True
        solved=m.fit_sensor_supported_pinch(q,root,near,rotations,frame)
        assert m.retarget_contacts and m.retarget_contacts[0]['applied']
        assert m.retarget_contacts[0]['pad_gap_mm']<m.retarget_contacts[0]['pad_gap_before_mm']
        assert not m.retarget_contacts[0]['force_feedback'] and not m.retarget_contacts[0]['geometry_changed']
        untouched=[c-1 for f,chain in enumerate(CHAINS) if f not in (0,finger) for c in chain]
        assert np.array_equal(solved[untouched],q[untouched])
        assert np.array_equal(m.rest_np,rest) and np.array_equal(m.rest_j_np,joints)
        assert np.array_equal(m.dimension_parameters,dims) and m.shape_version==version


def test_calibration_acceptance_requires_real_contact_observations_and_allows_small_overlap():
    from moxian_web.mano_fit import candidate_validation
    geo={'noncontact_penetration_m':.003,'contact_gap_m':.004,'contact_stages':['pinch_'+f for f in ('index','middle','ring','pinky')]}
    assert candidate_validation(10.,8.,geo,.005)==(True,True)
    # Reduced position error cannot legitimize missing contact data or a
    # fabricated default distance; all four guided pairs must be evaluated.
    assert not candidate_validation(10.,8.,{**geo,'contact_stages':[]},.005)[0]
    assert not candidate_validation(10.,8.,{**geo,'contact_gap_m':.04},.005)[0]
    assert candidate_validation(10.,8.,{**geo,'noncontact_penetration_m':.008},.005)[0]
    assert not candidate_validation(10.,11.,geo,.005)[0]
    assert not candidate_validation(10.,float('nan'),geo,.005)[0]


def test_runtime_pose_is_identical_when_tactile_flags_change():
    from unittest.mock import patch
    models=[]
    for _ in range(2):
        m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20));m.bone_reference=np.array([[.03,.02,.02]]*5)
        frame,p,r=_emf_observation(m,np.zeros((15,3),np.float32),1000,np.eye(3),np.zeros(3),np.tile(np.eye(3),(5,1,1)))
        m.last_frame=frame;m.install_flat_reference({'pose':'flat_open','hand_side':'left','observation_space':'raw_emf_sensor','positions_local':p.tolist(),'rotations_local':r.reshape(5,9).tolist()})
        models.append(m)
    # A near-pinch input is deliberately identical. Only the tactile bit changes.
    with patch.object(ManoSurface,'fit_sensor_supported_pinch',side_effect=AssertionError('Runtime must not call contact closure')):
        for i,level in enumerate([0,.2,.4,.6,.6,.4,.2,0]):
            a=np.zeros((15,3));a[:12:3,0]=level;a[1:12:3,0]=level
            b=models[0].anatomy.basis;q=Rotation.from_matrix(b@Rotation.from_euler('XYZ',a).as_matrix()@b.swapaxes(-2,-1)).as_rotvec().astype(np.float32)
            frame,_,_=_emf_observation(models[0],q,1000+i*50,np.eye(3),np.zeros(3),np.tile(np.eye(3),(5,1,1)))
            outputs=[]
            for k,m in enumerate(models):
                f=copy.deepcopy(frame);f['contact']={'ready':True,'patches':[{'id':p,'contact':bool(k)} for p in PATCH_IDS]}
                outputs.append(m.track(f,steps=1)[0]);assert not m.retarget_contacts
            assert np.array_equal(outputs[0],outputs[1])


def test_known_capture_gestures_constrain_training_bends_without_resizing():
    from moxian_web.mano_fit import constrain_capture_pose_
    m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20));rest=m.rest_np.copy()
    a=np.zeros((3,15,3));a[:,:,0]=.7;b=m.anatomy.basis
    q=torch.tensor(Rotation.from_matrix((b@Rotation.from_euler('XYZ',a.reshape(-1,3)).as_matrix().reshape(3,15,3,3)@b.swapaxes(-2,-1)).reshape(-1,3,3)).as_rotvec().reshape(3,15,3),device=m.device,dtype=torch.float32)
    constrain_capture_pose_(m,q,[{'stage':s} for s in ('open','mcp90','hook')])
    out=m.anatomy.angles(q.cpu().numpy())
    assert np.abs(out[0,:12,0]).max()<=np.deg2rad(5.01)
    assert np.min(out[1,:12:3,0])>=np.deg2rad(74.99)
    assert np.abs(out[1,[1,2,4,5,7,8,10,11],0]).max()<=np.deg2rad(7.01)
    assert np.max(out[2,:12:3,0])<=np.deg2rad(8.01)
    assert np.array_equal(m.rest_np,rest)


def test_emf_only_cannot_identify_radial_or_surface_tip_length_scales():
    m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20));m.flat_reference={'reference_id':'unit'}
    m.sensor_joint_position=.6*(m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]])
    q=torch.zeros((1,15,3),device=m.device);root=torch.eye(3,device=m.device)[None]
    def sensors():
        with torch.no_grad():_,j,r,_=m.decode(q,root)
        return j[0,[c[-1] for c in CHAINS]].cpu().numpy()+(r[0,[c[-1] for c in CHAINS]].cpu().numpy()@m.sensor_offsets()[:,:,None]).squeeze(-1)
    base=sensors();summary=m.dimension_summary();x=np.zeros(20);x[2:15:3]=np.log(1.1);x[15:]=np.log(1.1);m.set_dimensions(x)
    assert np.allclose(sensors(),base,atol=1e-8)
    changed=m.dimension_summary();assert changed['index']['surface_length_mm']>summary['index']['surface_length_mm']
    assert changed['index']['proximal_width_mm']>summary['index']['proximal_width_mm']*1.05
    assert 5<changed['index']['proximal_width_mm']<50


def test_surface_distance_beyond_four_cm_is_not_a_missing_query_sentinel():
    import trimesh
    m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20));m.build_binding({'assignments':default_mapping('left'),'version':'far','confirmed':PATCH_IDS.copy()})
    with torch.no_grad():v,j,r,_=m.decode(torch.zeros((1,15,3),device=m.device),torch.eye(3,device=m.device)[None])
    v=v[0].cpu().numpy();g=CollisionSurface(m).query(v,j[0].cpu().numpy(),r[0].cpu().numpy())
    sid=int(np.flatnonzero(np.array(m.mapping['assignments'])==0)[0]);target=PATCH_IDS.index('pinky_distal')
    mesh=trimesh.Trimesh(vertices=v,faces=m.faces[m.patch_faces[target]],process=False)
    _,distance,_=trimesh.proximity.closest_point_naive(mesh,g['points'][sid:sid+1])
    assert distance[0]>.04
    assert np.isclose(abs(g['gap'][sid,target]),distance[0],atol=1e-5)


def test_together_side_strips_cover_adjacent_fingers_and_follow_new_geometry():
    if not DEFAULT_MANO.exists() or not torch.cuda.is_available():pytest.skip('Local MANO/CUDA required')
    m=ManoSurface(DEFAULT_MANO);m.dimension_parameters=np.zeros(20);m.set_shape(np.zeros(41))
    regions=m.together_regions()
    assert len(regions)==9 and {(r['a'],r['b']) for r in regions}=={(1,2),(2,3),(3,4)}
    for r in regions:
        assert len(r['ids_a'])==len(r['ids_b'])==8
        assert set(m.labels[r['ids_a']])<=set(CHAINS[r['a']])
        assert set(m.labels[r['ids_b']])<=set(CHAINS[r['b']])
    version=m.shape_version;dims=m.dimension_parameters.copy();dims[16]=np.log(1.15);m.set_dimensions(dims)
    assert m.shape_version!=version and m.together_regions() is not regions
    q=torch.zeros((1,15,3),device=m.device,requires_grad=True)
    v,*_=m.decode(q,torch.eye(3,device=m.device)[None])
    losses=[torch.cdist(v[:,r['ids_a']],v[:,r['ids_b']]).amin() for r in m.together_regions()]
    torch.stack(losses).sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0

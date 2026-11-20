"""Deep overlaps must preserve the facing side rather than cross the mesh."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from moxian_web.cart_objective import CartHand
from moxian_web.cart_contacts import SourceContacts


def mesh_array(mesh,device):
    return wp.Mesh(points=wp.array(np.asarray(mesh.vertices,np.float32),dtype=wp.vec3,device=device),
        indices=wp.array(np.asarray(mesh.faces,np.int32).ravel(),dtype=int,device=device),support_winding_number=True)


@pytest.mark.parametrize('device',['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu'])
def test_retract_to_known_facing_side_before_nearest_query(device):
    box=trimesh.creation.box(extents=(.02,.02,.02))
    hand=CartHand.__new__(CartHand);hand.device=device;hand.full_meshes=[mesh_array(box,device)];hand.pad_meshes={}
    # The point crossed the centre; ordinary nearest selects +x (the back side).
    point=torch.tensor([[.008,0.,0.]],device=device)
    nearest,_,valid=hand.nearest(point,0)
    assert valid.item() and nearest[0,0].item()==pytest.approx(.01)
    direction=torch.tensor([[-1.,0.,0.]],device=device)
    hit,normal,valid,retracted=hand.nearest(point,0,escape=direction,return_retracted=True)
    assert valid.item() and retracted.item()
    torch.testing.assert_close(hit,torch.tensor([[-.01,0.,0.]],device=device),atol=1.e-7,rtol=0)
    torch.testing.assert_close(normal,direction)
    # No trusted direction: defer the contact instead of falling back to +x.
    _,_,valid=hand.nearest(point,0,escape=torch.zeros_like(direction))
    assert not valid.item()
    shallow=torch.tensor([[-.0099,0.,0.]],device=device)
    _,_,valid,retracted=hand.nearest(shallow,0,escape=direction,return_retracted=True)
    assert valid.item() and not retracted.item()
    # Exiting at an artificial cap or a different semantic surface is not valid.
    backside=box.copy();backside.update_faces(backside.face_normals[:,0]>.9)
    hand.pad_meshes[0]=mesh_array(backside,device)
    _,_,valid=hand.nearest(point,0,palmar=True,escape=direction)
    assert not valid.item()


def source_boxes(offset=.005,extents=(.02,.02,.02),rotation=None):
    meshes=[]
    for x,size in ((-.1,(.02,)*3),(0.,(.02,)*3),(offset,extents)):
        mesh=trimesh.creation.box(extents=size);mesh.apply_translation((x,0,0));meshes.append(mesh)
    vertices=np.vstack([m.vertices for m in meshes]).astype(np.float32)
    if rotation is not None:vertices=vertices@rotation.T
    faces=np.vstack([m.faces+8*i for i,m in enumerate(meshes)])
    class Model:
        bone_names=['right_wrist','right_index_proximal','right_middle_proximal']
        weights=torch.tensor(np.repeat(np.eye(3),8,axis=0),dtype=torch.float32)
        q0=torch.zeros(1)
        def __call__(self,q):return torch.tensor(vertices)[None],None,None
    topology=dict(faces=faces.tolist(),sid_face_ids=[12]*460,sid_barycentric=[[1/3]*3]*460,sid_patch_indices=[0]*460)
    return SourceContacts(topology,Model()),vertices


@pytest.mark.parametrize('rotated',[False,True])
def test_deep_source_overlap_selects_opposing_original_material_faces(rotated):
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_rotvec([.3,-.2,.4]).as_matrix() if rotated else np.eye(3)
    detector,vertices=source_boxes(rotation=rotation)
    original=vertices.copy()
    contacts=detector.detect(vertices,np.zeros((21,3)),dict(timestamp_ms=0.),include_all=True)
    assert len(contacts['self_pairs'])==1
    pair=contacts['self_pairs'][0]
    assert pair['witness_resolution']=='directional_retraction'
    points=np.asarray(pair['points_m'])@rotation
    normals=np.asarray(pair['normals'])@rotation
    for region,point,normal,material in zip(pair['regions'],points,normals,pair['material']):
        expected=(.01,1.) if region=='index_proximal' else (-.005,-1.)
        assert point[0]==pytest.approx(expected[0],abs=1.e-7)
        assert normal[0]==pytest.approx(expected[1],abs=1.e-6)
        reconstructed=(vertices[material['vertices']]*np.asarray(material['barycentric'])[:,None]).sum(0)
        np.testing.assert_allclose(reconstructed@rotation,point,atol=1.e-7)
    assert np.dot(*np.asarray(pair['normals']))<-.99
    assert pair['separation_m']==pytest.approx(.015,abs=1.e-7)
    # Native depths are still nearest signed distances, not the retraction path.
    assert pair['depth_m']<pair['separation_m']
    assert contacts['contact_resolution']['retracted_samples']>0
    assert contacts['self_penetrations']['count']>1
    np.testing.assert_array_equal(vertices,original)


def test_engulfed_coincident_centres_are_unresolved_not_guessed():
    detector,vertices=source_boxes(offset=0.,extents=(.04,.04,.04))
    contacts=detector.detect(vertices,np.zeros((21,3)),dict(timestamp_ms=0.),include_all=True)
    assert not contacts['self_pairs']
    assert contacts['contact_resolution']['unresolved_samples']>0
    assert contacts['self_penetrations']['count']>0


def test_robot_contact_retraction_keeps_gradients_through_both_original_poses():
    box=trimesh.creation.box(extents=(.02,)*3)
    hand=CartHand.__new__(CartHand);hand.device='cpu'
    hand.full_meshes=[mesh_array(box,'cpu'),mesh_array(box,'cpu')];hand.pad_meshes={}
    hand.surface=SimpleNamespace(meshes=[box,box])
    hand.samples=lambda owner,dense:(torch.tensor([[.01 if owner==0 else -.01,0.,0.]]),
                                    torch.tensor([[1. if owner==0 else -1.,0.,0.]]))
    p=torch.tensor([[[0.,0.,0.],[.005,0.,0.]]],requires_grad=True)
    r=torch.eye(3).expand(1,2,3,3)
    distance,angle,pa,pb,na,nb,valid=hand.pair(p,r,0,1)
    assert valid.item() and angle.item()==pytest.approx(0.)
    assert pa[0,0].item()==pytest.approx(.01) and pb[0,0].item()==pytest.approx(-.005)
    distance.sum().backward()
    # Gradient descent moves source left and target right, separating the links.
    assert p.grad[0,0,0]>0 and p.grad[0,1,0]<0 and p.grad.isfinite().all()


def test_unresolved_contact_has_no_false_pull_and_fails_acceptance(monkeypatch):
    import json
    from moxian_web.cart_objective import CartObjective,policy_settings
    from moxian_web.cart_training import quality_summary
    torch.set_num_threads(1)
    hand=CartHand('cpu');objective=CartObjective(hand,policy_settings())
    q=((hand.low+hand.high)/2)[None].requires_grad_()
    joints=hand.frames(q)[2][0].detach().tolist()
    row=dict(joints_m=joints,contacts=dict(tactile_ready=True,self_pairs=[],object_contacts=[
        dict(region='index_distal',position_m=joints[8],normal=[1.,0.,0.],force=1.)]))
    def unresolved(query,owner,**kwargs):
        return torch.zeros_like(query),torch.zeros_like(query),torch.zeros(query.shape[:-1],dtype=torch.bool)
    monkeypatch.setattr(hand,'nearest',unresolved)
    loss,parts,quality=objective(q,row)
    assert torch.isfinite(loss).all() and parts['contact'].item()==0 and parts['normal'].item()==0
    assert not quality['feasible'].item() and not quality['query_valid'].item()
    assert quality['unresolved_contact_count'].item()==1
    summary=quality_summary([quality])
    assert summary['unresolved_contact_frames']==1 and summary['gap_max_mm'] is None
    json.dumps(summary,allow_nan=False)
    loss.sum().backward();assert q.grad.isfinite().all()

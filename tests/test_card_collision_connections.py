"""Mechanical joints are excluded; nonconnected native penetration remains visible."""
import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from moxian_web.wuji_surface import WujiSurface
from retarget.baseline.card.collision import WujiMeshCollision,_native_penetration
from retarget.env.grasp.pose_gpu import ModelFK


@pytest.fixture(scope='module')
def native():
    torch.set_num_threads(1)
    surface=WujiSurface()
    return surface,WujiMeshCollision(surface,'cpu'),ModelFK(surface.model,torch.device('cpu'),torch.float32)


def test_native_parent_and_child_never_form_collision_pair(native):
    surface,audit,_=native
    for child,parent in enumerate(audit.parent_owners):
        if parent>=0:
            assert not audit.allowed_mask[child,parent]
            assert not audit.allowed_mask[parent,child]
    # All four joints of every native finger, including the palm/base joint.
    assert np.count_nonzero(audit.parent_owners>=0)==20
    for f in range(5):
        base=1+4*f
        assert not audit.allowed_mask[0,base]
        for k in range(3):assert not audit.allowed_mask[base+k,base+k+1]
    assert audit.allowed_mask[4,8]  # Thumb/index distal surfaces must still be checked.


def test_report_cannot_export_stale_or_accidental_depth_on_mechanical_pair(native):
    _,audit,_=native
    depth=torch.zeros(21,21)
    for child,parent in enumerate(audit.parent_owners):
        if parent>=0:depth[child,parent]=12.
    report=audit.report(depth,True)
    assert not report['pair_depths_mm'] and report['max_penetration_mm']==0.
    depth[4,8]=1.2
    report=audit.report(depth,True)
    assert list(report['pair_depths_mm'])==['thumb_link4__index_link4']
    assert report['max_penetration_mm']==pytest.approx(1.2)
    assert audit.report(depth,False)['pair_depths_mm']=={}


def test_mcp_collar_excludes_only_local_joint_connection():
    device='cpu'
    def arr(a,dtype=float):return wp.array(np.asarray(a),dtype=dtype,device=device)
    box=trimesh.creation.box(extents=[.01]*3)
    mesh=wp.Mesh(arr(box.vertices,wp.vec3),arr(box.faces.ravel(),int),support_winding_number=True)
    poses=arr(np.zeros((1,3,3)),wp.vec3)
    rotations=arr(np.broadcast_to(np.eye(3),(1,3,3,3)).copy(),wp.mat33)
    allowed=np.zeros((3,3),np.int32);allowed[0,2]=1
    def query(radius):
        depth=wp.zeros((1,3,3),device=device);invalid=wp.zeros(1,dtype=int,device=device)
        wp.launch(_native_penetration,(1,1,3),[arr([mesh.id]*3,wp.uint64),arr([[.003,0,0]],wp.vec3),arr([0],int),
            arr([[-.005]*3]*3,wp.vec3),arr([[.005]*3]*3,wp.vec3),arr([0,0,radius]),arr(allowed,int),
            poses,rotations,depth,invalid],device=device)
        assert invalid.numpy()[0]==0
        return depth.numpy()[0,0,2]
    assert query(.006)==0.  # Inside the actual joint collar.
    assert query(.001)==pytest.approx(2.,abs=1e-4)  # Outside collar: genuine penetration stays.
    allowed[0,2]=0
    assert query(.001)==0.  # Explicit mechanical adjacency always wins.


def test_neutral_is_clear_and_nonadjacent_penetration_is_preserved(native):
    _,audit,fk=native
    q=torch.zeros(2,26)
    q[1,6:]=torch.tensor([.64584655,-.57532471,1.22095430,-.11351934,
        .45659775,-.02537840,-.27771652,-.27140945,.24809290,.06261039,.55680192,-.12089592,
        .75052029,-.25887671,.50799608,.01866602,1.05431426,-.69800001,.92833167,-.12064862])
    depth,valid=audit.query(*fk.frames(q),dense=True)
    neutral=audit.report(depth[0],valid[0]);pinch=audit.report(depth[1],valid[1])
    assert neutral['query_valid'] and not neutral['pair_depths_mm']
    assert pinch['query_valid']
    assert pinch['pair_depths_mm']['thumb_link4__pinky_link4']>4.
    for pair in pinch['pair_depths_mm']:
        a,b=map(audit.names.index,pair.split('__'))
        assert audit.allowed_mask[a,b]

import numpy as np
import torch
import mujoco
import pytest
from scipy.spatial.transform import Rotation
from retarget.env.grasp.pose_gpu import ModelFK,euler_xyz_to_matrix,matrix_to_euler_xyz,WujiPoseAdapter


def test_native_model_fk_and_root_rotation():
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=mujoco.MjModel.from_xml_path('data/grasp/assets/wuji2/right.xml')
    fk=ModelFK(model,device,torch.float64)
    q=torch.randn(3,model.nq,device=device,dtype=torch.float64)*.3
    p,r=fk.frames(q);data=mujoco.MjData(model)
    for i in range(len(q)):
        data.qpos[:]=q[i].cpu().numpy();mujoco.mj_kinematics(model,data)
        np.testing.assert_allclose(p[i].cpu(),data.xpos,atol=1e-10)
        np.testing.assert_allclose(r[i].cpu(),data.xmat.reshape(-1,3,3),atol=1e-10)
    angles=q[:,:3]
    np.testing.assert_allclose(euler_xyz_to_matrix(angles).cpu(),Rotation.from_euler('XYZ',angles.cpu()).as_matrix(),atol=1e-10)
    torch.testing.assert_close(euler_xyz_to_matrix(matrix_to_euler_xyz(r[:,1])),r[:,1])


def test_gpu_ik_keeps_device_and_reduces_original_bone_residual():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    model=mujoco.MjModel.from_xml_path('data/grasp/assets/wuji2/right.xml')
    adapter=WujiPoseAdapter(model,'cuda:0')
    q=adapter.neutral.clone();q[6:]=.3;q=q.clamp(adapter.low+1.e-4,adapter.high-1.e-4)
    human=adapter.keypoints(q)
    result=adapter.initialize(human)
    assert result.is_cuda and torch.isfinite(result).all()
    target=torch.nn.functional.normalize(human[1:].reshape(5,4,3)[:,1:]-human[1:].reshape(5,4,3)[:,:-1],dim=-1)
    def cost(v):
        p=adapter.keypoints(v)[1:].reshape(5,4,3)
        return (torch.nn.functional.normalize(p[:,1:]-p[:,:-1],dim=-1)-target).square().sum()
    assert cost(result[0])<cost(adapter.neutral)

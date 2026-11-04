import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,warp as wp,mujoco,numpy as np
from retarget.env.grasp.mano_solver import ManoDecoder
from retarget.env.grasp.mano_adapter import ManoCollisionGeometry
from retarget.initialization.dexonomy.util.torch_util import torch_matrix_to_quaternion

device=sys.argv[1] if len(sys.argv)>1 else 'cuda:0'
torch.set_num_threads(1);wp.init();torch.cuda.set_device(device)
model=ManoDecoder(ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl',device)
geometry=ManoCollisionGeometry(model,4)
variables=torch.randn((4,30),device=device,generator=torch.Generator(device=device).manual_seed(17))*.08
with torch.no_grad(),wp.ScopedDevice(device),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(torch.device(device)))):
    vertices,joints,frames=model(variables,full=True)
    joints=joints.clone();frames=frames.clone()
    for row,other in ((2,6),(3,12)):
        joints[row,geometry.joint_order[3]]=joints[row,geometry.joint_order[other]]
        frames[row,3]=frames[row,other]
    depth,overflow=geometry.query(joints,frames)
    q=torch.cat((joints[:,geometry.joint_order],torch_matrix_to_quaternion(frames)),-1).reshape(4,-1).cpu().numpy()
    actual=depth.cpu().numpy()
expected=[]
for value in q:
    data=mujoco.MjData(geometry.cpu);data.qpos[:]=value
    mujoco.mj_kinematics(geometry.cpu,data);mujoco.mj_collision(geometry.cpu,data)
    expected.append(max([0.]+[-float(c.dist) for c in data.contact]))
error=float(np.max(np.abs(actual-expected)))
assert max(expected)>0 and not overflow.any() and error<1.e-4,(actual,expected,error)
report=dict(samples=4,case='includes_nonadjacent_body_overlap',maximum_depth_error_m=error,gpu_depths_m=actual.tolist(),cpu_depths_m=expected)
path=ROOT/'output/mano_shared_adapter_collision_audit.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

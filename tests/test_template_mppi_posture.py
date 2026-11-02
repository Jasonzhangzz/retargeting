import mujoco,numpy as np,pytest,torch
from retarget.config.run import ROOT
from retarget.env.grasp.template_mppi_posture import GraspPosture
from retarget.env.grasp.template_mppi_inputs import load_template_contacts
from retarget.env.grasp.template_mppi import TemplateMPPISettings,TemplatePoseObjective
from retarget.env.grasp.primitives import primitive_mesh

@pytest.fixture(scope='module')
def hand():return mujoco.MjModel.from_xml_path(str(ROOT/'data/grasp/assets/wuji2/right.xml'))

def posture(hand,name,device='cpu'):
 t=load_template_contacts(name,ROOT/'data/grasp/templates',hand,device='cpu')
 return GraspPosture(hand,t,int(name.split('_')[0]),device,TemplateMPPISettings()),t

def test_crossed_index_middle_rejected_and_order_is_root_invariant(hand):
 p,t=posture(hand,'24_Tip_Pinch');q=p.fk.tensor(np.r_[np.zeros(6),t.finger_qpos]);before=t.to_dict()
 assert float(p.batch(q[None])[1][0])==0
 q[11]=np.deg2rad(35)
 cost,cross,gaps=p.batch(q[None]);assert float(cross[0])>.02
 transformed=q.clone();transformed[:6]=q.new_tensor([.3,-.2,.1,.6,-.3,.4])
 torch.testing.assert_close(p.batch(transformed[None])[2],gaps,atol=1e-12,rtol=1e-12)
 assert t.to_dict()==before

def test_posture_gradient_and_padding_are_consistent(hand):
 p,t=posture(hand,'24_Tip_Pinch');q=p.fk.tensor(np.r_[np.zeros(6),t.finger_qpos]);q[11]=np.deg2rad(35)
 jac=torch.func.jacfwd(lambda x:p.single(x)[0])(q);assert torch.isfinite(jac).all()
 eps=1e-6;delta=torch.zeros_like(q);delta[11]=eps
 finite=(p.single(q+delta)[0]-p.single(q-delta)[0])/(2*eps)
 torch.testing.assert_close(jac[:,11],finite,atol=1e-6,rtol=1e-6)
 single=p.batch(q[None]);batch=p.batch(q[None].repeat(4,1))
 for a,b in zip(single,batch):torch.testing.assert_close(b,a.expand_as(b))

def test_idle_fingers_have_broad_bands_while_active_fingers_can_open(hand):
 p,t=posture(hand,'16_Lateral');lo=torch.full((26,),-2.);hi=torch.full((26,),2.);p.bounds(lo,hi)
 assert p.idle==[3,4,5]
 assert lo[14] == pytest.approx(np.deg2rad(45))
 assert lo[10] == pytest.approx(np.deg2rad(-5))
 assert hi[10] > 1.5 and hi[14] > 1.5
 assert p.metadata()['reference_1x_pose_used'] is False

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cpu_cuda_posture_agree(hand):
 a,t=posture(hand,'29_Stick');b,_=posture(hand,'29_Stick','cuda:0');q=a.fk.tensor(np.r_[np.zeros(6),t.finger_qpos])
 for cpu,gpu in zip(a.batch(q[None]),b.batch(q.cuda()[None])):torch.testing.assert_close(cpu,gpu.cpu(),atol=1e-9,rtol=1e-9)

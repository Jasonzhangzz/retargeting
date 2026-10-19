"""Reference twist and analytic velocity-loss gradients, including held targets."""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from forceaware.targets import object_reference_velocity,_interpolate_pose
from forceaware import mpc_loss as fa


@wp.kernel
def velocity_loss(job:fa.LossJob,output:wp.array(dtype=float)):
    world=wp.tid()
    output[world]=fa._object_velocity_squared(job,world)


def test_reference_velocity_matches_pose_derivative_and_quaternion_sign():
    initial=Rotation.from_euler('xyz',[.4,-.3,.2])
    frames=np.arange(5)
    rotation=Rotation.from_rotvec(np.array([0.,0.,.2])[None]*frames[:,None])*initial
    poses=np.c_[rotation.as_quat(),np.c_[frames*.01,frames*-.02,frames*.03]]
    poses[1::2,:4]*=-1.
    query=np.array([.25,1.7,2.4]);dt=.1
    actual=object_reference_velocity(poses,query,ref_dt=dt,end_frame=4)
    eps=1e-4
    plus=_interpolate_pose(poses,query+eps);minus=_interpolate_pose(poses,query-eps)
    numeric_linear=(plus[:,4:]-minus[:,4:])/(2*eps*dt)
    numeric_angular=(Rotation.from_quat(plus[:,:4])*Rotation.from_quat(minus[:,:4]).inv()).as_rotvec()/(2*eps*dt)
    np.testing.assert_allclose(actual[:,:3],numeric_linear,rtol=2e-3,atol=1e-5)
    np.testing.assert_allclose(actual[:,3:],numeric_angular,rtol=3e-3,atol=2e-3)
    np.testing.assert_allclose(actual[:,:3],np.tile([.1,-.2,.3],(3,1)),atol=1e-7)
    np.testing.assert_allclose(actual[:,:2],np.tile([.1,-.2],(3,1)),atol=1e-7)


def test_velocity_zero_at_clipped_endpoint_and_fractional_hold():
    poses=np.zeros((4,7));poses[:,3]=1.;poses[:,4]=np.arange(4)
    query=np.array([.3,1.2,1.5,2.,3.])
    value=object_reference_velocity(poses,query,ref_dt=.1,end_frame=1.5)
    np.testing.assert_allclose(value[:2,0],10.)
    np.testing.assert_array_equal(value[2:],0.)
    np.testing.assert_array_equal(object_reference_velocity(poses,query,ref_dt=.1,end_frame=3,hold=True),0.)


def test_velocity_loss_world_frame_and_gradients_against_finite_differences():
    device='cpu';job=fa.LossJob();job.model.object_qpos=1;job.model.object_dof=2
    job.weights.object_velocity=.2;job.weights.object_angular_velocity=.07
    rotation=Rotation.from_euler('xyz',[.4,-.2,1.1])
    q=np.zeros((1,8),np.float32);q[0,4:8]=rotation.as_quat()[[3,0,1,2]]
    v=np.array([[99.,99.,.1,-.2,.3,.4,-.5,.6]],np.float32)
    target=np.array([.03,.02,-.01,-.2,.1,.07],np.float32)
    job.state.qpos=wp.array(q,dtype=float,device=device,requires_grad=True)
    job.state.qvel=wp.array(v,dtype=float,device=device,requires_grad=True)
    job.target.object_velocity_world=wp.array(target,dtype=float,device=device)
    def fill_arrays(node):
        for name,field in node._cls.vars.items():
            value=getattr(node,name)
            if isinstance(field.type,wp.array) and value is None:
                setattr(node,name,wp.zeros((1,)*field.type.ndim,dtype=field.type.dtype,device=device))
            elif hasattr(value,'_cls'):fill_arrays(value)
    fill_arrays(job)
    output=wp.zeros(1,dtype=float,device=device,requires_grad=True)
    with wp.Tape() as tape:wp.launch(velocity_loss,1,inputs=[job],outputs=[output],device=device)
    tape.backward(grads={output:wp.ones(1,device=device)})
    expected=.2**2*np.sum((v[0,2:5]-target[:3])**2)+.07**2*np.sum((rotation.apply(v[0,5:8])-target[3:])**2)
    np.testing.assert_allclose(output.numpy()[0],expected,rtol=1e-6)
    assert np.all(job.state.qvel.grad.numpy()[0,:2]==0.)
    def evaluate(qp,vp):
        job.state.qpos.assign(qp);job.state.qvel.assign(vp)
        wp.launch(velocity_loss,1,inputs=[job],outputs=[output],device=device)
        return float(output.numpy()[0])
    for source,gradient,is_q in ((q,job.state.qpos.grad.numpy(),True),(v,job.state.qvel.grad.numpy(),False)):
        for col in range(source.shape[1]):
            plus,minus=source.copy(),source.copy();plus[0,col]+=2e-4;minus[0,col]-=2e-4
            fd=(evaluate(plus,v)-evaluate(minus,v))/(4e-4) if is_q else (evaluate(q,plus)-evaluate(q,minus))/(4e-4)
            np.testing.assert_allclose(gradient[0,col],fd,rtol=.005,atol=1e-5)

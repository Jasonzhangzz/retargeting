"""Input-only anatomical wrist/bone IK; no task optimization."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
FINGERS={'thumb':(12,1,1),'index':(0,2,5),'middle':(3,3,9),'ring':(9,4,13),'pinky':(6,5,17)}
def nearest_sdf(surface,points):
    """Exact triangle-mesh signed-distance nearest points, with safe KD culling."""
    from retarget.env.geometry import closest_triangles
    points=np.asarray(points,float);flat=points.reshape(-1,3)
    output=np.empty_like(flat);normal=np.empty_like(flat)
    for row,point in enumerate(flat):
        _,seed=surface.triangle_tree.query(point)
        first=closest_triangles(point[None],surface.triangles[[seed]])[0,0]
        upper=np.linalg.norm(first-point)
        ids=np.asarray(surface.triangle_tree.query_ball_point(point,upper+surface.max_triangle_radius+1e-10),int)
        candidates=closest_triangles(point[None],surface.triangles[ids])[0]
        choice=np.argmin(np.sum((candidates-point)**2,axis=1))
        output[row]=candidates[choice];normal[row]=surface.normals[ids[choice]]
    delta=flat-output;distance=np.linalg.norm(delta,axis=1)
    sign=np.where(np.sum(delta*normal,axis=1)>=0.,1.,-1.)
    phi=sign*distance
    gradient=np.where((distance>1e-12)[:,None],sign[:,None]*delta/np.maximum(distance[:,None],1e-12),normal)
    # This is p - phi(p) grad(phi), using the closest-feature SDF gradient.
    projected=flat-phi[:,None]*gradient
    np.testing.assert_allclose(projected,output,atol=1e-10)
    return projected.reshape(points.shape),normal.reshape(points.shape),phi.reshape(points.shape[:-1])

def directions(points):
    delta=np.diff(points,axis=0)
    return delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)

def palm_frame(wrist, roots):
    """Anatomical frame from the wrist and five MCP/CMC roots, independent of grasp."""
    origin=np.mean(np.r_[wrist[None],roots],axis=0)
    x=roots[1]-roots[4];x/=np.linalg.norm(x)
    forward=np.mean(roots[1:],axis=0)-wrist
    z=np.cross(x,forward);z/=np.linalg.norm(z)
    y=np.cross(z,x)
    return origin,np.stack((x,y,z),axis=-1)

def retarget_from_mano(model, joints):
    """Direct palm + absolute bone-direction IK, with no feasible robot reference.

    This routine deliberately accepts no baseline trajectory, contact targets,
    object poses or masks. The robot neutral geometry defines the palm frame;
    subsequent IK warm starts come only from the preceding generated frame.
    """
    import mujoco
    from time import perf_counter
    started=perf_counter();data=mujoco.MjData(model)
    qids=np.arange(26);joint_ids=np.array([np.flatnonzero(model.jnt_qposadr==c)[0] for c in qids])
    lo,hi=model.jnt_range[joint_ids].T
    neutral=np.clip(model.qpos0[:26],lo+1e-8,hi-1e-8)
    data.qpos[:26]=neutral;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    chains={};robot_roots=[]
    for name,(_,robot,mano) in FINGERS.items():
        bodies=[model.body(f'right_finger{robot}_link{i}').id for i in (2,3,4)]
        site=model.site(f'right_{name}_tip').id
        chains[name]=(bodies,site,mano,np.arange(6+4*(robot-1),6+4*robot))
        robot_roots.append(data.xpos[bodies[0]].copy())
    robot_origin,robot_basis=palm_frame(np.zeros(3),np.asarray(robot_roots))
    output=np.empty((len(joints),26));errors=[];evaluations=[]
    # TRF can terminate prematurely when all flexion coordinates start within
    # 1e-8 of a lower bound. Use a generic, slightly flexed neutral seed.
    previous=np.clip(neutral,lo+.1,hi-.1)
    for frame,human in enumerate(joints):
        roots=human[[FINGERS[name][2] for name in FINGERS]]
        origin,basis=palm_frame(human[0],roots)
        rotation=basis@robot_basis.T
        base=previous.copy();base[:3]=origin-rotation@robot_origin
        # The XML uses serial X/Y/Z hinge joints: R = Rx @ Ry @ Rz.
        base[3:6]=Rotation.from_matrix(rotation).as_euler('XYZ')
        if frame:base[3:6]+=2*np.pi*np.round((previous[3:6]-base[3:6])/(2*np.pi))
        for name,(bodies,site,mano,cols) in chains.items():
            desired=directions(human[mano:mano+4])
            def evaluate(values):
                q=base.copy();q[cols]=values
                data.qpos[:26]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                points=np.r_[data.xpos[bodies],data.site_xpos[site][None]].copy();jac=[]
                for body in bodies:
                    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                    mujoco.mj_jacBody(model,data,jp,jr,body);jac.append(jp[:,cols])
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacSite(model,data,jp,jr,site);jac.append(jp[:,cols])
                vec=np.diff(points,axis=0);length=np.maximum(np.linalg.norm(vec,axis=1),1e-12);unit=vec/length[:,None]
                derivative=np.einsum('bij,bjk->bik',(np.eye(3)[None]-unit[:,:,None]*unit[:,None,:])/length[:,None,None],np.diff(jac,axis=0))
                return (unit-desired).ravel(),derivative.reshape(-1,4)
            fitted=least_squares(lambda v:evaluate(v)[0],np.clip(previous[cols],lo[cols]+1e-4,hi[cols]-1e-4),
                jac=lambda v:evaluate(v)[1],bounds=(lo[cols],hi[cols]),max_nfev=64,
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
            base[cols]=fitted.x;errors.append(float(np.sqrt(np.mean(fitted.fun**2))));evaluations.append(int(fitted.nfev))
        output[frame]=base;previous=base
    return output,dict(method='MANO_palm_frame_absolute_chain_IK',
        original_wuji_reference_used=False,object_or_contact_objective_used=False,
        initialization='slightly_flexed_model_neutral_then_previous_generated_frame',
        chain_direction_rms=float(np.mean(errors)),chain_direction_rms_max=float(np.max(errors)),
        evaluations=sum(evaluations),seconds=perf_counter()-started)

"""Humanoid task costs and contact geometry, independent of the simulator."""
import numpy as np
import warp as wp
from retarget.spider_pose import quaternion_error_squared, zero_safe_sqrt

vec16 = wp.types.vector(length=16,dtype=float)
vec12 = wp.types.vector(length=12,dtype=float)
vec5 = wp.types.vector(length=5,dtype=float)
BALANCE_PARTS = ('com_xy','com_height','dcm','vertical_velocity','collapse')

TERMS = ('root_position', 'root_rotation', 'joints', 'robot_velocity',
         'keypoint_position', 'keypoint_rotation', 'upright', 'balance',
         'foot_slip', 'action_reference', 'action_smoothness', 'penetration',
         'contact_wrench', 'hand_position', 'consensus', 'anchor')
SPIDER_TERMS = ('spider_pose',)+TERMS[1:]


def stack_host(arrays):
    """One batched device transfer instead of synchronizing every horizon step."""
    import torch
    with torch.no_grad():
        return torch.stack([wp.to_torch(array).detach() for array in arrays]).cpu().numpy()


def stack_host(arrays):
    """One batched device transfer instead of synchronizing every horizon step."""
    import torch
    with torch.no_grad():
        return torch.stack([wp.to_torch(array).detach() for array in arrays]).cpu().numpy()


@wp.struct
class Model:
    spider_cost: int
    object_qpos: int
    bodies: wp.array(dtype=int)
    robot_bodies: wp.array(dtype=int)
    mass: wp.array(dtype=float)
    ipos: wp.array(dtype=wp.vec3)
    joint_qpos: wp.array(dtype=int)
    robot_dofs: wp.array(dtype=int)
    role_bodies: wp.array(dtype=int)
    weights: wp.array(dtype=float)
    foot_centers: wp.array2d(dtype=wp.vec3)
    foot_radii: wp.array2d(dtype=float)
    foot_counts: wp.array(dtype=int)
    hand_centers: wp.array(dtype=wp.vec3)
    hand_radii: wp.array(dtype=float)
    object_geom_position: wp.vec3
    object_geom_rotation: wp.mat33
    object_halfsize: wp.vec3
    box_object: int
    root: int
    torso: int
    object_dof: int
    realization_weight: float
    wrench_robust_scale: float
    object_body: int
    robot_mass: float
    object_mass: float
    scale: float
    dt: float
    ground: float
    balance_xy_scale: float
    balance_z_scale: float
    balance_velocity_scale: float
    balance_height_margin: float
    balance_height_scale: float
    upright_scale: float


@wp.struct
class Targets:
    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    ctrl: wp.array2d(dtype=float)
    position: wp.array2d(dtype=wp.vec3)
    rotation: wp.array2d(dtype=wp.mat33)
    com: wp.array(dtype=wp.vec3)
    wrench: wp.array2d(dtype=wp.vec3)
    object_wrench: wp.array2d(dtype=wp.vec3)
    hand_position: wp.array2d(dtype=wp.vec3)
    mask: wp.array2d(dtype=float)
    local: wp.array2d(dtype=wp.vec3)
    anchor: wp.array2d(dtype=wp.vec3)
    stance: wp.array2d(dtype=wp.vec3)
    z: wp.array3d(dtype=wp.vec3)
    dual: wp.array3d(dtype=wp.vec3)
    forces: wp.array3d(dtype=wp.vec3)
    floor_point: wp.array2d(dtype=wp.vec3)
    floor_force: wp.array2d(dtype=wp.vec3)
    floor_torque: wp.array2d(dtype=wp.vec3)
    penalty: wp.array(dtype=float)


@wp.struct
class Stage:
    model: Model
    target: Targets
    index: int
    normalizer: float
    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    position: wp.array2d(dtype=wp.vec3)
    rotation: wp.array2d(dtype=wp.mat33)
    com: wp.array(dtype=wp.vec3)
    previous_com: wp.array(dtype=wp.vec3)
    previous_position: wp.array2d(dtype=wp.vec3)
    previous_rotation: wp.array2d(dtype=wp.mat33)
    ctrl: wp.array2d(dtype=float)
    previous_ctrl: wp.array2d(dtype=float)
    distance: wp.array2d(dtype=float)
    active: wp.array2d(dtype=int)
    terms: wp.array2d(dtype=float)
    loss: wp.array(dtype=float)
    query_points: wp.array2d(dtype=wp.vec3)
    balance_parts: wp.array2d(dtype=float)
    constraint_force: wp.array2d(dtype=float)
    physical_contact: int


@wp.func
def squared(v: wp.vec3):
    return wp.dot(v, v)


@wp.func
def robust_squared(value: float, scale: float):
    # Pseudo-Huber in residual norm; rational form avoids cancellation at zero.
    return 2.*value/(wp.sqrt(1.+value/(scale*scale))+1.)


@wp.func
def orientation_error(a: wp.mat33, b: wp.mat33):
    return 0.5*(wp.cross(wp.vec3(a[0,0],a[1,0],a[2,0]), wp.vec3(b[0,0],b[1,0],b[2,0]))
                + wp.cross(wp.vec3(a[0,1],a[1,1],a[2,1]), wp.vec3(b[0,1],b[1,1],b[2,1]))
                + wp.cross(wp.vec3(a[0,2],a[1,2],a[2,2]), wp.vec3(b[0,2],b[1,2],b[2,2])))


@wp.func
def rotation_squared(a: wp.mat33, b: wp.mat33):
    value = float(0.)
    for i in range(3):
        for j in range(3):
            delta = a[i,j]-b[i,j]
            value += 0.5*delta*delta
    return value


@wp.func
def foot_support(job: Stage, world: int, foot: int) -> wp.vec4:
    m=job.model
    body=m.role_bodies[foot+2]
    rotation=job.rotation[world,body]
    origin=job.position[world,body]
    minimum=float(1.e20)
    for j in range(m.foot_counts[foot]):
        center=origin+rotation@m.foot_centers[foot,j]
        minimum=wp.min(minimum,center[2]-m.foot_radii[foot,j])
    cx=float(0.);cy=float(0.);cz=float(0.);radius=float(0.);count=float(0.)
    for j in range(m.foot_counts[foot]):
        local=m.foot_centers[foot,j]
        center=origin+rotation@local
        if center[2]-m.foot_radii[foot,j]<=minimum+0.001:
            cx+=local[0];cy+=local[1];cz+=local[2]
            radius+=m.foot_radii[foot,j];count+=1.
    divisor=wp.max(count,1.)
    return wp.vec4(cx/divisor,cy/divisor,cz/divisor,radius/divisor)


@wp.func_grad(foot_support)
def foot_support_selection_gradient(job: Stage, world: int, foot: int, adj_ret: wp.vec4):
    # Output contains only fixed cloud coordinates/radii. Its selected set is
    # locally constant, hence its exact derivative is zero away from switches.
    # Differentiate the pose transform in contact_point, not the discrete set.
    return


@wp.func
def contact_point(job: Stage, world: int, role: int) -> wp.vec3:
    m=job.model
    body=m.role_bodies[role]
    rotation=job.rotation[world,body]
    origin=job.position[world,body]
    point=origin+rotation@job.target.local[job.index,role]
    if role>=2:
        support=foot_support(job,world,role-2)
        point=origin+rotation@wp.vec3(support[0],support[1],support[2])-wp.vec3(0.,0.,support[3])
    elif m.box_object!=0:
        ob=m.object_body
        object_rotation=job.rotation[world,ob]
        center=wp.transpose(m.object_geom_rotation)@(wp.transpose(object_rotation)@(origin+rotation@m.hand_centers[role]-job.position[world,ob])-m.object_geom_position)
        size=m.object_halfsize
        nearest=wp.vec3(wp.clamp(center[0],-size[0],size[0]),wp.clamp(center[1],-size[1],size[1]),wp.clamp(center[2],-size[2],size[2]))
        normal=wp.vec3(0.)
        if wp.abs(center[0])<=size[0] and wp.abs(center[1])<=size[1] and wp.abs(center[2])<=size[2]:
            axis=int(0);margin=size[0]-wp.abs(center[0])
            for j in range(1,3):
                if size[j]-wp.abs(center[j])<margin:
                    axis=j;margin=size[j]-wp.abs(center[j])
            normal[axis]=wp.where(center[axis]>=0.,1.,-1.)
        else:
            difference=center-nearest
            normal=difference/wp.max(wp.length(difference),1.e-9)
        surface=center-m.hand_radii[role]*normal
        point=job.position[world,ob]+object_rotation@(m.object_geom_position+m.object_geom_rotation@surface)
    return point


@wp.func
def realized_wrench(job: Stage, w: int) -> vec12:
    m=job.model
    f,t,of,ot=wp.vec3(0.),wp.vec3(0.),wp.vec3(0.),wp.vec3(0.)
    if job.physical_contact!=0:
        force=wp.vec3(job.constraint_force[w,0],job.constraint_force[w,1],job.constraint_force[w,2])
        torque=job.rotation[w,m.root]@wp.vec3(job.constraint_force[w,3],job.constraint_force[w,4],job.constraint_force[w,5])
        torque-=wp.cross(job.com[w]-job.position[w,m.root],force)
        scale=wp.max(m.robot_mass*9.81,1.)
        f=force/scale;t=torque/(scale*.5)
        if m.object_body>=0:
            d=m.object_dof
            force_object=wp.vec3(job.constraint_force[w,d],job.constraint_force[w,d+1],job.constraint_force[w,d+2])
            torque_object=job.rotation[w,m.object_body]@wp.vec3(job.constraint_force[w,d+3],job.constraint_force[w,d+4],job.constraint_force[w,d+5])
            torque_object-=wp.cross(job.rotation[w,m.object_body]@m.ipos[m.object_body],force_object)
            oscale=wp.max(m.object_mass*9.81,1.)
            of=force_object/oscale;ot=torque_object/(oscale*.3)
    return vec12(f[0],f[1],f[2],t[0],t[1],t[2],of[0],of[1],of[2],ot[0],ot[1],ot[2])


@wp.func
def balance_parts(job: Stage, world: int) -> vec5:
    m,t,k=job.model,job.target,job.index
    error=job.com[world]-t.com[k]
    velocity_error=wp.vec3(0.)
    if k>0:
        velocity_error=(job.com[world]-job.previous_com[world]-t.com[k]+t.com[k-1])/m.dt
    omega=wp.sqrt(9.81/wp.max(t.com[k][2]-m.ground,0.2))
    dcm=error+velocity_error/omega
    collapse=wp.max(t.position[k,m.root][2]-job.position[world,m.root][2]-m.balance_height_margin,0.)/m.balance_height_scale
    xy=(error[0]*error[0]+error[1]*error[1])/(m.balance_xy_scale*m.balance_xy_scale)
    height=error[2]*error[2]/(m.balance_z_scale*m.balance_z_scale)
    divergent=0.5*(dcm[0]*dcm[0]+dcm[1]*dcm[1])/(m.balance_xy_scale*m.balance_xy_scale)
    vertical=0.25*velocity_error[2]*velocity_error[2]/(m.balance_velocity_scale*m.balance_velocity_scale)
    return vec5(xy,height,divergent,vertical,collapse*collapse)


@wp.func
def qpos_quat(job: Stage, w: int, adr: int):
    return wp.quat(job.qpos[w,adr+1],job.qpos[w,adr+2],job.qpos[w,adr+3],job.qpos[w,adr])


@wp.func
def reference_quat(job: Stage, adr: int):
    k=job.index
    return wp.quat(job.target.qpos[k,adr+1],job.target.qpos[k,adr+2],job.target.qpos[k,adr+3],job.target.qpos[k,adr])


@wp.func
def spider_pose_squared(job: Stage, w: int):
    m,t,k=job.model,job.target,job.index
    position_scale=float(1.)
    if m.object_body<0:position_scale=3.
    value=float(0.)
    for j in range(3):value+=position_scale*position_scale*wp.pow(job.qpos[w,j]-t.qpos[k,j],2.)
    value+=quaternion_error_squared(qpos_quat(job,w,3),reference_quat(job,3))
    for j in range(m.joint_qpos.shape[0]):
        adr=m.joint_qpos[j]
        value+=wp.pow(job.qpos[w,adr]-t.qpos[k,adr],2.)
    if m.object_body>=0:
        adr=m.object_qpos
        for j in range(3):value+=wp.pow(job.qpos[w,adr+j]-t.qpos[k,adr+j],2.)
        value+=.09*quaternion_error_squared(qpos_quat(job,w,adr+3),reference_quat(job,adr+3))
    return value


@wp.func
def spider_loss_terms(job: Stage, w: int):
    m,t,k=job.model,job.target,job.index
    consensus=float(0.);anchor=float(0.)
    for role in range(4):
        point=contact_point(job,w,role)
        if role<2 and m.object_body>=0:
            point=wp.transpose(job.rotation[w,m.object_body])@(point-job.position[w,m.object_body])
        if k>0:consensus+=t.mask[k,role]*squared((point-t.z[k,w,role]+t.dual[k,w,role])/m.scale)
        anchor+=t.mask[k,role]*squared((t.z[k,w,role]-t.anchor[k,role])/m.scale)
    values=vec16(0.)
    values[0]=zero_safe_sqrt(spider_pose_squared(job,w))*job.normalizer
    values[14]=consensus*t.penalty[0]*job.normalizer
    values[15]=anchor*t.penalty[1]*job.normalizer
    return values


@wp.func
def loss_terms(job: Stage, w: int):
    m, t, k = job.model, job.target, job.index
    if m.spider_cost!=0:return spider_loss_terms(job,w)
    c0 = float(0.)
    c1 = float(0.)
    c2 = float(0.)
    c3 = float(0.)
    c4 = float(0.)
    c5 = float(0.)
    c6 = float(0.)
    c7 = float(0.)
    c8 = float(0.)
    c9 = float(0.)
    c10 = float(0.)
    c11 = float(0.)
    c12 = float(0.)
    c13 = float(0.)
    c14 = float(0.)
    hand = float(0.)
    root = job.position[w,m.root]
    c0 = squared(root-t.position[k,m.root])
    c1 = rotation_squared(job.rotation[w,m.root],t.rotation[k,m.root])
    for j in range(m.joint_qpos.shape[0]):
        adr = m.joint_qpos[j]
        d = job.qpos[w,adr]-t.qpos[k,adr]
        c2 += d*d
    for j in range(m.robot_dofs.shape[0]):
        adr = m.robot_dofs[j]
        d = job.qvel[w,adr]-t.qvel[k,adr]
        c3 += d*d
    for i in range(m.bodies.shape[0]):
        body_id = m.bodies[i]
        # Track body shape relative to the pelvis. Absolute translation is
        # already tracked by root_position and COM: repeating it at all 14
        # bodies lets path lag overwhelm joint posture and recovery.
        c4 += squared((job.position[w,body_id]-root)-(t.position[k,body_id]-t.position[k,m.root]))
        c5 += rotation_squared(job.rotation[w,body_id],t.rotation[k,body_id])
    # Reference-relative torso tilt: bending, running and crouching remain valid.
    tilt, tilt_ref = job.rotation[w,m.torso], t.rotation[k,m.torso]
    c6 = squared(wp.vec3(tilt[0,2]-tilt_ref[0,2],tilt[1,2]-tilt_ref[1,2],tilt[2,2]-tilt_ref[2,2]))/(m.upright_scale*m.upright_scale)
    com = job.com[w]
    # Dimensionless position/velocity/height viability, relative to the
    # demonstrated gait. No static support hull is imposed during running.
    bp=balance_parts(job,w)
    c7=bp[0]+bp[1]+bp[2]+bp[3]+bp[4]
    net_force, net_torque = wp.vec3(0.), wp.vec3(0.)
    object_force, object_torque = wp.vec3(0.), wp.vec3(0.)
    obj_pos, obj_rot, obj_com = wp.vec3(0.), wp.identity(n=3,dtype=float), wp.vec3(0.)
    if m.object_body >= 0:
        obj_pos = wp.vec3(job.position[w,m.object_body])
        obj_rot = wp.mat33(job.rotation[w,m.object_body])
        obj_com = obj_pos+obj_rot@m.ipos[m.object_body]
    for role in range(4):
        body = m.role_bodies[role]
        if role<2 and m.object_body>=0:
            center=job.position[w,body]+job.rotation[w,body]@m.hand_centers[role]
            center_ref=t.hand_position[k,role]
            hand+=t.mask[k,role]*squared(center-center_ref)
        point = contact_point(job,w,role)
        x = wp.vec3(point)
        z = wp.vec3(t.z[k,w,role])
        force = wp.vec3(t.forces[k,w,role])
        if role < 2 and m.object_body >= 0:
            x = wp.transpose(obj_rot)@(point-obj_pos)
            z = obj_pos+obj_rot@z
            object_force += force
            object_torque += wp.cross(z-obj_com,force)
            force = -force
        net_force += force
        net_torque += wp.cross(z-com,force)
        if k > 0:
            residual = (x-t.z[k,w,role]+t.dual[k,w,role])/m.scale
            c13 += t.mask[k,role]*squared(residual)
        c14 += t.mask[k,role]*squared((t.z[k,w,role]-t.anchor[k,role])/m.scale)
        if role >= 2:
            # Velocity of the SAME material foot point in consecutive poses.
            # The selected heel/toe may change without falsely treating COP
            # motion as slip, and z cannot move to evade this physical term.
            local=wp.transpose(job.rotation[w,body])@(point-job.position[w,body])
            old_point = job.previous_position[w,body]+job.previous_rotation[w,body]@local
            slip = (point-old_point)/m.dt
            c8 += t.mask[k,role]*(slip[0]*slip[0]+slip[1]*slip[1])
    desired_force = t.wrench[k,0] + m.robot_mass*20.*(t.com[k]-com)
    desired_force += m.robot_mass*4.*wp.vec3(t.qvel[k,0]-job.qvel[w,0],t.qvel[k,1]-job.qvel[w,1],t.qvel[k,2]-job.qvel[w,2])
    desired_torque = t.wrench[k,1]+20.*orientation_error(job.rotation[w,m.torso],t.rotation[k,m.torso])
    scale = wp.max(m.robot_mass*9.81,1.)
    c12 = squared((net_force-desired_force)/scale)+squared((net_torque-desired_torque)/(scale*0.5))
    if m.object_body >= 0:
        support = t.floor_force[k,w]
        object_force += support
        object_torque += t.floor_torque[k,w]-wp.cross(obj_com,support)
        object_scale = wp.max(m.object_mass*9.81,1.)
        c12 += squared((object_force-t.object_wrench[k,0])/object_scale)
        c12 += squared((object_torque-t.object_wrench[k,1])/(object_scale*0.3))
    if job.physical_contact!=0:
        actual=realized_wrench(job,w)
        object_scale=wp.max(m.object_mass*9.81,1.)
        for j in range(3):
            c12+=m.realization_weight*(wp.pow(net_force[j]/scale-actual[j],2.)+wp.pow(net_torque[j]/(scale*.5)-actual[3+j],2.))
            if m.object_body>=0:
                c12+=m.realization_weight*(wp.pow(object_force[j]/object_scale-actual[6+j],2.)+wp.pow(object_torque[j]/(object_scale*.3)-actual[9+j],2.))
    for j in range(job.ctrl.shape[1]):
        d = job.ctrl[w,j]-t.ctrl[k,j]
        v = job.ctrl[w,j]-job.previous_ctrl[w,j]
        c9 += d*d
        c10 += v*v
    for j in range(job.distance.shape[1]):
        if job.active[w,j] > 0:
            d = wp.min(job.distance[w,j],0.)
            c11 += d*d
    if m.object_body>=0:c12=robust_squared(c12,m.wrench_robust_scale)
    return vec16(c0*m.weights[0]*job.normalizer,c1*m.weights[1]*job.normalizer,c2*m.weights[2]*job.normalizer,c3*m.weights[3]*job.normalizer,c4*m.weights[4]*job.normalizer,c5*m.weights[5]*job.normalizer,c6*m.weights[6]*job.normalizer,c7*m.weights[7]*job.normalizer,c8*m.weights[8]*job.normalizer,c9*m.weights[9]*job.normalizer,c10*m.weights[10]*job.normalizer,c11*m.weights[11]*job.normalizer,c12*m.weights[12]*job.normalizer,hand*m.weights[13]*job.normalizer,c13*t.penalty[0]*job.normalizer,c14*t.penalty[1]*job.normalizer)


@wp.kernel
def whole_body_loss(job: Stage):
    world = wp.tid()
    values = loss_terms(job,world)
    total = float(0.)
    for i in range(16):
        total += values[i]
    job.loss[world] = total


@wp.kernel(enable_backward=False)
def diagnose_terms(job: Stage):
    world = wp.tid()
    values = loss_terms(job,world)
    for i in range(16):
        job.terms[world,i] = values[i]
    for role in range(4):
        job.query_points[world,role]=contact_point(job,world,role)
    parts=balance_parts(job,world)
    for i in range(5):
        job.balance_parts[world,i]=parts[i]*job.model.weights[7]*job.normalizer


@wp.kernel(enable_backward=False)
def pack_snapshot(job: Stage, output: wp.array3d(dtype=float)):
    w=wp.tid()
    k=job.index
    actual=realized_wrench(job,w)
    for j in range(12):output[k,w,39+j]=actual[j]
    for j in range(3):
        output[k,w,j]=job.com[w][j]
        output[k,w,36+j]=job.qvel[w,j]
    for role in range(4):
        point=contact_point(job,w,role)
        for j in range(3):
            output[k,w,3+role*3+j]=point[j]
    for i in range(3):
        for j in range(3):
            output[k,w,15+i*3+j]=job.rotation[w,job.model.torso][i,j]
    if job.model.object_body>=0:
        b=job.model.object_body
        for i in range(3):
            output[k,w,24+i]=job.position[w,b][i]
            for j in range(3):
                output[k,w,27+i*3+j]=job.rotation[w,b][i,j]


@wp.kernel(enable_backward=False)
def compute_com(job: Stage):
    world=wp.tid()
    value=wp.vec3(0.)
    for i in range(job.model.robot_bodies.shape[0]):
        body=job.model.robot_bodies[i]
        value+=job.model.mass[body]*(job.position[world,body]+job.rotation[world,body]@job.model.ipos[body])/job.model.robot_mass
    job.com[world]=value


@wp.kernel(enable_backward=False)
def build_controls(variables: wp.array2d(dtype=float), reference: wp.array2d(dtype=float),
                   scale: float, previous: wp.array(dtype=float), lower: wp.array(dtype=float),
                   upper: wp.array(dtype=float), left: int, right: int, fraction: float,
                   output: wp.array2d(dtype=float)):
    world, joint = wp.tid()
    n = output.shape[1]
    a = previous[joint]
    if left >= 0:
        a = reference[left,joint]+scale*variables[world,left*n+joint]
    b = reference[right,joint]+scale*variables[world,right*n+joint]
    output[world,joint] = wp.clamp((1.-fraction)*a+fraction*b,lower[joint],upper[joint])


@wp.kernel(enable_backward=False)
def build_controls_from_dense_base(variables: wp.array2d(dtype=float), base: wp.array2d(dtype=float),
                   boundary: wp.array(dtype=float), scale: float, lower: wp.array(dtype=float), upper: wp.array(dtype=float),
                   left: int, right: int, fraction: float, step: int,
                   output: wp.array2d(dtype=float)):
    world,joint=wp.tid();n=output.shape[1]
    a=boundary[joint]
    if left>=0:a=variables[world,left*n+joint]
    b=variables[world,right*n+joint]
    output[world,joint]=wp.clamp(base[step,joint]+scale*((1.-fraction)*a+fraction*b),lower[joint],upper[joint])


@wp.kernel(enable_backward=False)
def add_losses(job: Stage, failure: wp.array(dtype=int), total: wp.array(dtype=float), terminal: int):
    w = wp.tid()
    value=job.loss[w]
    if failure[w]!=0:value=wp.inf
    if terminal!=0:
        for j in range(job.qpos.shape[1]):
            if not wp.isfinite(job.qpos[w,j]):value=wp.inf
        for j in range(job.qvel.shape[1]):
            if not wp.isfinite(job.qvel[w,j]):value=wp.inf
    total[w] += value



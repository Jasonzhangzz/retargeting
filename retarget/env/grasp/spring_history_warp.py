"""Fused execution of the v5 implicit contact integrator, without host iteration.

One CUDA thread owns a world and its contact history. The constitutive law,
four substeps, eight-iteration ceiling and five-way line search are unchanged.
The Newton system is solved by a 3x3 block Schur complement. Converged worlds
stop independently on the GPU at 1/100 of the final residual tolerance.
"""
import torch
import warp as wp


@wp.func
def rotation_increment(a: wp.vec3):
    theta=wp.length(a)
    x=float(1.)
    y=float(.5)
    if theta>1.e-4:
        x=wp.sin(theta)/theta
        half=wp.sin(.5*theta)/theta
        y=2.*half*half
    else:
        x=1.-theta*theta/6.
        y=.5-theta*theta/24.
    S=wp.skew(a)
    return wp.identity(n=3,dtype=float)+x*S+y*(S*S)


@wp.func
def force_and_tangent(v: wp.vec3,n: wp.vec3,compression: float,history: wp.vec3,
                      dt: float,kn: float,kt: float,cn: float,ct: float,mu: float):
    eye=wp.identity(n=3,dtype=float)
    P=eye-wp.outer(n,n)
    vt=v-wp.dot(v,n)*n
    xi=history-wp.dot(history,n)*n
    an=kn*dt+cn
    at=kt*dt+ct
    fn=wp.max(0.,kn*compression+an*wp.dot(v,n))
    trial=-kt*xi-at*vt
    length=wp.max(wp.length(trial),1.e-12)
    limit=mu*fn
    ft=wp.min(1.,limit/length)*trial
    J=wp.mat33(0.)
    if fn>0.:
        J=-an*wp.outer(n,n)
        if length<=limit:
            J=J-at*P
        else:
            direction=trial/length
            J=J+mu*an*wp.outer(direction,n)-(limit/length)*at*(eye-wp.outer(direction,direction))*P
    return -fn*n+ft,fn,ft,J


@wp.func
def force_only(v: wp.vec3,n: wp.vec3,compression: float,history: wp.vec3,
               dt: float,kn: float,kt: float,cn: float,ct: float,mu: float):
    vn=wp.dot(v,n)
    xi=history-wp.dot(history,n)*n
    fn=wp.max(0.,kn*compression+(kn*dt+cn)*vn)
    trial=-kt*xi-(kt*dt+ct)*(v-vn*n)
    ft=wp.min(1.,mu*fn/wp.max(wp.length(trial),1.e-12))*trial
    return -fn*n+ft,fn,ft


@wp.func
def wrench(b: int,contacts: int,v: wp.vec3,w: wp.vec3,
           arm: wp.array2d[wp.vec3],normal: wp.array2d[wp.vec3],compression: wp.array2d[float],
           history: wp.array2d[wp.vec3],allowed: wp.array2d[bool],
           dt: float,kn: float,kt: float,cn: float,ct: float,mu: float):
    f=wp.vec3(0.)
    t=wp.vec3(0.)
    for k in range(contacts):
        if allowed[b,k]:
            r=arm[b,k]
            force,unused_fn,unused_ft=force_only(v+wp.cross(w,r),normal[b,k],compression[b,k],history[b,k],dt,kn,kt,cn,ct,mu)
            f=f+force
            t=t+wp.cross(r,force)
    return f,t


@wp.kernel(enable_backward=False)
def integrate(local_points: wp.array2d[wp.vec3],local_normals: wp.array2d[wp.vec3],hand: wp.array2d[wp.vec3],
              allowed: wp.array2d[bool],inertia: wp.array[wp.mat33],inv_inertia: wp.array[wp.mat33],
              linear: wp.array[wp.vec3],momentum: wp.array[wp.vec3],center: wp.array[wp.vec3],rotation: wp.array[wp.mat33],
              free_rotation: wp.array[wp.mat33],history: wp.array2d[wp.vec3],
              arm: wp.array2d[wp.vec3],normal: wp.array2d[wp.vec3],compression: wp.array2d[float],
              final_fn: wp.array2d[float],final_ft: wp.array2d[wp.vec3],max_force: wp.array[float],
              converged: wp.array[bool],newton_steps: wp.array[int],
              contacts: int,steps: int,iterations: int,dt: float,mass: float,gravity: wp.vec3,
              kn: float,kt: float,cn: float,ct: float,mu: float,linear_tol: float,angular_tol: float):
    b=wp.tid()
    R=rotation[b]
    free_R=R
    com=center[b]
    v=linear[b]
    L=momentum[b]
    free_L=L
    good=bool(True)
    peak=float(0.)
    solves=int(0)
    eye=wp.identity(n=3,dtype=float)
    for step in range(steps):
        count=int(0)
        for k in range(contacts):
            if allowed[b,k]:
                r=R*local_points[b,k]
                n=R*local_normals[b,k]
                arm[b,k]=r
                normal[b,k]=n
                compression[b,k]=wp.dot(com+r-hand[b,k],n)
                h=history[b,k]
                history[b,k]=h-wp.dot(h,n)*n
                count=count+1
        Iw=R*inertia[b]*wp.transpose(R)
        Ii=R*inv_inertia[b]*wp.transpose(R)
        rhs_v=mass*(v+dt*gravity)
        rhs_w=L
        u=v+dt*gravity
        w=Ii*L
        for iteration in range(iterations):
            if count==0:
                break
            f=wp.vec3(0.)
            t=wp.vec3(0.)
            A=mass*eye
            B=wp.mat33(0.)
            C=wp.mat33(0.)
            D=Iw
            for k in range(contacts):
                if allowed[b,k]:
                    r=arm[b,k]
                    F,unused_fn,unused_ft,J=force_and_tangent(u+wp.cross(w,r),normal[b,k],compression[b,k],history[b,k],dt,kn,kt,cn,ct,mu)
                    S=wp.skew(r)
                    f=f+F
                    t=t+wp.cross(r,F)
                    A=A-dt*J
                    B=B+dt*J*S
                    C=C-dt*S*J
                    D=D+dt*S*J*S
            rv=mass*u-rhs_v-dt*f
            rw=Iw*w-rhs_w-dt*t
            ev=rv/mass
            ew=Ii*rw
            if wp.length(ev)<=.01*linear_tol and wp.length(ew)<=.01*angular_tol:
                break
            Ai=wp.inverse(A)
            schur=D-C*Ai*B
            dw=wp.inverse(schur)*(-rw+C*Ai*rv)
            du=Ai*(-rv-B*dw)
            if not wp.isfinite(du) or not wp.isfinite(dw):
                du=wp.vec3(0.)
                dw=wp.vec3(0.)
            best=float(1.e30)
            next_u=u
            next_w=w
            scale=float(1.)
            for attempt in range(5):
                if attempt==4:
                    scale=0.
                uv=u+scale*du
                uw=w+scale*dw
                ff,tt=wrench(b,contacts,uv,uw,arm,normal,compression,history,allowed,dt,kn,kt,cn,ct,mu)
                er_v=(mass*uv-rhs_v-dt*ff)/(mass*linear_tol)
                er_w=(Ii*(Iw*uw-rhs_w-dt*tt))/angular_tol
                metric=wp.length_sq(er_v)+wp.length_sq(er_w)
                if wp.isfinite(metric) and metric<best:
                    best=metric
                    next_u=uv
                    next_w=uw
                scale=.5*scale
            u=next_u
            w=next_w
            solves=solves+1
        f=wp.vec3(0.)
        t=wp.vec3(0.)
        for k in range(contacts):
            fn=float(0.)
            ft=wp.vec3(0.)
            if allowed[b,k]:
                r=arm[b,k]
                vp=u+wp.cross(w,r)
                n=normal[b,k]
                F,fn,ft=force_only(vp,n,compression[b,k],history[b,k],dt,kn,kt,cn,ct,mu)
                f=f+F
                t=t+wp.cross(r,F)
                history[b,k]=-(ft+ct*(vp-wp.dot(vp,n)*n))/kt
                good=good and wp.isfinite(F)
            final_fn[b,k]=fn
            final_ft[b,k]=ft
            peak=wp.max(peak,fn)
        good=good and wp.isfinite(u) and wp.isfinite(w)
        good=good and wp.length((mass*u-rhs_v-dt*f)/mass)<=linear_tol
        good=good and wp.length(Ii*(Iw*w-rhs_w-dt*t))<=angular_tol
        # Invalid states remain invalid, while preserving the last finite step.
        if not wp.isfinite(u):
            u=wp.vec3(0.)
        if not wp.isfinite(w):
            w=wp.vec3(0.)
        v=u
        L=Iw*w
        com=com+dt*v
        R=rotation_increment(dt*w)*R
        for k in range(contacts):
            if allowed[b,k]:
                n=R*local_normals[b,k]
                gap=wp.dot(hand[b,k]-com-R*local_points[b,k],n)
                h=history[b,k]
                h=h-wp.dot(h,n)*n
                if final_fn[b,k]<=0. or gap>0.:
                    h=wp.vec3(0.)
                history[b,k]=h
        free_w=free_R*inv_inertia[b]*wp.transpose(free_R)*free_L
        free_R=rotation_increment(dt*free_w)*free_R
    linear[b]=v
    momentum[b]=L
    center[b]=com
    rotation[b]=R
    free_rotation[b]=free_R
    max_force[b]=peak
    converged[b]=good
    newton_steps[b]=solves


def integrate_fused(rollout,linear,momentum,center,rotation,local_points,local_normals,hand,history,inertia,inv,allowed):
    """Torch-owned storage permits both eager use and CUDA graph capture."""
    s=rollout.settings
    c=rollout.spring
    b,k=allowed.shape
    center=center.clone()
    rotation=rotation.clone()
    free_rotation=torch.empty_like(rotation)
    arm=torch.empty_like(hand)
    normal=torch.empty_like(hand)
    compression=torch.empty_like(allowed,dtype=hand.dtype)
    fn=torch.empty_like(compression)
    ft=torch.empty_like(hand)
    peak=torch.empty_like(center[:,0]).contiguous()
    good=torch.empty(b,device=hand.device,dtype=torch.bool)
    solves=torch.empty(b,device=hand.device,dtype=torch.int32)
    vec=lambda x:wp.from_torch(x.contiguous(),dtype=wp.vec3)
    mat=lambda x:wp.from_torch(x.contiguous(),dtype=wp.mat33)
    with wp.ScopedDevice(rollout.device),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(hand.device)),sync_enter=False,sync_exit=False):
        wp.launch(integrate,dim=b,inputs=[vec(local_points),vec(local_normals),vec(hand),wp.from_torch(allowed),mat(inertia),mat(inv),
            vec(linear),vec(momentum),vec(center),mat(rotation),mat(free_rotation),vec(history),vec(arm),vec(normal),
            wp.from_torch(compression),wp.from_torch(fn),vec(ft),wp.from_torch(peak),wp.from_torch(good),wp.from_torch(solves),
            k,s.horizon*c.substeps,c.newton_iterations,s.timestep/c.substeps,s.mass,wp.vec3(*s.gravity),
            c.normal_stiffness_n_m,c.tangent_stiffness_n_m,c.normal_damping_ns_m,c.tangent_damping_ns_m,s.friction,
            c.linear_residual_m_s,c.angular_residual_rad_s],device=rollout.device)
    return linear,momentum,center,rotation,history,good,fn,ft,peak,free_rotation,solves

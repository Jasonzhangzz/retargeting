"""Fused CUDA execution of the analytic, time-sequential Riccati/box-QP.

One cooperative CUDA block per record; this is not a time-parallel solver.
Same projected Newton/Armijo rules as Robokit's box-QP, FP64 value functions,
FP32 stored gains, and the same absolute-control feedback rollout.
"""
import torch
import warp as wp


@wp.func
def clamp64(x:wp.float64,lo:wp.float64,hi:wp.float64):return wp.clamp(x,lo,hi)


@wp.func
def free64(x:wp.float64,g:wp.float64,lo:wp.float64,hi:wp.float64,valid:wp.float64):
    f=valid
    if (x<=lo and g>wp.float64(0.)) or (x>=hi and g<wp.float64(0.)):f=wp.float64(0.)
    return f


@wp.func
def rounded64(x:wp.float64):return wp.float64(wp.float32(x))


_BACKWARD={}


def backward_kernel(dimension):
    if dimension in _BACKWARD:return _BACKWARD[dimension]
    Dc=wp.constant(dimension)
    def backward(d:int,h:int,cg:wp.array3d(dtype=wp.float64),ch:wp.array4d(dtype=wp.float64),ctrl:wp.array3d(dtype=wp.float32),
                 lo:wp.array(dtype=wp.float64),hi:wp.array(dtype=wp.float64),valid:wp.array(dtype=wp.float64),eye:wp.array2d(dtype=wp.float64),
                 mu_all:wp.array(dtype=wp.float64),ff:wp.array3d(dtype=wp.float32),fb:wp.array4d(dtype=wp.float32),ok:wp.array(dtype=wp.int32)):
        b=wp.tid();mu=mu_all[b];ok[b]=1
        ident=wp.tile_load(eye,shape=(Dc,Dc));mask=wp.tile_load(valid,shape=(Dc,));maskcol=wp.tile_reshape(mask,shape=(Dc,1))
        outer=wp.tile_matmul(maskcol,wp.tile_transpose(maskcol))
        vg=wp.tile_zeros(shape=(Dc,),dtype=wp.float64);vh=wp.tile_zeros(shape=(Dc,Dc),dtype=wp.float64)
        next_k=wp.tile_zeros(shape=(Dc,),dtype=wp.float64)
        for reverse in range(h):
            t=h-1-reverse
            quu=wp.tile_load(ch[b,t],shape=(Dc,Dc),offset=(d,d))+vh
            quu=wp.tile_map(wp.mul,quu,outer)
            qu=wp.tile_map(wp.mul,wp.tile_load(cg[b,t],shape=(Dc,),offset=(d,))+vg,mask)
            qx=wp.tile_map(wp.mul,wp.tile_load(cg[b,t],shape=(Dc,)),mask)
            w=ch[b,t,0,0]
            u=wp.tile_zeros(shape=(Dc,),dtype=wp.float64)
            for i in range(Dc):
                if i<d:u[i]=wp.float64(ctrl[b,t,i])
            lower=wp.tile_load(lo,shape=(Dc,))-u;upper=wp.tile_load(hi,shape=(Dc,))-u
            diag=wp.tile_zeros(shape=(Dc,),dtype=wp.float64)
            for i in range(Dc):diag[i]=mu*mask[i]+(wp.float64(1.)-mask[i])
            factor=wp.tile_cholesky(wp.tile_diag_add(quu,diag))
            k=wp.tile_cholesky_solve(factor,-qu)
            outside=wp.float64(0.)
            for i in range(Dc):
                if k[i]<lower[i] or k[i]>upper[i]:outside=wp.float64(1.)
            free=mask
            if outside>wp.float64(0.):
                k=wp.tile_map(clamp64,next_k,lower,upper)
                value=wp.float64(1e300)
                for iteration in range(100):
                    qk=wp.tile_reshape(wp.tile_matmul(quu,wp.tile_reshape(k,shape=(Dc,1))),shape=(Dc,))
                    grad=qu+mu*k+qk
                    free=wp.tile_map(free64,k,grad,lower,upper,mask)
                    gradnorm=wp.tile_sum(wp.tile_map(wp.mul,wp.tile_map(wp.mul,grad,grad),free))[0]
                    newvalue=wp.tile_sum(wp.tile_map(wp.mul,k,qu+wp.float64(.5)*(qk+mu*k)))[0]
                    small=value-newvalue<wp.float64(1e-12)*(wp.float64(1.)+wp.abs(newvalue))
                    if wp.tile_sum(free)[0]==wp.float64(0.) or wp.sqrt(gradnorm)<wp.float64(1e-8) or small:break
                    value=newvalue
                    freecol=wp.tile_reshape(free,shape=(Dc,1));freeouter=wp.tile_matmul(freecol,wp.tile_transpose(freecol))
                    for i in range(Dc):diag[i]=mu*free[i]+wp.float64(1.)-free[i]
                    factor=wp.tile_cholesky(wp.tile_diag_add(wp.tile_map(wp.mul,quu,freeouter),diag))
                    direction=wp.tile_cholesky_solve(factor,-wp.tile_map(wp.mul,grad,free))
                    slope=wp.tile_sum(wp.tile_map(wp.mul,grad,direction))[0]
                    alpha=wp.float64(1.);trial=k
                    while alpha>wp.float64(1e-12):
                        trial=wp.tile_map(clamp64,k+alpha*direction,lower,upper)
                        qt=wp.tile_reshape(wp.tile_matmul(quu,wp.tile_reshape(trial,shape=(Dc,1))),shape=(Dc,))
                        trialcost=wp.tile_sum(wp.tile_map(wp.mul,trial,qu+wp.float64(.5)*(qt+mu*trial)))[0]
                        if trialcost-value<=wp.float64(.1)*alpha*slope:break
                        alpha*=wp.float64(.6)
                    if alpha<=wp.float64(1e-12):break
                    k=trial
                freecol=wp.tile_reshape(free,shape=(Dc,1));freeouter=wp.tile_matmul(freecol,wp.tile_transpose(freecol))
                for i in range(Dc):diag[i]=mu*free[i]+wp.float64(1.)-free[i]
                factor=wp.tile_cholesky(wp.tile_diag_add(wp.tile_map(wp.mul,quu,freeouter),diag))
            rhs=wp.tile_map(wp.mul,ident,w*wp.tile_matmul(wp.tile_reshape(free,shape=(Dc,1)),wp.tile_transpose(maskcol)))
            gains=wp.tile_upper_solve(wp.tile_transpose(factor),wp.tile_lower_solve(factor,rhs))
            gains=wp.tile_map(rounded64,wp.tile_map(wp.mul,gains,wp.tile_matmul(wp.tile_reshape(free,shape=(Dc,1)),wp.tile_transpose(maskcol))))
            k=wp.tile_map(rounded64,k);next_k=k
            for i in range(Dc):
                if i<d:
                    ff[b,t,i]=wp.float32(k[i])
                    for j in range(Dc):
                        if j<d:fb[b,t,i,j]=wp.float32(gains[i,j])
            qk=wp.tile_matmul(quu,wp.tile_reshape(k,shape=(Dc,1)))+wp.tile_reshape(qu,shape=(Dc,1))
            vg=qx+wp.tile_reshape(wp.tile_matmul(wp.tile_transpose(gains),qk),shape=(Dc,))-w*k
            qgain=wp.tile_matmul(quu,gains)-w*ident
            vh=w*ident+wp.tile_matmul(wp.tile_transpose(gains),qgain)-w*gains
            vh=wp.float64(.5)*(vh+wp.tile_transpose(vh))
            vh=wp.tile_map(wp.mul,vh,outer)
    kernel=wp.kernel(backward,enable_backward=False)
    _BACKWARD[dimension]=kernel
    return kernel


vec32=wp.types.vector(length=32,dtype=wp.float32)


@wp.kernel(enable_backward=False)
def rollout(d:int,h:int,nalpha:int,ctrl:wp.array3d(dtype=wp.float32),ff:wp.array3d(dtype=wp.float32),fb:wp.array4d(dtype=wp.float32),
            alphas:wp.array(dtype=wp.float32),lo:wp.array(dtype=wp.float32),hi:wp.array(dtype=wp.float32),trial:wp.array3d(dtype=wp.float32)):
    world=wp.tid();b=world//nalpha;alpha=alphas[world%nalpha];dx=vec32(0.)
    for t in range(h):
        nextdx=vec32(0.)
        for j in range(d):
            feedback=wp.float32(0.)
            for m in range(d):feedback+=fb[b,t,j,m]*dx[m]
            u=wp.clamp(ctrl[b,t,j]+alpha*ff[b,t,j]+feedback,lo[j],hi[j]);trial[t,world,j]=u
            nextdx[j]=u-ctrl[b,t,j]
        dx=nextdx


def setup(solver):
    # Stable registration order lets isolated worker processes reuse the same
    # compiled module regardless of whether the human or robot is solved first.
    for dimension in (3,20,21):backward_kernel(dimension)
    d=solver.d;device=solver.q.device
    mask=torch.ones(d,device=device,dtype=torch.float64)
    lo=torch.zeros_like(mask);hi=lo.clone();lo[:d]=solver.lower;hi[:d]=solver.upper
    solver.native_buffers=[mask,lo,hi,torch.eye(d,device=device,dtype=torch.float64)]
    solver.native_wp=[wp.from_torch(x) for x in solver.native_buffers]


def fused_backward(solver):
    mask,lo,hi,eye=solver.native_wp
    wp.launch_tiled(backward_kernel(solver.d),dim=solver.b,inputs=[solver.d,solver.t,solver._cost_grad,solver._cost_hess,solver._ctrls,lo,hi,mask,eye,solver._regularization],outputs=[solver._feedforward,solver._feedback,solver._backward_ok],block_dim=128,device=solver.device)


def fused_rollout(solver):
    wp.launch(rollout,dim=solver.b*solver.k,inputs=[solver.d,solver.t,solver.k,solver._ctrls,solver._feedforward,solver._feedback,solver.alpha_wp,solver._lo,solver._hi],outputs=[solver.trials],block_dim=128,device=solver.device)

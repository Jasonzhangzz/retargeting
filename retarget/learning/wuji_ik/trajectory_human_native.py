"""Fused causal human finger FK, pose Jacobian and reference QP on CUDA.

One CUDA thread solves one finger with adaptive evaluation/retry budgets. No
fixed-budget masked FK replays, CPU linear algebra, or numerical derivatives.
"""
import torch
import warp as wp
from .trajectory_human_causal import box_qp_impl
f64=wp.float64
V5=wp.types.vector(length=5,dtype=f64);V6=wp.types.vector(length=6,dtype=f64)
J65=wp.types.matrix(shape=(6,5),dtype=f64)
M35=wp.types.matrix(shape=(3,5),dtype=f64)


@wp.func
def cross_matrix(v:wp.vec3d):
    return wp.mat33d(f64(0.),-v[2],v[1],v[2],f64(0.),-v[0],-v[1],v[0],f64(0.))


@wp.func
def rotation_log(r:wp.mat33d):
    theta=wp.acos(wp.clamp((r[0,0]+r[1,1]+r[2,2]-f64(1.))*f64(.5),f64(-1.),f64(1.)))
    value=wp.vec3d(f64(0.))
    if theta>f64(3.141591653589793):
        plus=r+wp.identity(n=3,dtype=f64);best=int(0);length=f64(0.)
        for j in range(3):
            column=wp.vec3d(plus[0,j],plus[1,j],plus[2,j]);size=wp.dot(column,column)
            if size>length:best=j;length=size
        if length>f64(1e-24):value=wp.vec3d(plus[0,best],plus[1,best],plus[2,best])*(theta/wp.sqrt(length))
    elif theta>=f64(1e-10):value=wp.vec3d(r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1])*(theta/(f64(2.)*wp.sin(theta)))
    return value


@wp.func
def finger_evaluate(f:int,q:V5,nq:int,root_r:wp.mat33d,root_p:wp.vec3d,xyz:wp.array2d(dtype=wp.vec3d),rot:wp.array2d(dtype=wp.mat33d),axis:wp.array2d(dtype=wp.vec3d),qi:wp.array2d(dtype=int),target_p:wp.vec3d,target_r:wp.mat33d):
    r=root_r;p=root_p;origins=M35(f64(0.));axes=M35(f64(0.))
    for s in range(xyz.shape[0]):
        p+=r*xyz[s,f];r=r*rot[s,f];j=qi[s,f]
        if j>=0:
            a=r*axis[s,f]
            for c in range(3):origins[c,j]=p[c];axes[c,j]=a[c]
            k=cross_matrix(axis[s,f]);r=r*(wp.identity(n=3,dtype=f64)+wp.sin(q[j])*k+(f64(1.)-wp.cos(q[j]))*(k*k))
    relative=wp.transpose(r)*target_r;v=rotation_log(relative);inverse=-v;theta=wp.length(inverse)
    if theta>f64(3.141591653589793):inverse=rotation_log(wp.transpose(relative));theta=wp.length(inverse)
    k=cross_matrix(inverse);block=wp.identity(n=3,dtype=f64)-f64(.5)*k+(k*k)/f64(12.)
    if theta>=f64(1e-8):
        sn=wp.sin(theta);den=f64(1.)-wp.cos(theta);alpha=f64(1.)/(theta*theta)-sn/(f64(2.)*theta*den);beta=theta*f64(.5)*sn/den
        block=alpha*wp.outer(inverse,inverse)+beta*wp.identity(n=3,dtype=f64)+f64(.5)*k
    residual=V6(f64(0.));jacobian=J65(f64(0.));dp=(p-target_p)*f64(300.)
    for c in range(3):residual[c]=dp[c];residual[c+3]=v[c]*f64(5.729577951308232)
    for j in range(nq):
        ax=wp.vec3d(axes[0,j],axes[1,j],axes[2,j]);origin=wp.vec3d(origins[0,j],origins[1,j],origins[2,j]);jp=wp.cross(ax,p-origin)*f64(300.);jw=-(block*(wp.transpose(r)*ax))*f64(5.729577951308232)
        for c in range(3):jacobian[c,j]=jp[c];jacobian[c+3,j]=jw[c]
    return residual,jacobian,wp.length(p-target_p),wp.length(v)


@wp.func
def solve_finger(b:int,f:int,nq:int,initial:V5,reference:V5,lower:V5,upper:V5,multiplier:f64,budget:int,
    root_r:wp.mat33d,root_p:wp.vec3d,xyz:wp.array2d(dtype=wp.vec3d),rot:wp.array2d(dtype=wp.mat33d),axis:wp.array2d(dtype=wp.vec3d),qi:wp.array2d(dtype=int),target_p:wp.vec3d,target_r:wp.mat33d,
    h:wp.array3d(dtype=f64),g:wp.array2d(dtype=f64),lo:wp.array2d(dtype=f64),hi:wp.array2d(dtype=f64),matrix:wp.array3d(dtype=f64),factor:wp.array3d(dtype=f64),free:wp.array2d(dtype=int),candidate:wp.array2d(dtype=f64),delta:wp.array2d(dtype=f64)):
    q=initial;count=int(0)
    residual,jac,pos,ang=finger_evaluate(f,q,nq,root_r,root_p,xyz,rot,axis,qi,target_p,target_r)
    while count<budget:
        cost=wp.dot(residual,residual)
        if not wp.isfinite(cost) or (pos<f64(1e-8) and ang<f64(1e-6)):break
        damping=f64(.000025)*cost+f64(1e-6)
        for j in range(5):
            posture=f64(1.)
            if f==0:posture=f64(.01)
            elif j>=3:posture=f64(0.)
            gradient=f64(0.)
            for row in range(6):gradient+=f64(.5)*jac[row,j]*residual[row]
            g[b,j]=gradient+posture*(q[j]-reference[j])+f64(20.)*(q[j]-initial[j])
            for i in range(5):
                value=f64(0.)
                for row in range(6):value+=jac[row,i]*jac[row,j]
                if i==j:value+=damping+posture+f64(20.)
                if i==j and j>=nq:value=f64(1e10)
                h[b,i,j]=value
            cap=multiplier*f64(3.141592653589793/120.)
            lo[b,j]=wp.max(f64(.5)*(lower[j]-q[j]),-cap);hi[b,j]=wp.min(f64(.5)*(upper[j]-q[j]),cap)
            if j>=nq:lo[b,j]=f64(0.);hi[b,j]=f64(0.);g[b,j]=f64(0.)
        box_qp_impl(b,h,g,lo,hi,matrix,factor,free,candidate,delta)
        accepted=bool(False);alpha=f64(1.)
        for trial in range(4):
            if count>=budget:break
            proposal=q
            for j in range(5):proposal[j]+=alpha*delta[b,j]
            next_r,next_j,next_p,next_a=finger_evaluate(f,proposal,nq,root_r,root_p,xyz,rot,axis,qi,target_p,target_r);next_cost=wp.dot(next_r,next_r);count+=1
            if wp.isfinite(next_cost) and next_cost<=cost+f64(1e-12)*wp.max(f64(1.),wp.abs(cost)):
                q=proposal;residual=next_r;jac=next_j;pos=next_p;ang=next_a;accepted=True;break
            alpha*=f64(.5)
        if not accepted:break
    return q,pos,count


@wp.kernel(enable_backward=False)
def causal_fingers(state:wp.array2d(dtype=f64),reference:wp.array2d(dtype=f64),lower:wp.array2d(dtype=f64),upper:wp.array2d(dtype=f64),
    root_r:wp.mat33d,root_p:wp.vec3d,xyz:wp.array2d(dtype=wp.vec3d),rot:wp.array2d(dtype=wp.mat33d),axis:wp.array2d(dtype=wp.vec3d),qi:wp.array2d(dtype=int),position:wp.array2d(dtype=wp.vec3d),rotation:wp.array2d(dtype=wp.mat33d),
    h:wp.array3d(dtype=f64),g:wp.array2d(dtype=f64),lo:wp.array2d(dtype=f64),hi:wp.array2d(dtype=f64),matrix:wp.array3d(dtype=f64),factor:wp.array3d(dtype=f64),free:wp.array2d(dtype=int),candidate:wp.array2d(dtype=f64),delta:wp.array2d(dtype=f64),counts:wp.array(dtype=int)):
    b=wp.tid();f=b%5;instance=b//5;nq=int(4)
    if f==0:nq=5
    initial=V5(f64(0.));ref=V5(f64(0.));lb=V5(f64(0.));ub=V5(f64(0.))
    for j in range(5):initial[j]=state[b,j];ref[j]=reference[f,j];lb[j]=lower[f,j];ub[j]=upper[f,j]
    q,error,count=solve_finger(b,f,nq,initial,ref,lb,ub,f64(1.),40,root_r,root_p,xyz,rot,axis,qi,position[instance,f],rotation[instance,f],h,g,lo,hi,matrix,factor,free,candidate,delta);counts[b]=count
    for retry in range(2):
        if error>f64(.005):
            multiplier=f64(2.)
            if retry==1:multiplier=f64(4.)
            trial,trial_error,number=solve_finger(b,f,nq,initial,ref,lb,ub,multiplier,36,root_r,root_p,xyz,rot,axis,qi,position[instance,f],rotation[instance,f],h,g,lo,hi,matrix,factor,free,candidate,delta);counts[b]+=number
            if trial_error<error:q=trial;error=trial_error
    for j in range(5):state[b,j]=q[j]


class NativeCausalHuman:
    def __init__(self,base,seed,lower,upper):
        wp.init();self.base=base;self.b=len(base.position);device=seed.device;self.ids=torch.tensor([[0,1,2,3,4],[5,6,7,8,0],[9,10,11,12,0],[13,14,15,16,0],[17,18,19,20,0]],device=device);valid=torch.ones(5,5,device=device,dtype=torch.float64);valid[1:,4]=0
        self.seed=(seed[self.ids]*valid).contiguous();self.state=self.seed[None].expand(self.b,-1,-1).clone().reshape(-1,5);self.lower=(lower[self.ids]*valid).contiguous();self.upper=(upper[self.ids]*valid).contiguous();fk=base.fk
        self.xyz=torch.stack(fk.chain_xyz).contiguous();self.rot=torch.stack(fk.chain_R).contiguous();self.axes=torch.stack(fk.chain_axis).contiguous();indices=torch.stack(fk.chain_index).int();active=torch.stack(fk.chain_active).bool();starts=torch.tensor([0,5,9,13,17],device=device);self.qi=torch.where(active,indices-starts[None],-1).int().contiguous()
        self.root_r=wp.mat33d(*fk.root_R.cpu().flatten().tolist());self.root_p=wp.vec3d(*fk.root_p.cpu().tolist());n=self.b*5
        self.h=torch.empty(n,5,5,device=device,dtype=torch.float64);self.g=torch.empty(n,5,device=device,dtype=torch.float64);self.lo=torch.empty_like(self.g);self.hi=torch.empty_like(self.g);self.matrix=torch.empty_like(self.h);self.factor=torch.empty_like(self.h);self.free=torch.empty(n,5,device=device,dtype=torch.int32);self.candidate=torch.empty_like(self.g);self.delta=torch.empty_like(self.g);self.counts=torch.empty(n,device=device,dtype=torch.int32)
    def reset(self):self.state.copy_(self.seed[None].expand(self.b,-1,-1).reshape(-1,5))
    @torch.no_grad()
    def step(self,position,rotation,*,stream=None):
        if stream is None:
            if not hasattr(self,"external_stream"):self.external_stream=wp.stream_from_torch(torch.cuda.current_stream(position.device))
            stream=self.external_stream
        wp.launch(causal_fingers,self.b*5,inputs=[wp.from_torch(v) for v in [self.state,self.seed,self.lower,self.upper]]+[self.root_r,self.root_p,wp.from_torch(self.xyz,dtype=wp.vec3d),wp.from_torch(self.rot,dtype=wp.mat33d),wp.from_torch(self.axes,dtype=wp.vec3d),wp.from_torch(self.qi),wp.from_torch(position.contiguous(),dtype=wp.vec3d),wp.from_torch(rotation.contiguous(),dtype=wp.mat33d)]+[wp.from_torch(v) for v in [self.h,self.g,self.lo,self.hi,self.matrix,self.factor,self.free,self.candidate,self.delta,self.counts]],stream=stream,device=wp.get_device(str(position.device)))
        q=self.state.reshape(self.b,5,5);return torch.cat((q[:,0],q[:,1,:4],q[:,2,:4],q[:,3,:4],q[:,4,:4]),-1)

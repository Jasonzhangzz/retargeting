"""Stateful GPU human IK preserving the reference's QP, retries and pose-only acceptance."""
import torch
import warp as wp
from robokit.opt.ilqr_optimizer import _compute_cholesky_func,_compute_cho_solve_func


@wp.func
def box_qp_impl(b:int,h:wp.array3d(dtype=wp.float64),g:wp.array2d(dtype=wp.float64),lo:wp.array2d(dtype=wp.float64),hi:wp.array2d(dtype=wp.float64),
    matrix:wp.array3d(dtype=wp.float64),factor:wp.array3d(dtype=wp.float64),free:wp.array2d(dtype=wp.int32),candidate:wp.array2d(dtype=wp.float64),x:wp.array2d(dtype=wp.float64)):
    for i in range(5):x[b,i]=wp.clamp(wp.float64(0.),lo[b,i],hi[b,i]);free[b,i]=1
    for iteration in range(28):
        for i in range(5):
            for j in range(5):matrix[b,i,j]=h[b,i,j]
        ok=_compute_cholesky_func(matrix[b],wp.float64(0.),free[b],factor[b])
        if ok!=0:
            for i in range(5):
                value=-g[b,i]
                for j in range(5):
                    if free[b,j]==0:value-=h[b,i,j]*x[b,j]
                candidate[b,i]=value*wp.float64(free[b,i])
            _compute_cho_solve_func(factor[b],candidate[b])
            fraction=wp.float64(1.);block=int(-1)
            for i in range(5):
                if free[b,i]!=0:
                    direction=candidate[b,i]-x[b,i];trial=wp.float64(1.)
                    if direction>wp.float64(1e-12):trial=(hi[b,i]-x[b,i])/direction
                    elif direction<wp.float64(-1e-12):trial=(lo[b,i]-x[b,i])/direction
                    if trial<fraction:fraction=trial;block=i
            fraction=wp.max(wp.float64(0.),fraction)
            for i in range(5):
                if free[b,i]!=0:x[b,i]+=fraction*(candidate[b,i]-x[b,i])
            if block>=0:free[b,block]=0;continue
        worst=wp.float64(1e-9);release=int(-1)
        for i in range(5):
            value=g[b,i]
            for j in range(5):value+=h[b,i,j]*x[b,j]
            violation=wp.float64(0.)
            if free[b,i]==0:
                if wp.abs(x[b,i]-lo[b,i])<wp.float64(1e-9):violation=-value
                else:violation=value
            if violation>worst:worst=violation;release=i
        if release<0:break
        free[b,release]=1


@wp.kernel
def _box_qp(h:wp.array3d(dtype=wp.float64),g:wp.array2d(dtype=wp.float64),lo:wp.array2d(dtype=wp.float64),hi:wp.array2d(dtype=wp.float64),matrix:wp.array3d(dtype=wp.float64),factor:wp.array3d(dtype=wp.float64),free:wp.array2d(dtype=wp.int32),candidate:wp.array2d(dtype=wp.float64),x:wp.array2d(dtype=wp.float64)):
    b=wp.tid();box_qp_impl(b,h,g,lo,hi,matrix,factor,free,candidate,x)


class CausalHumanCUDA:
    def __init__(self,base,seed,lower,upper):
        wp.init()
        self.base=base;self.call=torch.compile(base,fullgraph=True,dynamic=False);self.seed=seed;self.lower,self.upper=lower,upper;self.state=seed[None].expand(len(base.position),-1).clone();self.graph=None
        self.ids=torch.tensor([[0,1,2,3,4],[5,6,7,8,0],[9,10,11,12,0],[13,14,15,16,0],[17,18,19,20,0]],device=seed.device)
        self.valid=torch.ones(5,5,device=seed.device,dtype=torch.float64);self.valid[1:,4]=0
        self.posture=self.valid.clone();self.posture[0]=.01;self.posture[1:,3]=0
        n=len(self.state)*5;self.matrix=torch.empty(n,5,5,device=seed.device,dtype=torch.float64);self.factor=torch.empty_like(self.matrix);self.free=torch.empty(n,5,device=seed.device,dtype=torch.int32);self.candidate=torch.empty(n,5,device=seed.device,dtype=torch.float64);self.delta=torch.empty_like(self.candidate)
    def pack(self,q):return q[:,self.ids]*self.valid
    def unpack(self,q):return torch.cat((q[:,0],q[:,1,:4],q[:,2,:4],q[:,3,:4],q[:,4,:4]),-1)
    def evaluate(self,q):
        r,j=self.call(q);pose=r[:,:30].reshape(-1,5,6);jac=j[:,:30].reshape(-1,5,6,21).gather(-1,self.ids[None,:,None].expand(len(q),-1,6,-1))*self.valid[None,:,None]
        return pose,jac,(pose[...,:3].norm(dim=-1)/300),(pose[...,3:].norm(dim=-1)/5.729577951308232)
    def core(self,initial,multiplier,budget):
        q=initial.clone();count=torch.zeros(len(q),5,device=q.device,dtype=torch.int32);stopped=torch.zeros_like(count,dtype=torch.bool)
        for _ in range(budget):
            pose,j,pos,ang=self.evaluate(q);cost=pose.square().sum(-1);active=(count<budget)&~stopped&~((pos<1e-8)&(ang<1e-6));padded=self.pack(q)
            damping=.0001*(.5*pose).square().sum(-1)+1e-6
            h=j.transpose(-1,-2)@j+torch.diag_embed(self.posture+20*self.valid+damping[...,None])
            h=h+torch.diag_embed((1-self.valid)*1e10)
            g=.5*(j*pose[...,None]).sum(-2)+self.posture*(padded-self.pack(self.seed[None]))+20*self.valid*(padded-self.pack(initial))
            cap=multiplier*3.141592653589793/120
            lo=torch.maximum(.5*(self.pack(self.lower[None])-padded),padded.new_tensor(-cap))*self.valid
            hi=torch.minimum(.5*(self.pack(self.upper[None])-padded),padded.new_tensor(cap))*self.valid
            wp.launch(_box_qp,len(q)*5,inputs=[wp.from_torch(h.reshape(-1,5,5).contiguous()),wp.from_torch(g.reshape(-1,5).contiguous()),wp.from_torch(lo.reshape(-1,5).contiguous()),wp.from_torch(hi.reshape(-1,5).contiguous()),wp.from_torch(self.matrix),wp.from_torch(self.factor),wp.from_torch(self.free),wp.from_torch(self.candidate),wp.from_torch(self.delta)],device=wp.get_device(str(q.device)))
            delta=self.delta.reshape(len(q),5,5);accepted=torch.zeros_like(active)
            for alpha in (1.,.5,.25,.125):
                testing=active&~accepted&(count<budget);trial=self.unpack(padded+alpha*delta);trial_pose,_,_,_=self.evaluate(trial);new_cost=trial_pose.square().sum(-1)
                ok=testing&torch.isfinite(new_cost)&(new_cost<=cost+1e-12*torch.maximum(torch.ones_like(cost),cost.abs()))
                q=self.unpack(torch.where(ok[...,None],self.pack(trial),self.pack(q)));accepted|=ok;count+=testing.int()
            stopped|=active&~accepted
        return q,self.evaluate(q)[2]
    def body(self):
        initial=self.state.clone();q,error=self.core(initial,1.,40)
        for multiplier in (2.,4.):
            retry=error>.005;candidate,candidate_error=self.core(initial,multiplier,36);select=retry&(candidate_error<error)
            q=self.unpack(torch.where(select[...,None],self.pack(candidate),self.pack(q)));error=torch.where(select,candidate_error,error)
        self.state.copy_(q)
    @torch.no_grad()
    def step(self,position,rotation):
        self.base.position.copy_(position);self.base.rotation.copy_(rotation)
        caller=torch.cuda.current_stream(position.device)
        if not hasattr(self,"torch_stream"):self.torch_stream=torch.cuda.Stream(device=position.device)
        self.torch_stream.wait_stream(caller);stream=wp.stream_from_torch(self.torch_stream)
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(stream):
            if self.graph is None:
                saved=self.state.clone();self.body();torch.cuda.synchronize();self.state.copy_(saved);self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=self.torch_stream):
                    with wp.ScopedCapture(stream=stream,external=True):self.body()
                self.state.copy_(saved)
            self.graph.replay()
        caller.wait_stream(self.torch_stream)
        return self.state.clone()
    def reset(self):self.state.copy_(self.seed[None])

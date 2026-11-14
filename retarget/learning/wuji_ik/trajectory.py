"""Compact trajectory normal equations; no network-anchor loss or dense TD matrix.

Frames are flattened, with explicit segment boundaries. Spatial blocks may be
fully dense; time couplings are represented by one lower off-diagonal block.
"""
from dataclasses import dataclass, field
import math
import torch


@dataclass
class TrajectoryResult:
    q: torch.Tensor
    initial_cost: torch.Tensor
    cost: torch.Tensor
    history: list = field(default_factory=list)
    backend: str = ''
    function_evaluations: int = 0
    gradient_evaluations: int = 0


class TrajectoryObjective:
    def __init__(self, spatial, offsets, reference, weight):
        self.spatial = spatial
        self.offsets = tuple(int(v) for v in offsets)
        if len(self.offsets)<2 or self.offsets[0]!=0 or any(b<=a for a,b in zip(self.offsets,self.offsets[1:])):
            raise ValueError('Offsets must delimit nonempty continuous segments')
        if not math.isfinite(weight) or weight<0:
            raise ValueError('History weight must be finite and nonnegative')
        self.weight = float(weight)
        self.reference = reference
        n = self.offsets[-1]
        self.starts = torch.tensor(self.offsets[:-1],device=reference.device)
        self.segment = torch.repeat_interleave(torch.arange(len(self.starts),device=reference.device),
            torch.tensor([b-a for a,b in zip(self.offsets,self.offsets[1:])],device=reference.device))
        self.previous = (torch.arange(n,device=reference.device)-1).clamp_min(0)
        self.link = torch.ones(n,device=reference.device,dtype=torch.bool)
        self.link[self.starts] = False
        self.following = torch.cat((self.link[1:], self.link.new_zeros(1)))

    def residuals(self,q):
        r,j = self.spatial(q)
        previous=q[self.previous].clone()
        previous[self.starts]=self.reference
        return r,j,q-previous

    def reduce(self,values):
        return torch.zeros(len(self.starts),device=values.device,dtype=values.dtype).index_add(0,self.segment,values)

    def evaluate(self,q,normal=True):
        r,j,delta=self.residuals(q)
        cost=self.reduce(.5*r.square().sum(-1)+.5*self.weight*delta.square().sum(-1))
        if not normal:return cost
        g=(j*r[...,None]).sum(1)+self.weight*delta
        correction=torch.cat((delta[1:]*self.link[1:,None],torch.zeros_like(delta[:1])),0)
        g=g-self.weight*correction
        eye=torch.eye(q.shape[-1],device=q.device,dtype=q.dtype)
        a=j.transpose(-1,-2)@j+self.weight*(1+self.following.to(q.dtype))[:,None,None]*eye
        lower=-self.weight*self.link.to(q.dtype)[:,None,None]*eye.expand(len(q),-1,-1)
        return cost,g,a,lower

    def local_cost(self,q):
        r,j,delta=self.residuals(q)
        return .5*r.square().sum(-1)+.5*self.weight*delta.square().sum(-1)


def block_matvec(a,lower,x,segment):
    result=(a@x[...,None]).squeeze(-1)
    result[1:]+=(lower[1:]@x[:-1,...,None]).squeeze(-1)
    result[:-1]+=(lower[1:].transpose(-1,-2)@x[1:,...,None]).squeeze(-1)
    return result


def block_pcr(a,lower,rhs,segment):
    """Parallel cyclic reduction reference. O(N d²) storage, O(log T) depth.

    CPU is a correctness reference. Production CUDA uses native_pcr below.
    """
    n=len(a); index=torch.arange(n,device=a.device)
    diagonal=a.clone();left=lower.clone()
    right=torch.cat((lower[1:].transpose(-1,-2),torch.zeros_like(lower[:1])),0)
    b=rhs.clone();stride=1
    while stride<n:
        li=(index-stride).clamp_min(0);ri=(index+stride).clamp_max(n-1)
        lm=((index>=stride)&(segment==segment[li]))[:,None,None]
        rm=((index+stride<n)&(segment==segment[ri]))[:,None,None]
        alpha=-torch.linalg.solve(diagonal[li].transpose(-1,-2),left.transpose(-1,-2)).transpose(-1,-2)*lm
        beta=-torch.linalg.solve(diagonal[ri].transpose(-1,-2),right.transpose(-1,-2)).transpose(-1,-2)*rm
        diagonal=diagonal+alpha@right[li]+beta@left[ri]
        b=b+(alpha@b[li,...,None]+beta@b[ri,...,None]).squeeze(-1)
        left=alpha@left[li];right=beta@right[ri]
        stride*=2
    return torch.linalg.solve(diagonal,b[...,None]).squeeze(-1)


def active_system(a,lower,g,q,lo,hi,damping,segment):
    active=((q<=lo+1e-7)&(g>0))|((q>=hi-1e-7)&(g<0))
    free=(~active).to(q.dtype)
    eye=torch.eye(q.shape[-1],device=q.device,dtype=q.dtype)
    matrix=a*free[:,:,None]*free[:,None,:]+torch.diag_embed(damping[:,None]*free+(1-free))
    previous=torch.cat((torch.zeros_like(free[:1]),free[:-1]),0)
    off=lower*free[:,:,None]*previous[:,None,:]
    return matrix,off,-g*free


class BandedLM:
    def __init__(self,objective,iterations=16,*,native=True,max_step=.35):
        self.objective,self.iterations,self.native,self.max_step=objective,iterations,native,max_step

    @torch.no_grad()
    def __call__(self,initial,lower,upper):
        obj=self.objective;q=initial.clamp(lower,upper).clone();initial_cost=obj.evaluate(q,False)
        damping=q.new_full((len(obj.starts),),.01);history=[]
        solve=block_pcr
        if self.native:
            if q.device.type!='cuda':raise ValueError('Native banded LM requires CUDA')
            from .trajectory_cuda import native_pcr
            solve=native_pcr
        for k in range(self.iterations):
            cost,g,a,off=obj.evaluate(q)
            matrix,off,rhs=active_system(a,off,g,q,lower,upper,damping[obj.segment],obj.segment)
            step=solve(matrix,off,rhs,obj.segment)
            candidate=(q+step.clamp(-self.max_step,self.max_step)).clamp(lower,upper)
            candidate_cost=obj.evaluate(candidate,False);actual=candidate-q
            predicted=-obj.reduce((g*actual).sum(-1)+.5*(actual*block_matvec(a,off,actual,obj.segment)).sum(-1))
            reduction=cost-candidate_cost;rho=reduction/predicted.clamp_min(1e-30)
            accept=torch.isfinite(candidate_cost)&(reduction>0)&(predicted>0)&(rho>1e-3)
            q=torch.where(accept[obj.segment,None],candidate,q)
            damping=torch.where(accept,torch.where(rho>.75,damping*.5,damping),damping*2).clamp(1e-7,1e10)
        cost=obj.evaluate(q,False)
        return TrajectoryResult(q,initial_cost,cost,history,'native_cuda_block_pcr_lm' if self.native else 'reference_block_pcr_lm',2*self.iterations+2,self.iterations)


def segmented_iir(q,offsets,alpha=.2,round_float32=True):
    """Associative affine prefix scan, O(N d) storage and O(log T) depth.

    The scan has the same recurrence as SDK IIR. Parallel arithmetic is not
    bitwise equal to a sequential FP32 recurrence; causal mode uses the latter.
    """
    work=q.float() if round_float32 else q
    n=len(q);starts=torch.tensor(offsets[:-1],device=q.device);segment=torch.repeat_interleave(
        torch.arange(len(starts),device=q.device),torch.tensor([b-a for a,b in zip(offsets,offsets[1:])],device=q.device))
    a=work.new_full((n,1),1-alpha);b=work*alpha;a[starts]=0;b[starts]=work[starts]
    index=torch.arange(n,device=q.device);stride=1
    while stride<n:
        previous=(index-stride).clamp_min(0);valid=((index>=stride)&(segment==segment[previous]))[:,None]
        b=torch.where(valid,b+a*b[previous],b);a=torch.where(valid,a*a[previous],a);stride*=2
    return b.to(q.dtype)


class CUDABandedLM:
    """One captured iteration; replay budget does not duplicate the graph."""
    def __init__(self,objective,initial,lower,upper):
        if initial.device.type!='cuda':raise ValueError('CUDA required')
        self.objective=objective;self.q=initial.clone();self.lower,self.upper=lower,upper
        self.damping=initial.new_full((len(objective.starts),),.01);self.graph=None
    def iteration(self):
        from .trajectory_cuda import native_pcr
        obj=self.objective;q=self.q
        cost,g,a,off=obj.evaluate(q);matrix,masked_off,rhs=active_system(a,off,g,q,self.lower,self.upper,self.damping[obj.segment],obj.segment)
        step=native_pcr(matrix,masked_off,rhs,obj.segment,max_length=max(e-s for s,e in zip(obj.offsets,obj.offsets[1:])))
        candidate=(q+step.clamp(-.35,.35)).clamp(self.lower,self.upper);after=obj.evaluate(candidate,False);actual=candidate-q
        predicted=-obj.reduce((g*actual).sum(-1)+.5*(actual*block_matvec(a,off,actual,obj.segment)).sum(-1));reduction=cost-after;rho=reduction/predicted.clamp_min(1e-30)
        accepted=torch.isfinite(after)&(reduction>0)&(predicted>0)&(rho>1e-3)
        self.q.copy_(torch.where(accepted[obj.segment,None],candidate,q));self.damping.copy_(torch.where(accepted,torch.where(rho>.75,self.damping*.5,self.damping),self.damping*2).clamp(1e-7,1e10))
    @torch.no_grad()
    def __call__(self,initial,iterations):
        import warp as wp
        wp.init()
        self.q.copy_(initial.clamp(self.lower,self.upper));self.damping.fill_(.01)
        caller=torch.cuda.current_stream(initial.device)
        if not hasattr(self,"torch_stream"):self.torch_stream=torch.cuda.Stream(device=initial.device)
        self.torch_stream.wait_stream(caller);stream=wp.stream_from_torch(self.torch_stream)
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(stream):
            if self.graph is None:
                self.iteration();torch.cuda.synchronize();self.q.copy_(initial.clamp(self.lower,self.upper));self.damping.fill_(.01)
                self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=self.torch_stream):
                    with wp.ScopedCapture(stream=stream,external=True):self.iteration()
                self.q.copy_(initial.clamp(self.lower,self.upper));self.damping.fill_(.01)
            for _ in range(iterations):self.graph.replay()
        caller.wait_stream(self.torch_stream)
        return TrajectoryResult(self.q.clone(),self.objective.evaluate(initial,False),self.objective.evaluate(self.q,False),backend='native_cuda_block_pcr_lm_graph',function_evaluations=2*iterations+2,gradient_evaluations=iterations)


def robot_diagnostics(spatial,q,offsets,reference,*,causal=False,lower=None,upper=None):
    """Scoring-only term decomposition and projected-gradient stationarity."""
    obj=TrajectoryObjective(spatial,offsets,reference,.01);r,j,delta=obj.residuals(q)
    terms={name:.5*r[:,s:e].square().sum(-1) for name,s,e in [('laplacian',0,63),('direction',63,111),('thumb_relative',111,113),('antireverse',113,123)]};terms['history']=.005*delta.square().sum(-1)
    g=(j*r[...,None]).sum(1)+.01*delta
    if not causal:g-=.01*torch.cat((delta[1:]*obj.link[1:,None],torch.zeros_like(delta[:1])),0)
    if lower is not None:g=torch.where(((q<=lower+1e-7)&(g>0))|((q>=upper-1e-7)&(g<0)),0,g)
    norm=g.abs().amax(-1)
    return terms,dict(projected_gradient_inf=norm,stationary=norm<=1e-6,criterion='per-frame projected gradient <=1e-6',information='fixed_previous_q' if causal else 'joint_trajectory_gradient')

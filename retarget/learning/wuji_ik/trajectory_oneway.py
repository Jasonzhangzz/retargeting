"""Causal stationarity updates, deliberately distinct from joint least squares.

`frozen_parallel` freezes the previous *outer iteration*: block Jacobi.
`causal_prefix` solves a lower-bidiagonal linearization with a GPU affine scan.
Neither is a symmetric Hessian with its upper triangle removed. Acceptance is
per frame, so neither future costs nor a future rejection can affect the past.
"""
import torch
from .trajectory import TrajectoryResult


def lower_bidiagonal_solve(diagonal, lower, rhs, segment):
    """FP64-capable reference affine scan, O(N d²) storage, log(N) depth."""
    transition = -torch.linalg.solve(diagonal, lower)
    value = torch.linalg.solve(diagonal, rhs.unsqueeze(-1)).squeeze(-1)
    index = torch.arange(len(rhs), device=rhs.device)
    stride = 1
    while stride < len(rhs):
        previous = (index-stride).clamp_min(0)
        valid = (index >= stride) & (segment == segment[previous])
        value = value + torch.where(valid[:, None],
            (transition @ value[previous, :, None]).squeeze(-1), 0.)
        transition = torch.where(valid[:, None, None], transition @ transition[previous], 0.)
        stride *= 2
    return value


class OneWayLM:
    def __init__(self, objective, initial, lower, upper, *, mode='causal_prefix', native=True):
        if mode not in ('causal_prefix', 'frozen_parallel'):
            raise ValueError(mode)
        self.objective = objective
        self.q = initial.clone()
        self.lower, self.upper = lower, upper
        self.mode, self.native = mode, native
        self.damping = initial.new_full((len(initial),), .01)
        self.graph = None

    def iteration(self):
        obj, q = self.objective, self.q
        r, j, delta = obj.residuals(q)
        g = (j*r[..., None]).sum(1) + obj.weight*delta
        eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
        a = j.transpose(-1,-2)@j + obj.weight*eye
        active = ((q <= self.lower+1e-7)&(g > 0)) | ((q >= self.upper-1e-7)&(g < 0))
        free = (~active).to(q.dtype)
        matrix = a*free[:,:,None]*free[:,None,:] + torch.diag_embed(self.damping[:,None]*free + (1-free))
        previous_free = torch.cat((torch.zeros_like(free[:1]), free[:-1]))
        off = -obj.weight*obj.link.to(q.dtype)[:,None,None]*eye*free[:,:,None]*previous_free[:,None,:]
        if self.mode == 'frozen_parallel':
            off = torch.zeros_like(off)
        if self.native:
            from .trajectory_oneway_cuda import native_lower_scan
            step = native_lower_scan(matrix, off, -g*free, obj.segment,
                max(e-s for s,e in zip(obj.offsets,obj.offsets[1:])))
        else:
            step = lower_bidiagonal_solve(matrix, off, -g*free, obj.segment)
        candidate = (q + step.clamp(-.35,.35)).clamp(self.lower,self.upper)
        # Prefix steps account for proposed predecessor motion. This shifted
        # local merit is NOT the joint cost, and decisions never reduce by segment.
        previous = q[obj.previous].clone()
        if self.mode == 'causal_prefix':
            previous = candidate[obj.previous].clone()
        previous[obj.starts] = obj.reference
        shifted_g = (j*r[...,None]).sum(1) + obj.weight*(q-previous)
        before = .5*r.square().sum(-1) + .5*obj.weight*(q-previous).square().sum(-1)
        cr, _ = obj.spatial(candidate)
        after = .5*cr.square().sum(-1) + .5*obj.weight*(candidate-previous).square().sum(-1)
        actual = candidate-q
        predicted = -(shifted_g*actual).sum(-1) - .5*(actual*(a@actual[...,None]).squeeze(-1)).sum(-1)
        rho = (before-after)/predicted.clamp_min(1e-30)
        accepted = torch.isfinite(after)&(after < before)&(predicted > 0)&(rho > 1e-3)
        self.q.copy_(torch.where(accepted[:,None],candidate,q))
        self.damping.copy_(torch.where(accepted,torch.where(rho > .75,self.damping*.5,self.damping),self.damping*2).clamp(1e-7,1e10))

    @torch.no_grad()
    def __call__(self, initial, iterations):
        if iterations < 0:
            raise ValueError('Negative update budget')
        self.q.copy_(initial.clamp(self.lower,self.upper)); self.damping.fill_(.01)
        if initial.device.type == 'cuda' and self.native:
            import warp as wp
            wp.init()
            caller = torch.cuda.current_stream(initial.device)
            if not hasattr(self,'stream'): self.stream = torch.cuda.Stream(device=initial.device)
            self.stream.wait_stream(caller)
            warp_stream = wp.stream_from_torch(self.stream)
            with torch.cuda.stream(self.stream), wp.ScopedStream(warp_stream):
                if self.graph is None:
                    self.iteration(); self.stream.synchronize()
                    self.q.copy_(initial.clamp(self.lower,self.upper)); self.damping.fill_(.01)
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph,stream=self.stream):
                        with wp.ScopedCapture(stream=warp_stream,external=True): self.iteration()
                    self.q.copy_(initial.clamp(self.lower,self.upper)); self.damping.fill_(.01)
                for _ in range(iterations): self.graph.replay()
            caller.wait_stream(self.stream)
        else:
            for _ in range(iterations): self.iteration()
        return TrajectoryResult(self.q.clone(), self.objective.evaluate(initial,False),
            self.objective.evaluate(self.q,False), backend=self.mode,
            function_evaluations=2*iterations+2,gradient_evaluations=iterations)


class CausalHumanRollout:
    """Source-rule human history, independent of the robot's new backend."""
    def __init__(self,pipeline,positions,rotations,offsets):
        from .batched_lm import HumanPoseObjective
        from .trajectory_human_native import NativeCausalHuman
        self.p=pipeline;self.positions=positions.clone();self.rotations=rotations.clone()
        self.starts=torch.tensor(offsets[:-1],device=positions.device)
        self.lengths=torch.tensor([e-s for s,e in zip(offsets,offsets[1:])],device=positions.device)
        b=len(self.starts)
        base=HumanPoseObjective(pipeline.hfk,positions[:b].clone(),rotations[:b].clone(),pipeline.seed,pipeline.posture)
        self.human=NativeCausalHuman(base,pipeline.seed,pipeline.hlo,pipeline.hhi)
        self.index=torch.zeros((),dtype=torch.long,device=positions.device)
        self.out=positions.new_zeros(len(positions),21);self.max_length=max(e-s for s,e in zip(offsets,offsets[1:]));self.graph=None

    def body(self,stream):
        ids=self.starts+torch.minimum(self.index,self.lengths-1)
        q=self.human.step(self.positions[ids],self.rotations[ids],stream=stream)
        self.out[ids]=torch.where((self.index<self.lengths)[:,None],q,self.out[ids]);self.index.add_(1)

    @torch.no_grad()
    def __call__(self,positions,rotations):
        import warp as wp
        self.positions.copy_(positions);self.rotations.copy_(rotations)
        caller=torch.cuda.current_stream(positions.device)
        if not hasattr(self,'stream'):self.stream=torch.cuda.Stream(device=positions.device)
        self.stream.wait_stream(caller);ws=wp.stream_from_torch(self.stream)
        with torch.cuda.stream(self.stream),wp.ScopedStream(ws):
            if self.graph is None:
                self.human.reset();self.index.zero_();self.body(ws);self.stream.synchronize()
                self.human.reset();self.index.zero_();self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=self.stream):
                    with wp.ScopedCapture(stream=ws,external=True):self.body(ws)
            self.human.reset();self.index.zero_()
            for _ in range(self.max_length):self.graph.replay()
        caller.wait_stream(self.stream)
        return self.out.clone()


class FrozenHistoryRobot:
    """Online LM: completed FP32 history frozen, source previous-q initializer.

    Native Robokit LM replaces finite SLSQP, so this is a causal replacement,
    not an assertion of identical numerical stopping or identical solutions.
    """
    def __init__(self,objectives,initial,lower,upper,*,seed_mode='previous'):
        import copy
        from .batched_lm import JointAnchorObjective
        from .robokit_lm import RobokitCUDALM
        if seed_mode not in ('previous','network'):raise ValueError(seed_mode)
        self.objectives=objectives;obj=objectives[0];self.seed_mode=seed_mode
        self.starts=obj.starts;self.lengths=torch.tensor([e-s for s,e in zip(obj.offsets,obj.offsets[1:])],device=initial.device)
        self.index=torch.zeros((),dtype=torch.long,device=initial.device);self.reference=obj.reference.clone();self.initial=initial.clone()
        self.out=torch.zeros_like(initial);self.filtered=torch.zeros(len(obj.starts),initial.shape[1],device=initial.device,dtype=torch.float32);self.filtered_out=torch.zeros_like(initial)
        self.lower,self.upper=lower,upper;self.raws=[];self.engines={};self.graphs={}
        self.fields=('s','l','d','directions','normal','cosine','sine','thumb_weight')
        for stage,objective in enumerate(objectives):
            full=objective.spatial.raw;raw=copy.copy(full)
            for name in self.fields:setattr(raw,name,getattr(full,name)[self.starts].clone())
            self.raws.append(raw)
            for first in (True,False):
                count=([50,80] if first else [1,10])[stage]
                self.engines[stage,first]=RobokitCUDALM(JointAnchorObjective(raw,self.reference,.01),iterations=count,compile=True,graph_mode='none')

    def body(self,first):
        ids=self.starts+torch.minimum(self.index,self.lengths-1);active=self.index<self.lengths
        for raw,obj in zip(self.raws,self.objectives):
            for name in self.fields:getattr(raw,name).copy_(getattr(obj.spatial.raw,name)[ids])
        q=self.reference.clone() if self.seed_mode=='previous' else self.initial[ids].clone()
        for stage in (0,1):q=self.engines[stage,first](q,self.lower,self.upper).q
        raw=q.clamp(self.lower,self.upper).float();self.reference.copy_(raw.double())
        if first:self.filtered.copy_(raw)
        else:self.filtered.copy_(self.filtered+(raw-self.filtered)*.2)
        self.out[ids]=torch.where(active[:,None],raw.double(),self.out[ids]);self.filtered_out[ids]=torch.where(active[:,None],self.filtered.double(),self.filtered_out[ids]);self.index.add_(1)

    def reset(self):
        self.reference.copy_(self.objectives[0].reference);self.index.zero_();self.filtered.zero_()

    @torch.no_grad()
    def __call__(self,initial):
        import warp as wp
        self.initial.copy_(initial);caller=torch.cuda.current_stream(initial.device)
        if not hasattr(self,'stream'):self.stream=torch.cuda.Stream(device=initial.device)
        self.stream.wait_stream(caller);ws=wp.stream_from_torch(self.stream)
        with torch.cuda.stream(self.stream),wp.ScopedStream(ws):
            if not self.graphs:
                self.reset();self.body(True);self.stream.synchronize();self.reset()
                self.graphs[True]=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graphs[True],stream=self.stream):
                    with wp.ScopedCapture(stream=ws,external=True):self.body(True)
                self.reset();self.graphs[True].replay();self.body(False);self.stream.synchronize()
                self.graphs[False]=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graphs[False],stream=self.stream):
                    with wp.ScopedCapture(stream=ws,external=True):self.body(False)
            self.reset();self.graphs[True].replay()
            for _ in range(1,int(self.lengths.max().item())):self.graphs[False].replay()
        caller.wait_stream(self.stream)
        return self.out.clone(),self.filtered_out.clone()

"""Full causal frame CUDA graph: EMF -> human -> targets -> robot -> IIR.

Only graph replays remain in the host time loop. Ended segment outputs are
frozen; their private padded solver states cannot affect another segment.
"""
import math
import torch
import warp as wp
from .batched_lm import HumanPoseObjective,RobotRetargetObjective,JointAnchorObjective
from .trajectory_human_native import NativeCausalHuman
from .trajectory import TrajectoryObjective


class CausalFrameGraph:
    def __init__(self,pipeline,positions,rotations,offsets,backend):
        wp.init();self.p=pipeline;self.backend=backend;self.offsets=offsets;self.b=len(offsets)-1;self.n=len(positions);self.max_length=max(e-s for s,e in zip(offsets,offsets[1:]));self.device=positions.device
        self.positions=positions.clone();self.rotations=rotations.clone();self.starts=torch.tensor(offsets[:-1],device=self.device);self.lengths=torch.tensor([e-s for s,e in zip(offsets,offsets[1:])],device=self.device);self.index=torch.zeros((),device=self.device,dtype=torch.long)
        base=HumanPoseObjective(pipeline.hfk,self.positions[:self.b].clone(),self.rotations[:self.b].clone(),pipeline.seed,pipeline.posture);self.human=NativeCausalHuman(base,pipeline.seed,pipeline.hlo,pipeline.hhi)
        self.reference=pipeline.tensor(pipeline.robot.bounds.mean(1)).expand(self.b,-1).clone();self.filtered=torch.zeros(self.b,20,device=self.device,dtype=torch.float32)
        self.raw_out=torch.zeros(self.n,20,device=self.device,dtype=torch.float64);self.filtered_out=torch.zeros_like(self.raw_out);self.human_out=torch.zeros(self.n,21,device=self.device,dtype=torch.float64);self.evaluations=torch.zeros(self.b*5,device=self.device,dtype=torch.int64)
        self.ids=torch.tensor([8,12,16,20],device=self.device);self.edges=torch.tensor([(a,b) for f in range(1,5) for a,b in zip([0]+list(range(1+4*f,4+4*f)),range(1+4*f,5+4*f))],device=self.device);self.neighbors=positions.new_ones(21);self.neighbors[1:4]=0;self.eye=torch.eye(21,device=self.device,dtype=positions.dtype)
        q=pipeline.seed.expand(self.b,-1);target=self.targets(pipeline.sfk.forward(q)[0]);self.raws=[RobotRetargetObjective(pipeline.rfk,*target,stage=stage) for stage in [1,2]]
        self.engines={};self.first=None;self.later=None;self.torch_stream=torch.cuda.Stream(device=self.device);self.wp_stream=wp.stream_from_torch(self.torch_stream)

    def targets(self,skeleton):
        skeleton=skeleton.float().double();s=(skeleton-skeleton[:,:1])@self.p.alignment;thumb=s[:,4];fingers=s[:,self.ids];distance=(fingers-thumb[:,None]).norm(dim=-1);denominator=1/(1+math.exp(-.009/.0015))-.5
        weights=((torch.sigmoid((.009-distance)/.0015)-.5)/denominator).clamp(0,1)*(distance<.009);center=(weights[...,None]*fingers).sum(1)/weights.sum(1)[:,None].clamp_min(1e-30);midpoint=(thumb+center)*.5
        source=s.clone();source[:,4]=thumb+(1-(1-weights).prod(1))[:,None]*(midpoint-thumb);source[:,self.ids]=fingers+weights[...,None]*(midpoint[:,None]-fingers)
        distances=(source[:,:,None]-source[:,None,:]).norm(dim=-1);adjacency=torch.exp(-50*distances)*self.neighbors[None,None,:]*(1-self.eye);laplacian=self.eye-adjacency/adjacency.sum(-1,keepdim=True).clamp_min(1e-30);delta=source[:,self.edges[:,1]]-source[:,self.edges[:,0]];directions=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return source,laplacian,laplacian@source,self.edges,directions

    def update_targets(self,target):
        s,l,d,edges,directions=target;u=s[:,3]-s[:,2];v=s[:,4]-s[:,3];u=u/u.norm(dim=-1,keepdim=True).clamp_min(1e-8);v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-8);cross=torch.linalg.cross(u,v);sine=cross.norm(dim=-1);cosine=(u*v).sum(-1);normal=cross/sine[:,None].clamp_min(1e-8);t=((cosine-math.cos(.5))/(math.cos(.35)-math.cos(.5))).clamp(0,1);weight=(.5*t.square()*(3-2*t)*(sine>1e-8)).sqrt()
        for raw in self.raws:
            for name,value in [('s',s),('l',l),('d',d),('directions',directions),('normal',normal),('cosine',cosine),('sine',sine),('thumb_weight',weight)]:getattr(raw,name).copy_(value)

    def body(self,first):
        ids=self.starts+torch.minimum(self.index,self.lengths-1);active=self.index<self.lengths
        hq=self.human.step(self.positions[ids],self.rotations[ids],stream=self.wp_stream);self.update_targets(self.targets(self.p.sfk.forward(hq)[0]));q=self.reference.clone()
        for stage in (1,2):
            count=(50 if stage==1 else 80) if first else (1 if stage==1 else 10);key=(stage,first)
            if key not in self.engines:
                if self.backend=='causal_lm':
                    from .robokit_lm import RobokitCUDALM
                    engine=RobokitCUDALM(JointAnchorObjective(self.raws[stage-1],self.reference,.01),iterations=count,compile=True,graph_mode='none')
                else:
                    from .trajectory_lbfgs import RobokitTrajectoryLBFGS
                    from .trajectory_pipeline import CompiledSpatial
                    obj=TrajectoryObjective(CompiledSpatial(self.raws[stage-1]),list(range(self.b+1)),self.reference,.01);engine=RobokitTrajectoryLBFGS(obj,count)
                self.engines[key]=engine
            q=self.engines[key](q,self.p.rfk.lower,self.p.rfk.upper).q
        raw=q.clamp(self.p.rfk.lower,self.p.rfk.upper).float();self.reference.copy_(raw.double())
        if first:self.filtered.copy_(raw)
        else:self.filtered.copy_(self.filtered+(raw-self.filtered)*.2)
        self.raw_out[ids]=torch.where(active[:,None],raw.double(),self.raw_out[ids]);self.filtered_out[ids]=torch.where(active[:,None],self.filtered.double(),self.filtered_out[ids]);self.human_out[ids]=torch.where(active[:,None],hq,self.human_out[ids]);self.evaluations.add_(self.human.counts.long());self.index.add_(1)

    def reset(self):
        self.index.zero_();self.human.reset();self.reference.copy_(self.p.tensor(self.p.robot.bounds.mean(1))[None]);self.filtered.zero_();self.evaluations.zero_()

    @torch.no_grad()
    def __call__(self,positions,rotations):
        self.positions.copy_(positions);self.rotations.copy_(rotations);caller=torch.cuda.current_stream(self.device);self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(self.wp_stream):
            if self.first is None:
                self.reset();self.body(True);torch.cuda.synchronize();self.reset();self.first=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.first,stream=self.torch_stream):
                    with wp.ScopedCapture(stream=self.wp_stream,external=True):self.body(True)
                self.reset();self.first.replay();self.body(False);torch.cuda.synchronize();self.reset();self.first.replay();self.later=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.later,stream=self.torch_stream):
                    with wp.ScopedCapture(stream=self.wp_stream,external=True):self.body(False)
            self.reset();self.first.replay()
            for _ in range(1,self.max_length):self.later.replay()
        caller.wait_stream(self.torch_stream)
        return self.human_out.clone(),self.raw_out.clone(),self.filtered_out.clone()

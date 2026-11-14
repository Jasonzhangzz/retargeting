"""Isolated causal-stage graph replays; diagnostic times are not additive."""
import time
import torch
from .trajectory_causal_graph import CausalFrameGraph
from .batched_lm import RobotRetargetObjective,prepare_robot_targets


class StageGraph(CausalFrameGraph):
    def __init__(self,pipeline,positions,rotations,offsets,backend,mode,human_q,raw_q):
        super().__init__(pipeline,positions,rotations,offsets,backend);self.mode=mode;self.saved_human=human_q;self.saved_raw=raw_q
        self.full_targets=RobotRetargetObjective(pipeline.rfk,*prepare_robot_targets(pipeline.sfk.forward(human_q)[0],pipeline.alignment))
    def body(self,first):
        ids=self.starts+torch.minimum(self.index,self.lengths-1);active=self.index<self.lengths
        if self.mode=='human':
            h=self.human.step(self.positions[ids],self.rotations[ids],stream=self.wp_stream);self.human_out[ids]=torch.where(active[:,None],h,self.human_out[ids])
        elif self.mode=='targets':self.update_targets(self.targets(self.p.sfk.forward(self.saved_human[ids])[0]))
        elif self.mode=='robot':
            for raw in self.raws:
                for name in ['s','l','d','directions','normal','cosine','sine','thumb_weight']:getattr(raw,name).copy_(getattr(self.full_targets,name)[ids])
            q=self.reference.clone()
            for stage in (1,2):
                key=(stage,first);count=(50 if stage==1 else 80) if first else (1 if stage==1 else 10)
                if key not in self.engines:
                    if self.backend=='causal_lm':
                        from .robokit_lm import RobokitCUDALM
                        from .batched_lm import JointAnchorObjective
                        engine=RobokitCUDALM(JointAnchorObjective(self.raws[stage-1],self.reference,.01),iterations=count,compile=True,graph_mode='none')
                    else:
                        from .trajectory_lbfgs import RobokitTrajectoryLBFGS
                        from .trajectory_pipeline import CompiledSpatial
                        from .trajectory import TrajectoryObjective
                        engine=RobokitTrajectoryLBFGS(TrajectoryObjective(CompiledSpatial(self.raws[stage-1]),list(range(self.b+1)),self.reference,.01),count)
                    self.engines[key]=engine
                q=self.engines[key](q,self.p.rfk.lower,self.p.rfk.upper).q
            q=q.clamp(self.p.rfk.lower,self.p.rfk.upper).float().double();self.reference.copy_(q);self.raw_out[ids]=torch.where(active[:,None],q,self.raw_out[ids])
        else:
            q=self.saved_raw[ids].float()
            if first:self.filtered.copy_(q)
            else:self.filtered.copy_(self.filtered+(q-self.filtered)*.2)
            self.filtered_out[ids]=torch.where(active[:,None],self.filtered.double(),self.filtered_out[ids])
        self.index.add_(1)


def profile_causal_stages(pipeline,positions,quaternions,offsets,backend,result,repeats=10):
    from .trajectory_pipeline import quaternion_matrix
    hp=pipeline.tensor(positions).float().double();hr=quaternion_matrix(pipeline.tensor(quaternions).float().double());report={}
    for mode in ['human','targets','robot','filter']:
        graph=StageGraph(pipeline,hp,hr,offsets,backend,mode,result.human_q,result.q_raw);samples=[];difference=None
        for repeat in range(2+repeats):
            torch.cuda.synchronize();start=time.perf_counter();h,q,f=graph(hp,hr);torch.cuda.synchronize();seconds=time.perf_counter()-start
            if repeat>=2:samples.append(seconds)
        if mode=='human':difference=float((h-result.human_q).abs().max())
        elif mode=='robot':difference=float((q-result.q_raw).abs().max())
        elif mode=='filter':difference=float((f-result.q_filtered).abs().max())
        report[mode]=dict(seconds=samples,max_output_difference_rad=difference)
        del graph,h,q,f
    report['scope']='Isolated graph replays on predicted human/robot trajectories, never SDK intermediates. Phase times include replay/collection overhead and are not additive; full production timing remains authoritative.'
    return report

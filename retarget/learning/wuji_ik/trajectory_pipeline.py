"""EMF-to-Wuji2 trajectory interface, preserving original spatial/history weights."""
from dataclasses import dataclass
from pathlib import Path
import importlib,sys,time
import numpy as np
import torch
from .batched_lm import BatchURDF,FastFingerURDF,HumanPoseObjective,RobotRetargetObjective,prepare_robot_targets,JointAnchorObjective
from .trajectory import TrajectoryObjective,CUDABandedLM,segmented_iir


def quaternion_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True);x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


class HumanSpatial:
    def __init__(self,base,error_gain=.5):self.base,self.error_gain=base,error_gain
    def __call__(self,q):
        r,j=self.base(q);scale=torch.cat((r.new_full((30,),self.error_gain**.5),r.new_ones(q.shape[-1])))
        return r*scale,j*scale[None,:,None]


class CompiledSpatial:
    def __init__(self,raw,compile=True):
        self.raw=raw;self.call=torch.compile(raw,fullgraph=True,dynamic=False) if compile else raw
    def __call__(self,q):return self.call(q)


@dataclass
class EMFTrajectoryResult:
    human_q:torch.Tensor
    q_raw:torch.Tensor
    q_filtered:torch.Tensor
    spatial_cost:torch.Tensor
    total_cost:torch.Tensor
    timings:dict
    solver_traces:dict
    information:str
    initial_q:torch.Tensor|None=None
    objective_terms:dict|None=None
    convergence:dict|None=None


class EMFTrajectorySolver:
    """One hand/glove configuration. Inputs are TX EMF; network consumes wrist features.

    `offsets` delimit complete independent segments. Joint route uses all frames;
    causal route owns and resets history at each segment. Model/asset loading is
    outside solve(); feature construction and neural inference are inside it.
    """
    def __init__(self,reference,urdf,side,*,device='cuda',compile=True,model=None,normalizer=None,crocoddyl_module=None):
        sys.path.insert(0,str(reference));self.reference=Path(reference);self.side=side;self.device=torch.device(device);self.compile=compile;self.model,self.normalizer=model,normalizer;self.crocoddyl_module=crocoddyl_module
        from human_ik_reference.kinematics import HumanHandKinematics
        from human_ik_reference.parameters import default_parameters
        from retarget_reference.kinematics import HandKinematics
        pre=importlib.import_module("retarget_reference.preprocess")
        from .features import joint_names
        from retarget_reference.session import builtin_parameters
        actual=builtin_parameters('WujiHand2')
        expected=dict(lambda_warm=1.,lambda_lap=150.,lambda_pose=.1,lambda_smooth=.01,lambda_thumb_rel=.5,lambda_antirev=20.)
        if any(actual[k]!=v for k,v in expected.items()):raise ValueError('Reference objective changed; port its weights before benchmarking')
        self.pre=pre;self.human=HumanHandKinematics(side,urdf,sdk_tx_rounding=True);self.robot=HandKinematics('WujiHand2',side);self.params=default_parameters(side)
        if list(self.robot.joint_names)!=joint_names(side):raise ValueError('Network and URDF joint order differ')
        if any(self.params[k]!=v for k,v in dict(w_pos=.3,w_rot=5.729577951308232,error_gain=.5,history_weight=20.,posture_weight=1.,thumb_posture_scale=.01,error_to_m=1.,target_scale=1.,damping_scale=.0001,damping_floor=1e-6,max_step_scale=1/120,bound_scale=.5,retry_base=np.pi).items()):raise ValueError('Reference human objective changed')
        tensor=lambda x:torch.as_tensor(x,device=self.device,dtype=torch.float64)
        self.tensor=tensor;self.seed=tensor(np.concatenate(self.params['seed']));bounds=np.concatenate(self.params['bounds']);self.hlo=tensor(bounds[:,0]);self.hhi=tensor(bounds[:,1])
        self.hfk=FastFingerURDF(urdf,self.human.joint_names,['thumb_rx_coil','index_rx_coil','middle_rx_coil','ring_rx_coil','pinky_rx_coil'],device=self.device,root_transform=self.human._tx_inverse)
        self.sfk=FastFingerURDF(urdf,self.human.joint_names,self.human.skeleton_links,device=self.device)
        self.rfk=FastFingerURDF(self.robot.urdf_path,self.robot.joint_names,self.robot.landmark_names,device=self.device)
        posture=np.ones(21);posture[:5]=self.params['thumb_posture_scale']**.5;posture[[8,12,16,20]]=0;self.posture=tensor(posture)
        neutral=self.human.forward(self.seed.cpu().numpy())['skeleton'][:,:3,3].astype(np.float32).astype(np.float64)
        operator=pre.OPERATOR2MANO_RIGHT if side=='right' else pre.OPERATOR2MANO_LEFT;rz=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        self.alignment=tensor(pre.estimate_frame(neutral)@operator@rz.T);self.urdf=Path(urdf)
        self.engines={};self.objectives={}

    def sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    def network(self,positions,quaternions,offsets):
        from .features import wrist_tx_transform,wrist_emf_features
        from .metrics import autocast_context
        if self.model is None:raise ValueError('Provide a model/normalizer or explicit initial_q')
        transform=wrist_tx_transform(self.urdf);features=[torch.from_numpy(wrist_emf_features(positions[s:e],quaternions[s:e],transform)) for s,e in zip(offsets,offsets[1:])]
        inputs=torch.nn.utils.rnn.pad_sequence(features,batch_first=True).to(self.device);lengths=torch.tensor([e-s for s,e in zip(offsets,offsets[1:])],device=self.device);sides=torch.full((len(features),),0 if self.side=='right' else 1,device=self.device,dtype=torch.long)
        with autocast_context(self.device):prediction=self.model(inputs,sides,lengths)
        radians=self.normalizer.denormalize(prediction.float(),sides).double();return torch.cat([radians[i,:e-s] for i,(s,e) in enumerate(zip(offsets,offsets[1:]))])

    def solve(self,positions,quaternions,offsets,*,backend='banded_lm',**kwargs):
        if backend in ('causal_prefix','frozen_parallel','frozen_history'):
            return self.solve_oneway(positions,quaternions,offsets,backend=backend,**kwargs)
        if backend.startswith('causal_'):
            if kwargs:raise ValueError('Causal route uses the fixed reference update budgets')
            return self.solve_causal(positions,quaternions,offsets,backend=backend)
        return self.solve_joint(positions,quaternions,offsets,backend=backend,**kwargs)

    @torch.no_grad()
    def solve_oneway(self,positions,quaternions,offsets,*,backend='causal_prefix',iterations=32,stage1_iterations=4,initial_q=None):
        """Source-rule causal human reconstruction, followed by one-way robot IK.

        Explicit initial_q must itself be causal if past/present information is
        required. frozen_history uses previous raw q, so needs no neural seed.
        """
        from .trajectory_oneway import CausalHumanRollout,OneWayLM,FrozenHistoryRobot
        if offsets[0]!=0 or offsets[-1]!=len(positions):raise ValueError('Invalid segments')
        if backend not in ('causal_prefix','frozen_parallel','frozen_history'):raise ValueError(backend)
        self.sync();start=time.perf_counter()
        initial=self.tensor(self.robot.bounds.mean(1)).expand(len(positions),-1).clone() if backend=='frozen_history' else (self.network(positions,quaternions,offsets) if initial_q is None else initial_q.clone())
        self.sync();nn=time.perf_counter();hp=self.tensor(positions).float().double();hr=quaternion_matrix(self.tensor(quaternions).float().double())
        human_key=('oneway_human',tuple(offsets))
        if human_key not in self.engines:self.engines[human_key]=CausalHumanRollout(self,hp,hr,offsets)
        hq=self.engines[human_key](hp,hr);target=prepare_robot_targets(self.sfk.forward(hq)[0],self.alignment)
        key=('oneway',backend,tuple(offsets))
        if key not in self.objectives:
            self.objectives[key]=[TrajectoryObjective(CompiledSpatial(RobotRetargetObjective(self.rfk,*target,stage=stage),self.compile),offsets,self.tensor(self.robot.bounds.mean(1)).expand(len(offsets)-1,-1),.01) for stage in (1,2)]
        else:
            for stage,obj in enumerate(self.objectives[key],1):obj.spatial.raw.copy_targets_(RobotRetargetObjective(self.rfk,*target,stage=stage))
        objectives=self.objectives[key];self.sync();human_end=time.perf_counter();traces=[];q=initial
        if backend=='frozen_history':
            if key not in self.engines:self.engines[key]=FrozenHistoryRobot(objectives,initial,self.rfk.lower,self.rfk.upper)
            q,filtered=self.engines[key](initial)
            self.sync();robot_end=time.perf_counter()
        else:
            for stage,budget in ((0,stage1_iterations),(1,iterations)):
                ek=(key,stage)
                if ek not in self.engines:self.engines[ek]=OneWayLM(objectives[stage],q,self.rfk.lower,self.rfk.upper,mode=backend)
                result=self.engines[ek](q,budget);q=result.q;traces.append(result)
            q=q.float().double();self.sync();robot_end=time.perf_counter();filtered=segmented_iir(q,offsets)
        self.sync();end=time.perf_counter()
        from .trajectory import robot_diagnostics
        terms,state=robot_diagnostics(objectives[1].spatial,q,offsets,objectives[1].reference,causal=True,lower=self.rfk.lower,upper=self.rfk.upper)
        return EMFTrajectoryResult(hq,q,filtered,.5*objectives[1].spatial(q)[0].square().sum(-1),objectives[1].evaluate(q,False),dict(network=nn-start,human_and_targets=human_end-nn,robot=robot_end-human_end,filter=end-robot_end,total=end-start),dict(robot=traces), 'past_and_present_with_causal_initial_q',initial if backend!='frozen_history' else None,terms,state)

    def make_engine(self,key,obj,initial,lo,hi,backend):
        if key not in self.engines:
            if backend=='banded_lm':engine=CUDABandedLM(obj,initial,lo,hi)
            elif backend=='trajectory_lbfgs':
                from .trajectory_lbfgs import RobokitTrajectoryLBFGS
                engine=RobokitTrajectoryLBFGS(obj)
            elif backend=='gpu_ilqr':
                from .trajectory_ilqr import AnalyticILQR
                engine=AnalyticILQR(obj,initial,lo,hi)
            elif backend=='crocoddyl_boxfddp':
                from .trajectory_crocoddyl import CrocoddylTrajectory
                engine=CrocoddylTrajectory(obj,lo,hi,self.crocoddyl_module)
            else:raise ValueError('Unknown backend '+backend)
            self.engines[key]=engine
        return self.engines[key]

    @torch.no_grad()
    def solve_joint(self,positions,quaternions,offsets,*,backend='banded_lm',iterations=16,stage1_iterations=4,initial_q=None,human_initial=None):
        if offsets[0]!=0 or offsets[-1]!=len(positions):raise ValueError('Invalid segments')
        self.sync();start=time.perf_counter();initial=self.network(positions,quaternions,offsets) if initial_q is None else initial_q.clone();self.sync();nn=time.perf_counter()
        hp=self.tensor(positions).float().double();hr=quaternion_matrix(self.tensor(quaternions).float().double());n=len(hp)
        hstart=self.seed[None].expand(n,-1).clone() if human_initial is None else human_initial.clone()
        key=(backend,n,tuple(offsets))
        # Stable target addresses allow graph replay with fresh EMF; robot
        # graph-bound targets are refreshed after the human solve below.
        if key not in self.objectives:
            hraw=HumanSpatial(HumanPoseObjective(self.hfk,hp,hr,self.seed,self.posture));hs=CompiledSpatial(hraw,self.compile and backend!='crocoddyl_boxfddp');ho=TrajectoryObjective(hs,offsets,self.seed.expand(len(offsets)-1,-1),20.)
            self.objectives[key]=dict(hraw=hraw,h=ho)
        objects=self.objectives[key];objects['hraw'].base.position.copy_(hp);objects['hraw'].base.rotation.copy_(hr)
        he=self.make_engine((key,'human'),objects['h'],hstart,self.hlo,self.hhi,backend)
        h=self.run_engine(he,hstart,self.hlo,self.hhi,iterations,backend)
        skeleton=self.sfk.forward(h.q)[0];target=prepare_robot_targets(skeleton,self.alignment)
        if 'r1' not in objects:
            for stage in (1,2):
                raw=RobotRetargetObjective(self.rfk,*target,stage=stage);spatial=CompiledSpatial(raw,self.compile and backend!='crocoddyl_boxfddp');objects[f'r{stage}']=TrajectoryObjective(spatial,offsets,self.tensor(self.robot.bounds.mean(1)).expand(len(offsets)-1,-1),.01)
        else:
            for stage in (1,2):objects[f'r{stage}'].spatial.raw.copy_targets_(RobotRetargetObjective(self.rfk,*target,stage=stage))
        self.sync();human_end=time.perf_counter()
        results=[];q=initial
        for stage,budget in [(1,stage1_iterations),(2,iterations)]:
            if budget==0:continue
            engine_key=(key,'robot',stage)
            engine=self.make_engine(engine_key,objects[f'r{stage}'],q,self.rfk.lower,self.rfk.upper,backend)
            result=self.run_engine(engine,q,self.rfk.lower,self.rfk.upper,budget,backend);q=result.q;results.append(result)
        self.sync();robot_end=time.perf_counter();filtered=segmented_iir(q,offsets);self.sync();end=time.perf_counter()
        from .trajectory import robot_diagnostics
        terms,convergence=robot_diagnostics(objects['r2'].spatial,q,offsets,objects['r2'].reference,lower=self.rfk.lower,upper=self.rfk.upper)
        return EMFTrajectoryResult(h.q,q,filtered,objects['r2'].spatial(q)[0].square().sum(-1)*.5,objects['r2'].evaluate(q,False),dict(network=nn-start,human_and_targets=human_end-nn,robot=robot_end-human_end,filter=end-robot_end,total=end-start),dict(human=h,robot=results), 'past_present_and_future',initial,terms,convergence)

    def run_engine(self,engine,q,lo,hi,iterations,backend):
        if backend=='trajectory_lbfgs':engine.iterations=iterations;return engine(q,lo,hi)
        return engine(q,iterations)

    @torch.no_grad()
    def solve_causal(self,positions,quaternions,offsets,*,backend='causal_lm'):
        from .trajectory_causal_graph import CausalFrameGraph
        self.sync();start=time.perf_counter();hp=self.tensor(positions).float().double();hr=quaternion_matrix(self.tensor(quaternions).float().double());key=(backend,'full_causal',tuple(offsets))
        if key not in self.engines:self.engines[key]=CausalFrameGraph(self,hp,hr,offsets,backend)
        engine=self.engines[key];hq,raw,filtered=engine(hp,hr);self.sync();end=time.perf_counter()
        target=prepare_robot_targets(self.sfk.forward(hq)[0],self.alignment);spatial=RobotRetargetObjective(self.rfk,*target);obj=TrajectoryObjective(spatial,offsets,self.tensor(self.robot.bounds.mean(1)).expand(len(offsets)-1,-1),.01)
        from .trajectory import robot_diagnostics
        terms,convergence=robot_diagnostics(spatial,raw,offsets,obj.reference,causal=True,lower=self.rfk.lower,upper=self.rfk.upper)
        return EMFTrajectoryResult(hq,raw,filtered,.5*spatial(raw)[0].square().sum(-1),obj.evaluate(raw,False),dict(network=0.,human_and_targets=None,robot=None,filter=None,total=end-start),dict(human=dict(actual_function_evaluations=int(engine.evaluations.sum()),scope='includes padded inactive-segment computations'),robot='whole-frame graph; first50/80, later1/10 update caps; not SLSQP evaluation caps'),'past_and_present',None,terms,convergence)

    @torch.no_grad()
    def solve_causal_eager_reference(self,positions,quaternions,offsets,*,backend='causal_lm'):
        from .trajectory_human_native import NativeCausalHuman
        from .robokit_lm import RobokitCUDALM
        from .trajectory_lbfgs import RobokitTrajectoryLBFGS
        b=len(offsets)-1;lengths=[e-s for s,e in zip(offsets,offsets[1:])];n=len(positions);self.sync();start=time.perf_counter()
        hp=self.tensor(positions).float().double();hr=quaternion_matrix(self.tensor(quaternions).float().double());key=(backend,'causal',b)
        if key not in self.objectives:
            base=HumanPoseObjective(self.hfk,hp[:b].clone(),hr[:b].clone(),self.seed,self.posture)
            human=NativeCausalHuman(base,self.seed,self.hlo,self.hhi)
            reference=self.tensor(self.robot.bounds.mean(1))[None].expand(b,-1).clone()
            self.objectives[key]=dict(human=human,reference=reference,engines={})
        state=self.objectives[key];human=state['human'];human.reset();reference=state['reference'];reference.copy_(self.tensor(self.robot.bounds.mean(1))[None]);filtered=None
        qs=torch.empty(n,20,device=self.device,dtype=torch.float64);fs=torch.empty_like(qs);hs=torch.empty(n,21,device=self.device,dtype=torch.float64);human_seconds=0.;robot_seconds=0.;filter_seconds=0.
        for t in range(max(lengths)):
            ids=torch.tensor([s+min(t,length-1) for s,length in zip(offsets,lengths)],device=self.device)
            self.sync();hbegin=time.perf_counter();hq=human.step(hp[ids],hr[ids]);skeleton=self.sfk.forward(hq)[0];target=prepare_robot_targets(skeleton,self.alignment);self.sync();human_seconds+=time.perf_counter()-hbegin
            rbegin=time.perf_counter();q=reference.clone()
            for stage in (1,2):
                name=f'r{stage}'
                if name not in state:state[name]=RobotRetargetObjective(self.rfk,*target,stage=stage)
                else:state[name].copy_targets_(RobotRetargetObjective(self.rfk,*target,stage=stage))
                count=(50 if stage==1 else 80) if t==0 else (1 if stage==1 else 10)
                engine_key=(stage,t==0)
                if engine_key not in state['engines']:
                    if backend=='causal_lm':engine=RobokitCUDALM(JointAnchorObjective(state[name],reference,.01),iterations=count,compile=self.compile)
                    else:
                        objective=TrajectoryObjective(CompiledSpatial(state[name],self.compile),list(range(b+1)),reference,.01)
                        engine=RobokitTrajectoryLBFGS(objective,count)
                    state['engines'][engine_key]=engine
                q=state['engines'][engine_key](q,self.rfk.lower,self.rfk.upper).q
            raw=q.clamp(self.rfk.lower,self.rfk.upper).float();reference.copy_(raw.double());self.sync();robot_seconds+=time.perf_counter()-rbegin
            fbegin=time.perf_counter()
            if filtered is None:filtered=raw.clone()
            else:filtered=filtered+(raw-filtered)*np.float32(.2)
            for i,(s,length) in enumerate(zip(offsets,lengths)):
                if t<length:qs[s+t]=raw[i].double();fs[s+t]=filtered[i].double();hs[s+t]=hq[i]
            self.sync();filter_seconds+=time.perf_counter()-fbegin
        skeleton=self.sfk.forward(hs)[0];target=prepare_robot_targets(skeleton,self.alignment);raw_obj=RobotRetargetObjective(self.rfk,*target);obj=TrajectoryObjective(raw_obj,offsets,self.tensor(self.robot.bounds.mean(1)).expand(b,-1),.01)
        self.sync();end=time.perf_counter()
        return EMFTrajectoryResult(hs,qs,fs,.5*raw_obj(qs)[0].square().sum(-1),obj.evaluate(qs,False),dict(network=0.,human_and_targets=human_seconds,robot=robot_seconds,filter=filter_seconds,total=end-start),dict(human='reference QP/pose-only line search, budgets40/36/36, max448 trial evaluations per finger per frame',robot='LM/L-BFGS update caps50/80 first,1/10 later; distinct from SLSQP evaluation caps'),'past_and_present')

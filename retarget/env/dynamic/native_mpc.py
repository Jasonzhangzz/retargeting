"""Original Robokit SimMPC, with GPU force costs and named-state restoration."""
from dataclasses import asdict
import numpy as np
import warp as wp
from robokit.helpers.sim_mpc import SimMPCConfig,SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import StateTask,ControlTask
from robokit.terms.task import Task
from robokit.terms.simulation import SimContext
from retarget.env.contact.robokit_rollout import StatefulSimMPC
from .execution import GPUExecution
from .rollout import contact_cost,CostConfig


@wp.kernel
def force_penalty(force:wp.array2d(dtype=float),phase:wp.array(dtype=float),clock:wp.array(dtype=int),target:float,weight:float,out:wp.array(dtype=float)):
    w=wp.tid();first=float(0.);second=float(0.)
    for f in range(5):
        value=force[w,f]
        if value>first:second=first;first=value
        else:second=wp.max(second,value)
    frame=wp.min(clock[0]+clock[1]+1,phase.shape[0]-1)
    deficit=wp.max(target-second,0.)/target
    out[w]+=weight*phase[frame]*deficit*deficit


class ForceTask(Task):
    def __init__(self,phase,cost):self.phase,self.config=phase,cost
    def register(self,sim):self.sim=sim
    def resolve(self,values,context):
        self.clock=context[SimContext.CLOCK];sim=self.sim
        self.phase=wp.array(self.phase,dtype=float,device=sim.device)
        self.weights=wp.array(list(asdict(self.config).values()),dtype=float,device=sim.device)
        self.force=wp.zeros((sim.nworld,5),dtype=float,device=sim.device)
        owners=[]
        for body in sim.mjm.geom_bodyid:
            b=int(body);name=sim.mjm.body(b).name
            while not name.startswith('right_finger') and b>0:b=int(sim.mjm.body_parentid[b]);name=sim.mjm.body(b).name
            owners.append(int(name[len('right_finger')])-1 if name.startswith('right_finger') else -1)
        self.owners=wp.array(owners,dtype=int,device=sim.device)
    def reset(self,sim):pass
    def compute_weighted_cost(self,values,*,out_cost):
        self.force.zero_();d=self.sim._d
        wp.launch(contact_cost,dim=d.contact.dist.shape[0],inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.dist,d.contact.efc_address,d.efc.force,self.sim.mjm.geom('tracked_object_geom').id,self.owners,self.force,self.weights,1,out_cost])
        wp.launch(force_penalty,dim=self.sim.nworld,inputs=[self.force,self.phase,self.clock,self.config.grasp_force_target,self.config.grasp_force,out_cost])


def make_planner(model,initial,reference,phase,*,samples=128,iterations=6,horizon=160,node_steps=40,execute_steps=20,seed=42,device='cuda:0',contact_reference=None,contact_weight=1.,contact_scale=.01):
    if not str(device).startswith('cuda'):raise ValueError('CUDA MJWarp required')
    weights=np.r_[np.full(3,np.sqrt(.025/26)/.05),np.full(23,np.sqrt(.025/26)/.5),np.full(3,20.),np.full(3,1.)]
    # Native fine-step sampling. Coarse 4x rollouts caused a CUDA failure on
    # the mustard case; use the common 300 Hz dynamics for every candidate.
    # All times here are PHYSICS steps, never 30 Hz source frames.
    stages=[SimMPCStage(iterations)]
    sigma=np.r_[np.full(3,.01),np.full(3,.05),np.full(20,.1)]
    cfg=SimMPCConfig(num_samples=samples,horizon_steps=horizon,node_steps=node_steps,ctrl_steps=execute_steps,
        stages=stages,ctrl_std=sigma,draw_first_node=True,noise_decay=.7,std_growth=2.,terminal_scale=3.,min_improvement=-1.,seed=seed)
    cfg.add(StateTask(reference,np.zeros((len(reference),model.nv)),weights,squared=True))
    cfg.add(ControlTask(.01,reference[:,:26],norm=True))
    cfg.add(ForceTask(phase,CostConfig()))
    if contact_reference is not None:
        from .contact_reference import ContactAnchorTask
        if len(contact_reference.weight)!=len(reference):raise ValueError('Contact and pose reference timelines differ')
        cfg.add(ContactAnchorTask(contact_reference,weight=contact_weight,scale=contact_scale))
    with wp.ScopedDevice(device):
        sim=MjWarpSim(model,1,initial,substeps=1,nconmax=128,njmax=768,device=device)
        planner=StatefulSimMPC(cfg,sim,reference[:,:26])
    return planner

"""CUDA-JIT contact/tracking objectives over Robokit MuJoCo-Warp rollouts."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
import warp as wp
from robokit.sim import MjWarpSim


@dataclass(frozen=True)
class CostConfig:
    object_position: float=1.
    object_rotation: float=.25
    hand_reference: float=.025
    control_reference: float=.01
    control_smoothness: float=.002
    penetration: float=.2
    excessive_force: float=.05
    object_position_scale: float=.05
    object_rotation_scale: float=.5
    penetration_limit: float=.005
    force_limit: float=100.
    terminal: float=3.
    grasp_force: float=.10
    grasp_force_target: float=1.2

    def __post_init__(self):
        values=asdict(self)
        if not np.isfinite(list(values.values())).all() or min(values.values())<0 or min(self.object_position_scale,self.object_rotation_scale,self.penetration_limit,self.force_limit)<=0:
            raise ValueError('Invalid cost scales/weights')


@wp.kernel
def broadcast_state(source:wp.array2d(dtype=float),dest:wp.array2d(dtype=float)):
    w,j=wp.tid();dest[w,j]=source[0,j]


@wp.kernel
def set_control(controls:wp.array3d(dtype=float),step:int,ctrl:wp.array2d(dtype=float)):
    w,j=wp.tid();ctrl[w,j]=controls[w,step,j]


@wp.kernel
def tracking_cost(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),ctrl:wp.array2d(dtype=float),
                  ref:wp.array2d(dtype=float),previous:wp.array2d(dtype=float),clock:wp.array(dtype=int),
                  weights:wp.array(dtype=float),finger_force:wp.array2d(dtype=float),phase:wp.array(dtype=float),step:int,horizon:int,cost:wp.array(dtype=float)):
    w=wp.tid();frame=wp.min(clock[0]+step+1,ref.shape[0]-1)
    position=float(0.);hand=float(0.);reference=float(0.);smooth=float(0.)
    valid=bool(True)
    for j in range(q.shape[1]):valid=valid and wp.isfinite(q[w,j])
    for j in range(v.shape[1]):valid=valid and wp.isfinite(v[w,j]) and wp.abs(v[w,j])<10000.
    for j in range(3):position+=(q[w,26+j]-ref[frame,26+j])**2.0
    dot=float(0.);nq=float(0.);nr=float(0.)
    for j in range(4):
        dot+=q[w,29+j]*ref[frame,29+j];nq+=q[w,29+j]**2.0;nr+=ref[frame,29+j]**2.0
    angle=2.*wp.acos(wp.clamp(wp.abs(dot)/wp.sqrt(wp.max(nq*nr,1.e-12)),0.,1.))
    for j in range(26):
        scale=wp.where(j<3,.05,.5)
        d=q[w,j]-ref[frame,j]
        if j>=3 and j<6:d=wp.atan2(wp.sin(d),wp.cos(d))
        hand+=(d/scale)**2.0/26.
        reference+=((ctrl[w,j]-ref[wp.max(frame-1,0),j])/scale)**2.0/26.
        smooth+=((ctrl[w,j]-previous[w,j])/scale)**2.0/26.
        previous[w,j]=ctrl[w,j]
    value=weights[0]*position/(weights[7]*weights[7])+weights[1]*(angle/weights[8])**2.0+weights[2]*hand+weights[3]*reference+weights[4]*smooth
    first=float(0.);second=float(0.)
    for finger in range(5):
        fn=finger_force[w,finger]
        if fn>first:second=first;first=fn
        else:second=wp.max(second,fn)
    deficit=wp.max(weights[13]-second,0.)/wp.max(weights[13],1.e-6)
    value+=weights[12]*phase[frame]*deficit*deficit
    terminal=wp.where(step==horizon-1,weights[11],1.)
    if valid:cost[w]+=terminal*value/float(horizon)
    else:cost[w]=wp.inf


@wp.kernel
def contact_cost(count:wp.array(dtype=int),world:wp.array(dtype=int),pair:wp.array(dtype=wp.vec2i),
                 dist:wp.array(dtype=float),address:wp.array2d(dtype=int),force:wp.array2d(dtype=float),
                 object_geom:int,owner:wp.array(dtype=int),finger_force:wp.array2d(dtype=float),weights:wp.array(dtype=float),horizon:int,cost:wp.array(dtype=float)):
    i=wp.tid()
    if i<count[0]:
        g=pair[i];w=world[i]
        if w>=0 and (g[0]==object_geom or g[1]==object_geom):
            depth=wp.max(0.,-dist[i]-weights[9])/weights[9]
            normal=float(0.);adr=address[i,0]
            # The scene uses elliptic cones: first constraint is normal force.
            if adr>=0 and adr<force.shape[1]:normal=wp.max(0.,force[w,adr])
            other=wp.where(g[0]==object_geom,g[1],g[0])
            finger=owner[other]
            if finger>=0:wp.atomic_add(finger_force,w,finger,normal)
            excess=wp.max(0.,normal-weights[10])/weights[10]
            wp.atomic_add(cost,w,(weights[5]*depth*depth+weights[6]*excess*excess)/float(horizon))


class RobokitRollout:
    def __init__(self,model,state,reference,horizon,population,*,substeps=10,device='cuda:0',cost=CostConfig(),capture=True,contact_phase=None):
        if not str(device).startswith('cuda'):raise ValueError('Dynamic retargeting requires CUDA MJWarp; CPU fallback is disabled')
        self.model,self.horizon,self.population,self.device=model,horizon,population,device
        self.config=cost
        with wp.ScopedDevice(device):
            self.sim=MjWarpSim(model,population,state,substeps=substeps,nconmax=128,njmax=768,device=device)
            self.ref=wp.array(np.ascontiguousarray(reference,np.float32),dtype=float,device=device)
            self.clock=wp.zeros(1,dtype=int,device=device)
            self.controls=wp.zeros((population,horizon,model.nu),dtype=float,device=device)
            self.cost=wp.zeros(population,dtype=float,device=device)
            self.previous=wp.zeros((population,model.nu),dtype=float,device=device)
            self.weights=wp.array(list(asdict(cost).values()),dtype=float,device=device)
            phase=np.zeros(len(reference),np.float32) if contact_phase is None else np.asarray(contact_phase,np.float32)
            if phase.shape!=(len(reference),) or not np.isfinite(phase).all() or np.any((phase<0)|(phase>1)):raise ValueError('Invalid grasp phase')
            self.phase=wp.array(phase,dtype=float,device=device)
            self.finger_force=wp.zeros((population,5),dtype=float,device=device)
            owners=[]
            for body in model.geom_bodyid:
                b=int(body);name=model.body(b).name
                while not name.startswith('right_finger') and b>0:b=int(model.body_parentid[b]);name=model.body(b).name
                owners.append(int(name[len('right_finger')])-1 if name.startswith('right_finger') else -1)
            self.owners=wp.array(owners,dtype=int,device=device)
            self.start={}
            # Restore every persistent numeric integration field exposed by mjw.
            for name in ('qpos','qvel','act','history','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','eq_active','mocap_pos','mocap_quat','userdata','time'):
                if not hasattr(self.sim._d,name) or not hasattr(state,name):continue
                dest=getattr(self.sim._d,name)
                if not isinstance(dest,wp.array) or not dest.size:continue
                value=np.asarray(getattr(state,name))
                shape=dest.numpy().shape
                buffer=wp.empty_like(dest)
                buffer.assign(np.broadcast_to(value,shape).copy())
                self.start[name]=(buffer,dest,shape)
        self.cost_torch=wp.to_torch(self.cost)
        self.ctrl_torch=wp.to_torch(self.controls)
        self.graph=None
        # Initialize lazy Robokit physics kernels outside outer graph capture.
        with wp.ScopedDevice(device),wp.ScopedStream(self.sim.stream):
            self.set_state(state,0);self.ctrl_torch.copy_(torch.as_tensor(reference[:horizon,:26],device=device)[None]);torch.cuda.synchronize(device)
            self._run();wp.synchronize_device(device)
            if capture:
                with wp.ScopedCapture(device=device) as ctx:self._run()
                self.graph=ctx.graph

    def set_state(self,state,frame):
        for name,(source,_,shape) in self.start.items():source.assign(np.broadcast_to(np.asarray(getattr(state,name)),shape).copy())
        self.clock.assign([frame])

    def _run(self):
        self.cost.zero_()
        for source,dest,_ in self.start.values():wp.copy(dest,source)
        if hasattr(self.sim._d,'overflow'):self.sim._d.overflow.zero_()
        wp.copy(self.previous,self.sim.data.ctrl)
        self.sim.forward()
        for step in range(self.horizon):
            wp.launch(set_control,dim=self.sim.data.ctrl.shape,inputs=[self.controls,step,self.sim.data.ctrl],device=self.device)
            self.sim.step()
            d=self.sim._d;self.finger_force.zero_()
            wp.launch(contact_cost,dim=d.contact.dist.shape[0],inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.dist,d.contact.efc_address,d.efc.force,self.model.geom('tracked_object_geom').id,self.owners,self.finger_force,self.weights,self.horizon,self.cost],device=self.device)
            wp.launch(tracking_cost,dim=self.population,inputs=[self.sim.data.qpos,self.sim.data.qvel,self.sim.data.ctrl,self.ref,self.previous,self.clock,self.weights,self.finger_force,self.phase,step,self.horizon,self.cost],device=self.device)

    def __call__(self,controls):
        if tuple(controls.shape)!=(self.population,self.horizon,26):raise ValueError('Fixed rollout population required')
        with wp.ScopedDevice(self.device),wp.ScopedStream(self.sim.stream):
            stream=wp.stream_to_torch(self.sim.stream)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):self.ctrl_torch.copy_(controls)
            if self.graph is None:self._run()
            else:wp.capture_launch(self.graph)
            wp.synchronize_stream(self.sim.stream)
        # Severe capacity overflows invalidate candidates, never just log them.
        if hasattr(self.sim._d,'overflow'):
            import mujoco_warp as mjw
            flags=wp.to_torch(self.sim._d.overflow).reshape(self.population,-1)
            ignored=int(mjw.OverflowType.ITERATIONS|mjw.OverflowType.LS_ITERATIONS) if hasattr(mjw,'OverflowType') else 0
            bad=((flags.to(torch.int64)&~ignored)!=0).any(1)
            self.cost_torch.masked_fill_(bad,torch.inf)
        return self.cost_torch.clone()


class Evaluator:
    def __init__(self,model,state,reference,horizon,population,**kwargs):
        self.batch=RobokitRollout(model,state,reference,horizon,population,**kwargs)
        self.validation=RobokitRollout(model,state,reference,horizon,1,**kwargs)

    def set_state(self,state,frame):
        self.batch.set_state(state,frame);self.validation.set_state(state,frame)

    def __call__(self,controls):
        return (self.validation if len(controls)==1 else self.batch)(controls)

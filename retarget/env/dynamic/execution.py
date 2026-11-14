"""All executed physics, grasp initialization and baseline replay use MJWarp."""
import mujoco
import numpy as np
import torch
import warp as wp
from robokit.sim import MjWarpSim


@wp.kernel
def pin_object(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),pose:wp.array(dtype=float)):
    i=wp.tid()
    if i<7:q[0,26+i]=pose[i]
    if i<6:v[0,26+i]=0.


@wp.kernel
def collect_contact(count:wp.array(dtype=int),pair:wp.array(dtype=wp.vec2i),dist:wp.array(dtype=float),
                    address:wp.array2d(dtype=int),efc_force:wp.array2d(dtype=float),
                    owners:wp.array(dtype=int),object_geom:int,slot:int,forces:wp.array2d(dtype=float),depth:wp.array(dtype=float)):
    i=wp.tid()
    if i<count[0]:
        g=pair[i]
        if g[0]==object_geom or g[1]==object_geom:
            other=wp.where(g[0]==object_geom,g[1],g[0]);f=owners[other]
            if f>=0:
                adr=address[i,0]
                if adr>=0 and adr<efc_force.shape[1]:wp.atomic_add(forces,slot,f,wp.max(0.,efc_force[0,adr]))
            wp.atomic_max(depth,slot,wp.max(0.,-dist[i]))


class GPUExecution:
    def __init__(self,model,state=None,*,substeps=10,device='cuda:0'):
        if not str(device).startswith('cuda'):raise ValueError('Dynamic retargeting requires CUDA MJWarp; CPU fallback is disabled')
        self.model,self.device,self.substeps=model,device,substeps
        state=mujoco.MjData(model) if state is None else state
        with wp.ScopedDevice(device):
            self.sim=MjWarpSim(model,1,state,substeps=substeps,nconmax=128,njmax=768,device=device)
            self.ctrl=wp.to_torch(self.sim.data.ctrl)
            owners=[]
            for body in model.geom_bodyid:
                b=int(body);name=model.body(b).name
                while not name.startswith('right_finger') and b>0:b=int(model.body_parentid[b]);name=model.body(b).name
                owners.append(int(name[len('right_finger')])-1 if name.startswith('right_finger') else -1)
            self.owners=wp.array(owners,dtype=int,device=device)
        self.stream=wp.stream_to_torch(self.sim.stream)

    def command(self,value):
        # Explicit stream hand-off for Torch control buffers and Warp graphs.
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):self.ctrl.copy_(torch.as_tensor(value,dtype=torch.float32,device=self.device).reshape(1,26))

    def snapshot(self):
        """CPU MjData is a storage/serialization container, never stepped."""
        wp.synchronize_stream(self.sim.stream)
        out=mujoco.MjData(self.model)
        for name in ('qpos','qvel','act','history','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','eq_active','mocap_pos','mocap_quat','userdata'):
            if not hasattr(self.sim._d,name) or not hasattr(out,name):continue
            value=getattr(self.sim._d,name)
            if isinstance(value,wp.array) and value.size:
                target=getattr(out,name);x=value.numpy()
                if x.size==target.size:target[:]=x.reshape(target.shape)
        out.time=float(self.sim.data.time.numpy().reshape(-1)[0])
        return out

    def establish(self,hand,object_pose,frames=12):
        if frames<0:raise ValueError('Negative grasp initialization length')
        q=np.r_[np.clip(self.model.qpos0[:26],*self.model.actuator_ctrlrange.T),object_pose].astype(np.float32)
        q[:6]=hand[:6]
        if frames==0:q[:26]=hand
        with wp.ScopedDevice(self.device),wp.ScopedStream(self.sim.stream):
            self.sim.data.qpos.assign(q[None]);self.sim.data.qvel.zero_();self.command(q[:26])
            self.sim.opt.gravity.zero_();self.sim.forward()
            pose=wp.array(np.asarray(object_pose,np.float32),dtype=float,device=self.device)
            original_substeps=self.sim.substeps;self.sim.substeps=1
            # Rebuild only the step graph for one-physics-step pinning.
            self.sim._graphs.pop('step',None)
            for frame in range(frames):
                self.command(q[:26]+(hand-q[:26])*(frame+1)/max(frames,1))
                for _ in range(self.substeps):
                    wp.launch(pin_object,dim=7,inputs=[self.sim.data.qpos,self.sim.data.qvel,pose],device=self.device)
                    self.sim.step()
                    wp.launch(pin_object,dim=7,inputs=[self.sim.data.qpos,self.sim.data.qvel,pose],device=self.device)
            self.sim.substeps=original_substeps;self.sim._graphs.pop('step',None)
            self.sim.opt.gravity.assign(np.asarray(self.model.opt.gravity,np.float32).reshape(1,3))
            self.sim.forward();wp.synchronize_stream(self.sim.stream)
        return self.snapshot()

    def advance(self,controls):
        controls=np.asarray(controls,np.float32)
        if controls.ndim!=2 or controls.shape[1]!=26 or not len(controls):raise ValueError('Controls must have nonempty shape [T,26]')
        n=len(controls)
        with wp.ScopedDevice(self.device),wp.ScopedStream(self.sim.stream):
            force=wp.zeros((n,5),dtype=float,device=self.device);depth=wp.zeros(n,dtype=float,device=self.device)
            qs=[];vs=[]
            for slot,value in enumerate(controls):
                self.command(value);self.sim.step()
                # Forces are MJWarp elliptic-constraint normal forces from the
                # final physics substep of each control interval.
                d=self.sim._d
                wp.launch(collect_contact,dim=d.contact.dist.shape[0],inputs=[d.nacon,d.contact.geom,d.contact.dist,d.contact.efc_address,d.efc.force,self.owners,self.model.geom('tracked_object_geom').id,slot,force,depth],device=self.device)
                wp.synchronize_stream(self.sim.stream)
                qs.append(self.sim.data.qpos.numpy()[0].copy());vs.append(self.sim.data.qvel.numpy()[0].copy())
            return np.asarray(qs),np.asarray(vs),force.numpy(),depth.numpy()

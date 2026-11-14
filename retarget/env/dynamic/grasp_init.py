"""Common GPU sampling initialization; object pinned only before the episode."""
from dataclasses import asdict
import time
import mujoco,numpy as np,warp as wp
from robokit.sim import MjWarpSim
from .rollout import contact_cost,CostConfig


@wp.kernel
def pin(q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),pose:wp.array(dtype=float)):
    w,i=wp.tid()
    if i<7:q[w,26+i]=pose[i]
    if i<6:v[w,26+i]=0.


@wp.kernel
def depth_kernel(count:wp.array(dtype=int),world:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),distance:wp.array(dtype=float),obj:int,out:wp.array(dtype=float)):
    i=wp.tid()
    if i<count[0] and (geom[i][0]==obj or geom[i][1]==obj):wp.atomic_max(out,world[i],wp.max(0.,-distance[i]))


def establish_sampled(model,state,reference,*,samples=128,steps=90,seed=42,device='cuda:0'):
    if not str(device).startswith('cuda'):raise ValueError('CUDA MJWarp required')
    started=time.perf_counter();rng=np.random.default_rng(seed);n=samples+1
    controls=np.repeat(reference[None],n,axis=0)
    noise=rng.normal(size=(samples,26))*np.r_[np.full(3,.003),np.full(3,.025),np.tile([.12,.035,.16,.16],5)]
    noise[:,6:]+=np.tile([.10,0.,.12,.12],5)
    controls[1:]+=noise;controls=np.clip(controls,*model.actuator_ctrlrange.T).astype(np.float32)
    with wp.ScopedDevice(device):
        sim=MjWarpSim(model,n,state,substeps=1,nconmax=128,njmax=768,device=device)
        with wp.ScopedStream(sim.stream):
            sim.opt.gravity.zero_();sim.data.ctrl.assign(controls)
            pose=wp.array(state.qpos[26:33].astype(np.float32),dtype=float,device=device)
            sim.forward()
            for _ in range(steps):
                wp.launch(pin,dim=(n,7),inputs=[sim.data.qpos,sim.data.qvel,pose]);sim.step()
                wp.launch(pin,dim=(n,7),inputs=[sim.data.qpos,sim.data.qvel,pose])
            owners=[]
            for b in model.geom_bodyid:
                b=int(b);name=model.body(b).name
                while not name.startswith('right_finger') and b>0:b=int(model.body_parentid[b]);name=model.body(b).name
                owners.append(int(name[len('right_finger')])-1 if name.startswith('right_finger') else -1)
            force=wp.zeros((n,5),dtype=float,device=device);cost=wp.zeros(n,dtype=float,device=device);depth=wp.zeros(n,dtype=float,device=device)
            weights=wp.array(list(asdict(CostConfig()).values()),dtype=float,device=device);owner=wp.array(owners,dtype=int,device=device)
            d=sim._d;obj=model.geom('tracked_object_geom').id
            wp.launch(contact_cost,dim=d.contact.dist.shape[0],inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.dist,d.contact.efc_address,d.efc.force,obj,owner,force,weights,1,cost])
            wp.launch(depth_kernel,dim=d.contact.dist.shape[0],inputs=[d.nacon,d.contact.worldid,d.contact.geom,d.contact.dist,obj,depth])
            wp.synchronize_stream(sim.stream)
            fs=force.numpy();ps=depth.numpy();q=sim.data.qpos.numpy();v=sim.data.qvel.numpy()
            squeeze=np.sort(fs,axis=1)[:,-2]
            score=100*np.maximum(1.2-squeeze,0)**2+1e7*np.maximum(ps-.005,0)**2+cost.numpy()+.001*np.square((controls-reference)/np.r_[np.full(3,.01),np.full(3,.1),np.full(20,.2)]).mean(1)
            valid=np.isfinite(q).all(1)&np.isfinite(v).all(1)&np.isfinite(score)
            score[~valid]=np.inf
            if not valid.any():raise RuntimeError('No finite grasp initialization candidate')
            best=int(score.argmin());result=mujoco.MjData(model)
            for name in ('qpos','qvel','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied'):
                value=getattr(d,name).numpy()[best];getattr(result,name)[:]=value
            result.time=float(d.time.numpy()[best])
    return result,controls[best],dict(seconds=time.perf_counter()-started,samples=n,physics_steps=steps,seed=seed,selected=best,initial_two_finger_load_N=float(squeeze[best]),initial_maximum_penetration_mm=float(ps[best]*1000),finger_force_N=fs[best].tolist(),grasp_established=bool(squeeze[best]>=1 and ps[best]<=.005),backend='MJWarp pinned-object sampling; object released for both controllers')

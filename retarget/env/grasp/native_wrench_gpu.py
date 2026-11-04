"""CUDA original six-force trials plus the authorized six pure-torque trials."""
import torch


def slerp(a,b,t):
    a=a/a.norm(dim=-1,keepdim=True);b=b/b.norm(dim=-1,keepdim=True)
    dot=(a*b).sum(-1,keepdim=True);b=torch.where(dot<0,-b,b);dot=dot.abs().clamp(0,1)
    angle=torch.acos(dot);sine=torch.sin(angle).clamp_min(1e-12)
    value=(torch.sin((1-t)*angle)*a+torch.sin(t*angle)*b)/sine
    linear=(1-t)*a+t*b
    value=torch.where(dot>.999999,linear,value)
    return value/value.norm(dim=-1,keepdim=True)


def original_displacement(reference,current):
    translation=(reference[:,:3]-current[:,:3]).norm(dim=-1)
    # Match native np_get_delta_pose, including its quaternion-sign convention.
    inverse=reference[:,3:]/reference[:,3:].square().sum(-1,keepdim=True)
    scalar=(inverse*current[:,3:]).sum(-1)
    scalar=torch.where(scalar.abs()>1,torch.ones_like(scalar),scalar)
    return translation,torch.rad2deg(2*torch.acos(scalar))


def torque_displacement(reference,current):
    translation=(reference[:,:3]-current[:,:3]).norm(dim=-1)
    a=reference[:,3:]/reference[:,3:].norm(dim=-1,keepdim=True)
    b=current[:,3:]/current[:,3:].norm(dim=-1,keepdim=True)
    angle=torch.rad2deg(2*torch.acos((a*b).sum(-1).abs().clamp(0,1)))
    return translation,angle


def evaluate(physics,grasp,squeeze,pregrasp,pregrasp_count,object_pose,radius,eligible,*,object_body):
    batch=len(grasp);device=grasp.device
    if physics.worlds!=12*batch:raise ValueError('Allocate twelve independent trial worlds per input')
    if physics.cpu.opt.noslip_iterations!=2:raise ValueError('Original evaluation requires exactly two NoSlip iterations')
    repeat=lambda x:x.repeat_interleave(12,0)
    g,s,pre,n,obj,rad=map(repeat,(grasp,squeeze,pregrasp,pregrasp_count,object_pose,radius))
    valid=repeat(eligible);worlds=len(g)
    first=torch.where((n>0)[:,None],pre[:,0],g)
    first=torch.where(torch.isfinite(first),first,g)
    physics.reset(torch.cat((first,obj),-1))
    max_segments=int(n.max())+1
    row=torch.arange(worlds,device=device)
    for segment in range(max_segments):
        a=pre[row,torch.full_like(n,segment).minimum((n-1).clamp_min(0))]
        b=pre[row,torch.full_like(n,segment+1).minimum((n-1).clamp_min(0))]
        a=torch.where((segment>=n)[:,None],g,a)
        b=torch.where((segment>=n-1)[:,None],g,b)
        b=torch.where((segment>=n)[:,None],s,b)
        mask=valid&(segment<=n)
        for step in range(1,6):
            t=step/5;control=(1-t)*a+t*b
            control[:,3:7]=slerp(a[:,3:7],b[:,3:7],t)
            physics.set_ctrl(control);physics.advance(5,mask=mask)
    current=torch.cat((physics.tensor('xpos')[:,object_body],physics.tensor('xquat')[:,object_body]),-1)
    preload_translation,preload_rotation=torque_displacement(obj,current)
    preload_ok=(preload_translation<.05)&(preload_rotation<15.)
    trial=torch.arange(worlds,device=device)%12;torque_trial=trial>=6
    axes=torch.tensor([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]],device=device)
    direction=axes[trial%6]
    mass=torch.as_tensor(physics.cpu.body_subtreemass[object_body],device=device,dtype=torch.float32).expand(worlds)
    force_n=mass*9.8;torque_nm=force_n*rad
    force=direction*force_n[:,None]*(~torque_trial)[:,None]
    bodies=torch.tensor([object_body],device=device,dtype=torch.long)
    point=physics.tensor('subtree_com')[:,object_body,None]
    physics.apply_point_forces(bodies,point,force[:,None])
    physics.tensor('xfrc_applied')[:,object_body,3:]=direction*torque_nm[:,None]*torque_trial[:,None]
    active=valid&torch.where(torque_trial,preload_ok,torch.ones_like(valid))
    passed=active.clone();last_translation=preload_translation;last_rotation=preload_rotation
    simulated=torch.zeros(worlds,device=device,dtype=torch.int32)
    for _ in range(10):
        physics.advance(50,mask=active);simulated+=50*active.to(torch.int32)
        current=torch.cat((physics.tensor('xpos')[:,object_body],physics.tensor('xquat')[:,object_body]),-1)
        dt,dr=original_displacement(obj,current);tt,tr=torque_displacement(obj,current)
        dt=torch.where(torque_trial,tt,dt);dr=torch.where(torque_trial,tr,dr)
        last_translation=torch.where(active,dt,last_translation);last_rotation=torch.where(active,dr,last_rotation)
        good=torch.isfinite(dt)&torch.isfinite(dr)&(dt<.05)&(dr<15.)&~physics.overflow()
        passed&=~active|good;active&=good
    flags=passed.reshape(batch,12)
    return dict(native_gsr_pass=flags[:,:6].all(-1),torque_pass=flags[:,6:].all(-1),combined_pass=flags.all(-1),
        directions_pass=flags,preload_translation_m=preload_translation.reshape(batch,12),
        preload_rotation_deg=preload_rotation.reshape(batch,12),translation_m=last_translation.reshape(batch,12),
        rotation_deg=last_rotation.reshape(batch,12),simulated_steps=simulated.reshape(batch,12),
        torque_nm=torque_nm.reshape(batch,12),force_n=force_n.reshape(batch,12))

#!/usr/bin/env python3
"""Small physics-refinement experiment using Dex-Supermimic's PPO runner.

Run with Dex-Supermimic/.venv/bin/python. Does not modify upstream task defaults.
This is a privileged simulation grasp-refinement experiment, not a deployed
human-history-only mapper. Exported grasps require independent mesh validation.
"""
from __future__ import annotations
import argparse,json,math,time
from functools import partial
from pathlib import Path
import numpy as np
import torch
import mujoco


def primitive_spec(radius,height,kind,mass):
    from dex_supermimic.assets.inhand_cube import _get_spec
    spec=_get_spec()
    typ=mujoco.mjtGeom.mjGEOM_SPHERE if kind=='sphere' else mujoco.mjtGeom.mjGEOM_CYLINDER
    size=[radius,0,0] if kind=='sphere' else [radius,height/2,0]
    for name in ('cube','cube_visual'):
        geom=spec.geom(name);geom.type=typ;geom.size=size
    geom=spec.geom('cube');geom.mass=mass;geom.friction=[.8,.005,.0001]
    return spec


def native_collision_spec(base_fn):
    spec=base_fn()
    # Reverse only the task's proximal capsule overlay; retain its actuators,
    # tip sites and calibration tag. The underlying Wuji2 meshes are unchanged.
    disabled=[g for g in spec.geoms if g.name.endswith('_col_mesh_disabled')]
    for mesh in disabled:
        original=mesh.name.removesuffix('_mesh_disabled')
        capsule=spec.geom(original);capsule.name=original+'_capsule_disabled'
        capsule.contype=0;capsule.conaffinity=0
        mesh.name=original;mesh.contype=1;mesh.conaffinity=1;mesh.friction=[.8,.005,.0001]
    # The overlay deletes the native adjacent-link exclusions because its L1
    # capsules are disabled. Restoring L1 meshes also requires restoring these
    # exclusions, otherwise the palm collides with finger hinges by ~14mm.
    from dex_supermimic.assets.wuji_hand2 import _get_spec,wuji_hand2_xml_path
    native=_get_spec(wuji_hand2_xml_path())
    while spec.excludes:spec.delete(spec.excludes[0])
    for exclusion in native.excludes:
        spec.add_exclude(bodyname1=exclusion.bodyname1,bodyname2=exclusion.bodyname2)
    return spec


def grasp_quantities(env):
    obj=env.scene['object'];data=env.scene['grasp_contacts'].data
    found=(data.found>0).float()
    normal=data.normal; tangent=data.tangent
    bitangent=torch.cross(normal,tangent,dim=-1)
    theta=torch.arange(8,device=env.device)*math.pi/4
    force=normal[:,:,None,:]+.8*(theta.cos()[None,None,:,None]*tangent[:,:,None,:]+
                                theta.sin()[None,None,:,None]*bitangent[:,:,None,:])
    lever=(data.pos-obj.data.root_link_pos_w[:,None,:])/env.primitive_radius
    torque=torch.cross(lever[:,:,None,:].expand_as(force),force,dim=-1)
    G=torch.cat((force,torque),-1).flatten(1,2).transpose(1,2)
    mask=found[:,:,None].expand(-1,-1,8).flatten(1)
    G=G*mask[:,None,:]
    denom=mask.sum(-1,keepdim=True).clamp_min(1)
    weights=mask/denom
    # Positive, normalized contact-cone weights. This is a dense proxy,
    # not a certificate; final CPU LP checks the exact finite cone.
    for _ in range(12):
        residual=(G@weights[:,:,None]).squeeze(-1)
        gradient=(G.transpose(1,2)@residual[:,:,None]).squeeze(-1)
        weights=weights*torch.exp((-gradient*.2).clamp(-5,5))+.001*mask/denom
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    balance=torch.linalg.vector_norm((G@weights[:,:,None]).squeeze(-1),dim=-1)
    gram=(G@G.transpose(1,2))/denom[:,:,None]
    smallest=torch.linalg.eigvalsh(gram).clamp_min(0)[:,0].sqrt()
    quality=torch.exp(-balance/.2)*(smallest/.1).clamp(0,1)*(found.sum(-1)>=3)
    reference=obj.data.default_root_state[:,:3]+env.scene.env_origins
    displacement=torch.linalg.vector_norm(obj.data.root_link_pos_w-reference,dim=-1)
    penetration=(-data.dist.squeeze(-1)).clamp_min(0).amax(-1)
    return quality,displacement,found.sum(-1),penetration,balance


def grasp_reward(env):
    quality,distance,count,penetration,balance=grasp_quantities(env)
    velocity=torch.linalg.vector_norm(env.scene['object'].data.root_link_lin_vel_w,dim=-1)
    # The object is floating in air; there is no table support or pose tether.
    near=torch.exp(-(distance/.025).square())
    reward=10*near+4*quality+near*count.clamp_max(6)-2*velocity.clamp_max(3)
    reward-=20*(penetration-.002).clamp_min(0)/.01
    return reward


def disturbance_pulse(env,env_ids):
    body=int(env.scene['object'].indexing.body_ids[0])
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    env.sim.data.xfrc_applied[ids,body,:]=0
    phase=env.episode_length_buf[ids]
    direction=torch.eye(3,device=env.device)[ids%3]
    direction*=torch.where(ids%2==0,1.,-1.)[:,None]
    active=((phase>=50)&(phase<60)).float()
    env.sim.data.xfrc_applied[ids,body,:3]=direction*active[:,None]*env.primitive_mass*9.81*.5


def make_config(args,z):
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    from dex_supermimic.tasks.dex_reorient.config.recipe import configure_wuji_ppo
    from mjlab.entity import EntityCfg
    from mjlab.managers import EventTermCfg,RewardTermCfg
    from mjlab.envs.mdp.events import reset_scene_to_default
    from mjlab.sensor import ContactSensorCfg,ContactMatch
    from dex_supermimic.assets.wuji_hand2 import WUJI_HAND2_TASK_NAMES
    cfg=make_env_cfg(args.num_envs,50,play=True,task='hand2-reorient')
    configure_wuji_ppo(cfg,50)
    cfg.scene.num_envs=args.num_envs;cfg.scene.env_spacing=.6
    cfg.scene.terrain=None;cfg.curriculum.clear();cfg.metrics.clear()
    cfg.episode_length_s=2.;cfg.auto_reset=True
    if args.native_collision:
        cfg.sim.mujoco.timestep=.002
        cfg.decimation=10
        cfg.sim.mujoco.iterations=50
        cfg.sim.nconmax=512
        cfg.sim.njmax=3000
    cfg.events={'reset_scene_to_default':EventTermCfg(func=reset_scene_to_default,mode='reset')}
    cfg.events['training_force_pulse']=EventTermCfg(func=disturbance_pulse,mode='interval',interval_range_s=(.02,.02))
    base=z['target_base_pose'][args.row].copy();obj=z['object_pose_world'][args.row].copy()
    base[2]+=1.;obj[2]+=1.
    q=z['target_qpos'][args.row]
    robot=cfg.scene.entities['robot']
    if args.native_collision:robot.spec_fn=partial(native_collision_spec,robot.spec_fn)
    robot.init_state=EntityCfg.InitialStateCfg(pos=tuple(base[:3]),rot=tuple(base[3:]),
        joint_pos={WUJI_HAND2_TASK_NAMES['joint'][str(n)]:float(v) for n,v in zip(z['joint_names'],q)},joint_vel={'.*':0.})
    robot.articulation.soft_joint_pos_limit_factor=1.
    obj_cfg=cfg.scene.entities['object']
    obj_cfg.init_state=EntityCfg.InitialStateCfg(pos=tuple(obj[:3]),rot=tuple(obj[3:]))
    obj_cfg.spec_fn=partial(primitive_spec,args.radius,args.height,args.kind,args.mass)
    cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='grasp_contacts',
        primary=ContactMatch(mode='body',pattern=('.*_palm_link','.*_finger[1-5]_link[1-4]','.*_finger[1-5]_tip_sensor'),entity='robot'),
        secondary=ContactMatch(mode='body',pattern='cube',entity='object'),
        fields=('found','pos','normal','tangent','dist'),reduce='mindist',num_slots=2))
    for term in cfg.rewards.values():term.weight=0.
    # Keep cage_escape active because the upstream SimReader reads its config.
    cfg.rewards['cage_escape'].weight=-1.
    cfg.rewards['hand_pose'].weight=-1.
    cfg.rewards['torque'].weight=-.01
    cfg.rewards['finger_collision'].weight=-.2
    cfg.rewards['grasp_quality']=RewardTermCfg(func=grasp_reward,weight=1.)
    cfg.actions['joint_pos'].action_scale=.25
    cfg.actions['joint_pos'].warmup_time_s=0.
    cfg.commands['reorient_command'].debug_vis=False
    return cfg


@torch.no_grad()
def evaluate(env,policy,steps=100):
    env.cfg.auto_reset=False
    drop_cfg=env.termination_manager.get_term_cfg('cage_drop')
    old_drop=drop_cfg.params['max_outside_steps']
    drop_cfg.params['max_outside_steps']=1000000
    obs=env.reset()[0]['policy'];env._manual_reset_pending.zero_()
    history=[];contact=[];nan=False
    objbody=int(env.scene['object'].indexing.body_ids[0])
    for k in range(steps):
        action=torch.zeros((env.num_envs,20),device=env.device) if policy is None else policy(obs)
        # Unassisted hold, then a modest 0.5g lateral pulse in +/- XYZ directions.
        env.sim.data.xfrc_applied[:,objbody,:]=0
        if 50<=k<60:
            direction=torch.eye(3,device=env.device)[torch.arange(env.num_envs,device=env.device)%3]
            direction*=torch.where(torch.arange(env.num_envs,device=env.device)%2==0,1.,-1.)[:,None]
            env.sim.data.xfrc_applied[:,objbody,:3]=direction*env.primitive_mass*9.81*.5
        ob,rew,terminated,truncated,_=env.step(action);obs=ob['policy']
        quality,distance,count,pen,balance=grasp_quantities(env)
        history.append(distance.clone());contact.append(count.clone())
        nan=nan or not bool(torch.isfinite(obs).all())
        if nan:break
    env.sim.data.xfrc_applied[:,objbody,:]=0
    drop_cfg.params['max_outside_steps']=old_drop
    distance=torch.stack(history);counts=torch.stack(contact)
    q=env.scene['robot'].data.joint_pos.clone()
    base=env.scene['robot'].data.root_link_pose_w.clone();base[:,:3]-=env.scene.env_origins
    obj=env.scene['object'].data.root_link_pose_w.clone();obj[:,:3]-=env.scene.env_origins
    base[:,2]-=1.;obj[:,2]-=1.
    return dict(hold_within_30mm_rate=float((distance.amax(0)<.03).float().mean()),
                mean_final_displacement_mm=float(distance[-1].mean()*1000),
                mean_contacts_last_10_steps=float(counts[-10:].mean()),
                mean_wrench_proxy=float(quality.mean()),finite=not nan,
                target_qpos=q.cpu().numpy(),target_base_pose=base.cpu().numpy(),object_pose_world=obj.cpu().numpy(),
                control_target_qpos=env.action_manager.get_term('joint_pos').processed_action.cpu().numpy(),
                actuator_force=env.scene['robot'].data.actuator_force.cpu().numpy(),
                max_displacement_m=distance.amax(0).cpu().numpy(),
                hold_pass=(distance.amax(0)<.03).cpu().numpy(),
                contact_pos=env.scene['grasp_contacts'].data.pos.cpu().numpy()-env.scene.env_origins.cpu().numpy()[:,None,:]-np.array([0,0,1]),
                contact_normal=env.scene['grasp_contacts'].data.normal.cpu().numpy(),
                contact_tangent=env.scene['grasp_contacts'].data.tangent.cpu().numpy(),
                contact_found=env.scene['grasp_contacts'].data.found.cpu().numpy())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pose-file',type=Path,required=True);ap.add_argument('--row',type=int,default=27)
    ap.add_argument('--kind',choices=('sphere','disk','cylinder'),default='disk')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=128)
    ap.add_argument('--iterations',type=int,default=150);ap.add_argument('--mass',type=float,default=.04)
    ap.add_argument('--native-collision',action='store_true')
    ap.add_argument('--actor-checkpoint',type=Path)
    ap.add_argument('--evaluate-only',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    z=np.load(args.pose_file,allow_pickle=False);args.radius=float(z['radius_m'][args.row]);args.height=float(z['height_m'][args.row])
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    wp.init();wp.set_device('cuda:0');torch.manual_seed(42);np.random.seed(42)
    env=ManagerBasedRlEnv(make_config(args,z),device='cuda:0')
    env.primitive_radius=args.radius;env.primitive_mass=args.mass
    env.reset()
    from dex_supermimic.assets.wuji_hand2 import WUJI_HAND2_TASK_NAMES
    reference={WUJI_HAND2_TASK_NAMES['joint'][n]:v for n,v in zip(z['joint_names'].tolist(),z['target_qpos'][args.row].tolist())}
    expected=np.array([reference[n] for n in env.scene['robot'].joint_names])
    actual=env.scene['robot'].data.joint_pos[0].cpu().numpy()
    if not np.allclose(expected,actual,atol=1e-5):
        raise ValueError(f'Initial pose was not imported exactly: {np.max(abs(expected-actual))}')
    cfg=DexSuperMimicCfg(task='hand2-reorient',algorithm='ppo',num_envs=args.num_envs,
        max_iterations=args.iterations,logger='none',compile=False,compile_policy=False,amp=False,
        wm_hidden=(32,32),ppo_rollout_steps=32,ppo_epochs=3,ppo_minibatches=4,
        ppo_init_std=.2,ppo_std_min=.03,ppo_lr=3e-4,log_interval=10,save_interval=max(1,args.iterations//2))
    env.max_common_steps=cfg.max_iterations*cfg.ppo_rollout_steps
    # Evaluate identical reset states before and after training.
    initial=evaluate(env,None)
    metadata={k:v for k,v in vars(args).items() if k!='output'}
    metadata={k:str(v) if isinstance(v,Path) else v for k,v in metadata.items()}
    metadata.update(experiment='primitive_grasp_refinement',runner='DexSuperMimicRunner PPO',
                    privileged_simulation_policy=True,world_model_training=False,
                    action='retargeted_qpos + clipped EMA residual, scale .25 rad',
                    contact_reward='normalized friction-cone wrench-balance and rank proxy, not a closure certificate',
                    gravity=9.81,terrain=False,object_pose_tether=False,friction=.8)
    (args.output/'experiment.json').write_text(json.dumps(metadata,indent=2))
    print('BASELINE',json.dumps({k:v for k,v in initial.items() if not isinstance(v,np.ndarray)}),flush=True)
    if not initial['finite']:
        (args.output/'failure.json').write_text(json.dumps(dict(status='failed',reason='Non-finite physical baseline; PPO was not started.')))
        env.close()
        raise RuntimeError('Non-finite physical baseline; refusing to train on invalid states')
    runner=None
    try:
        env.cfg.auto_reset=True
        runner=DexSuperMimicRunner(env,cfg,args.output/'training')
        with torch.no_grad():
            runner.actor.net[-1].weight[:20].zero_()
            runner.actor.net[-1].bias[:20].zero_()
        if args.actor_checkpoint:
            state=torch.load(args.actor_checkpoint,map_location='cuda:0',weights_only=False)
            runner.actor.load_state_dict(state['modules']['actor'])
        if not args.evaluate_only:runner.learn()
        final=evaluate(env,runner.actor.mean)
        summary=dict(baseline={k:v for k,v in initial.items() if not isinstance(v,np.ndarray)},
                     learned={k:v for k,v in final.items() if not isinstance(v,np.ndarray)})
        for name,result in (('baseline',initial),('learned',final)):
            np.savez_compressed(args.output/f'{name}-evaluation.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)},
                                joint_names=np.array(runner.reader.robot.joint_names))
        (args.output/'evaluation.json').write_text(json.dumps(summary,indent=2))
        print('EVALUATION',json.dumps(summary),flush=True)
    finally:
        if runner is not None:runner.close()
        env.close()


if __name__=='__main__':main()

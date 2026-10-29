#!/usr/bin/env python3
"""Execute pinned SPIDER model/reward functions in isolation and audit parity."""
import argparse,ast,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.humanoid import HumanoidProblem,Settings
from run_humanoid import task_paths


def source_functions():
    path=ROOT/'data/spider/source/spider/math.py'
    spec=importlib.util.spec_from_file_location('spider_math_audit',path)
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    text=(ROOT/'data/spider/source/spider/simulators/mjwp.py').read_text()
    names={'setup_mj_model','_weight_diff_qpos','_diff_qpos','get_reward','get_terminal_reward'}
    tree=ast.parse(text);module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    scope=dict(mujoco=mujoco,torch=torch,np=np,Config=SimpleNamespace,MJWPEnv=SimpleNamespace,quat_sub=math.quat_sub,
               wp=SimpleNamespace(to_torch=lambda x:x))
    exec(compile(module,str(ROOT/'data/spider/source/spider/simulators/mjwp.py'),'exec'),scope)
    return scope


def array_difference(first,second):
    if first.shape!=second.shape:return dict(equal=False,first_shape=list(first.shape),second_shape=list(second.shape))
    return dict(equal=bool(np.array_equal(first,second)),max_abs=float(np.max(np.abs(first-second))) if first.size else 0.)


def compare(output):
    source=source_functions();results=[]
    for task_name,override,embodiment in [('humanoid','humanoid','humanoid'),('humanoid-object','humanoid_object','humanoid_object')]:
        scene,reference=task_paths(ROOT/'data/spider',task_name)
        cfg=yaml.safe_load((ROOT/f'data/spider/source/examples/config/override/{override}.yaml').read_text())
        cfg.update(model_path=str(scene),device='cpu',contact_site_ids=[],nv=35 if task_name=='humanoid' else 41,
                   nu=29,nq_obj=0,base_pos_rew_scale=cfg.get('base_pos_rew_scale',1.),base_rot_rew_scale=cfg.get('base_rot_rew_scale',.3),terminal_rew_scale=1.)
        upstream=source['setup_mj_model'](SimpleNamespace(**cfg))
        ours=HumanoidProblem(scene,reference,settings=Settings())
        fields=('body_mass','body_inertia','body_ipos','body_iquat','dof_armature','dof_damping','dof_frictionloss',
                'jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange',
                'pair_geom1','pair_geom2','pair_friction','pair_solref','pair_solimp','geom_type','geom_pos','geom_size')
        parity={name:array_difference(np.asarray(getattr(upstream,name)),np.asarray(getattr(ours.model,name))) for name in fields}
        options={name:dict(spider=np.asarray(getattr(upstream.opt,name)).tolist(),ours=np.asarray(getattr(ours.model.opt,name)).tolist()) for name in ('timestep','iterations','ls_iterations','integrator','gravity','disableflags','enableflags','o_solref','o_solimp')}
        raw=np.load(reference)
        q=raw['qpos'][2:7].astype(np.float64).copy();ref=q.copy()
        q[:,:3]+=np.array([.03,-.02,.01]);q[:,7:36]+=.02
        v=raw['qvel'][2:7].astype(np.float64)
        # Public examples omit contact arrays: evaluate the actual source pose
        # reward separately rather than inventing original contact labels.
        cfg['contact_rew_scale']=0.;c=SimpleNamespace(**cfg)
        env=SimpleNamespace(data_wp=SimpleNamespace(qpos=torch.tensor(q,dtype=torch.float64),qvel=torch.tensor(v,dtype=torch.float64)))
        refs=(torch.tensor(ref[0]),torch.tensor(v[0]),None,None,None)
        reward=source['get_reward'](c,env,refs)
        if isinstance(reward,tuple):reward=reward[0]
        delta=np.empty(upstream.nv);manual=[]
        for actual in q:
            mujoco.mj_differentiatePos(upstream,delta,1.,ref[0],actual)
            w=source['_weight_diff_qpos'](c).numpy();manual.append(np.linalg.norm(w*delta))
        source_cost=-reward.numpy();pose_error=float(np.max(np.abs(source_cost-manual)))
        assert pose_error<2e-6,(task_name,pose_error)
        record=dict(task=task_name,scene=str(scene),scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                    structural_parity=parity,options=options,all_structural_fields_equal=all(v['equal'] for v in parity.values()),
                    source_cost='norm(weight * manifold_qpos_difference, 2) + velocity_scale*norm(qvel_difference,2) + sum(masked site distances) when contact_rew_scale>0 and sites are provided; contact_rew_scale gates the branch rather than multiplying its value',
                    source_has_explicit_com_or_upright_cost=False,source_pose_reward_numerical_max_error=pose_error,
                    source_pose_cost_examples=source_cost.tolist(),source_pose_weights=source['_weight_diff_qpos'](c).tolist(),
                    override=cfg,raw_reference_keys=raw.files,
                    raw_contact_data_available=all(k in raw.files for k in ('contact','contact_pos')),
                    source_default_contact_reward=float(yaml.safe_load((ROOT/f'data/spider/source/examples/config/override/{override}.yaml').read_text()).get('contact_rew_scale',0.)),
                    task_metadata=json.loads((scene.parent/'task_info.json').read_text()) if (scene.parent/'task_info.json').exists() else None,
                    remaining_or_deliberate_differences=['both use MJWarp implicitfast; ADMM uses finite-difference action gradients',
                      'whole-body balance/relative-keypoint/realized-wrench extension; object pose/velocity removed by user request',
                      'AdamW gradients and joint contact ADMM instead of SPIDER sampling',
                      'both simulate the original MJCF collision shapes and explicit contact pairs',
                      'MPC currently replans each reference frame (1/30s), SPIDER ctrl_dt=0.1s',
                      'local reference uses SLERP for quaternion targets; SPIDER interpolates raw channels with align_corners=True',
                      'reference dt is currently 1/30s; raw AMASS task metadata is recorded separately',
                      'SPIDER setup_env advances the initial state by one CPU physics step; local initialization starts exactly at reference qpos/qvel',
                      'SPIDER starts dense reference controls plus interpolated noise; ADMM interpolates absolute control knots with previous-control boundary',
                      'SPIDER scores post-step states; SPIDER cost mode excludes the fixed initial stage and uses terminal weight 1'])
        published_path=reference.parent/'trajectory_mjwp.npz'
        if published_path.exists():
            published=np.load(published_path)
            states=published['qpos'].reshape(-1,upstream.nq)
            times=published['time'].reshape(-1)
            factor=int(round(cfg['ref_dt']/cfg['sim_dt']))
            source_grid=np.minimum(np.arange(len(states)),len(raw['qpos'])*factor-1)*(len(raw['qpos'])-1)/(len(raw['qpos'])*factor-1)
            dense_ref=np.stack([np.interp(source_grid,np.arange(len(raw['qpos'])),raw['qpos'][:,j]) for j in range(upstream.nq)],axis=1)
            record['published_state_audit']=dict(physics_steps=len(states),recorded_dt=float(np.median(np.diff(times))),
                root_position_rms_m=float(np.sqrt(np.mean(np.sum((states[:,:3]-dense_ref[:,:3])**2,axis=1)))),
                root_height_min_m=float(states[:,2].min()),
                alignment='Original interp align_corners=True factor, last frame padding; published states, not rerun optimizer or matched compute budget')
        if task_name=='humanoid':
            legacy=mujoco.MjModel.from_xml_path(str(ROOT/'data/spider/processed/amass/assets/robots/unitree_g1/scene.xml'))
            record['previous_wrong_scene_differences']={name:array_difference(np.asarray(getattr(legacy,name)),np.asarray(getattr(upstream,name))) for name in fields}
            record['gain_ranges']=dict(previous=[float(legacy.actuator_gainprm[:,0].min()),float(legacy.actuator_gainprm[:,0].max())],task=[float(upstream.actuator_gainprm[:,0].min()),float(upstream.actuator_gainprm[:,0].max())])
            record['armature_ranges']=dict(previous=[float(legacy.dof_armature[6:].min()),float(legacy.dof_armature[6:].max())],task=[float(upstream.dof_armature[6:].min()),float(upstream.dof_armature[6:].max())])
        assert record['all_structural_fields_equal'],task_name
        results.append(record)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(results,indent=2)+'\n')
    for r in results:print(r['task'],'scene parity',r['all_structural_fields_equal'],'source reward numerical error',r['source_pose_reward_numerical_max_error'])
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/humanoid/spider_comparison.json');args=p.parse_args();compare(args.output)

#!/usr/bin/env python3
"""Compare frozen MuJoCo 3.3.3 virtual-force steps to CUDA mjwarp operators."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,hashlib,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(a):
    import mujoco,torch,warp as wp
    from retarget.env.grasp.native_mjwarp import NativePhysics
    root=a.root.resolve();cpu=mujoco.MjModel.from_xml_path(str(root/'reference_opt.xml'))
    metadata=json.loads((root/'reference_source.json').read_text())
    for category,flag_values in metadata['flags'].items():
        own=mujoco.mjtDisableBit if category=='disable' else mujoco.mjtEnableBit
        opposite=mujoco.mjtEnableBit if category=='disable' else mujoco.mjtDisableBit
        prefix='mjDSBL_' if category=='disable' else 'mjENBL_'
        otherprefix='mjENBL_' if category=='disable' else 'mjDSBL_'
        attr='disableflags' if category=='disable' else 'enableflags'
        otherattr='enableflags' if category=='disable' else 'disableflags'
        for name,enabled in flag_values.items():
            if hasattr(own,prefix+name):
                bit=int(getattr(own,prefix+name));value=int(getattr(cpu.opt,attr))
                setattr(cpu.opt,attr,(value|bit) if enabled else (value&~bit))
            elif hasattr(opposite,otherprefix+name):
                bit=int(getattr(opposite,otherprefix+name));value=int(getattr(cpu.opt,otherattr))
                setattr(cpu.opt,otherattr,(value&~bit) if enabled else (value|bit))
            elif enabled:raise ValueError('An active original option flag is unavailable: '+name)
    for name,value in metadata['numeric_options'].items():
        if not hasattr(cpu.opt,name):raise ValueError('Missing original option: '+name)
        if isinstance(value,list):getattr(cpu.opt,name)[:]=value
        else:setattr(cpu.opt,name,value)
    frozen=dict(np.load(root/'reference_model_all_arrays.npz'));differences={};unmatched={}
    physical_fields={'body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat','body_gravcomp',
        'jnt_axis','jnt_pos','jnt_range','jnt_margin','jnt_solimp','jnt_solref','jnt_stiffness',
        'geom_pos','geom_quat','geom_size','geom_solimp','geom_solref','geom_margin','geom_gap',
        'geom_friction','geom_condim','geom_contype','geom_conaffinity','eq_data','eq_solimp','eq_solref',
        'dof_frictionloss','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','actuator_gear',
        'actuator_ctrlrange','actuator_forcerange','qpos0','qpos_spring'}
    for key,value in frozen.items():
        if not hasattr(cpu,key):unmatched[key]='not present';continue
        actual=getattr(cpu,key)
        if actual.shape!=value.shape:unmatched[key]=dict(reference=list(value.shape),current=list(actual.shape));continue
        differences[key]=float(np.max(np.abs(actual-value))) if actual.size else 0.
        if key not in physical_fields:continue
        actual[:]=value
    # Sparse mass indices and other compiler caches are version-specific. Rebuild
    # them from physical parameters instead of copying their numeric indices.
    cpu.body_simple[:]=0;cpu.dof_simplenum[:]=0
    cpu.body_sameframe[:]=0;cpu.geom_sameframe[:]=0
    mujoco.mj_setConst(cpu,mujoco.MjData(cpu))
    # All MANO actuators use scalar joint transmission, matching native sparse actuator_moment.
    matrix=np.zeros((cpu.nu,cpu.nv),np.float32)
    for i in range(cpu.nu):
        if cpu.actuator_trntype[i]!=mujoco.mjtTrn.mjTRN_JOINT:raise ValueError('Non-joint transmission requires an explicit native matrix')
        j=cpu.actuator_trnid[i,0];matrix[i,cpu.jnt_dofadr[j]]=cpu.actuator_gear[i,0]
    refs=dict(np.load(root/'reference_steps.npz'));device=torch.device(a.device)
    np.testing.assert_allclose(matrix,refs['native_qpos_to_ctrl'],atol=1e-12)
    cpu_data=mujoco.MjData(cpu)
    for key in ('qpos','qvel','qacc_warmstart','ctrl','mocap_pos','mocap_quat'):getattr(cpu_data,key)[:]=refs[key]
    mujoco.mj_forward(cpu,cpu_data)
    mass=np.empty((cpu.nv,cpu.nv))
    try:mujoco.mj_fullM(cpu,cpu_data,mass)
    except TypeError:mujoco.mj_fullM(cpu,mass,cpu_data.qM)
    diagnostic={name:getattr(cpu_data,name).copy() for name in ('efc_type','efc_id','efc_J','efc_R','efc_D','efc_aref','efc_pos','efc_diagApprox','qacc','qacc_smooth','qfrc_passive','qfrc_actuator') if hasattr(cpu_data,name)}
    np.savez(root/'host_dynamics_arrays.npz',mass=mass,body_invweight0=cpu.body_invweight0,**diagnostic)
    cpu_data.qfrc_applied[:]=refs['force'];cpu_data.qvel[:]=0.
    cpu_version_rows=[]
    for i in range(10):
        mujoco.mj_step(cpu,cpu_data)
        cpu_version_rows.append(dict(step=i+1,qpos_max_error=float(np.max(np.abs(cpu_data.qpos-refs[f'step_{i+1}_qpos']))),qvel_max_error=float(np.max(np.abs(cpu_data.qvel-refs[f'step_{i+1}_qvel'])))))
    if a.host_only:
        (root/'host_version_parity.json').write_text(json.dumps(dict(version=mujoco.__version__,rows=cpu_version_rows),indent=2));return
    torch.cuda.set_device(device);torch.set_num_threads(1)
    start=time.perf_counter()
    engine=NativePhysics(cpu,1,device=a.device,qpos_to_ctrl=matrix,frozen_mesh_arrays=frozen)
    t=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
    engine.reset(t(refs['qpos'])[None])
    for key in ('qvel','qacc_warmstart','ctrl','mocap_pos','mocap_quat'):engine.tensor(key).copy_(t(refs[key])[None])
    engine.forward();bodies=torch.tensor(refs['hand_bodyids'],device=device,dtype=torch.long)
    engine.apply_forces(bodies,t(refs['hand_cpn_b'])[None],t(refs['target'])[None])
    force=engine.tensor('qfrc_applied').clone()
    force_error=(force[0]-t(refs['force'])).abs().max().item()
    n=int(engine.tensor('nacon')[0]);contact=engine.data.contact
    pairs=wp.to_torch(contact.geom)[:n].cpu().numpy();distance=wp.to_torch(contact.dist)[:n].cpu().numpy()
    normals=wp.to_torch(contact.frame)[:n].reshape(-1,3,3)[:,0].cpu().numpy()
    locations=wp.to_torch(contact.pos)[:n].cpu().numpy()
    reference_pairs=refs['contact_geom'];reference_distance=refs['contact_dist'];reference_normal=refs['contact_frame'].reshape(-1,3,3)[:,0]
    pair_rows=[]
    for i,pair in enumerate(reference_pairs):
        match=np.flatnonzero(np.all(pairs==pair,axis=1));reverse=np.flatnonzero(np.all(pairs==pair[::-1],axis=1))
        if not len(match) and not len(reverse):pair_rows.append(dict(pair=pair.tolist(),missing=True));continue
        j=int(match[0] if len(match) else reverse[0]);normal=normals[j] if len(match) else -normals[j]
        pair_rows.append(dict(pair=pair.tolist(),distance_reference=float(reference_distance[i]),distance_gpu=float(distance[j]),
            normal_cosine=float(np.dot(reference_normal[i],normal)),position_error_m=float(np.linalg.norm(locations[j]-refs['contact_pos'][i]))))
    contact_diagnostic=dict(reference_count=len(reference_pairs),gpu_count=n,rows=pair_rows)
    rows=[]
    torch.cuda.synchronize(device);preparation=time.perf_counter()-start
    for i in range(10):
        start=time.perf_counter();engine.step(1);torch.cuda.synchronize(device);elapsed=time.perf_counter()-start
        q=engine.tensor('qpos')[0];v=engine.tensor('qvel')[0]
        qerror=(q-t(refs[f'step_{i+1}_qpos'])).abs();verror=(v-t(refs[f'step_{i+1}_qvel'])).abs()
        rows.append(dict(step=i+1,qpos_max_error=float(qerror.max()),qvel_max_error=float(verror.max()),wall_seconds=elapsed,overflow=bool(engine.overflow().any())))
    report=dict(stage='operator_parity',complete=True,device=a.device,cpu_reference_mujoco='3.3.3',gpu_host_mujoco=mujoco.__version__,
        wp_version=wp.__version__,torch_version=torch.__version__,model_parameter_differences_before_restore=differences,unmatched_model_arrays=unmatched,cpu_version_steps=cpu_version_rows,
        initial_contact_comparison=contact_diagnostic,created_unix=time.time(),input_buffers_sha256=hashlib.sha256((root/'reference_model_all_arrays.npz').read_bytes()).hexdigest(),
        model_parameters_restored=True,complete_original_mesh_buffers_uploaded=True,model_options=dict(noslip=cpu.opt.noslip_iterations,cone=int(cpu.opt.cone),solver=int(cpu.opt.solver),integrator=int(cpu.opt.integrator),tolerance=cpu.opt.tolerance),
        force_max_error=force_error,preparation_seconds=preparation,steps=rows,
        force_pass=force_error<=1e-5,dynamics_pass=all(r['qpos_max_error']<=1e-5 and r['qvel_max_error']<=1e-4 and not r['overflow'] for r in rows),
        criterion='Ten fixed steps, identical initial pose/velocities/control and original spring force; no weakened collision or objective settings',production_launch_allowed=False)
    (root/'operator_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--host-only',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException:
        (a.root/'operator_failure.json').write_text(json.dumps(dict(complete=False,exception=traceback.format_exc()),indent=2));raise

"""Startup prewarm of the same fixed-shape CUDA losses used by calibration."""
import copy,time
import numpy as np
import torch
from .mano_runtime import ManoSurface
from .mano_fit import choose_rows,holdout_rows
from .official_mano import reference_from_frame
from .mano_gpu_objective import GPUManoObjective
from scripts.calibrate_wuji_mano import CHAINS


def warmup(model_path,profile,mapping,rows,progress,cancel,methods=('parameters','flashsac','delta')):
    m=ManoSurface(model_path,side=profile['side']);m.enable_official_fk();m.build_binding(mapping)
    from .training_options import training_rows
    motion=all(r.get('capture_protocol') in ('geort_motion_v2','contact_motion_v3') for r in rows)
    train=training_rows(rows,full_buffer=True) if motion else choose_rows(rows)[0]
    if any(not r.get('skeleton') for r in train):raise ValueError('GPU 预热等待匹配的骨架记录')
    m.install_flat_reference(reference_from_frame(next(r for r in rows if r['stage']=='open'),m.weights_hash,profile['serial'],'gpu-prewarm'),allow_coarse=True)
    if rows[0].get('capture_protocol')=='contact_motion_v3':
        from .contact_templates import prepare_templates
        rows=copy.deepcopy(rows);prepare_templates(m,rows,cancel);train=training_rows(rows,full_buffer=True)
    values=[m.official_fk.parameters(m,row,CHAINS) for row in train]
    from .mano_mixed_training import MixedGPUManoObjective
    from .mano_gpu_solver import solve_calibration_gpu
    from .mano_objective import SolverSettings
    for method in methods:
        if cancel.is_set():raise InterruptedError('GPU 预热已取消')
        if method=='delta':
            if not motion:continue
            from .mano_joint_delta import JointGPUManoObjective
            Objective=JointGPUManoObjective
        else:Objective=MixedGPUManoObjective if motion else GPUManoObjective
        o=Objective(m,train,np.asarray([v[0] for v in values]),np.asarray([v[1] for v in values]),training=True,shared=True,tactile=True,fixed_forward=True)
        o.settings=SolverSettings(adam_steps=2 if method!='parameters' else 1,lm_steps=0)
        if method=='flashsac':
            from .mano_calibrate_flashsac import solve_calibration_gpu as solve
        else:solve=solve_calibration_gpu
        solve(o,cancel,progress,max_seconds=3600,compile_loss=True)
        del o
    m.gpu_pose_enabled=False;m.solver_enabled=False;m.contact_pose_enabled=False
    valid=holdout_rows(rows)
    for size in dict.fromkeys((32,len(valid)%32 or 32,1)):
        if cancel.is_set():raise InterruptedError('GPU 预热已取消')
        chunk=valid[:size];values=[m.official_fk.parameters(m,row,CHAINS) for row in chunk]
        with torch.no_grad():m.decode(torch.as_tensor(np.asarray([v[0] for v in values]),device=m.device),torch.as_tensor(np.asarray([v[1] for v in values]),device=m.device))
    from .mano_mjwarp_check import cross_check
    cross_check(m,torch.as_tensor(np.asarray([v[0] for v in values]),device=m.device),torch.as_tensor(np.asarray([v[1] for v in values]),device=m.device),cancel)
    return dict(state='ready',training_frames=len(train),runtime_batch_sizes=[32,len(valid)%32 or 32,1])

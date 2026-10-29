#!/usr/bin/env python3
"""G1 retargeting with MJWarp dynamics and selectable action optimizers."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def task_paths(data_root,task_name):
    data_root=Path(data_root)
    folder=('processed/amass/unitree_g1/humanoid/sprint' if task_name=='humanoid'
            else 'processed/omomo/unitree_g1/humanoid_object/move_largebox')
    task=data_root/folder
    return task/'scene.xml',task/'0/trajectory_kinematic.npz'


def main(argv=None):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['humanoid', 'humanoid-object'], default='humanoid-object')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/spider')
    parser.add_argument('--scene', type=Path, help='Override G1 MJCF scene')
    parser.add_argument('--reference', type=Path, help='Override SPIDER (T,nq)/(T,nv)/(T,nu) NPZ')
    parser.add_argument('--ref-dt', type=float, default=None,
                        help='Default: exact upstream grid in spider mode; 1/30 in other modes')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--frames', type=int, help='Number of transitions; default entire clip')
    parser.add_argument('--horizon', type=int, default=8, help='Control knots per MPC window; default spans 0.8 s')
    parser.add_argument('--knot-stride', type=int, default=3, help='Reference frames per control knot')
    parser.add_argument('--admm-iterations', '--contact-rounds', type=int, default=1)
    parser.add_argument('--action-iterations', '--iterations', type=int, default=2, help='Action optimizer updates per ADMM round')
    parser.add_argument('--action-solver', choices=['adamw','adamw-ls','mppi','acados-gn','sample-lm'], default='adamw')
    parser.add_argument('--cost', choices=['spider','spider-contact','legacy'], default='spider',
                        help='Pinned SPIDER source problem on installed MJWarp (default; engine versions audited in config); spider-contact adds local contact constraints; legacy uses whole-body cost')
    parser.add_argument('--contact-lm-iterations',type=int,default=3,
                        help='Same sampled ComFree force LM as run.py, per contact set')
    parser.add_argument('--contact-sweeps',type=int,default=1)
    parser.add_argument('--acados-difference',choices=['forward','central'],default='forward')
    parser.add_argument('--acados-fd-step',type=float,default=.01,help='Finite difference in normalized knot coordinates (times control_scale radians)')
    parser.add_argument('--acados-trust-radius',type=float,default=.15,help='Maximum knot target change in radians per GN update')
    parser.add_argument('--acados-damping',type=float,default=.01)
    parser.add_argument('--acados-globalization',choices=['linesearch','cauchy'],default='cauchy')
    parser.add_argument('--acados-initial-iterations',type=int,default=0,help='GN updates on the first action block only; subsequent windows use action-iterations')
    parser.add_argument('--line-search-trials', type=int, default=6)
    parser.add_argument('--line-search-contraction', type=float, default=.5)
    parser.add_argument('--line-search-armijo', type=float, default=1.e-4)
    parser.add_argument('--mppi-samples', type=int, default=64, help='Total trajectories per MPPI update, chunked over --multistarts rollout worlds')
    parser.add_argument('--mppi-noise-std', type=float, default=.05, help='Joint target knot noise standard deviation in radians')
    parser.add_argument('--mppi-temperature', type=float, default=1., help='Gibbs temperature in full objective cost units')
    parser.add_argument('--mppi-update', choices=['safeguarded','weighted-mean'], default='safeguarded', help='Best of nominal/mean/samples, or the MPPI weighted mean directly')
    parser.add_argument('--mppi-single-rollout',action='store_true',help='Evaluate common nominal/mean/export controls in one world instead of all sampling slots')
    parser.add_argument('--sample-lm-samples',type=int,default=64)
    parser.add_argument('--sample-lm-rank',type=int,default=32)
    parser.add_argument('--sample-lm-noise-std',type=float,default=.05)
    parser.add_argument('--sample-lm-fd-step',type=float,default=.02,help='Normalized action-coordinate difference step')
    parser.add_argument('--sample-lm-trust-radius',type=float,default=.03,help='Maximum LM knot displacement in radians')
    parser.add_argument('--sample-lm-damping',type=float,default=.01)
    parser.add_argument('--sample-lm-reuse-samples',action=argparse.BooleanOptionalAction,default=False,
                        help='Fit the first LM model from sampled residuals; disable for central differences')
    parser.add_argument('--multistarts', type=int, default=8, help='Independent AdamW starts; MPPI sample batch size; acados globalization candidates')
    parser.add_argument('--multistart-noise-std',type=float,help='AdamW start noise in actuator radians; default 0.05 for manipulation, 0.003 for locomotion; first start retains the warm nominal')
    parser.add_argument('--adamw-lr', type=float, default=.3)
    parser.add_argument('--adamw-weight-decay', type=float, default=.01)
    parser.add_argument('--adamw-persist', action='store_true')
    parser.add_argument('--device', default=os.environ.get('FORCEAWARE_DEVICE','cuda:0'))
    parser.add_argument('--mjwarp-fd-step', type=float, default=.01,
                        help='Central-difference step in normalized action coordinates')
    parser.add_argument('--mjwarp-fd-batch', type=int, default=64,
                        help='MJWarp worlds used to batch action-gradient probes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--contact-radius-mm', type=float, default=5.,help='Hand-object point radius')
    parser.add_argument('--foot-contact-radius-mm',type=float,default=150.,help='Ground footstep adaptation radius')
    parser.add_argument('--contact-samples', type=int, default=8)
    parser.add_argument('--contact-qp-iterations', type=int, default=48)
    parser.add_argument('--substeps', type=int, default=2,help='Physics steps per reference frame; 2 matches SPIDER 60 Hz')
    parser.add_argument('--trajectory-cache',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--contact-realization-weight',type=float,default=1.)
    parser.add_argument('--wrench-robust-scale',type=float,default=1.,help='Pseudo-Huber scale for normalized wrench residuals')
    parser.add_argument('--rho', type=float, default=1.)
    parser.add_argument('--contact-scale', type=float, default=.05, help='Consensus normalization length in meters')
    parser.add_argument('--anchor-weight', type=float, default=None,
                        help='Default: 0 for exact spider; .05 for contact/legacy modes')
    parser.add_argument('--hand-contact-threshold', type=float, default=.04, help='Reference collision-surface gap in meters')
    parser.add_argument('--foot-contact-threshold', type=float, default=.035)
    parser.add_argument('--contact-mask', type=Path, help='Explicit binary (T,4) NPY; LH,RH,LF,RF')
    parser.add_argument('--weights', type=Path, help='JSON overriding named humanoid cost weights')
    for name,default in [('balance-xy-scale',.15),('balance-z-scale',.08),('balance-velocity-scale',.5),('balance-height-margin',.08),('balance-height-scale',.05),('upright-scale',.25)]:
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--output-base', type=Path, default=ROOT/'outputs/humanoid')
    parser.add_argument('--case-name', default=None)
    parser.add_argument('--viewer', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    if args.anchor_weight is None:
        args.anchor_weight=0. if args.cost=='spider' else .05
    if args.cost in ('spider','spider-contact') and args.action_solver!='adamw':
        parser.error('SPIDER task mode uses the shared run.py AdamW; use --cost legacy for other experimental solvers')
    if min(args.contact_lm_iterations,args.contact_sweeps)<1:
        parser.error('Contact LM iterations and sweeps must be positive')
    if args.cost in ('spider','spider-contact') and args.weights:
        parser.error('--weights configures the legacy task; SPIDER mode uses the upstream task weights')
    if min(args.horizon, args.admm_iterations, args.action_iterations, args.substeps,args.knot_stride,args.multistarts,args.contact_samples,args.contact_qp_iterations) < 1:
        parser.error('Iteration counts, substeps and horizon must be positive')
    if args.line_search_trials < 1 or args.mppi_samples < 2:
        parser.error('Positive line-search trials and at least two MPPI samples required')
    if not 0 < args.line_search_contraction < 1 or not 0 < args.line_search_armijo < 1:
        parser.error('Line-search contraction and Armijo constant must lie in (0,1)')
    if any(not math.isfinite(v) or v <= 0 for v in (args.mppi_noise_std,args.mppi_temperature)):
        parser.error('MPPI noise and temperature must be finite and positive')
    if args.action_solver in ('mppi','acados-gn','sample-lm') and args.adamw_persist:
        parser.error('--adamw-persist applies to AdamW solvers')
    if args.action_solver=='acados-gn' and args.multistarts<2:
        parser.error('acados-gn requires at least two candidate rollout slots')
    if args.acados_initial_iterations<0:parser.error('acados initial iterations must be nonnegative')
    if min(args.sample_lm_rank,args.sample_lm_samples)<1:
        parser.error('Sampled LM rank and sample count must be positive')
    if args.sample_lm_reuse_samples and args.sample_lm_samples<args.sample_lm_rank+1:
        parser.error('Sample-fit LM requires more samples than subspace dimensions')
    if any(not math.isfinite(v) or v<=0 for v in (args.sample_lm_noise_std,args.sample_lm_fd_step,args.sample_lm_trust_radius,args.sample_lm_damping)):
        parser.error('Sampled LM scales must be finite and positive')
    if any(not math.isfinite(v) or v<=0 for v in (args.acados_fd_step,args.acados_trust_radius,args.acados_damping)):
        parser.error('acados difference step, trust radius and damping must be finite and positive')
    if args.start_frame < 0 or (args.frames is not None and args.frames < 1):
        parser.error('Invalid start frame or frame count')
    for key in ['ref_dt', 'rho', 'contact_scale', 'hand_contact_threshold', 'foot_contact_threshold','balance_xy_scale','balance_z_scale','balance_velocity_scale','balance_height_margin','balance_height_scale','upright_scale','wrench_robust_scale']:
        if key=='ref_dt' and args.ref_dt is None:
            continue
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
    if not math.isfinite(args.contact_realization_weight) or args.contact_realization_weight < 0:
        parser.error('contact realization weight must be finite and nonnegative')
    if args.multistart_noise_std is not None and (not math.isfinite(args.multistart_noise_std) or args.multistart_noise_std<0):
        parser.error('multistart noise must be finite and nonnegative')
    if not math.isfinite(args.anchor_weight) or args.anchor_weight < 0:
        parser.error('anchor weight must be finite and nonnegative')
    if args.seed < 0 or any(not math.isfinite(v) or v<0 for v in (args.contact_radius_mm,args.foot_contact_radius_mm)):
        parser.error('seed and contact radius must be nonnegative')
    if not math.isfinite(args.adamw_lr) or args.adamw_lr <= 0 or not math.isfinite(args.adamw_weight_decay) or args.adamw_weight_decay < 0:
        parser.error('AdamW requires a positive finite learning rate and nonnegative decay')
    name = args.case_name or datetime.now().strftime('g1_%Y%m%d_%H%M%S_%f')
    if Path(name).name != name or name in ('.', '..'):
        parser.error('case-name must be one directory name')
    scene,reference=task_paths(args.data_root,args.task)
    scene=args.scene or scene
    reference=args.reference or reference
    if not scene.is_file() or not reference.is_file():
        parser.error('Missing SPIDER data; run python scripts/fetch_spider_data.py')
    if args.cost=='spider':
        if args.action_iterations < 2:
            parser.error('Strict consensus ADMM needs at least 2 action updates total (one per block)')
        if not math.isfinite(args.mjwarp_fd_step) or args.mjwarp_fd_step<=0 or args.mjwarp_fd_batch<1:
            parser.error('Positive finite MJWarp difference step and batch required')
        from retarget.humanoid_spider_exact import run as run_exact
        output=run_exact(args,scene,reference,args.output_base/name)
        print(f'Saved {output}')
        return output
    import json
    from retarget.humanoid import HumanoidProblem, Settings, Weights, run
    overrides = json.loads(args.weights.read_text()) if args.weights else {}
    if not isinstance(overrides, dict) or any(k not in Weights.__dataclass_fields__ for k in overrides):
        parser.error('Unknown weight key; see retarget/humanoid.py:Weights')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in overrides.values()):
        parser.error('Weights must be finite and nonnegative')
    if args.task=='humanoid' and args.cost=='legacy':overrides={'joints':3.,**overrides}
    settings = Settings(cost_mode='spider' if args.cost=='spider-contact' else args.cost,mjwarp_fd_step=args.mjwarp_fd_step,
                        mjwarp_fd_batch=args.mjwarp_fd_batch,contact_lm_iterations=args.contact_lm_iterations,
                        contact_sweeps=args.contact_sweeps,horizon=args.horizon, admm_iterations=args.admm_iterations,
                        action_solver=args.action_solver,
                        **{name:getattr(args,name) for name in ('acados_difference','acados_fd_step','acados_trust_radius','acados_damping','acados_globalization','acados_initial_iterations')},
                        **{name:getattr(args,name) for name in ('sample_lm_samples','sample_lm_rank','sample_lm_noise_std','sample_lm_fd_step','sample_lm_trust_radius','sample_lm_damping','sample_lm_reuse_samples')},
                        **{name:getattr(args,name) for name in ('line_search_trials','line_search_contraction','line_search_armijo','mppi_samples','mppi_noise_std','mppi_temperature','mppi_update','mppi_single_rollout')},
                        action_iterations=args.action_iterations, substeps=args.substeps,trajectory_cache=args.trajectory_cache,contact_realization_weight=args.contact_realization_weight,wrench_robust_scale=args.wrench_robust_scale,
                        rho=args.rho, contact_scale=args.contact_scale,
                        anchor_weight=args.anchor_weight,
                        knot_stride=args.knot_stride,multistarts=args.multistarts,multistart_noise_std=args.multistart_noise_std,
                        adamw_lr=args.adamw_lr,adamw_weight_decay=args.adamw_weight_decay,
                        adamw_persist=args.adamw_persist,seed=args.seed,device=args.device,
                        contact_radius=args.contact_radius_mm/1000,foot_contact_radius=args.foot_contact_radius_mm/1000,contact_samples=args.contact_samples,
                        contact_qp_iterations=args.contact_qp_iterations,
                        hand_contact_threshold=args.hand_contact_threshold,
                        foot_contact_threshold=args.foot_contact_threshold,
                        **{name:getattr(args,name) for name in ('balance_xy_scale','balance_z_scale','balance_velocity_scale','balance_height_margin','balance_height_scale','upright_scale')})
    problem = HumanoidProblem(scene, reference, ref_dt=args.ref_dt or 1/30, settings=settings,
                              weights=Weights(**overrides) if args.cost=='legacy' else None, contact_mask=args.contact_mask)
    if args.start_frame >= len(problem.qref)-1:
        parser.error('start-frame must precede the last reference frame')
    output = run(problem, args.output_base/name, start_frame=args.start_frame,
                 frames=args.frames, viewer=args.viewer)
    print(f'Saved {output}')
    return output


if __name__ == '__main__':
    main()

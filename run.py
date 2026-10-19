#!/usr/bin/env python3
"""Run fixed-template MANO → Wuji retargeting, or replay a saved result."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Set before importing NumPy/Warp; avoid nested BLAS thread pools.
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if '--replay' in argv:
        argv.remove('--replay')
        from retarget.viewer import main as replay
        return replay(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    solvers = parser.add_mutually_exclusive_group()
    solvers.add_argument('--fa', dest='hand_solver', action='store_const', const='lbfgs', help='Legacy FA/L-BFGS (40 updates)')
    solvers.add_argument('--adamw', dest='hand_solver', action='store_const', const='adamw', help='AdamW contact-surface and control update (default)')
    solvers.add_argument('--sqp', dest='hand_solver', action='store_const', const='acados-rti', help='Direct trigrasp ComFree symbolic SQP-RTI (one update, requires acados)')
    parser.set_defaults(hand_solver='adamw')
    parser.add_argument('--experiment', type=Path, default=ROOT/'data/example', help='Prepared input directory')
    parser.add_argument('--output-base', type=Path, default=ROOT/'outputs', help='Result root')
    parser.add_argument('--case-name', help='Unique run name; default includes solver and timestamp')
    parser.add_argument('--executor', choices=['comfree', 'mujoco'], default='comfree')
    parser.add_argument('--duration', type=float, default=5.4, help='Simulated seconds, positive multiple of 0.01')
    parser.add_argument('--iterations', type=int, help='FA steps per window (40), AdamW hand steps per contact round (1), SQP=1')
    parser.add_argument('--multistarts', type=int, default=None,
                        help='Independent starts for FA/AdamW; SQP-RTI always uses one warm-started trajectory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', action='store_true', help='Perturb MANO fingers and regenerate object templates by nearest SDF projection')
    parser.add_argument('--noise-finger', choices=['all','thumb','index','middle','ring','pinky'], default='all')
    parser.add_argument('--noise-amplitude', type=float, default=.35, help='MANO joint-angle perturbation in radians (default .35)')
    parser.add_argument('--noise-seed', type=int, help='Optional reproducible random joint biases; omitted uses the benchmark flexion bias')
    parser.add_argument('--noise-source', type=Path, help='Raw MANO/semantic correspondence source; defaults to --experiment')
    parser.add_argument('--noise-mano-models', type=Path, help='Directory containing MANO_RIGHT.pkl; also MANO_MODEL_DIR or ~/.config/retarget/mano.json')
    parser.add_argument('--noise-manopth', type=Path, help='Manopth checkout directory; also MANOPTH_DIR or local MANO config')
    parser.add_argument('--trust-radius', type=float, default=1., help='SQP node bound in balanced coordinates')
    parser.add_argument('--threads', type=int, default=8, help='SQP CPU worker threads')
    parser.add_argument('--adamw-lr', type=float, default=.3)
    parser.add_argument('--adamw-weight-decay', type=float, default=.01)
    parser.add_argument('--adamw-persist', action='store_true', help='Shift and retain AdamW moments across windows')
    parser.add_argument('--contact-rounds', type=int, default=1, help='AdamW contact/action ADMM rounds per window; --iterations hand steps per round')
    parser.add_argument('--contact-solver', choices=['sampled-lm','qp'], default='sampled-lm', help='Object contact update: joint set sampling + ComFree batched force LM, or wrench QP')
    parser.add_argument('--contact-samples', type=int, default=8, help='Total sampled contact combinations, including the incumbent')
    parser.add_argument('--contact-lm-iterations', type=int, default=3, help='Batched force LM steps per contact combination')
    parser.add_argument('--contact-qp-iterations', type=int, default=96, help='Projected accelerated-gradient iterations for each convex wrench QP')
    parser.add_argument('--contact-radius-mm', type=float, default=5., help='Hard contact displacement limit from MANO points in mm; 0 fixes the points')
    parser.add_argument('--no-contact-exploration', action='store_true', help='Freeze AdamW object templates while retaining the same physical wrench and ADMM terms')
    parser.add_argument('--no-admm', action='store_true', help='AdamW hand-only updates with fixed object templates; disable consensus, dual updates and contact memory')
    parser.add_argument('--contact-sweeps', type=int, default=1, help='Contact-set selection or projected QP contact steps per object block')
    parser.add_argument('--contact-wrench-weight', type=float, default=1., help='ComFree prediction or QP wrench residual weight in the object block')
    parser.add_argument('--contact-motion-tolerance-mm', type=float, default=.1, help='Hold LM contact sets below this prospective equivalent pose error (mm)')
    parser.add_argument('--contact-wrench-tolerance', type=float, default=.05, help='Hold QP contact sets below this gravity-normalized wrench residual')
    parser.add_argument('--contact-min-improvement', type=float, default=.01, help='Minimum relative task-residual improvement while the contact set remains infeasible')
    parser.add_argument('--contact-rho', type=float, default=.1, help='ADMM penalty on contact residuals normalized by the search radius (default .1)')
    parser.add_argument('--viewer', action=argparse.BooleanOptionalAction, default=True,
                        help='Live MuJoCo viewer, enabled by default; --no-viewer for headless runs')
    parser.add_argument('--replay', action='store_true', help='Replay mode: use --replay --help for options')
    args = parser.parse_args(argv)
    if args.multistarts is None:
        args.multistarts = 1 if args.hand_solver == 'acados-rti' else 8
    if args.iterations is None:
        args.iterations = 40 if args.hand_solver == 'lbfgs' else 1
    if args.iterations < 1 or args.multistarts < 1 or args.threads < 1:
        parser.error('iterations, multistarts and threads must be positive')
    if args.hand_solver == 'acados-rti' and args.iterations != 1:
        parser.error('--sqp performs exactly one RTI update; use --iterations 1')
    if args.hand_solver == 'acados-rti' and args.multistarts != 1:
        parser.error('--sqp uses one warm-started RTI trajectory; --multistarts must be 1')
    import math
    if not math.isfinite(args.noise_amplitude) or args.noise_amplitude < 0:
        parser.error('--noise-amplitude must be finite and nonnegative')
    if args.noise_seed is not None and args.noise_seed < 0:
        parser.error('--noise-seed must be nonnegative')
    if (args.no_contact_exploration or args.no_admm) and args.hand_solver != 'adamw':
        parser.error('--no-contact-exploration and --no-admm apply to --adamw')
    if min(args.contact_rounds,args.contact_sweeps,args.contact_samples,args.contact_lm_iterations,args.contact_qp_iterations) < 1:
        parser.error('contact round, sweep, sample and solver iteration counts must be positive')
    if (not all(math.isfinite(v) for v in (args.contact_radius_mm, args.contact_wrench_weight, args.contact_rho))
            or args.contact_radius_mm < 0 or args.contact_wrench_weight < 0 or args.contact_rho <= 0):
        parser.error('contact search requires finite nonnegative radius/wrench weight and positive rho')
    if (not all(math.isfinite(v) for v in (args.contact_motion_tolerance_mm,args.contact_wrench_tolerance,args.contact_min_improvement))
            or min(args.contact_motion_tolerance_mm,args.contact_wrench_tolerance,args.contact_min_improvement) < 0
            or args.contact_min_improvement >= 1):
        parser.error('contact tolerances must be finite and nonnegative; min-improvement must be below one')
    if not math.isfinite(args.duration) or args.duration <= 0 or abs(args.duration/.01-round(args.duration/.01)) > 1.e-7:
        parser.error('--duration must be a positive multiple of 0.01')
    if not math.isfinite(args.trust_radius) or args.trust_radius <= 0:
        parser.error('--trust-radius must be finite and positive')
    if not math.isfinite(args.adamw_lr) or args.adamw_lr <= 0 or not math.isfinite(args.adamw_weight_decay) or args.adamw_weight_decay < 0:
        parser.error('AdamW needs a finite positive learning rate and nonnegative decay')
    if args.case_name is None:
        label = {'lbfgs':'fa', 'adamw':'adamw', 'acados-rti':'sqp'}[args.hand_solver]
        args.case_name = label+('_noise' if args.noise else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if Path(args.case_name).name != args.case_name or args.case_name in ('.','..'):
        parser.error('--case-name must be a single directory name')
    # Diagnose missing compiled dependencies before allocating GPU state.
    if args.hand_solver == 'acados-rti':
        from retarget.support.acados_env import ensure_acados_env
        ensure_acados_env()
    if args.noise:
        from retarget.mano_noise import prepare_run
        prepare_run(args)
    from retarget.workflow import run
    return run(args)


if __name__ == '__main__':
    main()

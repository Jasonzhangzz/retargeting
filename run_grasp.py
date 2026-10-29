#!/usr/bin/env python3
"""Synthesize static Wuji grasps on independent meshes and test force closure."""
import argparse
from datetime import datetime
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--object', type=Path, nargs='+', help='One or more OBJ/STL meshes; coordinates in metres before --scale')
    source.add_argument('--primitives', '--demo', dest='primitives', action='store_true', help='Generate taxonomy-associated primitive families and dimension sweeps')
    source.add_argument('--list-types', action='store_true', help='List all numbered Shadow types, primitive associations and provenance')
    source.add_argument('--replay', type=Path, help='Saved grasp case directory')
    parser.add_argument('--stage', choices=['grasp','initial','prediction','formed'], default='grasp',
                        help='Replay the next-step grasp, initialization, prediction or physically formed grasp')
    parser.add_argument('--dexonomy-root', type=Path, default=ROOT/'third_party/Dexonomy')
    parser.add_argument('--templates', nargs='+', default=['1'], help='Grasp IDs 1–33 (e.g. 6 or 1 6 27), full raw_anno names, or all')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/grasps')
    parser.add_argument('--name', default=None, help='Unique batch directory name')
    parser.add_argument('--archive-results', action='store_true',
                        help='Keep full solver intermediates for debugging; default persists numeric wrist-frame grasps only')
    parser.add_argument('--scale', type=float, default=1., help='Uniform mesh scale; use .001 for millimetres')
    parser.add_argument('--hand-model', choices=['wuji2', 'wuji1'], default='wuji2',
                        help='Grasp hand: WUJI Hand 2.2 right (default), or the original Hand 1')
    parser.add_argument('--primitive', choices=['cylinder', 'sphere', 'cuboid', 'disk', 'plate'], help='Filter the associated primitive families')
    parser.add_argument('--primary-shape-only', action='store_true',
                        help='Use the first associated primitive per grasp type, giving one case per size')
    parser.add_argument('--radius-scales', nargs='+', type=float, default=[.75, 1., 1.25], help='Independent radius / planar size factors for primitive sweeps')
    parser.add_argument('--height-scales', nargs='+', type=float, default=[1.], help='Independent cylinder height / disk or plate thickness factors')
    parser.add_argument('--radius', type=float, help='Override primitive base radius in metres')
    parser.add_argument('--height', type=float, help='Override primitive base full height / thickness in metres')
    parser.add_argument('--friction',type=float,default=None,help='Sliding friction; default inherits the run.py physics reference')
    parser.add_argument('--physics-reference',type=Path,help='Reference XML for object contact parameters; default is run.py data/example scene')
    parser.add_argument('--impratio',type=float,help='Friction-to-normal impedance ratio, inherited from the physics reference')
    parser.add_argument('--contact-solref',type=float,nargs=2,help='MuJoCo contact solref (time constant/damping ratio, or direct negative stiffness/damping)')
    parser.add_argument('--contact-solimp',type=float,nargs=5,help='MuJoCo contact solimp (five values)')
    parser.add_argument('--planning-gravity',type=float,default=9.81,help='Downward acceleration in the planning input (default: 9.81)')
    parser.add_argument('--load-conditioning',action=argparse.BooleanOptionalAction,default=True,
                        help='CUDA actuator-feasible preload input and physical initialization (default: on)')
    parser.add_argument('--preload-ramp-seconds',type=float,default=.3,
                        help='Ramp from hand-gravity compensation to the load command before release; 0 gives a step')
    parser.add_argument('--initialization-retries',type=int,default=2,
                        help='Additional input attempts after physical failure: alternate candidate policy, then resample; all are retained')
    parser.add_argument('--init-selection',choices=['interior','matching'],default='matching',
                        help='Use matching-loss order (default), or prefer original candidates away from sharp rims')
    parser.add_argument('--closing-depth-mm',type=float,default=None,help='Hand reference closure along original contact normals; default inherits -target_phi (4 mm); 0 disables')
    parser.add_argument('--closing-seconds',type=float,default=.3,help='Duration of the hand closing reference; object target remains constant')
    parser.add_argument('--formation-seconds',type=float,default=1.,help='Physical preload formation duration before stability checking')
    parser.add_argument('--hold-controller',choices=['mpc','fixed'],default='mpc',help='Shared run.py rolling MPC (default), or old first-command hold ablation')
    parser.add_argument('--physics-dt',type=float,default=.00025,help='CUDA executor timestep; solver still uses the run.py 10 ms grid')
    parser.add_argument('--physics-graph',action=argparse.BooleanOptionalAction,default=True,
                        help='Replay identical physics substeps in a CUDA graph (default: on)')
    parser.add_argument('--starts', type=int, default=1, help='Dexonomy initial poses per object')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--contact-rounds', '--rounds', dest='contact_rounds', type=int, default=1, help='AdamW only: same ADMM rounds as run.py')
    parser.add_argument('--iterations', '--hand-iterations', dest='iterations', type=int, default=1, help='AdamW only: same hand steps per round as run.py')
    parser.add_argument('--multistarts', type=int, default=8, help='AdamW only: same parallel solver starts as run.py')
    parser.add_argument('--grasp-optimizer', choices=['cem', 'adamw'], default=None,
                        help='GPU default: CARD CEM with explicit pose prediction/refinement; legacy default: AdamW')
    parser.add_argument('--cem-population', type=int, default=1024, help='Parallel static configurations per node')
    parser.add_argument('--cem-iterations', type=int, default=16, help='Total annealed CEM generations')
    parser.add_argument('--cem-elite-fraction', type=float, default=.05)
    parser.add_argument('--cem-temperature', type=float, default=.1)
    parser.add_argument('--cem-translation-sigma-mm', type=float, default=2.)
    parser.add_argument('--cem-rotation-sigma', type=float, default=.016)
    parser.add_argument('--cem-joint-sigma', type=float, default=.05)
    parser.add_argument('--cem-final-noise', type=float, default=.2, help='Final / initial sampling noise ratio')
    parser.add_argument('--cem-self-clearance-mm', type=float, default=.5,
                        help='Hard clearance between enclosing finger capsules; default 0.5 mm')
    parser.add_argument('--explicit-pose', action=argparse.BooleanOptionalAction, default=True,
                        help='CEM: predict gravity/contact motion with independent servo preload and refine using joint gradients')
    parser.add_argument('--explicit-horizon', type=int, default=4)
    parser.add_argument('--explicit-dt', type=float, default=.1)
    parser.add_argument('--explicit-sigma', type=float, default=.5, help='Projected contact solver relaxation, in (0, 1]')
    parser.add_argument('--explicit-contact-iterations', type=int, default=64)
    parser.add_argument('--explicit-preload-limit', type=float, default=.5, help='Maximum independent finger servo offset in radians; actual actuator limits also apply')
    parser.add_argument('--explicit-motion-tolerance-mm', type=float, default=2.)
    parser.add_argument('--explicit-rotation-tolerance-deg', type=float, default=5.)
    parser.add_argument('--explicit-pose-weight', type=float, default=1.e6,
                        help='Scale upstream terminal cost (5000*m^2 + 50*quaternion error) to CARD cost units')
    parser.add_argument('--explicit-translation-metric', type=float, default=50., help='Quasi-static Q translation diagonal; not physical mass')
    parser.add_argument('--explicit-rotation-metric', type=float, default=.1, help='Quasi-static Q rotation diagonal; not physical inertia')
    parser.add_argument('--explicit-contact-cutoff-mm', type=float, default=1., help='Near-contact activation distance; capped at 1 mm')
    parser.add_argument('--explicit-refinement-steps', type=int, default=32)
    parser.add_argument('--explicit-refinement-step-size', type=float, default=1., help='Gradient proposal size in initial CEM noise units')
    parser.add_argument('--explicit-line-search-steps', type=int, default=8)
    parser.add_argument('--gpu-nodes', type=int, default=8,
                        help='Independent scene graphs submitted together; 0 submits all scenes (VRAM permitting)')
    parser.add_argument('--legacy-serial', action='store_true',
                        help='Reference path: serial tasks and host-driven shared solver')
    parser.add_argument('--solver-config', type=Path, default=ROOT/'data/example/forceaware.yaml', help='Shared solver weights and time grid; input scene/references are rebuilt')
    parser.add_argument('--object-mass', type=float, default=.2, help='Object mass in kg for the shared dynamics input')
    parser.add_argument('--init-iterations', type=int, default=60, help='Original Dexonomy Torch Adam initialization steps')
    parser.add_argument('--init-points', type=int, default=24, help='Original Dexonomy object surface point samples')
    parser.add_argument('--init-inplane', type=int, default=8, help='In-plane rotations per sampled contact pair')
    parser.add_argument('--init-device', default='cuda', help='CUDA device for original Dexonomy initialization, e.g. cuda:0')
    parser.add_argument('--no-template-fit', action='store_true',
                        help='Ablation: retain the old bone-only Wuji template mapping')
    parser.add_argument('--template-refinements',type=int,default=2,
                        help='Maximum input hand fits to original Dexonomy surface contacts; stops when valid and within contact tolerance')
    parser.add_argument('--contact-samples', type=int, default=8)
    parser.add_argument('--contact-qp-iterations', '--qp-iterations', dest='contact_qp_iterations', type=int, default=96)
    parser.add_argument('--contact-radius-mm', type=float, default=5.)
    parser.add_argument('--contact-lm-iterations', type=int, default=3)
    parser.add_argument('--contact-sweeps', type=int, default=1)
    parser.add_argument('--contact-wrench-weight', type=float, default=1.)
    parser.add_argument('--contact-motion-tolerance-mm', type=float, default=.1)
    parser.add_argument('--contact-wrench-tolerance', type=float, default=.05)
    parser.add_argument('--contact-min-improvement', type=float, default=.01)
    parser.add_argument('--contact-rho', '--rho', dest='contact_rho', type=float, default=.1)
    parser.add_argument('--adamw-lr', '--lr', dest='adamw_lr', type=float, default=.3)
    parser.add_argument('--adamw-weight-decay', type=float, default=.01)
    parser.add_argument('--adamw-persist', action='store_true')
    parser.add_argument('--no-admm', action='store_true')
    parser.add_argument('--no-contact-exploration', action='store_true')
    parser.add_argument('--contact-tolerance-mm', type=float, default=2.)
    parser.add_argument('--penetration-tolerance-mm', type=float, default=2.)
    parser.add_argument('--cone-edges', type=int, default=16, help='Inscribed Coulomb cone edges for independent closure LP')
    parser.add_argument('--viewer',action=argparse.BooleanOptionalAction,default=True,
                        help='Show each grasp / physical test live (default: on; headless falls back to saved visualization)')
    parser.add_argument('--render',action=argparse.BooleanOptionalAction,default=True,
                        help='Save grasp images and physics video (default: on)')
    parser.add_argument('--drop-test',action=argparse.BooleanOptionalAction,default=None,
                        help='Host-driven gravity/MPC validation, requires --legacy-serial (legacy default: on)')
    parser.add_argument('--settle-seconds',type=float,default=.5,help='Minimum settling time after preload formation')
    parser.add_argument('--settle-mode',choices=['fixed-object','free'],default='fixed-object',
                        help='Temporarily hold the object while the hand settles, then fully release it (default); or prepare with a free object')
    parser.add_argument('--settle-timeout',type=float,default=3.,help='Maximum settling time; unstable cases skip the gravity test')
    parser.add_argument('--drop-seconds',type=float,default=2.,help='Gravity test duration after settling')
    parser.add_argument('--drop-distance-mm',type=float,default=30.,help='Object displacement relative to the settled hand that counts as escape/drop')
    parser.add_argument('--follow',action='store_true',help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    removed = {'--cem-sigma', '--cem-min-sigma', '--cem-trust-radius', '--cem-gradient-steps',
               '--cem-gradient-lr', '--cem-optimize-pose', '--no-cem-optimize-pose',
               '--contact-depth-mm', '--grasp-penetration-weight', '--grasp-endpoint-weight'}
    used = [value.split('=')[0] for value in arguments if value.split('=')[0] in removed]
    if used:
        parser.error('Removed dynamics CEM options: '+', '.join(used)+
                     '. Use --cem-joint-sigma, --cem-translation-sigma-mm, '
                     '--cem-rotation-sigma and --cem-final-noise for sampling; '
                     'the new explicit-model gradient uses --explicit-refinement-steps '
                     'and --explicit-refinement-step-size.')
    args = parser.parse_args(arguments)
    args.grasp_optimizer = args.grasp_optimizer or ('adamw' if args.legacy_serial else 'cem')
    if args.legacy_serial and args.grasp_optimizer != 'adamw':
        parser.error('CEM runs on GPU nodes; omit --legacy-serial')
    if args.drop_test is None:
        args.drop_test = args.legacy_serial
    if args.drop_test and not args.legacy_serial:
        parser.error('--drop-test uses the host-driven rolling validator; use --legacy-serial --drop-test')
    # MuJoCo chooses its GL backend at import time, including imports made
    # while resolving the reference physics XML. The viewer uses a child.
    if args.render:
        os.environ.setdefault('MUJOCO_GL','egl')
    from retarget.grasp_physics import resolve_physics_args
    resolve_physics_args(args)
    args.hand_solver = 'adamw'
    args.contact_solver = 'sampled-lm'
    args.dexonomy_root = args.dexonomy_root.expanduser().resolve()
    args.solver_config = args.solver_config.expanduser().resolve()
    import math
    if (args.explicit_horizon < 1 or args.explicit_refinement_steps < 0 or args.explicit_line_search_steps < 1
            or args.explicit_contact_iterations < 1 or args.explicit_sigma > 1 or args.explicit_preload_limit <= 0 or
            not all(math.isfinite(v) and v > 0 for v in (args.explicit_dt, args.explicit_sigma,
                args.explicit_pose_weight, args.explicit_translation_metric, args.explicit_rotation_metric,
                args.explicit_contact_cutoff_mm, args.explicit_refinement_step_size, args.explicit_preload_limit,
                args.explicit_motion_tolerance_mm, args.explicit_rotation_tolerance_deg))):
        parser.error('Invalid explicit contact model horizon, metric, weight or refinement settings')
    if (args.cem_population < 2 or args.cem_iterations < 1
            or not 0 < args.cem_elite_fraction <= 1
            or not all(math.isfinite(v) and v > 0 for v in
                (args.cem_temperature, args.cem_translation_sigma_mm,
                 args.cem_rotation_sigma, args.cem_joint_sigma))
            or not 0 < args.cem_final_noise <= 1
            or not math.isfinite(args.cem_self_clearance_mm) or args.cem_self_clearance_mm < 0):
        parser.error('Invalid CARD CEM population, elite fraction, temperature or noise schedule')
    if (not math.isfinite(args.impratio) or args.impratio<=0
            or not (all(math.isfinite(v) for v in args.contact_solref) and
                    (all(v>0 for v in args.contact_solref) or
                     (all(v<=0 for v in args.contact_solref) and any(v<0 for v in args.contact_solref))))
            or not all(math.isfinite(v) for v in args.contact_solimp)
            or not (0<=args.contact_solimp[0]<=1 and 0<=args.contact_solimp[1]<=1
                    and args.contact_solimp[2]>0 and 0<args.contact_solimp[3]<1 and args.contact_solimp[4]>=1)
            or not math.isfinite(args.planning_gravity) or args.planning_gravity<0):
        parser.error('Invalid impratio, contact solref/solimp, or planning gravity')
    if min(args.starts, args.contact_rounds, args.iterations, args.multistarts, args.init_iterations,
           args.contact_samples, args.contact_qp_iterations, args.contact_lm_iterations, args.contact_sweeps,
           args.init_points, args.init_inplane) < 1 or args.cone_edges < 4:
        parser.error('Solver budgets must be positive; --cone-edges must be at least 4')
    if args.template_refinements < 0:
        parser.error('--template-refinements must be nonnegative')
    if args.gpu_nodes < 0:
        parser.error('--gpu-nodes must be nonnegative')
    if args.adamw_persist and not args.legacy_serial:
        parser.error('--adamw-persist is for --legacy-serial; GPU nodes have independent optimizer state')
    if args.initialization_retries<0:
        parser.error('--initialization-retries must be nonnegative')
    if not math.isfinite(args.preload_ramp_seconds) or not 0<=args.preload_ramp_seconds<=args.formation_seconds:
        parser.error('--preload-ramp-seconds must be nonnegative and covered by --formation-seconds')
    if (not all(math.isfinite(v) and v>0 for v in
            (args.settle_seconds,args.settle_timeout,args.drop_seconds,args.drop_distance_mm,
             args.closing_seconds,args.formation_seconds,args.physics_dt))
            or args.settle_timeout<max(args.settle_seconds,.2)):
        parser.error('Invalid settling/test duration or drop distance; settling timeout must cover the minimum and stability window')
    if args.closing_depth_mm is not None and (not math.isfinite(args.closing_depth_mm) or args.closing_depth_mm<0):
        parser.error('--closing-depth-mm must be finite and nonnegative')
    if args.formation_seconds<args.closing_seconds:
        parser.error('--formation-seconds must cover --closing-seconds')
    if not math.isclose(round(.01/args.physics_dt)*args.physics_dt,.01,abs_tol=1e-9) or args.physics_dt>.01:
        parser.error('--physics-dt must divide the 10 ms MPC grid')
    if (not all(math.isfinite(v) and v > 0 for v in (args.scale, args.object_mass,
            args.contact_rho, args.adamw_lr, args.contact_tolerance_mm, args.penetration_tolerance_mm))
            or not math.isfinite(args.friction) or args.friction < 0 or args.seed < 0):
        parser.error('Invalid scale, tolerances, solver parameters, friction or seed')
    if (not all(math.isfinite(v) and v >= 0 for v in (args.contact_radius_mm,args.contact_wrench_weight,args.adamw_weight_decay,
            args.contact_motion_tolerance_mm,args.contact_wrench_tolerance,args.contact_min_improvement))
            or args.contact_min_improvement >= 1):
        parser.error('Invalid solver weights or contact tolerances')
    if not all(math.isfinite(v) and v > 0 for v in args.radius_scales+args.height_scales+[
            v for v in (args.radius, args.height) if v is not None]):
        parser.error('Primitive dimensions and scale factors must be finite and positive')
    if args.primitives and args.scale != 1.:
        parser.error('Use --radius-scales and --height-scales for primitives; --scale is for external meshes')
    from retarget.grasp_primitives import catalog
    if args.list_types:
        import json
        print(json.dumps(catalog(args.dexonomy_root), indent=2))
        return
    if args.templates == ['all']:
        args.templates = [row['template'] for row in catalog(args.dexonomy_root)]
    elif not args.replay:
        resolved=[]
        for template in args.templates:
            try:
                number=int(template)
            except ValueError:
                resolved.append(template)
                continue
            if not 1<=number<=33:
                parser.error('--templates numeric IDs must be between 1 and 33')
            matches=sorted((args.dexonomy_root/'assets/hand/shadow/raw_anno').glob(f'{number}_*.yaml'))
            if len(matches)!=1:
                parser.error(f'Expected one Dexonomy annotation for grasp {number}, found {len(matches)} in {args.dexonomy_root}')
            resolved.append(matches[0].stem)
        args.templates=resolved
    if args.viewer and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        args.viewer = False
        print('[visualization] No display server: live window disabled; saved images/video follow --render.',flush=True)
    if args.render:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    from retarget.grasp import run, replay
    if args.replay:
        if args.follow:
            from retarget.grasp_preview import follow
            return follow(args.replay.expanduser().resolve())
        if args.drop_test and args.stage=='grasp':
            from retarget.grasp_preview import simulate_case
            return simulate_case(args.replay.expanduser().resolve(),args)
        return replay(args.replay.expanduser().resolve(), viewer=args.viewer,
                      render=args.render, stage=args.stage)
    args.name = args.name or 'grasp_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a single directory name')
    destination = args.output.expanduser().resolve()/args.name
    if not destination.exists():
        from retarget.grasp_results import method_name, register_results
        register_results(destination, method_name(args.grasp_optimizer, args.explicit_pose), args.hand_model)
    if args.archive_results:
        return run(args)
    from retarget.grasp_compact import run_compact
    return run_compact(args)


if __name__ == '__main__':
    main()

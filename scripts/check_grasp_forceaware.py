"""Diagnose original native ForceAware dynamics on a prepared grasp reference."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("native_scene", type=Path)
    parser.add_argument("--viewer-scene", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/grasp_forceaware_validation"))
    parser.add_argument("--duration", type=float, default=.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--directions", type=int, default=1)
    parser.add_argument("--servo-integration", choices=("upstream_euler", "implicit_actuator"), default="upstream_euler")
    args = parser.parse_args(argv)
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment
    require_environment(); activate_upstream()
    if args.servo_integration == "implicit_actuator":
        from retarget.dynamics.dexforge_servo_integration import activate_servo_integration
        activate_servo_integration()
    import warp as wp
    from retarget.env.grasp.forceaware_reference import prepare_reference, make_config
    from retarget.env.grasp.forceaware_validation import diagnose_native

    wp.init(); wp.set_device(args.device)
    artifacts = prepare_reference(args.experiment, args.output / "reference", args.native_scene,
                                  viewer_xml=args.viewer_scene)
    cfg = make_config(artifacts, duration_seconds=max(args.duration, .1))
    return diagnose_native(artifacts, cfg, args.output, duration_seconds=args.duration,
                           directions=args.directions, skip_gradients=args.skip_gradients)


if __name__ == "__main__":
    main()

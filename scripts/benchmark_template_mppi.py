#!/usr/bin/env python3
"""Measure warm static-grasp scoring separately from one-time GPU setup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    import warp as wp
    from retarget.common.results import write_json
    from retarget.env.grasp.projected_comfree import ClosureSettings
    from retarget.env.grasp.template_mppi import TemplateMPPISettings, TemplatePoseObjective
    from retarget.env.grasp.template_mppi_inputs import TemplateContacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--response-mode", default="equilibrium", choices=("equilibrium", "finite_horizon"))
    parser.add_argument("--gn-iterations", type=int, default=8)
    parser.add_argument("--torsional-friction-m", type=float, default=.02)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.worlds, args.repeats) < 1:
        parser.error("worlds and repeats must be positive")
    if args.gn_iterations < 0:
        parser.error("gn-iterations must be nonnegative")
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"
    wp.init()
    case = args.case.resolve()
    template = TemplateContacts(**json.loads((case / "template_contacts.json").read_text()))
    with np.load(case / "grasp.npz") as raw:
        q = raw["qpos"].copy()
    closure = ClosureSettings(response_mode=args.response_mode, cuda_graph=not args.eager,
                              gn_iterations=args.gn_iterations,
                              torsional_friction_m=args.torsional_friction_m)
    objective = TemplatePoseObjective(case / "input_object.obj", TemplateMPPISettings(closure=closure))
    objective.bind(template, q)
    objective.set_batch_size(args.worlds)
    controls = torch.as_tensor(q, device="cuda:0", dtype=torch.float32)[None, None].repeat(args.worlds, 1, 1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    objective.evaluate(controls)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - started

    def measure(function):
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            function()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        return dict(mean_seconds=float(np.mean(samples)), median_seconds=float(np.median(samples)), samples=samples)

    # Share Torch/Warp's active stream, as the actual candidate scorer does.
    with torch.no_grad(), wp.ScopedDevice("cuda:0"), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        geometry = objective._geometry(controls[:, 0])
        projected, normals = geometry[2:4]
        timings = dict(
            fk_collision_projection=measure(lambda: objective._geometry(controls[:, 0])),
            closure_gn=measure(lambda: objective.closure_solver.solve(projected, normals)),
            complete_candidate_score=measure(lambda: objective.evaluate(controls)),
        )
    score = timings["complete_candidate_score"]["median_seconds"]
    report = dict(gpu=torch.cuda.get_device_name(), case=str(case), contacts=len(template.body_names),
        worlds=args.worlds, response_mode=args.response_mode, gn_iterations=args.gn_iterations, eager=args.eager,
        torsional_friction_m=args.torsional_friction_m,
        force_effort_weight=closure.force_effort_weight,
        first_score_including_graph_setup_seconds=cold_seconds, timings=timings,
        estimated_8192_sample_score_seconds=8192 / args.worlds * score,
        scope="Warm scoring estimate excludes initialization, MPPI reductions, guarded means, diagnostics and physical evaluation")
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

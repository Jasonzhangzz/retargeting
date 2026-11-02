#!/usr/bin/env python3
"""Continue the original objective with identical template/root search bounds."""
import argparse
from dataclasses import replace
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
    from retarget.config.run import load
    from retarget.env.grasp.template_mppi import TemplatePoseObjective, TemplateMPPISettings
    from retarget.env.grasp.template_mppi_inputs import TemplateContacts
    from retarget.solver.spider_mppi import optimize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--types", type=int, nargs="+", default=[1, 10, 11, 33])
    parser.add_argument("--iterations", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"
    wp.init()
    config = load(args.batch / "config.yaml")
    entries = json.loads((args.batch / "manifest.json").read_text())["cases"]
    reports = []
    args.output.mkdir(parents=True, exist_ok=True)
    for typ in args.types:
        entry = min((e for e in entries if int(e["template"].split("_")[0]) == typ),
                    key=lambda e: abs(e["primitive"]["size_scale"] - 1.))
        case = (args.batch / entry["summary"]).parent.resolve()
        template = TemplateContacts(**json.loads((case / "template_contacts.json").read_text()))
        original = np.asarray(json.loads((case / "initialization.json").read_text())["initial_qpos"])
        q = np.load(case / "grasp.npz")["qpos"]
        settings = TemplateMPPISettings(**json.loads((case / "projection_force_closure.json").read_text())["settings"])
        objective = TemplatePoseObjective(case / "input_object.obj", settings)
        objective.bind(template, original)
        objective.set_batch_size(config.solver.worlds)
        solver = replace(config.solver, iterations=args.iterations,
                         improvement_check_steps=args.iterations + 1)
        noise = torch.tensor([settings.pos_noise] * 3 + [settings.rot_noise] * 3
                             + [settings.joint_noise] * 20, device="cuda")
        mean = torch.tensor(q, dtype=torch.float32, device="cuda")[None]
        before = objective.final_diagnostics(q)
        trace, histories = [q.copy()], []
        def evaluate(candidates):
            candidates.copy_(objective.project(candidates))
            return objective.evaluate(candidates)
        begun = time.perf_counter()
        with torch.no_grad():
            for stage in range(settings.stages):
                mean, detail = optimize(mean, evaluate, solver, knots=1, knot_steps=1,
                    channel_noise=noise * settings.final_stage_noise ** stage,
                    seed=config.seed + 104729 * stage, first_noise=1., last_noise=1.,
                    return_best=True, guard_mean=settings.guard_mean)
                histories.append(detail)
                trace.append(mean[0].cpu().numpy().copy())
        after = objective.final_diagnostics(mean[0])
        target = args.output / entry["name"]
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / "continuation.npz", qpos=np.asarray(trace))
        report = dict(template=entry["template"], case=str(case), original=before, continued=after,
            histories=histories, seconds=time.perf_counter() - begun,
            additional_sampled=sum(h["sampled"] for h in histories),
            original_root_bounds_preserved=True, original_fixed_contacts_preserved=True)
        write_json(target / "continuation.json", report)
        reports.append(report)
        write_json(args.output / "mppi_continuation.json", reports)
        print(entry["template"], "cost", before["cost"], after["cost"],
              "coverage", before["coverage"], after["coverage"],
              "gap mm", np.median(abs(np.asarray(before["signed_distance_m"]))) * 1000,
              np.median(abs(np.asarray(after["signed_distance_m"]))) * 1000, flush=True)
        del objective
        import gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

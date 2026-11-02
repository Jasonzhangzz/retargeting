#!/usr/bin/env python3
"""Audit diagnostic IK witnesses with the unmodified original GPU objective."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import gc
    import numpy as np
    import torch
    import warp as wp
    from retarget.common.results import write_json
    from retarget.env.grasp.template_mppi import TemplatePoseObjective, TemplateMPPISettings
    from retarget.env.grasp.template_mppi_inputs import TemplateContacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--types", type=int, nargs="+", default=[1, 6, 10, 11, 15, 17, 26, 28, 30, 33])
    args = parser.parse_args()
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"
    wp.init()
    entries = json.loads((args.batch / "manifest.json").read_text())["cases"]
    reports = []
    for typ in args.types:
        entry = min((e for e in entries if int(e["template"].split("_")[0]) == typ),
                    key=lambda e: abs(e["primitive"]["size_scale"] - 1.))
        case = (args.batch / entry["summary"]).parent.resolve()
        target = args.diagnosis / entry["name"]
        template = TemplateContacts(**json.loads((case / "template_contacts.json").read_text()))
        initial = np.asarray(json.loads((case / "initialization.json").read_text())["initial_qpos"])
        settings = TemplateMPPISettings(**json.loads((case / "projection_force_closure.json").read_text())["settings"])
        objective = TemplatePoseObjective(case / "input_object.obj", settings)
        objective.bind(template, initial)
        objective.set_batch_size(8)
        candidates = {"original": np.load(case / "grasp.npz")["qpos"]}
        for name in ("constrained_control", "object_allowance_0mm", "object_allowance_0.5mm",
                     "object_allowance_1mm", "object_allowance_2mm"):
            path = target / f"{name}.npz"
            if path.is_file():
                candidates[name] = np.load(path)["qpos"][-1]
        report = dict(template=entry["template"], case=str(case), candidates={})
        for name, q in candidates.items():
            diagnostic = objective.final_diagnostics(q)
            report["candidates"][name] = diagnostic
            print(entry["template"], name, diagnostic["coverage"], diagnostic["cost"],
                  "HO/HH mm", diagnostic["max_hand_object_penetration_m"] * 1000,
                  diagnostic["max_hand_hand_penetration_m"] * 1000,
                  "projected", diagnostic["projected_closure"], flush=True)
        reports.append(report)
        write_json(target / "gpu_verification.json", report)
        write_json(args.diagnosis / "gpu_verified_candidates.json", reports)
        del objective
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

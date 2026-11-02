#!/usr/bin/env python3
"""Summarize a grasp run without changing its evaluation denominator."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    summary = {}
    for name in ("baseline", "evaluation"):
        path = root / name / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        reasons = Counter()
        for line in (root / name / "directions.jsonl").read_text().splitlines():
            direction = json.loads(line)
            if not direction["success"]:
                reasons.update(direction["reasons"] or ["unspecified_failure"])
        minimum = report["minimum_case_success_rate"]
        summary[name] = {k: report[k] for k in ("protocol", "total_cases", "total_trials", "total_directions",
                                               "macro_success_rate", "minimum_case_success_rate", "all_cases_passed")}
        summary[name].update(worst_cases=[r["case_id"] for r in report["cases"] if r["success_rate"] == minimum],
                             failing_cases=[r for r in report["cases"] if r["successes"] < r["trials"]],
                             failure_reasons=dict(reasons.most_common()))
    metrics = root / "metrics.jsonl"
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if rows:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
            for ax, key in zip(axes.flat, ("real_reward", "real_failure", "wm_state", "wm_contact", "actor_loss", "critic_loss")):
                ax.plot([r["iteration"] for r in rows], [r[key] for r in rows])
                ax.set(title=key, xlabel="iteration")
                ax.grid(alpha=.2)
            fig.savefig(root / "training_curves.png", dpi=160)
            plt.close(fig)
            summary["last_training_metrics"] = rows[-1]
    status = root / "status.json"
    if status.exists():
        summary["status"] = json.loads(status.read_text())
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

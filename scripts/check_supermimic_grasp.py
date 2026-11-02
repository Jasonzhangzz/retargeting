#!/usr/bin/env python3
"""GPU preflight: representative variants, replay, imagination and save/restore."""
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    import torch
    from retarget.integrations.supermimic.dataset import GraspDataset
    from retarget.integrations.supermimic.train import GraspRunner, TrainConfig
    from retarget.integrations.supermimic.evaluate import evaluate
    ds = GraspDataset(args.manifest)
    # Type 9 orientation, pinch, thin plate, palm contact and disk.
    cases = tuple(next(i for i, case in enumerate(ds.cases) if case["type_id"] == kind)
                  for kind in (9, 11, 16, 24, 30))
    invalid_case = ds.ids["t10_disk_s09"]
    cases += (invalid_case,)
    num_envs = 2 * len(cases)
    cfg = TrainConfig(manifest=args.manifest, num_envs=num_envs, case_ids=cases, device=args.device,
                      wm_hidden=(64, 64), max_iterations=3, max_hours=1, buffer_steps=32,
                      warmup_steps=10, wm_window=4, wm_batches=1, wm_batch=8, wm_only_iters=1,
                      pi_batch=8, horizon=4, critic_epochs=1, critic_minibatches=1,
                      log_interval=1, save_interval=2, validation_repeats=1,
                      final_evaluation=False, log_dir=str(args.output))
    runner = GraspRunner(cfg)
    initial = {k: v.clone() for k, v in runner.actor.state_dict().items()}
    runner.learn()
    metrics = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
    assert any(row["actor_grad"] > 0 for row in metrics)
    assert runner.buffer.data["aux"][:runner.buffer.size, :, :6].max() > 0
    assert all(row["reward_parity"] == 0 for row in metrics)
    assert any(not torch.equal(initial[k], v) for k, v in runner.actor.state_dict().items() if "net." in k)
    path = runner.save("roundtrip.pt")
    expected = {k: v.clone() for k, v in runner.actor.state_dict().items()}
    with torch.no_grad():
        next(runner.actor.parameters()).add_(1)
    runner.load(path)
    assert all(torch.equal(expected[k], v) for k, v in runner.actor.state_dict().items())
    invalid_ref = next(i for i, record in enumerate(ds.records) if record["trial_id"] == "t10_disk_s09_seed01" and record["method"] == "admm")
    world = next(i for i, case in enumerate(runner.env.assignment) if case == invalid_case)
    runner.env.reset(torch.tensor([world], device=args.device), reference_ids=[invalid_ref])
    assert not runner.env.reference_valid[world], "Out-of-limit cached pose must remain a counted failure"
    runner.env.reset(split="test")
    audits = []
    for _ in range(20):
        runner.env.step(torch.zeros((num_envs, 20), device=args.device), training=False)
        audits.append(runner.env.meter.audit())
    assert max(a["loaded_native_rows"] for a in audits) > 0, "Audit needs actual loaded contacts"
    result = evaluate(runner.env, None, split="test", repeats=1, output=args.output / "baseline-smoke")
    report = dict(passed=True, contact_audits=audits, reward_parity=0,
                  actor_gradient=True, checkpoint_roundtrip=True, validation_boundary=True,
                  invalid_reference_rejected=True,
                  evaluation_cases=result["total_cases"])
    (args.output / "check-results.json").write_text(json.dumps(report, indent=2))
    print("TRAINING_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()

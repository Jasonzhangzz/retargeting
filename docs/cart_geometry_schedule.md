The geometry-to-policy controller is `scripts/continue_cart_after_geometry.py`.
It resumes an existing production run, including weights, replay buffers,
optimizer states and RNG states, and completes its 30,000 geometry updates.
Validation runs every 500 updates from the configured minimum warmup. A resumed
checkpoint beyond that minimum is also validated as the initial candidate.

`geometry_best.pth` retains the best validated collision model. Any checkpoint
passing every existing physical acceptance check outranks every failing
checkpoint. Within either category, selection minimizes the mean geometry error
normalized by the acceptance tolerances. The ranking is recorded in
`geometry_best.json`; it does not replace any physical acceptance check.

After update 30,000, the controller restores the selected weights and repeats
the complete geometry validation. Only a passing result freezes the collision
model and starts the configured policy training (1,000 epochs for the current
wrist-pose run). A rejection is terminal and records the failed checks without
starting policy training. It does not lower thresholds or automatically restart.

Run with the original Python environment, source checkout and GPU assignment:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/continue_cart_after_geometry.py \
  --root /path/to/run/cart --run-id RUN_ID --device cuda:0 \
  --previous-training-source /path/to/cart_training.before.py
```

Stop the original controller before launching this command. The normal training
lock and a controller lock prevent concurrent trainers. `--check-only` verifies
the source/configuration migration without writing files or starting training.
The first migration requires an archived copy matching the run's recorded
training-driver hash, rejects changes to any other geometry/model/data source,
and saves the previous configuration and migration hashes. Later restarts use
the same command and resume normally.

Durable run artifacts:

- `warmup.pth`: latest training state, saved every 100 updates and at completion.
- `geometry_best.pth` / `geometry_best.json`: best candidate weights and metrics.
- `geometry_acceptance.json`: final revalidation result and selected update.
- `geometry_schedule_status.json`: `policy_ready`, `policy_training` or `rejected`.
- `geometry_controller_status.json`: controller PID and terminal errors/results.
- `config.before_geometry_schedule.json` / `geometry_schedule_migration.json`:
  original configuration and the explicit driver/schedule migration record.

The existing W&B run ID is resumed, so geometry and policy curves remain together.

## Optional convergence schedule for link-SDF v10r5

`converged_best` retains the same candidate ranking and physical gates. It
allows completion before the maximum 30000 updates after minimum5000, with
validation every500 and six checks without a cumulative1% significant
improvement, all seven consecutive checks passing. Full revalidation of the
selected weights remains mandatory. The new history/decision lives in
`geometry_convergence.json`; policy resumes do not repeat an already accepted
early geometry stage. `scripts/enable_cart_sdf_early_stop.py` verifies a
schedule-only source/config migration, preserves the original configuration and
checkpoint hash, and rejects any network/loss/data source differences. The
original full-budget protocol remains available and unchanged for v10r4.

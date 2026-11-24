# Structured collision and contact geometry

Fresh floating-wrist runs use `fk_intrinsic_sdf_smooth_features_v5`. Legacy v9
and existing v10r3 checkpoints retain their original model definitions on load.
The policy architecture, future ten-frame EMF input, CARD term expressions,
bounded loss scaling, contact retention, wrist smoothing and 5 mm guard remain
the same. What changes is the differentiable geometry inside those expressions.

Self collision now composes exact FK with the intrinsic point neural SDF. Its
interface still takes only finger q20; global wrist pose cannot affect self
collision. The fixed 8-material-point clearance sampling and 54-point guard
sampling match the native reference. Grouped clearance and penetration energies
are summed, and depth uses logsumexp. Confirmed contact groups use penetration
energy in place of the clearance margin. A two-micrometre sigmoid smooths the
moving mechanical collar exclusion. There is no separately trained q-cost MLP
and no nearest-face or nearest-link query in this learned computation.

Contact uses a fixed-shape differentiable geometric feature decoder for each
palmar pad. It evaluates the plane, three edge and three vertex features of all
pad triangles, smoothly suppresses infeasible plane extensions, and aggregates
all candidates with a softmin. A barycentric barrier handles skinny triangles;
half-space distance alone is insufficient. Projection smoothing is 5 micrometres.
The normal uses a separate 50-micrometre distance temperature. Feature-energy
calculations use float64 to avoid cancellation in the sharp softmin. This is
explicit geometry encoded as buffers, not a learned neural SDF or a claim that
an MLP has successfully learned mesh Hessians. There is no argmin, top-k,
detached witness or custom straight-through backward in this projection.

Only the intrinsic SDF is learned in geometry warmup. Contact training scalars
are diagnostic values for the fixed projection layer. Native mesh queries remain
restricted to detached supervision and independent physical validation. Frozen
policy geometry is evaluated in chunks with activation recomputation, retaining
the exact q/wrist gradient path through FK and the full policy BPTT recurrence.

`scripts/train_cart_structured.py` starts a fresh run using only a previous
wrist run's recording IDs and hyperparameters:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_cart_structured.py \
  --root /path/to/new_run/cart --run-id NEW_RUN_ID \
  --reference-config /path/to/previous_run/config.json
```

No model or optimizer weights are imported. Add `--resume` only to resume that
same run after an interruption. The automatic full-budget schedule completes
30,000 geometry updates, saves `geometry_best.pth`, restores and independently
revalidates it, and starts the configured policy epochs only if every original
geometry gate passes. Rejections remain terminal and retain their failed checks.

The isolated repair validation uses the previous run's trained point SDF solely
to verify the new geometric construction before starting the fresh run. That
diagnostic checkpoint is not an initializer for the fresh training job.

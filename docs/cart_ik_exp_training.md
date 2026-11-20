# CART v6: future IK references and exponential policy costs

The policy receives current Wuji2 q and exactly the recorded `t+1…t+10`
wuji-retargeting IK states. IK inputs are radians `[B,10,20]`, normalized by
the native allowed joint domain to `[-1,1]`; validity is `[B,10]`. EMF remains
in raw recordings for provenance, but does not enter either policy encoder.

Attention uses five four-joint finger tokens per future frame to query the
future tactile positions and binary states. FSQ receives current q, 200
future joint values, ten validity bits and ten time offsets (240 encoder
inputs), with the existing FSQ levels and straight-through estimator. Invalid
IK/tactile tokens are masked, including all-invalid windows.

The zero-initialized decoder learns a correction bounded to ±0.1 in normalized
joint coordinates. Its final output remains the normalized delta from the
current state:

```
delta = clamp(normalize(ik[t+1]) - normalize(q[t]) + .1*decoder(context), -1, 1)
q[t+1] = unnormalize(clamp(normalize(q[t]) + delta, -1, 1))
```

An invalid first reference falls back to the current state. A fresh policy
therefore follows valid IK references immediately (subject to delta/joint
limits). Predicted states feed back throughout 16-step BPTT without detach;
IK never overwrites the feedback state between steps. Tests verify that a
later-frame loss reaches an earlier correction.

Training, full-session validation and runtime use the same IK seed semantics.
Recorded playback reads timestamp-matched cached IK and never solves it again.
Live operation computes IK for each buffered actual source frame, then uses
the same ten-frame input contract. Seek, time reversal, stream discontinuity
and model switch reset IK state/cache and policy state. A short final tail is
not extrapolated.

## Policy objective

For squared error `z`, let `C(z,s) = -expm1(-z/s²)`. A recovery kernel is
`C_multi(z,s,b) = .5*C(z,s) + .5*C(z,b*s)`. The wider kernel prevents immediate
gradient saturation for the initial tens-of-millimeters contact errors.

| Term | Scale | Weight / reduction |
| --- | --- | --- |
| Unit bone direction | 30° via unit-vector chord distance | 1; mean over bones |
| Adjacent action | 0.05 normalized joint displacement | 0.1; mean over joints |
| Contact position | 10 mm and 40 mm | 2; existing semantic/force balancing |
| Contact normal | `1-cos(30°)` | 0.5; existing semantic/force balancing |
| Self / active-taxel penetration | 1 mm and 6 mm | 2; mean of the two independent costs |
| Same-link contact retention | 10 mm distance reward | 1 |

The 10 mm contact scale comes from `CardSettings.radius_mm`; the wide
penetration scale uses CARD's 6 mm mechanical margin. CARD CEM's **raw** energy
uses squared millimeter errors and large squared weights, so its raw cost is
not inherently bounded. v6 takes its geometric scales rather than copying
those energy magnitudes. No CEM/Tube solve is called by policy training.

Each exponential component is dimensionless and bounded. With the current
force multipliers and semantic reduction, the policy batch objective is below
approximately 9.725. Geometry-model regression remains metric distance,
normal, sign and Eikonal supervision; it is a separate optimizer objective.
Physical acceptance gates remain 0.5 mm contact/penetration and 30° normals,
with the existing independent native-mesh audit and rejection rules.

Epoch logs now weight each batch mean by its actual window count; the final
1206-window batch no longer has the same epoch-average weight as 4096 windows.
Checkpoint state includes that denominator for exact restoration.

## Compatibility and cache reuse

Protocol `cart_diff_ik_v6` and new architecture IDs reject older EMF policy
weights. A new run may explicitly transfer only matching scene geometry /
collision replay from v5. The Neural SDF comparison additionally imports the
same verified articulated `best.pt`, with its separate fine-tuning rate.

`scripts/migrate_cart_ik_input_cache.py` supports this specific policy-only
change. It verifies legacy fingerprints, unchanged capture-function ASTs,
calibration, raw/capture/target/IK checksums, then backs up and updates three
metadata files. It never changes raw recordings, geometry, timestamps or IK
array bytes. Other changes require ordinary preprocessing. Use private data
directories when old/new runs coexist.

Every formal replacement run must have a **new run ID**, fresh policy optimizer
and fresh W&B history (`resume=never`). Keep previous curves/checkpoints for
comparison; do not continue a v5 run under a different loss scale.

## Validation performed locally

- 94 tests passed, including bounds, masks, no EMF input, both decoders,
  finite-difference exponential gradients, BPTT, FK/SDF gradients, online
  geometry updates, cache/playback and tracking.
- Both architectures passed length-16 synthetic preprocess/train/resume/
  closed-loop replay; interrupted and geometry-initialized resume parameter
  difference was exactly zero.
- Neural-SDF attention passed the same length-16 initialization/resume/replay
  path, also with zero resume difference.
- Verified real recordings retained identical target/IK bytes after metadata
  migration; a repeated migration was a no-op.
- Real-data, batch-8 GPU optimizer probes started at policy loss **2.2055**
  (baseline) and **2.4181** (Neural SDF). These small probes and synthetic runs
  are plumbing/scale checks, not evidence of geometric acceptance or final
  convergence. Independent native validation remains mandatory.

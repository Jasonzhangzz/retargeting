# WUJI2 primitive grasp tracking

`configs/grasp_tracking_mppi.yaml` uses MPPI to search the existing 46 static
grasp parameters: wrist pose (6), finger pose (20), and independent servo
offsets (20). The hold command is evaluated using the existing lightweight
quasi-static contact prediction, with gravity. This is a predicted hold test;
it is not a measured MuJoCo or hardware grasp success.

The four cost formulas are adapted from `retarget/task/dex/cost.py`. That dex
implementation has ForceAware/DexYCB provenance. Only the humanoid cost has the
repository's frozen original-SPIDER equivalence checks; the new grasp profile
does not claim to be an official SPIDER dex implementation.

## Objective

Select `task.options.objective: dex_tracking`; the existing `card` profile
remains the default. `tracking_weights` and `tracking_contact` optionally
override the dex weights and contact thresholds. The new profile scores:

- **penetration:** squared signed-depth excess with soft and deep thresholds,
  measured over the full hand surface;
- **approach:** the existing sigmoid-gated squared distance between fixed
  template hand points and their corresponding object-local surface anchors;
- **maintain:** squared excess of the signed gap above `target_phi` for each
  annotated contact's matched patch (`nearest` mode measures the whole link);
- **object_tracking:** object position, quaternion rotation, linear velocity,
  and angular velocity errors relative to its initial pose and zero velocity.

Template contact body IDs and local points are read from the mapped Dexonomy
annotation and remain fixed during MPPI. Object anchors are produced by the
input initialization and move with the predicted object. The desired object
pose stays fixed in the world. Contact gaps are geometric surface measurements,
not contact-force measurements.

The example selects `contact_mapping: semantic_patch`. Before gesture IK it
matches Shadow palm/finger-segment surface patches to WUJI2 in intrinsic
anatomical coordinates, preserving surface side and normalized position.
Gesture fitting uses the matched WUJI2 layout at the bone-seed pose rather
than reusing Shadow's absolute hand dimensions. Subsequent discrete query
selection stays inside each original patch; an empty patch is an input failure.
The previous `nearest` mapping remains available for comparison.
The patch matcher excludes fixed-mount samples from palm contacts, filters
occluded faces, and uses longitudinal bands and palm subregions. Source chart
widths are explicit anatomical scale assumptions, recorded alongside the
measured target-hand chart scales.

All four terms are evaluated at the candidate initial state and every predicted
state, including a fresh FK evaluation at the final state. The final state's
weight is `terminal`; costs are divided by the sum of state weights. The
original initial-state geometry hard constraints remain active. CARD style,
seed smoothing, and wrist-prior soft costs are not mixed into this profile.

The example uses a 1 mm contact depth target, a 2 mm soft penetration threshold,
and a 3 mm deep threshold. These grasp-specific values are explicit overrides;
they are not the original dex scene's contact settings.

## Reproduction

Prepare the pinned WUJI2 assets and copy the local Dexonomy templates, then run:

```bash
.venv-contact/bin/python scripts/setup_grasp_assets.py \
  --dexonomy-root /home/lab423/scsp/thirdparty/Dexonomy
.venv-contact/bin/python run.py --config configs/grasp_tracking_mppi.yaml
```

The example runs template `1_Large_Diameter` on three cylinders of radius
26.25, 35, and 43.75 mm, each 120 mm high and 0.2 kg, with seed 0. MPPI uses
population 256, up to 8 iterations, 32 simultaneous worlds, and no gradient
refinement. Prediction uses four 0.1 s steps. These are three size cases of one
primitive, not a grasp success-rate benchmark.

`physics_reference: null` selects standalone primitive contact parameters and
removes the dependency on a prepared DexYCB scene. `wrist_pose_prior: false`
disables the missing frozen-wrist-reference dependency. Asset source hashes,
model adaptation, and surface sample generation are recorded by the setup
script; no frozen wrist pose is fabricated.

## Results

Each case contains `objective.json`, `tracking_costs.npz`, `summary.json`, the
optimized grasp, predicted trajectory, and iteration history. The cost archive
contains initial/final per-term and per-state values plus penetration, contact
gap, and template-distance metrics. A predicted grasp passes only if the
initial geometry constraints pass, the object stays within 2 mm / 5 degrees,
the guided link gaps stay within 2 mm, and predicted penetration stays within
2 mm. `physical_hold_tested` remains false.

`semantic_matching.json` records source/target patch identities, normalized
coordinates, and the selected real surface queries. `semantic_patches.npz`
stores the allowed query masks. Initialization diagnostics verify patch
membership after all fitting stages.

## Small sample run (2026-10-31)

The three cases completed on the local RTX 4090 with the configuration above:
2,048 sampled candidates and 2,049 evaluations per case. The final output is
`output/grasp_tracking_mppi_semantic`; the previous nearest-mapping run remains
in `output/grasp_tracking_mppi`. The comparison includes both semantic input
matching and patch-specific maintenance gaps; it does not isolate their
individual effects or estimate a success rate.

| Cylinder radius (mm) | Nearest mapping: max drift (mm) | Semantic patches: max drift (mm) | Semantic: max rotation (deg) | Semantic: max penetration (mm) |
| ---: | ---: | ---: | ---: | ---: |
| 26.25 | 52.01 | 23.77 | 3.46 | 11.71 |
| 35.00 | 36.46 | 19.40 | 3.41 | 11.69 |
| 43.75 | 39.53 | 15.42 | 5.84 | 17.79 |

All final candidates pass the initial-state geometry hard constraints, and all
16 template contacts retain their assigned semantic patch through input
fitting. The largest case's original mean violates the stricter search-stage
geometry checks; its summary records that initial hard score as null and keeps
the raw initial tracking-term sum separately. MPPI finds a feasible candidate.

None of the three cases passes the predicted hold criteria: maximum guided
patch gaps are 23.91, 16.04, and 16.17 mm, and predicted penetration remains
above 2 mm. These are completed optimization runs with improved drift, not
successful stable grasps. No physical hold test was run.

### Failure diagnosis

Re-evaluating the saved candidates shows that only the thumb and index carry
nonzero predicted normal force at the initial state in all three cases.
Initial upward contact forces are 1.295, 1.264, and 1.562 N, below the 1.962 N
object weight. The other fingers and palm do not establish the intended power
grasp support. Geometry proximity alone therefore does not prove load support.

The inner contact solve also does not converge at 64 iterations. For the
unchanged 35 mm-radius candidate, increasing only the contact iterations to
1024 reduces the first-step linear normal constraint violation from 7.86 mm
to about 0.00012 mm and maximum predicted penetration from 11.69 to 1.57 mm.
Object drift remains 23.58 mm, and initial upward force remains only 1.272 N.
Thus poor support persists after this convergence check; the original large
penetration additionally reflects an under-converged prediction. The previous
low-budget scores should not be used as evidence of physically reliable
grasping. No MPPI reoptimization was performed in this diagnostic.

Full per-patch distances, predicted forces, and the contact-iteration comparison
are saved in the output directory as `contact_diagnosis.json`,
`contact_force_diagnosis.json`, and `contact_convergence_diagnosis.json`.

# CART training objective audit — 2026-11-19

## Scope and reproducibility

Read-only audit of the CART implementation at `f1d1c44a`. No production
training, recording, Tube asset or remote run was changed. The added
`scripts/audit_cart_objective.py` loads existing observations and checkpoints;
it never invokes preprocessing, `prepare_ik`, Tube or `train`.

The numerical experiments use the **local geometry initialization checkpoint**
`20261118T153108_5fd68a42/resume.pth`, not the latest remote policy or geometry.
Remote SSH connections repeatedly timed out during banner exchange during
this audit. These numbers establish initial-objective and implementation
problems; they are not a measurement of the remote models' current quality.

Independent recording: `20261118T135558_05a2628d`. Eight uniformly spaced
16-step windows give 128 window-frame evaluations; overlapping frames are not
claimed to be independent observations. All states and targets stay in their
recorded sessions, and cached original wuji-retargeting IK is joined by exact
timestamp. Direct q optimization experiments optimize independent frame
variables as a diagnostic. They are **not BPTT policy training**, a quality
certificate, a CEM solve or a trajectory cache.

Environment: local RTX 5060 Ti, Python environment
`/home/zz/miniconda3/envs/pi0`, PyTorch 2.10.0+cu128, Warp 1.17.0.

Example command:

```bash
/home/zz/miniconda3/envs/pi0/bin/python scripts/audit_cart_objective.py \
  --config output/moxian/cart/models/20261118T153108_5fd68a42/config.json \
  --checkpoint output/moxian/cart/models/20261118T153108_5fd68a42/resume.pth \
  --windows 8 --optimize-steps 200 \
  --output output/moxian/cart/audit_20261119/direct_optimization_v2.json
```

For the mesh-witness reference experiment add
`--optimization-geometry native-contact`. It uses fresh, detached native
surface witnesses and FK gradients for contact/normal losses; penetration
still uses the learned field. It is not a fully differentiable native mesh
optimizer and cannot establish a global optimum or infeasibility.

## 1. Loss scales are inconsistent with the intended balance

Current policy objective:

```
L = .05 * sum_bones ||unit(robot bone)-unit(source bone)||²
  + .005 * sum_joints ||q_next-q||²
  + 20 * semantic_mean(rho² * contact_position_error² / (1 mm)²)
  + 5 * semantic_mean(rho² * (1-normal_dot)/(1-cos(30 degrees)))
  + 100 * ((self_penetration/.5 mm)² + (taxel_penetration/.5 mm)²)
  + contact_keep_loss
```

Window/time losses are averaged. `rho` lies between 1 and 1.5 for object
contacts. Self-contact position also has a 0.02-weighted center tracking term.
The learned penetration depths are smooth maxima, whereas native audit uses
the maximum native depth. Those losses therefore need not numerically match
even with perfect proxy geometry.

The direction and smoothness weights were inherited from the original
retargeting optimization, while contact and penetration are normalized by
very tight millimeter tolerances and then multiplied by 20 and 100. Input
standardization does not equalize these objective gradients.

At exact per-frame IK, weighted gradient norms with respect to q are:

| Term | q-gradient L2 | Weighted scalar contribution |
| --- | ---: | ---: |
| Direction | 0.018405 | 0.09534 |
| Smoothness | 0.00005718 | 0.00002093 |
| Contact position | 3911.61 | 6556.59 |
| Contact normal | 4.718 | 6.684 |
| Penetration | 1718.60 | 454.67 |
| Contact keeping | 0.2263 | 0.9060 |

Contact-position gradient is about **212,500 times** direction gradient.
This is an actual gradient measurement, not an inference from scalar losses.
The shared policy gradient is also globally clipped to norm 10; clipping
does not restore relative contributions of the terms.

Ignoring other terms, a representative 0.5 mm penetration already contributes
100, and a 0.5 mm contact gap contributes 5 before `rho`. A single-term loss
below one corresponds to penetration under 0.05 mm or contact error under
0.224 mm before `rho`. Total loss below one is therefore not equivalent to the
specified 0.5 mm acceptance criterion. Simply dividing the whole objective by
a large constant would not repair the term imbalance or proxy bias.

Velocity/acceleration use real timestamps but acceleration's training weight
is **zero**. The very weak raw-radian adjacent-state penalty is the only
explicit action smoothness penalty in the active objective.

## 2. VLA and SuperMimic comparisons

OpenPI pi0 normalizes actions by dataset statistics (mean/std or quantiles),
then optimizes the mean squared error of a flow-matching velocity target
`noise-actions`. The mean is over action dimensions, followed by the training
reduction. This is not the same objective as millimeter-tolerance geometry.
Neither MSE nor model size guarantees initial loss below one.

Sources:
- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py
- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py

SuperMimic official source was inspected at
`698ea39bc901d8fe733b3c8621df3199d241f7e4`. Its tracking terms use
`weight*exp(-error_squared/std_squared)`, summed with weights totaling five,
then subtract action-rate and joint-limit penalties and multiply by dt.
The actor minimizes the batch mean of negative accumulated return plus
auxiliary losses. Positive tracking reward is bounded; the complete objective
is not globally bounded because of penalties/auxiliary terms/critic variants.
Logged policy reward is additionally divided by horizon and dt. Input
observations/goals have empirical normalizers. BPTT does not itself determine
the numeric loss scale.

Sources:
- https://github.com/HybridRobotics/supermimic/blob/698ea39bc901d8fe733b3c8621df3199d241f7e4/src/supermimic/utils/tracking_reward.py
- https://github.com/HybridRobotics/supermimic/blob/698ea39bc901d8fe733b3c8621df3199d241f7e4/src/supermimic/learners/policy_learner.py
- https://github.com/HybridRobotics/supermimic/blob/698ea39bc901d8fe733b3c8621df3199d241f7e4/src/supermimic/models.py

Do not replace all errors with narrow Gaussian rewards without considering
saturation: a 30 mm error at a 1 mm reward scale has almost no useful gradient.
Use meaningful task scales/curricula and retain penetration safety costs.

## 3. Existing IK is not a good solution to the current absolute contact cost

Using exact per-frame cached IK, without policy prediction or reset noise:

| Metric | Result |
| --- | ---: |
| Learned geometry loss | 7018.95 |
| Native geometry loss | 11543.37 |
| Native contact-gap P95 over resolved frames | 37.93 mm |
| Native maximum penetration | 4.81 mm |
| Mean bone-axis error | 15.48 degrees |
| Mean fingertip position difference, source vs robot | 34.01 mm |

Original wuji-retargeting IK optimizes its own Laplacian/direction/pose priors,
not CART's absolute glove-surface contact positions with 1 mm scaling. The
glove and native robot have different morphology. This experiment does not
prove the targets are impossible; a conservative reachability lower bound
did not prove any of the sampled object targets unreachable.

An additional landmark-definition mismatch exists: the original solver uses
the proximal/flexion joint origin for the first landmark in each finger;
CART's direction FK uses the next abduction origin. With the same q and each
model's own neutral canonical basis, these landmarks differ by about
14.5–14.9 mm. Distal joint origins agree to roughly 0.17–0.32 mm; tip sites
differ by about 1–2.6 mm. This is a landmark/asset-definition difference, not
evidence of a 329 mm physical FK bug (comparing native coordinates without
aligning the two models' root axes would produce a spurious large error).

## 4. Geometry proxy bias, insufficient on-policy coverage

Remove the policy and directly optimize all 128 q variables for 200 Adam
steps at lr .005, keeping geometry frozen:

| Objective / result | Initial | Final |
| --- | ---: | ---: |
| Learned total | 7018.95 | 269.13 |
| Native total | 11543.37 | 6525.32 |
| Native contact-gap P95 | 37.93 mm | 28.21 mm |

The q variables are easy to optimize under the learned objective, but that
objective is a poor proxy for native geometry. The model is being optimized
outside sufficiently accurate learned contact correspondences. This is not
an assertion that gradients are absent: contact-gradient descent improved
native contact at perturbations of 1e-4, 1e-3 and 1e-2 rad near IK. Local
descent direction can be useful while final proxy optima are badly biased.

On 457 actual object queries at IK:

- Teacher considers only 70.68% valid, native audit 99.34%.
- Learned contact-hit P95 error on teacher-valid queries: **23.43 mm**.
- Learned contact-hit P95 error on audit-valid queries: **22.92 mm**.
- Where both are valid, teacher/audit hit differences P95 are zero in this
  sample. Their **validity domains**, rather than jointly valid hits, disagree.
- After proxy q optimization, teacher validity falls to 65.43%; hit errors
  remain about 19–22 mm.

The regular contact holdout samples near native surfaces with normals close
to true surface normals. It does not cover the full distribution of far,
misaligned source contact queries produced by initial IK/rollouts. Its
roughly 5 mm P95 does not bound the above on-policy errors.

Concrete code mismatch: `ContactTeacher._contact_labels` also traces/rejects
outside queries whose nearest normal disagrees with the direction. Native
`contact_witness` only invokes the directional retraction inside the solid.
Rejected teacher samples are omitted from replay; policy contact evaluation
nevertheless assumes every learned contact prediction is valid. This permits
unsupported surface predictions to contribute optimistic loss/gradients.

A reference optimizer refreshing native contact witnesses every step (still
using the learned penetration field) reduces native contact loss from 532.96
to 24.82 and gap P95 to 15.23 mm in 200 steps. Its native total is 2004.14,
with maximum penetration 6.91 mm. This reinforces the geometry-proxy problem,
but is not a valid final robot policy or a proof that all constraints are
simultaneously achievable.

## 5. Initial-state distribution and reference-use mismatch

- Training windows seed q from cached IK plus up to 0.1 rad noise, then use
  uninterrupted 16-step policy delta feedback. This correctly retains BPTT.
- IK is used only at the window start. Later per-frame IK states/deltas are
  neither a reference motion input nor a residual baseline. A near-zero
  initial policy holds the initial pose and must learn motion tracking anew.
- Full-session validation and real-time runtime reset to neutral instead of
  the training IK state distribution. There is no corresponding explicit
  neutral-state curriculum in the current formal runs.
- Training windows reset to reference states every 16 steps; full closed-loop
  deployment may run for thousands of steps without such resets.

Thus current code is not simply a small correction to an already-followed IK
trajectory. A coherent design must either use the reference IK motion as a
baseline/conditioning, or teach the policy to track it and recover from all
deployment initial states. Replacing q with IK at every rollout step would
silently break the intended feedback/BPTT formulation and is not a fix.

## 6. Update budget and logging

5302 unique training windows with batch 4096 means only two optimizer steps
per epoch, or **400 total policy updates in 200 epochs**. Epoch count is not
comparable to SuperMimic training iterations. Overlapping windows are also
correlated, so increasing batch size does not add independent motions.

The `losses` epoch log averages minibatch means equally even though the two
batches contain 4096 and 1206 windows. It is not a sample-weighted epoch loss.
This affects that display, not the underlying per-batch objective. The
previous progress response used separately computed sample-weighted means.
W&B `train/*` currently logs the individual batch losses.

## Checks passing / remaining limits

48 tests passed across `test_cart_differentiable.py`,
`test_cart_initialization.py`, `test_cart_fast.py`, and
`test_cart_pretrained_sdf.py`. They cover normalization/masks, exact future
window packing, BPTT feedback, finite-difference FK/SDF gradients, frozen
geometry during policy updates, online geometry updates and state restore.
They do not certify geometry approximation quality or convergence.

## Corrective order

1. Unify teacher/native directional-contact definitions and log coverage on
   **actual rollout queries**, including rejection reasons. Train contact
   distances/normals/correspondences and check their input gradients near the
   actual operating manifold before trusting them for policy optimization.
   Enforce consistency of predicted hits with the distance field/surface.
2. Make the contact reference compatible with robot morphology and the
   intended good IK/CARD solution; align landmark definitions explicitly.
   Audit IK/CARD/native baselines before training a policy.
3. Replace the mixed inherited optimization weights with dimensionless,
   explicitly reduced terms and task-scale rewards/costs; measure per-term
   gradient contributions. Keep physical mm/degrees diagnostics unchanged.
4. Choose coherent reference-residual behavior and initial-state semantics
   shared by training, periodic validation, full-session evaluation and
   deployment; retain predicted-state feedback and BPTT.
5. Verify a single real minibatch can improve **native** geometry, then compare
   runs by optimizer updates/wall time, rather than increasing epochs blindly.

No production loss/configuration was changed mid-run, and no old/new loss
curves were spliced together.

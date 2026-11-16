> 历史实验记录：文中部分脚本已于2026-11-16清理；当前基线入口见 [baseline说明](../retarget/baseline/README.md)。

# Contact-space CEM

`solver.type: contact_cem` selects an explicit `ContactProblem`, rather than
silently interpreting a vector of joint controls as a contact strategy. Existing
CEM and GraspADMM entry points retain their original defaults.

The migrated benchmarks are self-contained: Sharpa pen, Sharpa coin and Unitree
H1 jog. Their scene definitions, reference schedules, actuator settings and
acceptance are adapted from OpenRoboKit commit
`736f112df93eb40930219501246032605202684a`. Asset revisions, original task hashes
and asset hashes are in `retarget/task/contact/SOURCE.json`. The library snapshot
is the user's local Robokit working tree based on that commit, not a claim of a
pristine checkout; `third_party/robokit/SOURCE.json` records its authoritative
per-file hashes. The vendored library files are unchanged. The MIT task-code
license, Apache Sharpa license and BSD H1 license are included.

## Installation and execution

Use two isolated Python 3.11 environments. Do not upgrade the GPU process to
MuJoCo 4: this pinned Warp version does not implement speculative contacts.

```bash
uv venv --python 3.11 .venv-contact
uv pip install --python .venv-contact/bin/python -r requirements.txt
bash scripts/setup_contact.sh

CUDA_VISIBLE_DEVICES=0 uv run --no-project --python .venv-contact/bin/python \
  python scripts/run_contact_benchmark.py --task sharpa_pen \
  --method contact_cem --output-name pen_contact --headless --save-video
```

The same command accepts `sharpa_coin`, `h1_jog`, `cem`, and `source_sampling`.
`--steps` creates a partial diagnostic, which can never pass full-task acceptance.
Output names must be new, preventing accidental replacement of existing results.
Canonical sample budgets and horizons come from the migrated original example factories in `retarget/task/contact/source/`; `tasks.py` exposes their static configuration summaries. Production
CLIs do not accept sample-count or iteration overrides.

The unified entry point also accepts the three `configs/contact_*.yaml` files.
Saved trajectories can be rendered or replayed with:

```bash
uv run --no-project --python .venv-contact/bin/python python \
  scripts/run_contact_benchmark.py --replay output/pen_contact
```

The fixed campaign is sequential, uses seeds 0/1/2 plus coin seed 10000, and
includes all three controllers and H1 reference tracking (39 jobs):

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python .venv-contact/bin/python \
  python scripts/run_contact_benchmark.py --campaign \
  --output-name contact_campaign --headless --save-video
```

Use `--campaign --resume --output-name contact_campaign` to continue unfinished
jobs; retries receive new directories and do not replace existing artifacts.
After the campaign, add executed contact/object metrics and joint actuator
torques derived from the recorded forces, transmission and joint force limits:

```bash
uv run --no-project --python .venv-contact/bin/python python \
  scripts/analyze_contact_benchmark.py output/contact_campaign
```

## Search and physics

The implementation follows the [official speculative pipeline](https://mujoco.readthedocs.io/en/4.0.0/computation/speculative.html).
In the pinned [4.0.0 continuous-collision source](https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_collision_continuous.c),
`mjc_advanceGeomPair` advances geom pairs, `evalGeomPairDist` calls GJK,
`isRayOccluded` filters obstructed sweeps, and `contactDist` constructs the
solver-effective distance. The [collision driver](https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_collision_driver.c)
dispatches these speculative pairs into the ordinary constraint pipeline.
Our diagnostic predictor uses public APIs; the native engine retains ownership
of constraint construction.

Each horizon discovers nearby link/object pairs using native distances. Distinct
finger or foot branches retain independent Gaussian distributions and elites.
The 2026-11-10 H1 audit found duplicate foot-order templates, a force penalty that
taxed normal support, and a decoder whose zero parameters changed the nominal
controls. The fixes change actual per-foot stance intervals, normalize H1 impact
regularization by body weight, and make zero contact residuals preserve the
whole-body nominal exactly. Stance normal velocity is zero; the separately
recorded `approach_normal_velocity` describes pre-impact motion. Unused event,
plane-normal and stick-tangent variables have zero sampling spread.
The original failure evidence and finite-difference replay are in
[`output/contact_failure_audit_20261110/REPORT.md`](../output/contact_failure_audit_20261110/REPORT.md).
The continuous variables are surface-point offsets, start time, duration,
approach direction, normal velocity and tangent velocity. Surface points are
projected onto primitives or the convex hull of a mesh part. Nonconvex objects
must be decomposed into convex geoms before loading.

At most three events are active in a proposal. Robokit `Robot`, `IKSolver`,
`LMOptimizerConfig`, `PositionTask`, `RotationTask`, `PositionLimit`,
`ConfigurationTask` and `SmoothnessTask` solve contact-frame goals with a
reference-posture prior. The IK-only model locks the measured/reference floating
base as a legal pose; executed physics retains the complete original model. The approach waypoint encodes
velocity; rolling adds a second surface waypoint with its changed normal.
Unactuated free joints are not teleported during execution. Stick suppresses
tangential target motion, slide tracks it, and roll follows a surface path while
penalizing contact slip. These are branch-specific geometric/dynamic objectives,
not a claim of enforcing an exact nonholonomic rolling constraint.

GPU proposals use independent native Warp states, restore every persistent
integration field, and recompute contacts at each step. Geometry, force peaks,
slip, friction, joints, acceleration, reference tracking and unexpected contacts
enter the cost. CPU4 adds observed contact timing. Candidate corrections are
always rolled out again; forces are never borrowed from a different trajectory.

Eight distinct shortlisted candidates, preserving each branch where possible,
and the previous plan are evaluated by a persistent CPU MuJoCo 4 process. CPU4
dispatches these rollouts to up to nine independent CPU processes, bounded by
the available CPU count; serial/parallel costs, states and predictions are
tested for exact equality. The execution state remains in the service process.
CPU4
cost selects the winner; CPU4 also executes the first controls and feeds the
measured state back into planning. No valid candidate produces an explicit
failure with partial artifacts. Only named arrays and controls cross processes;
MuJoCo binary models and internal pointers do not.

CPU4 enables `mjENBL_SPECULATIVE` and native CCD. Its engine performs swept
broadphase, GJK/conservative advancement, occlusion checks and effective-distance
constraint construction. The independent public-API predictor evaluates supplied
piecewise linear/geodesic geom paths. Before native validation it queries the
commanded joint path with measured free-body twist; afterwards it queries the
realized trajectory. Neither approximation guarantees actuator tracking.
Predictions expose `hit`, `no_collision`, `occluded`, or `uncertain`. Unresolved
command-path forecasts retain UNKNOWN and are adjudicated by the mandatory
native speculative rollout; they are never relabeled as no collision. Recorded
fallback metadata distinguishes observed native constraints, realized geometric
queries and engine-only evaluation. A kinematic proxy's iteration cap alone is
not a certificate that a dynamically simulated candidate is infeasible.

Native speculative sweeps use pre-step velocity, without new acceleration in
that step. Multi-step dynamics therefore re-predict on every step. A force that
accelerates a stationary body through a thin wall can still tunnel in one step;
the test suite includes this limitation. `effective_distance` and
`geometric_distance` are recorded separately. Contact force comes exclusively
from the constraint solve, never from TOI.

All three methods use the vendored original Robokit library in
`third_party/robokit`, with its per-file hashes in `SOURCE.json`.
`source_sampling` calls the original `SimMPC._optimize`, `GaussianSampler`,
interpolation, warm shift, task terms and `MjWarpSim` rollout. The migrated
factory prefixes preserve original initialization, callbacks, sensors and
configuration; local paths, bounded initialization retries and factory extraction
are listed in `retarget/task/contact/source/SOURCE_PATCHES.json`.

Full-joint CEM extends `GaussianSampler._move` with finite uniform elites and
uses the original `PopulationSolver` lifecycle. Contact CEM uses that same
sampler and lifecycle over contact parameters, then the original IK/LM and
SimMPC rollout/cost graphs. Neither path uses an independently rewritten MPPI,
physics rollout or analytic IK engine. The original Robokit loop selection kernel
handles populations above CUDA's 1024-thread block limit. CEM ends on the scored
last proposal batch, preserving the equal rollout budget; original MPPI retains
its final moved-mean evaluation and original coarse schedule.

CPU4 bridges named arrays to the original `SensorTask`, `ControlTask` and
`SupportTask` classes on Warp CPU; it adds the requested contact diagnostics and
penalties. CPU shortlist selection and native speculative execution are shared
extensions and differ from the original GPU-only examples. Results are therefore
not claimed to be a bitwise replay of those examples. Earlier independent
controller implementations and their results are retained only as diagnostics
under `output/contact_campaign_20261109_run`; they are excluded from final
Robokit-based comparisons.

H1 contact CEM now searches residuals about an online whole-body nominal supplied
by the original Robokit `SimMPC`, `GaussianSampler` and simulation terms through
`SourceNominalPlanner`. Generation uses the original jog goals. Tracking uses
only the frozen artifact's state, sensor and control targets; the new optimizer
does not read the reference generator's internal plans. Tracking adds explicit
root-height, root-orientation and joint-state errors, uses the original fine
rollout stages to preserve the frozen trace's intermediate controls, and retains
the best scored source incumbent across iterations. It also initializes lazily
created validation graphs with the current reference window.

This H1 method is a hybrid of whole-body source MPC and contact residual search.
It must not be presented as contact parameters independently synthesizing a gait.
The extra source-MPC rollouts, cold rollouts and time are recorded separately as
`nominal_gpu_rollouts`, `nominal_cold_gpu_rollouts` and `nominal_seconds`; total
planning latency is in `planning_seconds` / `steady_planning_mean_seconds`.
The contact-search budget and eight-candidate-plus-previous CPU4 quota are
unchanged. The zero-residual incumbent shares those eight slots and never skips
native validation. Comparisons must include the extra nominal computation.

Additional costs now use the integration input state consistently with the
actual step's force/geometry and original source sensor costs, on GPU and CPU4.
Source GPU scores are horizon means and CPU4 scores are horizon sums; they must
be normalized before numerical cross-backend comparisons. Actual integration
forces and post-forward execution labels remain distinct quantities.

## Evaluation and artifacts

Success is separate from optimization cost. Pen requires 15 seconds, at least
three turns and no 3 cm drop. Coin preserves the source round-trip, 150-degree
flips, penetration, self-overlap, jitter and drop criteria. H1 requires all eight
seconds, velocity error below 0.025 and no fall.
The source H1 error is the time mean of squared planar-velocity error plus
`(yaw_rate_error * pi / 180)**2`; the implementation preserves that source
scoring convention and threshold.

H1 source runs freeze a `reference.npz` with 401 states, 400 controls, joint names,
timestep, sensors and actual foot-contact labels. Tracking reads only that artifact and
applies 2 cm translation, 2-degree root tilt and seeded joint noise in ±0.02 rad.
Its velocity error is relative to the frozen motion's actual velocity; the
original 2 m/s command error is also retained. Its success additionally uses the
existing humanoid balance assessment. Contact IK uses the reference root at each
future waypoint and applies joint-pose corrections on top of the saved control
feedforward. A tracking pass establishes perturbed robot-reference tracking;
it does not establish original-command success or human-to-robot retargeting.
Foot-contact labels mean observed native normal constraint force above 2 mN.
Native force may act across a positive geometric gap because of contact margins
or speculative constraints. The audit separately counts force-bearing records
whose geom pairs geometrically touch (`mj_geomDistance <= 0`) and those with
positive gaps; full records preserve both geometric and effective distances.

Each run saves task/config/backend versions, relative-path scenes, trajectory,
warm starts, actuator forces, actual contacts and geometric/effective distances,
chosen events, predicted and observed impact times, every CPU validation cost,
optimizer history, step counts, cold/steady timing and optional video. Runtime
errors preserve partial results. Campaign `REPORT.md` includes every run, not
only successes. GPU and CPU4 rollouts are counted separately; LM and prediction
overhead are included in planning time.
New runs also record phase times for candidate discovery, contact LM, GPU
rollout and CPU4 validation. CPU candidate diagnostics separate commanded-path
forecasting, dynamics/cost and realized-path prediction. Concurrent candidate
times describe CPU work and are not additive with validation wall time.
`startup_seconds` covers scene/backend initialization; cold time to the first
control additionally includes first-window preparation and planning. H1's first
planning window intentionally has ten iterations versus four at steady state.

Checks:

```bash
uv run --no-project --python .venv-contact4/bin/python python -m pytest \
  tests/contact/test_prediction.py -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python .venv-contact/bin/python \
  python -m pytest tests/contact tests/core/test_solver_defaults.py \
  tests/core/test_unified_config.py -q -p no:cacheprovider
```

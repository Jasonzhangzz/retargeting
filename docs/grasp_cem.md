# Static CARD CEM grasp optimization

Grasp CLI defaults to `--hand-model wuji2`: the official WUJI Hand 2.2 bare
right hand, pinned in `assets/wuji2/source.json`. Use `--hand-model wuji1` for
the original model. The adapter keeps the six wrist coordinates and canonical
interface names while preserving Hand 2's actual 20 joint transforms, limits,
inertias and actuator parameters. Five fixed sensor pads and the mount are
included in collision geometry. Its 8880 surface queries are regenerated from
these meshes; no first-generation contact points are reused. Archived scenes
identify their own hand through an XML numeric marker (absent on old Hand 1
scenes), so replay and refinement keep the original model. The adapter selects
Euler integration, which the native ComFree backend requires; the unmodified
upstream MJCF remains available under `assets/wuji2/official/right.xml`.

`run_grasp.py` defaults to `--grasp-optimizer cem` on GPU. This completely
replaces the previous command-offset CEM/contact-adjoint algorithm.
`--grasp-optimizer adamw` and `--legacy-serial` retain the common dynamics solver.

The current default also enables `--explicit-pose`: a differentiable compliant contact predictor derived from the
[ASU explicit contact model](https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation/blob/87b320ffb4b40ab4f3aa344371f6e5e8d619aa7e/models/explicit_model.py)
adds a fifth, predicted-object-pose cost to CEM and supplies local refinement
gradients. `--no-explicit-pose` restores the previous four-cost geometric CARD
ablation. Old archived settings without this flag retain their original mode
when profiled or replayed; the refinement comparison script enables the new
mode unless explicitly disabled.

The implementation follows [CARD §3.3 and Appendix D](https://arxiv.org/html/2610.36539#S3.SS3),
with explicit adaptations for the existing static Dexonomy inputs. It is not a
reproduction of the paper's video, MANO sequence, force-estimation or tactile
controller pipeline. No success rate or timing from the previous dynamics CEM
applies to this implementation.

## Search and objective

The search variables are the wrist position, a local SO(3) rotation about the
input wrist orientation, and 20 Wuji finger joint angles. The object stays at
its input pose. Euler XYZ conversion is used only at the MuJoCo qpos boundary.
Candidates obey model joint limits, a 20 mm translation displacement cap and
0.25 rad per-joint displacement caps relative to the initialization.

Defaults are 1024 candidates, 16 generations, top 5% elites (rounded up, 52 at
the default population), temperature 0.1, and initial translation/rotation/joint
noise of 2 mm / 0.016 rad / 0.05 rad. The elite costs are standardized before a
stable softmax mean update. Noise is fixed by `sigma_i = sigma_0 * gamma**(i/K)`,
with `gamma=0.2`, `i=0,...,K-1`; the distribution at index K has the final noise
scale. It is not fitted from elite variance. An incumbent and, when population
permits, the mean are re-evaluated each generation; the best valid sample seen
is returned. The geometric CARD ablation has no rollout or gradient. The
default then uses the explicit prediction/refinement described below. Neither
CEM mode invokes the shared executor's dynamics adjoint, contact LM or ADMM.

All four cost groups use the paper's table-5 weights, multiplying residuals
before squaring. Geometric distances are converted from metres to millimetres
inside the objective; joint and rotation residuals remain radians.

- **Contact:** select a real pad sample per semantic link using proximity plus
  the remapped template contact preference (alpha 0.15 on middle/distal links,
  zero on proximal/palm). The preference uses the mapped fractional bone
  coordinate on the palmar centreline. Contact candidates are restricted to
  palmar query samples (normal dot local belly axis > 0.25), while penetration
  checks use the entire surface. Score reach, pad-normal/push-direction alignment and
  anti-slip at that selected point. Multiple annotations on one body are merged
  into one target, with force weighting when supplied.
- **Style:** preserve initialized fingertip pad apices, distal pad normals and
  the 15 semantic finger-bone directions in task space.
- **Smoothness:** preserve initialized finger angles, wrist translation and
  wrist rotation in the local SO(3) chart.
- **Collision:** query all hand surface samples (Hand 1: 5590; Hand 2: 8880) against the real object
  mesh SDF; use the existing inter-finger collision capsules for the soft
  clearance penalty. Finger capsules are enlarged as needed to enclose every
  vertex of the actual link mesh. Nonadjacent finger capsules are checked against the real
  palm mesh using nine points along each capsule segment and its radius.
  The adjacent base/proximal articulation is excluded from palm clearance;
  Wuji has an extra abduction base before each semantic proximal link.

## Explicit prediction and pose gradient

For each candidate, the object starts at the initial pose. Its target stays at
that pose for the complete prediction. The candidate has 26 posture variables
and **20 independent finger servo offsets**, bounded by
`--explicit-preload-limit` (default 0.5 rad) and actual actuator command limits.
The hold command is `u = q_candidate + offset`; geometry and export continue to
use the candidate posture, not the unloaded servo target. The clipped command
is saved independently, including its offsets in compact schema 4.

The lightweight model uses the actual hand FK and actuator stiffness. At every
prediction step it queries the current real object mesh and reselects an allowed,
facing hand-surface sample per collision body. Forbidden skin cannot contribute
force. Only contacts within 1 mm can enter the solver (the cutoff option may
reduce this margin, but cannot increase it). The actual signed separation enters
the contact solve. The positive half-gap/centimetre softplus response has been
removed. This is a sampled contact detector, not continuous mesh/mesh collision.
Discrete contact identities have no derivative; derivatives apply to the current
contact branch, with separate saved selections at every prediction step.

```
hold = clip(q_candidate + servo_offset - hand_gravity(q_candidate) / K_hand)
b = [mass * gravity, 0,
     clip(K_hand * (hold - predicted_hand), actuator_force_limits)
     + hand_gravity(predicted_hand)]
d_free = inverse(Q) * b
normal_force = max(normal_force - step * (J_normal * d + gap), 0)
tangent_force = project_Coulomb_pyramid(
    tangent_force - step * J_tangent * d, mu * normal_force)
d = d_free + inverse(Q) * J.T * contact_force
v = d / dt
```

The coupled contact iteration updates normal complementarity independently of
tangential friction, then projects tangential force into the four-ray Coulomb
pyramid. Tangential slip cannot generate normal compression. There are 64
under-relaxed projected steps, with `--explicit-sigma` in (0, 1]. The trace of the
contact admittance in normal/tangent coordinates bounds the step. Separate
iteration buffers preserve reverse AD through the solve. Gravity and servo
compression generate the forces; no independent force-balance objective or
freely assignable force variable is added. A separated contact outside the
activation margin has exactly zero Jacobian, ray and force.

Hand gravity is computed from every body mass and COM using the current FK and
joint Jacobians, including the wrist. The initial hold command compensates it,
and subsequent predicted states include their changing gravity torques. Actuator
command and force limits are applied. Both the saved command and the model use
this same convention. Physical formation keeps gravity enabled for these
compensated commands, settles with the object temporarily fixed, then releases
that fixture completely. Releasing an initially unloaded hand immediately is a
separate transient test; it does not establish a physically preloaded grasp.

`Q` remains a quasi-static displacement metric, not rigid-body inertia. Its
translation/rotation defaults are 50 and 0.1; the hand block uses inverse actual
position stiffness. Object mass enters gravity. This remains a compliant quasi-static approximation
without an inertial momentum state. Physical execution validation remains separate.

The motion cost tracks every predicted step (averaged over the horizon):

```
pose_cost = mean_over_steps(explicit_pose_weight * (
    5000 * squared_norm(predicted_position - initial_position)
    + 50 * (1 - dot(predicted_unit_quaternion, initial_unit_quaternion)**2)))
```

The initial object pose is the fixed target; predicted quaternions are normalized.
The default overall weight remains `1e6`; default horizon remains four 0.1 s steps.
Both contact attraction and motion cost are needed: attraction makes disconnected
fingers approach the IK contact targets, while motion gradients adjust posture
and compression through contact dynamics. The default is 32 local refinement iterations. The first half optimizes servo offsets with posture fixed;
the second half permits joint posture/preload refinement. Local reverse AD includes the
same contact, style and smoothness costs as CEM in addition to object motion.
The complete original objective and geometry hard checks accept or reject every
proposal. CEM population, generations and posture search caps remain unchanged;
only explicit mode appends servo offsets to its local variables. During CEM
these offsets stay zero and prediction is disabled. After the unchanged
geometric search, the selected posture is rescored with contact dynamics; this
complete score initializes local refinement. Thus motion error cannot displace
the contact-acquisition objective before contacts have formed.

Prediction acceptance checks the **maximum** translation/rotation over the whole
horizon (defaults 2 mm / 5 degrees), separately from numerical optimizer success.
An unstable prediction cannot pass final grasp acceptance even if it lowers cost.
`--explicit-motion-tolerance-mm` and `--explicit-rotation-tolerance-deg` control
these thresholds; this is model acceptance, not physical retention certification.

Useful controls additionally include `--explicit-contact-iterations`,
`--explicit-preload-limit`, `--explicit-refinement-steps`,
`--explicit-refinement-step-size` and `--explicit-line-search-steps`.

Hand 2's flat MCP housings produce false overlaps with a single enclosing
capsule. Its hard inter-finger check therefore also accepts an enclosing-box
separating-axis certificate with the same minimum 0.5 mm clearance. Both outer
volumes include every attached mesh vertex, including fixed sensor pads. A
separation certificate from either volume guarantees separation of the meshes;
overlapping boxes alone do not pass. Capsule soft penalties and all original
final audit thresholds remain unchanged. The legacy capsule audit may still
report approximate overlaps on a pose with certified mesh separation; both
results are recorded separately, and final feasibility still requires the
existing independent acceptance checks.

Object penetration above 2 mm has a soft cost; above 3 mm rejects a candidate.
Palm/finger clearance below 6 mm also rejects it. Inter-finger clearance keeps
the 6 mm soft margin and now additionally has a hard 0.5 mm minimum gap between
mesh-enclosing capsules (`--cem-self-clearance-mm`). This stricter local
extension prevents the contact objective from buying finger intersections.
When a whole population is infeasible, its elite mean moves toward lower
squared hard-constraint violation; only feasible candidates can enter the
returned incumbent. Reports count these restoration generations. Palm
clearance is a sampled capsule/mesh measurement, not an exact continuous
capsule/triangle minimum.

## Static input and output contract

The shared initialization uses deterministic multistart bone IK, fixed
annotated skin regions and a frozen bone reference. Contacts are optional for
IK; whole-hand dorsal clearance and bone-direction bounds preserve the shape.
Inputs failing these geometry/shape checks are saved as IK diagnostics and skip optimization;
see [grasp initialization](grasp_ik.md) for thresholds and archive compatibility.

The existing template mapping and Dexonomy initialization stay in place. Since
there is no previous frame, `q_previous=q_init`; the anti-slip reference is the
selected initial pad point. No temporal filtering is applied. The geometric
contact position comes from the original object's contact target; push direction
is the negative of its outward object normal. Style references come from the
initialized Wuji template, rather than an additional MANO demonstration.

Missing forces use `rho=1`, explicitly recorded as
`uniform_no_force_measurement`. Programmatic callers can provide optional
`GraspInitialization.contact_forces`: finite nonnegative magnitudes in N, one per
original annotation. These are saved in `card_input.npz` and grouped by semantic
body before applying the paper's force priority. Explicit model predictions are
reported separately and do not masquerade as measured inputs or silently alter
these CARD priorities. There is no new force-file CLI format.

CEM input construction saves a constant posture reference and does not apply a
load-command warm start or a closing reference. Explicit mode searches its own
servo compression offsets. Preload feasibility does not
gate shape-first IK; physical formation evaluates its own load requirements. The saved hold action includes the optimized finger servo offsets, clipped to
actual actuator command limits; clipping is recorded. Gravity simulation remains a separate validation operation.

Default persistence now uses numeric-only `grasps.npy`: 20 finger angles and
object pose in the actual wrist frame for both IK and optimized stages, plus
numeric identity/size/audit/cost fields. Hand assets are shared and primitive
geometry is generated in memory. Solver intermediates exist only in a temporary
directory; independent export/FK/closure checks run before cleanup. Use
`--archive-results` for the full diagnostic files described below. The compact
record has no velocity, so Viser disables dynamics playback for it. Schema 4
preserves the 20 clipped servo offsets and prediction acceptance metrics; schemas
1–3 remain readable. The size bar selects and snaps to actual saved scales.

In full-archive mode, `grasp.npz` exports the actual scored static configuration, zero velocity, the
selected local surface points/normals, and the initial template samples separately.
With `--explicit-pose`, `predicted_terminal_qpos` contains the predicted object
and compliant-hand state, while `qpos` remains the actual optimized static
grasp. `contacts.npz/explicit_predicted_qpos` stores all prediction states.
The target stays fixed at the initial object pose; output time is zero and
prediction horizon defaults to 0.4 s. With `--no-explicit-pose`, terminal qpos
equals the static result and the prediction length is zero. `contacts.npz` marks
`static_configuration=true` and `admm_active=false`; its zero dual is only a
schema placeholder. The independent contact, penetration and force-closure
audits evaluate the selected points and preserve their existing acceptance
thresholds (including the default 2 mm penetration tolerance).

`explicit_pose_gradient` and `explicit_pose_gradient_valid` in `contacts.npz`
store the final object-pose-cost gradient in the 26 optimization coordinates.
The JSON diagnostics also include predicted contact responses, active patches,
net object force, the initial-pose target, terminal drift, full objective before
and after refinement, and accepted updates. These predicted responses remain
separate from the external force-priority inputs and independent physical audits.

If all proposals are rejected, the initialization is saved with
`optimizer_success=false` and `termination=no_valid_candidate_initialization_saved`.
It is never reported as a feasible optimized grasp. Reports include cost groups,
hard-constraint status, contact selection, force source, noise and sampling
settings. Numerical search and final independent audit have separate statuses.

## Usage and migration

```bash
# Default paper budget, single scene group for a first smoke run.
conda run -n pi0 python run_grasp.py --primitives --templates 2 \
  --radius-scales 1 --gpu-nodes 1 --no-viewer --no-render

# Explicit budget override; these are not paper-budget results.
conda run -n pi0 python run_grasp.py --primitives --templates 17 \
  --radius-scales 1 --cem-population 128 --cem-iterations 8 \
  --no-viewer --no-render

# Original dynamics/ADMM path.
conda run -n pi0 python run_grasp.py --primitives --templates 2 \
  --grasp-optimizer adamw --no-viewer --no-render

# Paired comparison rebuilt from saved original initializations.
conda run -n pi0 python scripts/refine_grasp_results.py \
  --output outputs/card_comparison --methods cem adamw --audit
```

CEM-specific options are `--cem-population`, `--cem-iterations`,
`--cem-elite-fraction`, `--cem-temperature`, `--cem-translation-sigma-mm`,
`--cem-rotation-sigma`, `--cem-joint-sigma`, `--cem-final-noise`, and
`--cem-self-clearance-mm`.
`--contact-rounds`, `--iterations`, and `--multistarts` are AdamW budgets.

The old `--cem-sigma`, `--cem-min-sigma`, `--cem-trust-radius`,
`--cem-gradient-steps`, `--cem-gradient-lr`, `--[no-]cem-optimize-pose`,
`--contact-depth-mm`, `--grasp-penetration-weight`, and
`--grasp-endpoint-weight` are removed, with an explicit CLI migration error.
The comparison script no longer offers `cem-only`: CARD CEM is always
zeroth-order. Existing result directories remain replayable; old experiments
must be rebuilt before reoptimization. The comparison script writes new
experiments/results without overwriting the original inputs.

Viser gravity playback still starts from saved qpos/hold_action and simulates a
free object with the native Warp executor. Static optimization and force closure
do not certify that the object will be retained during playback.

## Verification status

CPU numerical tests independently check the four cost groups, elite softmax,
noise schedule, limits, SO(3)/Euler boundary, force priorities, hard rejection and
fallback behavior. Sphere, cylinder and cuboid tests run the real Wuji FK and
static numerical search on Warp CPU, comparing against independent MuJoCo FK.
These tests do not validate CUDA graph capture or GPU timing.

GPU tests cover node isolation, deterministic replay with host reads forbidden,
absence of dynamics/adjoints, exported pose/contact consistency and independent
closure agreement. GPU access is available through host execution. The collision-fix verification
includes CUDA replay, sphere/cylinder/cuboid export checks and a complete CLI
smoke. Full-matrix outcomes are recorded by each background run; no grasp
success-rate claim follows from numerical test passes alone.

## Full background matrix

`scripts/run_grasp_batch.py --run-dir PATH` solves all 31 supported types,
all 38 associated rigid shape families at radius scales 0.75/1/1.25 (114 cases).
Types 19 and 21 need articulated/multi-object assets and remain explicitly skipped.
The batch defaults to 4 independent CUDA scene branches, 1024 candidates and
32 generations to leave iterations for contact refinement after collision repair.
Initialization/file preparation remain sequential boundary work; this is one GPU
process, with parallel populations and scene branches rather than multiple
competing Python solver processes. A fresh directory is required. `status.json`
records progress and final/failed status; `results/` holds cases and summaries,
and independent export/closure checks are saved after solving.

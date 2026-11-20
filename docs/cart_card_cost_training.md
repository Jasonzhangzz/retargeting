# CARD raw cost with Neural SDF collision

The optional `--policy-objective card_raw_neural_collision_v1` trains the
existing future-IK residual policy with CARD's raw energy. It does not change
the default exponential objective or the already running Base process.

The fixed-wrist CARD terms are evaluated with the same expressions, weights,
native semantic pad apices and sum reductions as
`retarget/baseline/card/retarget.py:TactileRetarget.evaluate`. All lengths in
these costs are millimeters; joint angles are radians. The loss is the mean of
per-frame energies across the BPTT windows, without exponential compression.

For `rho = 1 + .5 * force / max_force`:

| W&B primary component | Expression per frame |
| --- | --- |
| `style_bone` | `20² * sum_f,b (1-dot(robot_bone, source_bone))` |
| `style_tip` | `3² * sum_f ||robot_pad_apex-source_pad_apex||²_mm` |
| `style_normal` | `60² * sum_f (1-dot(distal_pad_normal,source_normal))` |
| `contact_position` | `40² * sum_regions rho² ||CoP-target||²_mm` |
| `contact_direction` | `40² * 2 * 10² * sum_regions rho² (1-dot(normal,target_normal))` |
| `contact_fix` | `2.5² * sum_retained_regions rho² ||local_CoP-current_previous_material_CoP||²_mm` |
| `smooth_joint` | `.1² * sum_j (q_next-q_previous)²` |

Each finger has three semantic bones. Their chain consists of CARD's three
native region origins followed by its native distal pad apex, not the tip
site used by the former objective. The default `contact_style_weight=1`
keeps the style terms active for all fingers. Translational and rotational
smoothness are zero because the wrist is fixed. The default CARD profile
does not infer hand-hand pairs, add a joint-motion prior, or use the optional
Coulomb-force objective; those components are logged as zero.

Source style and contact targets are computed by the original
`TactileRetarget.style_targets` and `MoxianCARDSource.extract`, from the saved
glove motion and raw pressure records. The source's physical mesh is
reconstructed from the saved calibration. Every frame is checked against the
existing captured joints and exact timestamp. Separate checksummed `.npz`
caches contain these constants; raw captures, previous prepared targets, and
cached IK arrays are not regenerated or changed.

CoP selection uses CARD Eq. (2)'s native triangle query and its `.15` segment
prior. Discrete feature selection runs without a graph; the selected face,
edge or vertex projection is then recomputed with autograd. This includes
the derivative of CoP motion inside the selected feature. Previous predicted
states remain in the graph for material fixation and BPTT.

Collision keeps CARD's squared-mm cost expressions, replacing its capsule
and pseudo-object representations with signed Neural SDF queries:

```text
collision_self   = 1.5² * sum_valid_queries relu(6 - signed_gap_mm)²
collision_object = 3.5² * sum_active_taxel_link_queries relu(penetration_mm - 2)²
```

Self queries use deterministic moving native surface samples and existing
mechanical exclusions. External queries use confirmed active source taxels
against every robot link. These are the existing CART observations; they are
not a reconstruction of CARD's solid pseudo-object. Accordingly the collision
scalar is not numerically identical to CARD's capsule/pseudo-object scalar.
The articulated model receives the actual predicted `q` and points in its
native fixed-wrist frame. Its parameters are frozen during policy updates,
while derivatives through `q`, FK and query positions remain active. It still
receives two online geometry updates per policy update.

Two additional terms implement the requested contact continuity and 5 mm cap:

```text
link_continuity = 2.5² * sum_persistently_active_regions rho² * closest_pad_gap_mm²
penetration_cap = 40² * relu(max_neural_penetration_mm - 5)²
train/total = sum(all CARD components) + link_continuity + penetration_cap
```

The continuity term keeps the same semantic link against the current source
contact while allowing CoP sliding. It resets on contact deactivation, stream
changes or timestamp gaps. CARD's material-point fixation remains a separate
term. The finite cap penalty guides training; independent dense native-mesh
validation rejects any frame with penetration above **5 mm**. Contact position
and normal acceptance remain 0.5 mm and 30 degrees. Geometry-model precision
checks remain independent; an unused learned contact head no longer gates
acceptance because policy contacts now use native pad geometry.

All primary W&B terms already include their CARD weights. Familiar aliases
(`direction`, `contact`, `normal`, `smooth`, `penetration`, `contact_keep`)
are also logged, but must not be added again to the primary components.
`acceleration` remains a diagnostic with zero objective weight.

The replacement run uses a new run ID and fresh policy/optimizer, transferring
only the stopped run's trained articulated SDF, its geometry optimizer and
replay. The 16-step BPTT, ten measured future IK inputs, attention architecture,
learning rate and initial-state noise retain their previous settings.

Validation includes comparison of every active noncollision component against
the actual CARD evaluator, face/edge/vertex finite-difference projection
gradients, bone gradients, previous-state gradients, 5 mm cap checks and the
existing IK/BPTT/Neural-SDF/tracking regression tests. A remote controller must
also pass real-data optimizer probes and a full parallel-batch memory benchmark
before opening the formal W&B run.

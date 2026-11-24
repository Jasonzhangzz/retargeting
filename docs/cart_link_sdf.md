# Direct rigid-link SDF contact experiment

This independent experiment leaves the running v10r4 source and checkpoints
untouched. Its objective is `card_style_link_sdf_contacts_v6_wrist`, with geometry
`fk_independent_link_sdf_distance_contacts_v6`.

Each of 21 links has an independent 3→128→128→128→1 Softplus MLP, totaling
706,965 parameters. Matrices remain independent Muon parameters; grouped batched
GEMMs evaluate them efficiently. The input is link-local xyz in metres, scaled by
0.05. The sole output is signed distance in metres. Joint and wrist dependence
comes exclusively from differentiable FK. Spatial and joint Jacobians are actual
autograd derivatives of that distance, not independently predicted vectors.

Observed contact masks identify point/link pairs. Glove self-penetration records
identify anatomical link/link pairs and a common observed anchor. Contact loss
attracts each specified link's distance at its anchor to zero. Noncontact labels
do not create extra attractions; unknown taxel observations are not labeled as
noncontact. All mechanically allowed material-point/target-link pairs resist
penetration, including confirmed contacts. There is no whole-finger exemption.
The prior active-taxel all-link penalty retains CARD's 2mm allowance and
wobj_pseudo weight. An additional sum of per-query 5mm squared penetration hinges
applies to both material points and active taxels. A finite 54-point/link sample
set cannot prove continuous-mesh nonpenetration; true meshes are independently
validated. This is a soft penalty, not a hard constraint guarantee.

The retained-link continuity loss penalizes distance changes between consecutive
observed anchors and residual contact distance. The previous policy state stays
in the graph. CARD fingertip/bone/style tracking and wrist physical-dt velocity
and acceleration regularization remain. The full mean raw scalar cost is still
bounded by C/(C+1e8). Normal alignment at the contact and material-point projection
losses are replaced, not algebraically equivalent to CARD. Contact normals remain
physical diagnostics, with the original 30 degree feasibility criterion.
Independent policy validation measures gaps/normals on the native palmar pads
and maximum penetration on the native dense guard; a closed-link SDF zero on
the wrong side of a link therefore cannot masquerade as a valid pad contact.

There is no projection network, facet enumeration, distance-dependent min/top-k,
straight-through gradient, native query or learned q-only collision regressor in
the policy graph. Only supervision/independent evaluation uses native searches.
Link-ID grouping is fixed by observed labels, independent of continuous state.

## Policy distance graph and parallel queries

For an observed point x and specified link l, the optimized distance is
`d_l = f_l(R_l(q,w)^T (x - p_l(q,w)))`. FK, the inverse rigid transform,
the independent Softplus SDF and the loss reduction all retain autograd.
During policy updates only the SDF parameters are frozen: neither the query
coordinates nor current/previous predicted states are detached. Consequently
`∂L/∂(q,w) = ∂L/∂d_l · ∂f_l/∂x_l · ∂x_l/∂(q,w)` is the actual derivative
of the evaluated scalar objective. Contact continuity also differentiates
through the previous predicted state within the BPTT window.

Self collision transforms each fixed source material point through its source
FK and target inverse FK. It evaluates all 20520 mechanically allowed directed
point/link pairs per pose (54 guard samples per link), rather than selecting a
nearest pair. Active taxels are transformed against all 21 target links in one
ragged batch. Independent decoder matrices run as grouped batched GEMMs;
bounded chunks and activation recomputation limit memory without dropping
queries or detaching gradients. Contact attraction evaluates only observed
point/link and link/link labels, batching observations for each such group.
These fixed discrete labels/exclusions have no distance-dependent selection.

Contact energy is proportional to `d_l²`. Collision energy sums squared hinges
`relu(-1000*d_l - allowance_mm)²` across queries; the extra depth cap sums
`relu(-1000*d_l - 5)²`. The squared hinge is continuously differentiable,
including at the activation boundary. There is no optimized hard min/max over
links. Detached max depths/normals are reporting fields, not loss inputs.
Absolute wrist motion leaves self collision invariant, so self-collision wrist
gradients must be zero; contact and observed-taxel collision retain wrist
gradients because their observed points remain fixed.

`scripts/audit_cart_link_sdf_policy.py` checks these paths with a saved geometry
checkpoint and real observations, without any optimizer updates. It forbids
native queries/projections during policy forward/backward, checks component
gradients against finite differences, exercises a known penetrating fixture,
and checks 16-step BPTT into all five finger heads and the wrist head. This
checks graph correctness; native-mesh prediction accuracy remains a separate
geometry acceptance requirement.

SDF training uses link-balanced true-area-weighted samples mixed with padded-box
samples, signed distances, stable native first derivatives, Eikonal and sign
losses. Two-scale finite differences filter unstable derivative labels without
removing their scalar distance supervision. Geometry acceptance retains 0.5mm
P95, 1% penetration false negatives, spatial/FK gradient checks and native dense
collision/5mm-cap gradient and escape checks. Projection Jacobian gates are
replaced explicitly because this objective never uses a projection Jacobian.
No first-derivative or collision gate is waived to start policy.

The controller inherits recordings/split, seed, future10 EMF, 5 finger decoders,
6D wrist head, Muon+SGD, 16-step BPTT, bounded policy scaling, batch512 and 1000
epochs. SDF and policy both start from random parameters. It runs up to 30000
SDF updates, atomically selects/revalidates the best candidate and only starts
policy if the applicable geometry gates all pass. There are no automatic retries
or threshold relaxations. Resume restores optimizer, replay, RNG and budget.
Use an independent run root with its own `geort/training.lock`, model directory
and source snapshot; only immutable recording caches may be shared.

Entrypoint: `scripts/train_cart_link_sdf.py`. The paper and existing web deployment
are not switched by starting this comparison experiment.

Early stopping is now authorized for v10r5. Geometry validates every 500 updates
from update 2000. After at least 5000 updates, it can finish when seven consecutive
checks pass every applicable geometry gate and the last six show no cumulative
1% significant improvement in the normalized geometry selection score. Any failed
check resets patience. The best weights are restored and fully revalidated before
policy starts. Merely flat training loss never permits policy training. The
maximum stays 30000. Convergence history is persisted in
`geometry_convergence.json`; a resumed checkpoint prunes uncommitted future
validation observations. Original v10r4 keeps its original full-budget schedule.

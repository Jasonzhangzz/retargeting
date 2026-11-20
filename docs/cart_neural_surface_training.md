# CARD energy with learned nearest-surface geometry

`--policy-objective card_raw_neural_surface_v2` retains the fixed-wrist CARD
energy expressions and replaces native pad projection inside the policy graph
with the existing six-output contact MLP. The complete geometry surrogate has
the articulated signed-distance head plus a nearest-point/outward-normal head.
This is an approximation of CARD's geometry operator, not exact native cost
parity for arbitrary network weights.

The contact MLP receives local query xyz, link embedding, semantic-pad flag,
and direction. A **zero direction explicitly requests ordinary nearest-point
projection**. Nonzero directions retain the original directional escape mode.
The CoP query still includes CARD's .15 segment prior; contact continuity uses
the unbiased query. Previous-state material fixation retains its BPTT graph.

Only teacher-label generation and independent native-mesh audit use discrete
triangle queries. The policy forward/backward uses network predictions for
CoP position, CoP normal, and link-contact continuity. The learned surface is
restricted to the requested semantic link/pad, including the palm. Source
CARD targets and their cache implementation remain unchanged.

Teacher replay is collected from detached actual predicted states, including
biased and unbiased CARD queries, near-query jitter, and generic near/far mesh
queries. Teacher collection does not detach the retained policy rollout.
Nearest predictions receive ordinary native nearest-point/normal supervision,
without the directional teacher's ray retraction. Neural distance continues
receiving native signed-distance/normal supervision. SDF and contact parameters
are frozen only during the policy backward, retaining input/FK gradients.

The correction branch snapshots the stopped parent's last durable checkpoint,
fits its contact head with a separate bounded optimizer, resets incompatible
directional contact replay, then initializes the revised policy run from the
preserved policy, geometry and optimizers. The fitting optimizer is separate
from the inherited online geometry optimizer; this is a new branch, not exact
resume. The initial learned-projection screening uses 2 mm position P95 and 30 deg
normal P95 on validation recording queries. After 10,000 head updates,
position P95 is 2.92 mm (failed screening), and normal P95 is 25.33 degrees.
The experimental candidate continues policy/geometry learning with that failed
screening explicitly preserved; it is **not** accepted for inference. The
initial 2,000 updates used the original query distribution; another 8,000 used
broader generic pad queries, a lower learning rate, and stronger normal supervision. Final acceptance retains the stricter
0.5 mm geometry/contact position thresholds and 5 mm penetration ceiling.

Compare native and learned geometry using the same GPU, 4096 real parallel
windows, 16-step BPTT, BF16 attention and FP32 geometry. Two complete policy
updates per geometry operator include online SDF updates. Restore the intended
transfer weights after timing probes. Report complete step costs separately
from isolated geometry costs; do not infer whole-training speed from removal
of one operator. Authoritative launch, timing and quality evidence is under
`output/moxian/cart/remote_jobs/neural_surface_20261119` locally and the matching
remote run directory. The first-generation exact-pad run and artifacts remain
preserved.

The learned nearest-field precision sampler covers all 16 semantic pads in
zero-direction mode. It excludes unused whole-link escape modes; this changes
the model-precision denominator, not task contact/penetration validation or
acceptance thresholds. The same-run validation-only migration is recorded in
`validation_repair_receipt.json`.

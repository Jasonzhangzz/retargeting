# Direct configuration collision potential

`cart_q_collision.py` evaluates only the 20 joint angles. Finger/palm and
finger/finger dependency masks feed a smooth MLP. No FK, point transformation,
nearest-triangle ID or native query participates in this self-collision policy
graph. Its escape command is the negative gradient of the same self-collision
and 5 mm guard potential used by the policy, with outward joint-limit commands
removed. It is not a detached vector prediction or a custom backward rule.

CARD's original self-clearance expression (6 mm margin, squared millimetres,
weight 1.5) is supervised from its eight fixed material samples per link.
The additional 5 mm guard uses the Euclidean norm of positive penetrations at
54 fixed material samples per link. This norm bounds every sampled penetration
and provides a conservative guard when multiple points penetrate. The neural
network learns its value and joint Jacobian. The guard applies
`40**2 * relu(penetration_norm_mm - 5)**2`; it does not differentiate the
unstable maximum-depth predictor. Maximum depth remains a separately
supervised diagnostic. Geometry approximation is not a hard 5 mm guarantee:
the independent dense mesh audit still determines final acceptance.

The teacher computes native signed distances and their local feature
derivatives through both moving links. Native queries occur only during label
generation and independent evaluation. Value, Jacobian and direction losses
train the configuration potential. The replay and Muon/SGD state are saved in
the training checkpoint and updated from detached rollouts plus random and
perturbed configurations. Policy BPTT freezes geometry parameters while
retaining the derivative with respect to the predicted q.

Active observed glove points already use the canonical wrist frame. The
articulated SDF converts them to its native fixed-wrist frame and evaluates
`SDF(q, x_native, link)`. This preserves derivatives with respect to both the
observed point and q. Bone orientation uses exact FK and CARD's cosine cost;
contact position/normal/continuity use the learned nearest-pad field.

The complete policy cost remains `C / (C + 1e8)`, in `[0, 1)`. Raw CARD parts
remain separate diagnostics. Policy, geometry and configuration-field matrix
parameters use Muon, with momentum SGD for embeddings and vectors.

New-run options: `--initialize-q-collision-from CHECKPOINT`, optionally
`--initialize-from RUN --fresh-policy-optimizer`. A new branch can retain the
latest policy/SDF/contact weights while discarding policy momentum and the old
training cursor. `scripts/train_cart_q_collision.py` pretrains the new field.

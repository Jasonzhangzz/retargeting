# Shared q-conditioned collision field (v10r6)

The new experiment replaces v10r5 after its terminal geometry rejection. It
preserves the running v10r4 source and policy. SDF and policy start from random
parameters in a new isolated run. Reference recordings/hyperparameters only are
reused. Muon+SGD, future10 EMF, five policy finger heads plus wrist, 16-step BPTT,
batch512 and 1000 policy epochs remain. Geometry trains up to30000 updates with
the existing converged_best early stop and unchanged first-derivative tolerances.

Architecture: one q20 encoder (normalized angles, sin and cos; 60→128→128) and
projection128→256, computed once per pose. The point branch maps canonical
fixed-robot-wrist xyz3→256 and adds the q condition. A shared Softplus trunk
256→256→256→21 returns signed distances in metres. The21 rows of the final
linear layer are lightweight per-link heads, not21 independently executed MLPs.
No min/nearest-link selection or dynamic grouping/sorting is used. Target-link
motion is learned through q. Differentiable FK still generates source material
points and native labels. Observed points are inverse-transformed by the robot
wrist pose; the wrist and all previous policy states remain in autograd.

Both8-point clearance and54-point guard samples/source are retained, with all380
allowed directed link pairs. A network point query returns all21 link distances,
so each pose uses168+1134=1302 shared point-trunk evaluations, rather than23560
independent distance evaluations in v4 or20520 in v5. Mechanical exclusions remain
fixed and the small moving collar uses the same smooth weight as v4. Every
allowed pair resists penetration. Only exact observed link pairs have a zero
clearance target; other pairs retain6mm clearance. There are no finger-wide
penetration exemptions.

For positive penetration p_i=max(-1000 d_i,0) in mm, the self guard follows v4:

    N = sqrt(sum_i w_i p_i² + 1e-8)
    L_guard = max(N - 5, 0)²
    C_self = w_self² sum_i w_i max(m_i - 1000 d_i, 0)² + w_cap² L_guard

Here m_i is6mm or0 for an observed contact pair; clearance and guard use their
own fixed8/54-point sets. N conservatively bounds every sampled penetration;
several shallow penetrations may trigger this guard. It is a soft cost on finite
samples, not a guarantee of continuous-mesh separation. The escape direction is
-∇q C_self, obtained by differentiating the same scalar potential used by the
policy, not an independently predicted vector or the gradient of a hard min.

Contact attraction remains the first-derivative distance objective: each specified
link has d=0 at its observed anchor, and observed self-contact links both have d=0
at their common anchor. Contact continuity retains previous-state gradients.
This is intentionally not v4's expensive all-triangle projected contact/normal
objective. Active taxels evaluate all21 distances in one shared point pass;
the2mm object allowance remains and the5mm object guard uses all-query smooth
maximum depth as in v4. CARD style, tracking, wrist physical-time smoothing and
global C/(C+1e8) scaling remain unchanged.

Training uses grouped q/point replay:64 points/pose, normally32 poses/update.
Half the poses come from training-split IK with0.04rad noise; half are uniform
in the joint domain. Link-balanced true-area surface/box samples are transformed
by native FK. All21 native signed distances supervise the scalar heads, with
extra weight near surfaces. Selected per-point links provide spatial derivative,
Eikonal, sign and random-probe q-VJP supervision. Query points are fixed when
forming q derivatives. A small subset also carries native clearance/guard cost
gradients, trained through derivative losses and direction alignment. No branch
uses a separately predicted Jacobian. Two-scale native finite differences reject
unstable derivative labels while retaining their valid scalar distance labels.
All replay, optimizers, RNG and update counts are checkpointed.

Geometry acceptance reports0.5mm distanceP95,1% penetration false negatives,
spatial and articulated gradients, v4-style L2-penetration gradient alignment and
an additional >=0.8 cosine gate on the actual guard gradient. It also executes
small joint steps and recomputes native clearance/guard costs, including the
combined escape direction. Fixed native holdout queries and independent label
routing/FK/finite-difference audits are retained. No projection-Hessian gate is
claimed because no learned projection is used. Only accepted, best-restored,
revalidated geometry can start policy training; flat loss alone cannot.

Entrypoint: `scripts/train_cart_conditional_sdf.py`. New architecture/loss means
a fresh run; v4/v5 optimizer checkpoints cannot be silently resumed into it.

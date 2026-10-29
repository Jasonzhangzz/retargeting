# Grasp IK: shape first, optional contacts

Both CARD modes use the same IK. GPU bone seeding solves eight deterministic
starts for 96 LM iterations each and selects each finger's lowest finite
direction residual. Task and optimizer random streams are unchanged.

## IK acceptance

IK prepares a hand shape and object-relative placement; it does not certify a
formed grasp. Individual fingers and annotations may remain separated from
the object. Contact position/normal matching uses bounded residuals, with
normal alignment faded for distant contacts. Contact gaps, force closure,
sharp-edge proximity of detached targets and preload feasibility do not gate
shape-first IK. Final grasp acceptance still audits contact and closure.

The original mapped contact normals and the multistart bone reference remain
fixed through all fitting rounds. A selected point must stay in its annotated
normal region (dot > 0.5). Palmar annotations additionally require palmar-facing
samples (belly-normal dot > 0.25); lateral annotations retain their side.

All fifteen bone directions are measured in the palm frame. LM steps cannot
change any direction by more than 30 degrees from the reference. Nearly straight
non-thumb chains have stronger style weights, preserving extension gestures.
Object size cannot be matched by reversing a finger or changing the grasp type.

## Forbidden contact surfaces

Dorsal samples (local normal dot anatomical belly axis < -0.25) across the
whole hand must remain at least **1 mm** outside the actual object mesh. This
includes unannotated links and fingers. An explicit dorsal annotation on a
body grants only that annotation's normal region; side contacts stay legal.
The check uses the complete existing query assets (8880 samples on Hand 2),
not only the selected contact anchors. It is a sampled clearance constraint,
not a continuous triangle/triangle separation certificate.

Object-aware fitting refreshes exact GPU mesh projections between LM rounds
and differentiates the resulting local clearance planes. A 0.5 mm fitting
buffer accommodates changing mesh features. Final IK acceptance queries the
mesh again and checks forbidden clearance, whole-hand penetration, self
collision, joint limits and bone-direction deviation. A stale plane never
certifies the output. The default penetration tolerance on allowed surfaces
remains 2 mm.

The same forbidden mask is persisted in CARD inputs and enforced as a hard
1 mm constraint in both pure CEM and explicit scoring, including feasibility
restoration and refinement acceptance. Thus later search cannot buy dorsal
contact with a lower predicted-pose cost. Other explicit weights and sampling
budgets are unchanged.

## Failure records and compatibility

Unqualified shapes remain diagnostic records with
`termination=initialization_failed`, `optimizer_success=false` and
`feasible=false`. They skip optimization and physics, while other cases
continue. Archives retain a diagnostic qpos but expose only IK through
`stage_valid=[1,0]`. Missing contacts alone do not create this failure.

Numeric schema 3 adds IK policy, forbidden gap, bone-angle deviation and
IK validity. Schemas 1 and 2 remain readable. The viewer displays these
IK checks separately from final contact/force-closure acceptance.

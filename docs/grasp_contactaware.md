# ContactAware references for WUJI2 primitive grasping

`configs/grasp_contactaware_mppi.yaml` builds the hand reference before running
MPPI. It uses the pinned DexForge ContactAware solver at commit
`fa6d671948ef03d18514062748b94e445b63cd49`. The unchanged upstream source and its
licenses are in `third_party/DexForge/contactaware`; hashes are recorded in
`CONTACTAWARE_UPSTREAM.json`. This is a single static-key primitive adapter,
not a MANO-sequence frontend or an implementation of the full DexForge pipeline.

## Reference generation

1. Read the Shadow template's semantic contact patches and map them to actual
   WUJI2 surface queries. Bone IK, contact-layout fitting, and original global
   alignment provide only a numerical seed.
2. Directly call upstream `solve_joint_grasp` and its Clarabel QP. Jointly solve
   the 26 hand coordinates and object-surface anchors, with surface retraction,
   whole-hand nonpenetration, self-collision, joint limits, and anchor topology.
   The object pose stays fixed. Guided fingers can adapt beyond the old
   template bone-angle limit.
3. Evaluate the upstream wrench residual on actual robot material points and
   nearest object normals. The adapter treats selected mesh-face normals as
   piecewise constant and refreshes them at every nonlinear evaluation.
4. Require every annotated physical finger to have a point within 0.5 mm of
   its anchor and with an opposing normal. Nonmatching slots get an explicit
   inactive mask and release reason; a mandatory finger cannot be silently
   released. Whole-hand penetration must be at most 2 mm, and forbidden skin
   retains at least 1 mm clearance (1.1 mm SQP buffer).

Only an accepted solve creates `hand_reference_qpos`. Failed CA proposals retain
diagnostics and cannot be passed to MPPI through the old shape-first acceptance
path. The solved hand reference, refined anchors, and active mask are saved
separately from the template seed and actual initial state.

## MPPI objective and numerical checks

The objective includes full-hand penetration, active-point approach,
active-patch contact maintenance, stationary-object pose/velocity tracking,
and a fifth **hand_tracking** term. This last term directly uses the repository's
dex hand-reference formula with explicit position/rotation/finger coefficients
10 / 3 / 0.03. It reads only an explicitly marked ContactAware reference; a
template or seed cannot act as a fallback.

The example uses population 64, four MPPI iterations and 64 simultaneous worlds
for each of three cylinder sizes. The existing lightweight quasi-static
predictor scores four 0.1 s steps. The contact solve has an 8192-iteration cap
and checks normal feasibility/complementarity every 256 iterations against
0.1 mm. Candidates failing that numerical check receive an infinite hard score.
This normal check does not certify the full friction KKT system or a physical
hold test. The standalone MPPI workflow and original template profile remain
available separately.

## Run and inspect

```bash
uv pip install --python .venv-contact/bin/python -r requirements-grasp.txt
.venv-contact/bin/python run.py --config configs/grasp_contactaware_mppi.yaml
```

Each case saves `contactaware.json` (SQP history, per-finger/per-patch accuracy,
active contacts and release reasons), `contactaware_reference.npz`,
`input_*/card_input.npz` (explicit reference, initial hand state and masks),
`objective.json`, five-term `tracking_costs.npz`, `contact_solver.json`, the
predicted trajectory and video. The summary distinguishes reference feasibility,
numerical contact validity and predicted hold success. Actual physical holding
has not been tested by this workflow.

## Local sample results (2026-10-31)

The final three cases completed in `output/grasp_contactaware_mppi`. Each CA
reference has five active slots, one per required finger. All initial geometry
scores are finite, and the initial hand-reference cost at q0 is below 1e-6.
Each MPPI solve samples 256 candidates. The selected candidates pass the normal
residual gate; their contact iteration counts vary by step, up to 7424 in this
run. This is not a comparison of solver speed or a success-rate benchmark.

| Radius (mm) | CA max required-finger point error (mm) | Final normal residual (mm) | Predicted max object drift (mm) | Predicted max penetration (mm) |
| ---: | ---: | ---: | ---: | ---: |
| 26.25 | 0.456 | 0.0923 | 9.61 | 7.72 |
| 35.00 | 0.428 | 0.0999 | 9.12 | 2.82 |
| 43.75 | 0.421 | 0.0998 | 7.49 | 3.27 |

No case meets the complete predicted hold criteria. Five-finger reference
proximity does not imply that all fingers stay in contact or carry force after
MPPI changes the initial pose and servo offsets. Normal linear contact accuracy
also does not certify nonlinear whole-hand penetration or stable support. The
results retain separate contact-reference, numerical-residual, and dynamic-hold
flags so these outcomes cannot be conflated.

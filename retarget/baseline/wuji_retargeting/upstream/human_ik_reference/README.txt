Independent human-hand EMF IK reference (pinned SDK 2026.8.31)

Public API: HumanIKSession(side).step(positions[5,3], orientations_xyzw[5,4],
confidence[5]) -> dict with q[21], padded angles[5,5], skeleton[21,4,4],
tips[5,4,4], TX/RX transforms, confidence and pose errors. reset() restores
constructor seeds. Raw arrays contain decoded TX-frame poses, metres and XYZW.
Every step is a new numerical solve; use emf_retarget.EmfRetargetPipeline for
shared frame caching and conversion into Wuji1/Wuji2 robot joint angles.

kinematics.py: URDF FK/J, stable mathematical SO(3) reference.
sdk_rotation.py: observed native SO(3) branch/roundoff compatibility.
parameters.py: constructor-captured constants, seeds and physiological limits.
boxqp.py: scalar active-set box QP, independently implemented, no SDK import.
solver.py: weighted pose residual, damping, posture/history regularization,
bounded step, line search, retries, confidence and state handling.
assets/: the project's explicit CQQ left/right human URDF overrides.

The human URDF limits are not the actual solver bounds. Do not replace the
physiological defaults with the URDF's broad +/-pi values.
No SDK q, SDK warm state or per-frame SDK intermediate is used at runtime.
Native probes are separate validation tools. Source/asset provenance and
bounded numerical comparisons live in the parent directory's reports.
This is not an official SDK implementation or complete Python binding clone.

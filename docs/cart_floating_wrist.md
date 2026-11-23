# CART floating-wrist training

Select `--policy-objective card_raw_neural_self_contact_v5_wrist` with the attention architecture. This keeps the ten measured future EMF frames and five independent finger decoders, and adds a separate six-coordinate wrist decoder. The state is `[finger_q_rad (20), wrist_translation_m (3), wrist_rotation_vector_rad (3)]`.

The predicted wrist is the robot wrist relative to the fixed demonstrator-wrist frame. Robot points and normals are transformed into that frame before computing tracking, tactile contact, link continuity, and active-taxel penetration costs. Observed targets remain fixed. Palm contact terms participate in optimization again. The articulated SDF receives the original 20 finger angles and inverse-transformed query points, retaining wrist gradients through the query coordinates. The configuration-only self-collision field continues to receive only finger angles because a common rigid motion cannot change self collision. The existing 5 mm penetration penalty remains active.

Wrist smoothing uses actual observation timestamps. Let `h = 1/120 s`, `v = (t_k-t_{k-1})/dt`, and `omega = Log(R_k R_{k-1}^T)/dt`, with angular velocity expressed in the fixed frame. Acceleration is the difference of adjacent velocities divided by their mean time interval. Added costs are:

- Translation velocity: `wt² ||1000 h v||²`.
- Rotation velocity: `womega² ||h omega||²`.
- Translation acceleration: `wt² ||1000 h² a||²`.
- Rotation acceleration: `womega² ||h² alpha||²`.

Here `wt = sqrt(125)` and `womega = 10` are CARD's wrist weights. Finger smoothing retains its existing definition. All wrist terms enter the complete raw objective before the existing `C/(C+1e8)` normalization. W&B records `train/smooth_translation`, `train/smooth_rotation`, `train/wrist_translation_acceleration`, `train/wrist_rotation_acceleration`, wrist speeds, and relative pose offsets.

A fresh policy starts at identity wrist pose and zero predicted increments. Each wrist increment is scaled by 5 mm per translation coordinate and 0.05 rad per rotation-vector coordinate. Absolute components are projected to ±0.1 m and ±pi rad to prevent unbounded drift; these are component bounds, not task-space speed limits. Configuration files record these values and the new state/output version.

The training run saves the complete 26-coordinate policy and geometry, best and resumable checkpoints, and full-session trajectories containing both finger angles and wrist pose. This changes the model contract: the existing finger-only web runtime rejects these weights explicitly. Deploying them requires forwarding the wrist pose to a wrist-aware visualization/arm controller.

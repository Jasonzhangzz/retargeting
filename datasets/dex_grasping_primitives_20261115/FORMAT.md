# Robokit primitive retargeting

Source: /home/zhangzhe/dex_grasping/primitives (31 types, 1,247 size rows).
Two independent direct mappings: MANO -> calibrated 29-DOF glove, MANO -> native 20-DOF Wuji2.
Source MANO poses and object dimensions/poses are copied without alteration.

Files:
- glove/grids/<type>.npz and wuji2/grids/<type>.npz: paired source and retargeted poses.
- target_qpos: target joint angles in radians; joint_names gives their exact native order.
- target_base_pose: root position in metres, then quaternion WXYZ, in the original source world frame.
- target_keypoints_world: 21 semantic points in wrist, thumb, index, middle, ring, pinky order.
- object_pose_world: object XYZ + quaternion WXYZ, copied from the input.
- source_wuji_qpos_reference: dataset's old frozen/interpolated Wuji2 reference, NOT this run's output.
- valid, patch_joint_accepted, grasp_passed, rollout_*: source MANO diagnostics, NOT new target success.
- retarget_numerically_valid: finite values, valid quaternion and target joint limits only.
- config.json, model.xml: solver settings, native joint and point order, model provenance and independent FK checks.
- glove/glove-model.json: frozen calibrated glove model; its original fit-acceptance status is preserved.
- size-index.csv, type-summary.csv: all size rows and per-grasp summaries.
- comparison.png: skeleton comparison; identical object/world coordinates across each row.
- verification.json: aggregate checks and errors.

The two targets have no temporal warm start between dataset rows. The old anatomical-angle exporter was not used.
Positions and directions are retargeted with Robokit's batched population LM solver (16/4/1 candidates, 12/20/12 iterations).
The floating wrist is optimized. No object-contact, self-collision or force-closure objective was run.
These are kinematic references for subsequent physics refinement, not validated stable target grasps.

Example:
```python
import numpy as np
z=np.load('glove/grids/14_Tripod.npz',allow_pickle=False)
q=z['target_qpos']; base_xyz_wxyz=z['target_base_pose']
sizes=z['radius_m']; names=z['joint_names']
```

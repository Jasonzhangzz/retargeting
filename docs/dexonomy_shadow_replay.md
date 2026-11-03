# Dexonomy Shadow replay cache

Run all commands on the remote server from the repository root. The released
archives live in `data/dexonomy/`. First run `scripts/prepare_dexonomy_pilot.py` to scan the archives and extract only the selected pilot resources into `data/dexonomy/pilot_dataset_stratified/extracted`. Full extraction is resumable in `data/dexonomy/extracted` and is only required for the full run.
The shared meshes under `data/dexonomy/shadow_assets` use the repository's
Dexonomy hand XML and the pinned Menagerie mesh revision in `provenance.json`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../miniconda3/envs/pi0/bin/python scripts/replay_dexonomy_shadow.py \
  --dataset data/dexonomy/pilot_dataset_stratified --count 100 --output results/dexonomy_shadow_pilot100_20261103 \
  --wait-assets --auto-full --full-output results/dexonomy_shadow_full
```

The pilot contains 100 original grasps distributed across all available grasp
types, with hash-ordered objects and distinct available size endpoints. It does
not optimize poses or select samples by closure outcome. Each size is the scene
configuration scale multiplied by the record's per-row `scene_scale`; the latter
also scales scene translations. Nominal object mass is 0.1 kg, as in the current repository
closure setup, with scaled geometry and density chosen consistently from the released mesh metadata. Closure uses the actual compiled rigid-subtree mass, center of mass and inertia, including any convex-decomposition volume difference.

Original source hand qpos is xyz + quaternion wxyz + 22 Shadow joints. Full MuJoCo
qpos additionally includes the object free joint. Relative object pose maps the
object frame into the hand root frame. Nothing is converted to WUJI coordinates.

Outputs are separated by `grasp_type`, with the existing compact layout:
`manifest.json`, `poses.npz`, `templates.json.gz`, `metrics.json.gz`, and checksums.
`retarget.viewer.snapshot.get_snapshot(type_directory)` reads this schema. Each
case also has `cases/*.npz` for all three stages' contacts and closure witnesses,
and `models/*.xml` to restore the scene. The model XML references shared assets
on this server; it is not a portable standalone hand/object archive. The generic
snapshot reader supports Shadow, while the WUJI primitive-specific inspector
and training data consumers are not implicitly converted to Shadow support.

Contacts come from MuJoCo collision geometry at each original target pose and the actual post-execution pose,
without enlarged collision margins or pose fitting. Stored fields include:
contact midpoint, signed separation, hand/object surface endpoints, world object
outward normal, hand link IDs, geom IDs, link-local hand anchors, object-local
anchors, and world forces/torques acting on the object. Static constraint forces
at prescribed qpos and closure force witnesses are separate fields. No claim is
made that recomputed contacts match unpublished original contact labels.

Both grasp and squeeze stages call the repository's unchanged
`ProjectedComFreeClosure` with default settings: friction 0.6, torsional coefficient
0.02 m, nominal preload bound 20 N, and certificate tolerance 0.05. All 13 scenarios
(zero and positive/negative six-dimensional axes) are stored. The `task_success` flag and full-run gate use the closure certificate on actual contacts after three-stage execution and settling. Nominal grasp/squeeze certificates are also stored separately; nominal grasp targets can have a positive surface gap before closing. The actual contact points, rather than projected template anchors, are passed to the same solver. Penetration and joint-limit violations
are recorded independently and do not silently change this closure-only gate.

The replay additionally executes pregrasp -> grasp -> squeeze and settles for
0.2 s, with gravity disabled like the upstream evaluation environment. Its pose
error/remaining contact count is a separate diagnostic, not an official six-axis
physical robustness certificate. Static force closure alone is not a real-world
or lift-success guarantee.

`verification.json` independently checks file hashes, exact source hand poses,
contact-anchor FK, and certified wrench witnesses. Failed source records remain
in `failures.json` and count against the pilot pass rate. Only a completely verified pilot with at least 90/100 post-execution certificates (explicitly authorized after reviewing the pilot) automatically launches the
full background run. Otherwise the script saves the results and stops. The full
run is partitioned by grasp type and into 256-case shards, and skips verified
shards when restarted. Its log is `results/dexonomy_shadow_full/run.log` and each
type has a `status.json`. The published source 'successful' designation is not
substituted for the new certificate.

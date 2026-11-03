# Dexonomy Shadow static reconstruction

This workflow is the user's replacement for the earlier dynamic replay/closure
batch. All execution and cache files are on the remote server. Existing dynamic
pilot results remain intact; the old dynamic full process was stopped.

## Preview

`results/dexonomy_shadow_static_preview_20261103/index.html` contains the original
100-case type/object/size-stratified selection, with 300 images covering pregrasp,
grasp and squeeze. The gallery changes stage globally and allows a sample
selection within each grasp type. It is served on remote loopback port 8098;
forward this port in VS Code, then open http://127.0.0.1:8098/.

The static preview has 31 grasp types and 1,848 geometric contacts across all
three targets. It does not reuse dynamically executed poses from the earlier
91%-closure pilot. A nominal grasp target can have no geometric contact before
the squeeze target closes the fingers.

## Full cache

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../miniconda3/envs/pi0/bin/python -u scripts/cache_dexonomy_static.py \
  --mode full --dataset data/dexonomy \
  --output results/dexonomy_shadow_static_full --workers 8 --records-per-shard 32
```

The input archive is streamed directly, avoiding an initial recursive scan of
millions of extracted entries. Existing extracted scene/mesh resources are shared.
Each task handles up to 32 source record files of a single grasp type; sample
count per shard varies because each source file contains multiple grasps.

Output: `<grasp_type>/shards/<index>/` with the familiar compact layout:
`manifest.json`, `poses.npz`, `templates.json.gz`, `metrics.json.gz`,
plus `contacts.npz`, source fingerprints, checksums and verification.
No per-pose XML or full-dataset images are generated. Models are rebuilt from
shared original assets only when needed by the snapshot reader.

All source 29-dimensional Shadow poses are preserved (xyz, quaternion wxyz,
22 joints), for all three stages. Full MuJoCo qpos additionally contains the
object free joint. Scene scale and per-row extra scale are both applied.
Object pose in the hand frame maps object-local points into the hand root frame.

Contact arrays are ragged, using sample-major, then pregrasp/grasp/squeeze order:
`offsets[3*i+s]:offsets[3*i+s+1]`. They include world midpoint, signed distance,
hand/object surface endpoints, object outward normal in world, link-local hand
anchors/normals, object-local anchors, body/geom IDs. No contact-force arrays
exist in this schema. A non-contacting pose is retained with zero contacts.

Static queries call only kinematics and collision detection. They never call
`mj_step`, `mj_forward`, `mj_contactForce`, optimization or closure solvers.
Purely visual geometry is omitted from contact compilation via the supported
`discardvisual` compiler option; on-demand display models retain visual assets.
Contact equivalence between these two models is covered by tests. Saved geom IDs
refer to the collision-only model; use body names/points for display models.

`task_success` is null: static geometry is not a dynamic success or closure label.
Penetration/limit diagnostics are retained separately. Failed source records stay
in failure reports and the original data is not removed. Source fingerprints
must match when reusing completed shards.

`status.json` reports samples cached, records processed, contacts, errors,
per-type counts, elapsed time and measured throughput. `shards.jsonl` lists all
completed shards. On restart the archive stream is replayed and matching verified
shards are reused. Completion has an explicit completed/completed_with_errors state.

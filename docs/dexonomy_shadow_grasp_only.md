# Grasp-only Shadow cache and paginated viewer

This is the current user-selected workflow: extract only the original
`grasp_qpos` target, compute static geometric contacts for that target, and
provide rendered previews. Pregrasp and squeeze targets are not computed or
stored in this schema. Dynamic execution and force closure are not performed.

## Cache and process

- Entry point: `scripts/cache_dexonomy_grasp.py`
- Full cache: `results/dexonomy_shadow_grasp_full`
- Preview cache: `results/dexonomy_shadow_grasp_preview_20261103`
- Modules: `retarget/env/grasp/dexonomy_grasp_static.py`
- Viewer: `scripts/serve_dexonomy_grasp.py`
- URL: http://192.168.0.32:8098/

The 100 preview poses were selected for coverage across 31 grasp types. They
are not the total dataset size. The viewer defaults to the entire completed
cache, with 12 poses per page, type counts, next/previous, random page and
numeric jump controls. Its separate preview source exposes all 31 types while
the archive-ordered full cache is still processing earlier types.

Previously completed three-stage static shards were converted by selecting
their exact grasp pose and corresponding ragged contact ranges; no geometry
recomputation was needed. The migration reused 212,071 original grasp targets.
Future work resumes against the original source-record fingerprints. Progress
counters keep migrated totals and avoid double-counting reused shards.

## Layout

Each `<grasp_type>/shards/<index>` directory uses the existing compact file
layout, with storage tag `dexonomy_shadow_grasp_snapshot_v1`:
`manifest.json`, `poses.npz`, `contacts.npz`, `templates.json.gz`,
`metrics.json.gz`, source fingerprints, checksums and verification.

`stage_names` is exactly `["grasp"]`. Source arrays have shape `N x 1 x 29`;
full scene state is `N x 1 x 36`. There are no `pregrasp_qpos` or
`squeeze_qpos` arrays. Ragged contact offsets have length `N+1`, and each
pose's contact range is `offsets[i]:offsets[i+1]`.

A successful static reconstruction means the original hand/object poses and
geometry were loaded and cached without an error. It does not assert dynamic
grasp stability or force closure. Those labels remain unevaluated, and
`task_success` is null. A source grasp target can have a positive surface gap
and therefore no zero-margin geometric contact; it is retained with count zero.

## Progress

`status.json` exposes the actual cached grasp count, error count and per-type
counts. `shards.jsonl` indexes completed, verified shards. Archive order means
some grasp types become available before others. `state=running` means the full
release has not finished. `all_reconstructed` in the viewer API becomes true
only after a zero-error completed state.

The viewer displays every already cached pose by pagination and renders full
cache images lazily. It does not pre-render millions of images or cap the cache
at 100 poses. Routes: `/api/status`, `/api/list`, and `/image`.

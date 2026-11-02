# Highest verified grasp success policy

647/1610 (40.19%): use the raw local MPPI output when it passes the original native
GraspADMM protocol; otherwise retain the original ADMM output. This is a separate
validation fallback policy. Raw MPPI is 573/1610 (35.59%); ADMM is 524/1610 (32.55%).
All 1610 cases, including 963 failures, are retained. No best trajectory frame is selected.

`candidates.npz` preserves both source candidates and the chosen 27-coordinate pose.
`index.json` contains the exact required-body groups and actual object mesh for each trial.
`manifest.json`, `models/`, `poses/`, `trajectories/` and `assets/` are self-contained replay
inputs with relative paths. Physical traces are copied byte for byte. XML changes only
make mesh paths relative; all physical parameters are retained. Static previews use
the official WUJI2 geometry and show the chosen pose, without implying a physical test.

Original measured reports, configuration, hashes and code are retained under `evidence/`.
Absolute paths there are historical provenance, not replay dependencies. Success rates
for the original raw MPPI and the validated fallback policy remain separate.

```bash
.venv-contact/bin/python scripts/publish_grasp_results.py --verify --activate
.venv-contact/bin/python run.py --config configs/web_grasp_best.yaml
# Reevaluate selected candidates in the original pinned native environment:
.venv-contact/bin/python scripts/publish_grasp_results.py --reevaluate --ids t04_cylinder_s01_seed00 t28_sphere_s00_seed00
```

Full usage and measured timings: `docs/graspadmm_mppi.md`.

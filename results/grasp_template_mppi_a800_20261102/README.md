# Fixed-template MPPI: final dense size grid

All **640 final results across 31 grasp types** are included, including failures.
There are 31 retained, audited 1× anchors and 609 newly solved adjacent sizes.
The remote job used three A800 GPUs and 1,024 candidates per batch.

- Projected friction-cone closure: **610/640 (95.31%)**.
- Anatomical patch interior coverage ≥80%: **420/640 (65.63%)**.
- Type 9: 17/17 pass both checks.
- Type 16: 0/17 pass closure; Type 24: 32/45 pass closure.
- No adjacent object pose jump exceeds 3 mm or 3°; 440/609 pairs keep its pose unchanged.
- Total job time: 46.14 minutes; solver batch time: 44.39 minutes.

Projected closure is a virtual-contact force balance certificate. Native dynamic
physical evaluation was disabled. Patch coverage and projected closure are
reported separately, and failed final cases are never filtered out.

`cases/` contains each selected final pose, saved trajectory, exact object mesh,
fixed template, projected contact forces and torques, preload, geometry metrics,
and required initialization/continuity references. Retry directories, worker
logs and optimizer traces are omitted. `dense_report.md`, `dense_results.csv`
and `summary.json` preserve aggregate and per-case metrics.

Scene mesh paths and manifest entries are relative. References use paths relative
to the repository root. Absolute paths in original source/audit records are
historical provenance, not replay dependencies. `materialization.json` records
matching original and published SHA256 for all 1,920 pose, trajectory and force
witness files. Original remote audits are retained under `source_audits/`; the
local independent audit checks final FK, collisions, force/torque balance,
friction bounds, fixed template identity and adjacent pose continuity again.

From the repository root, with the dependencies in `requirements-template-mppi.txt`:

```bash
python scripts/audit_template_mppi_results.py results/grasp_template_mppi_a800_20261102
mkdir -p output
ln -s ../results/grasp_template_mppi_a800_20261102 output/grasp_template_mppi_a800_20261102
python visualize.py --batch grasp_template_mppi_a800_20261102 --port 8089
```

The output symlink is only needed once. Existing viewer configuration is unchanged.
The exporter is `scripts/publish_template_mppi_results.py`; full solver and remote
launch details are in `docs/grasp_template_mppi.md` and `docs/template_mppi_remote.md`.

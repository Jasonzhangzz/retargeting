# Paired ADMM → MPPI evaluation

Completed 1610 paired trials. Native protocol: `graspadmm_native_v1`.

ADMM: 524/1610 (32.55%). Raw local MPPI: 573/1610 (35.59%), change +3.04 percentage points.

Transitions: {'retained': 450, 'lost': 74, 'recovered': 123, 'neither': 963}. The native-validation fallback policy is separate: 647/1610 (40.19%).

The local MPPI objective differs from ADMM and from historical MPPI guidance; the final acceptance is identical. Every saved ADMM final pose and actual mesh is reused, without native-evaluation-guided sample selection.

| Type | Trials | ADMM | MPPI raw | Change | With validated fallback |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 5 | 9 | +4 | 10 |
| 2 | 50 | 33 | 39 | +6 | 40 |
| 3 | 50 | 41 | 45 | +4 | 45 |
| 4 | 50 | 50 | 42 | -8 | 50 |
| 5 | 50 | 47 | 30 | -17 | 47 |
| 6 | 50 | 0 | 1 | +1 | 1 |
| 7 | 50 | 0 | 0 | +0 | 0 |
| 8 | 50 | 0 | 3 | +3 | 3 |
| 9 | 50 | 3 | 1 | -2 | 3 |
| 10 | 110 | 16 | 29 | +13 | 32 |
| 11 | 50 | 38 | 42 | +4 | 44 |
| 12 | 50 | 19 | 15 | -4 | 20 |
| 13 | 50 | 0 | 0 | +0 | 0 |
| 14 | 50 | 0 | 0 | +0 | 0 |
| 15 | 50 | 48 | 45 | -3 | 50 |
| 16 | 50 | 0 | 2 | +2 | 2 |
| 17 | 50 | 50 | 46 | -4 | 50 |
| 18 | 50 | 24 | 24 | +0 | 28 |
| 20 | 50 | 1 | 0 | -1 | 1 |
| 22 | 50 | 8 | 16 | +8 | 18 |
| 23 | 50 | 0 | 0 | +0 | 0 |
| 24 | 50 | 0 | 0 | +0 | 0 |
| 25 | 50 | 12 | 13 | +1 | 19 |
| 26 | 50 | 30 | 40 | +10 | 40 |
| 27 | 50 | 0 | 0 | +0 | 0 |
| 28 | 50 | 42 | 50 | +8 | 50 |
| 29 | 50 | 9 | 26 | +17 | 27 |
| 30 | 50 | 0 | 4 | +4 | 4 |
| 31 | 50 | 24 | 29 | +5 | 34 |
| 32 | 50 | 24 | 22 | -2 | 29 |
| 33 | 50 | 0 | 0 | +0 | 0 |

See `timings.json`, `by_size.csv`, `trials.csv` and `audit.json`.

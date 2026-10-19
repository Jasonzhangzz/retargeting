# Contact solver check

Reproduce with `python scripts/benchmark_contact_solvers.py`. The bundled sugar-box sequence starts at frame 23, runs three MPC windows (0.09 simulated seconds), and uses eight starts, one outer round and one hand update per window. Warm timings average the last two windows; kernel compilation is excluded. Exact outputs and full logs are in `summary.json` and the method logs.

| Method | Whole window | Object block | GPU gather/LM/select/scatter | Feasible incumbents held |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 141.2 ms | — | — | — |
| Sampled LM | 139.4 ms | 28.6 ms | 1.46 ms | 0% |
| QP | 261.9 ms | 150.2 ms | — | 33.75% |

The object block includes preparing/evaluating the current physical rollout, patch geometry and diagnostics. The GPU figure includes the small statistics readback; it is a synchronized wall time, not a kernel event measurement. Reusing this rollout makes the hand portion cheaper, so block times should not be interpreted as independent end-to-end overheads. Maximum selected point shifts were 4.9954 mm (LM) and 4.9721 mm (QP), within the 5 mm limit; inactive points were unchanged.

This short contact-stage run checks timings and bounds. It does not establish complete grasp success or show LM reaching its 0.1 mm feasibility tolerance. Separate numeric tests verify that both solvers skip neighborhood work and hold an already feasible set despite lower hand geometry costs, and verify selected-contact history shifting and correspondence resets.

> 历史实验记录：文中部分脚本已于2026-11-16清理；当前基线入口见 [baseline说明](../retarget/baseline/README.md)。

# Wuji2 sparse temporal retargeting

The reference is `emf_wuji_retarget` (the independent SDK 2026.8.31
reconstruction), not a new SDK implementation. Existing SDK and training code
are not modified. Source and URDF hashes are recorded with each campaign.

## API

```python
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver

solver = EMFTrajectorySolver(reference_path, glove_urdf, 'right',
    device='cuda:0', model=frozen_model, normalizer=joint_normalizer,
    crocoddyl_module=compiled_extension_directory)
result = solver.solve(positions_tx_m, orientations_xyzw, offsets,
    backend='banded_lm', iterations=16, stage1_iterations=4)
# result.human_q, q_raw, q_filtered, spatial_cost, total_cost, timings
```

`offsets` start with zero, end with the frame count and delimit complete,
independent, continuous segments. Input receivers use the TX frame; only the
network's input is converted to wrist coordinates. `initial_q` and
`human_initial` optionally replace the joint route's neural q and constructor
human seed. They must never be taken from SDK outputs in production benchmarks.

Joint backends: `banded_lm`, `trajectory_lbfgs`, `gpu_ilqr`,
`crocoddyl_boxfddp`. They optimize the complete segment and permit future data to
affect earlier frames. Causal backends: `causal_lm`, `causal_lbfgs`; they use
constructor q at reset and previous raw q thereafter, without neural guesses.
Causal budgets follow 50/80 first-frame and 1/10 later-frame update caps; these
are algorithm updates, **not** SLSQP function-evaluation counts.

## Objective and structure

Robot weights: direction 1 (stage 1), direction .1 and Laplacian 150 (stage 2),
thumb-relative .5 with the original source-angle gate, antireverse 20 on the
selected distal joints, and history .01. Spatial preprocessing, pinch, masks,
physiological human limits and robot joint order come from the frozen reference.
No network-anchor penalty, total-offset box, acceleration term or dt scaling is
introduced. Each source residual receives the square root of its quadratic
weight; the human pose multipliers .3 and 5.729577951308232 are already residual
multipliers and must not be square-rooted again.

The human joint objective is `.5*error_gain*||W e||² + .5*posture +
.5*20*||h_t-h_(t-1)||²`. This matches the fixed-reference gradient, while its
Gauss–Newton curvature and full-cost acceptance differ from the reference's
pose-only acceptance. The fused causal human kernel preserves the reference's
QP, adaptive evaluation budgets, retries, pose-only acceptance and state update.
Both source history slots receive the last q, so their ordinary update reduces
to a first-difference penalty, not acceleration.

The point-graph Laplacian couples fingers: spatial blocks can be dense. Time
coupling is block tridiagonal. Native CUDA PCR stores diagonal and neighboring
blocks, never a dense `(T*d, T*d)` matrix. One graph iteration is captured and
replayed for different budgets. Activity masks remove outward-bound directions.
Damping and the .35-rad per-update LM cap are numerical solver mechanisms,
not additional objective terms.

Robokit supplies native L-BFGS directions/parallel trial machinery and the
Riccati/box-QP rules. The adapter checks projected Armijo steps and resets
curvature on active-set changes, including the previous step used by the
next curvature update. Native FP32 L-BFGS cost buffers are recentered at every
iterate against the actual FP64 cost, so a large absolute trajectory cost does
not erase small improvements. Trial FK and costs use an explicit vectorized
trial axis; they are not serialized Python line-search loops. Analytic iLQR fills both mixed Q-function
blocks, uses `x_next=u`, and has a sequential time backward/feedback pass; it is
not advertised as fully parallel over time. The production iLQR adapter fuses
the same FP64 Riccati recursion and projected-Newton box-QP into one cooperative
CUDA block per record, then fuses the FP32 feedback rollout per record/trial.
The original multi-kernel implementation remains available with `fused=False`
for numerical and performance comparison. Human/robot dimensions are
specialized to 21/20 to fit CUDA shared memory, and registered in a fixed order
so worker processes reuse the RAM compilation cache. The compact PCR linear solve has
logarithmic parallel depth. The alpha=.2 joint-route output filter uses an
associative prefix scan with linear memory. It agrees mathematically with the
SDK IIR but not necessarily bitwise with sequential FP32 arithmetic; the causal
route uses the sequential FP32 recurrence.

## Commands and provenance

- `scripts/prepare_wuji_trajectory_benchmark.py`: freezes source/assets and selects
  5 development plus 20 independent held-out source records. Existing validation
  contains only 19 independent source files; one unused sibling recording is
  selected by finite, confident EMF alone. Complete continuous context spans are
  retained. The normal source sequence stride can be 1 or 2; a stride of 2 at
  60 Hz is not a missing frame. No q error is used for selection.
- `scripts/build_wuji_trajectory_crocoddyl.py`: run with the isolated environment
  containing Crocoddyl 3.2.1, pybind11 and cmeel-eigen. It builds C++ action models
  and BoxFDDP; there are no Python callbacks in solver loops. Model derivatives
  are FP64 only. The extension records its source/build hashes.
- `scripts/verify_wuji_trajectory_objectives.py --root ...`: FP64 cost and
  analytic-gradient comparison against the frozen original FrameObjective,
  both stages and glove/hand configurations, including a straight thumb gate.
- `scripts/verify_wuji_trajectory_ilqr_fusion.py --root ...`: original and fused
  CUDA execution comparison with dense spatial blocks, active bounds,
  independent unequal-length segments and human/robot history weights.
- `scripts/benchmark_wuji_trajectory.py sdk --root ...`: original SDK reference,
  two warmups and ten measured repeats, including frame construction.
- `scripts/benchmark_wuji_trajectory.py run --root ... --backend banded_lm
  --configuration builtin --side right --split development --job-id example
  --budgets 1 2 4 8 16 32`: numerical IK benchmark, no neural optimization.
- `scripts/launch_wuji_trajectory_campaign.py --root ...`: finite single-owner
  sequential queue; requires completed numerical/pipeline validation and SDK
  evidence. Failed jobs stop for analysis; no automatic retries. Set
  `CONTROL.json.stop_requested=true` to stop. An explicit `--resume` retains
  completed units only when the candidate signature still matches.

Use one visible GPU and an 8-GiB PyTorch allocation cap. Warp, Inductor, Triton,
CUDA driver and temporary caches belong on RAM, with environments and evidence
on CPFS. Existing DiT training is left running. Latencies in a shared GPU
environment must be identified as such.

Filtered q is compared to filtered SDK q. A separate raw comparison uses SDK q
reconstructed from the FP32 IIR recurrence and is labeled approximate. SDK
human intermediates are loaded only after solving, for common-target scoring;
they never initialize the main solvers. All raw arrays, source timestamps,
segment boundaries, failures, cold compilation and steady measurements are
retained. Report frame-weighted and record-level errors separately from the
objective value; lower objective does not imply identical SDK q.

The final experiment manifest also retains candidate revisions. Completed
non-iLQR cases from before the CUDA fusion were reused only after every other
runtime source hash was verified unchanged; their original execution signatures
and original receipts remain available beside the compatibility proof. Old
slow iLQR measurements and the failed 32-wide shared-memory kernel are retained
as negative evidence rather than overwritten.

The human joint objective uses the symmetric Rodrigues dyad to recover the
rotation axis near pi. The earlier `R+I` approximation had an incorrect
axis derivative there. The inverse SO(3) Jacobian uses a stable cos/sin
cotangent ratio, avoiding an artificial singularity when floating-point
normalization places the rotation-vector norm one ulp above pi. Tests cover
all 21 human columns in Torch and C++, both gloves/sides, near zero and pi;
exactly pi is a geodesic branch cut, so only finite cost is claimed there.

Long L-BFGS vectors use a global-memory GPU two-loop recursion. The native
Robokit vector-free kernel keeps history in per-thread local matrices, which
caused several GiB of CUDA local-memory reservation at trajectory dimensions.
The wrapper retains Robokit's cost/gradient, projection, ten-pair history,
actual-step Armijo filter and parallel trials. Short causal vectors retain
the native direction kernel. Both paths match bounded high-precision tests.

The benchmark leaves headroom with a 6-GiB PyTorch allocation budget and
`PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6`. A separate single-owner
resource guard measures process-resident NVML memory, including Warp and the
CUDA context, and stops its controller if 8 GiB is exceeded. These are separate
limits. Previous failed attempts and resident-memory diagnostics are retained
in candidate revisions. No frames are removed to meet the budget.

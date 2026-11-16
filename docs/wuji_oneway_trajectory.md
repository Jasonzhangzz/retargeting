> 历史实验记录：文中部分脚本已于2026-11-16清理；当前基线入口见 [baseline说明](../retarget/baseline/README.md)。

# Causal and frozen-history Wuji2 comparison

These backends keep the original spatial weights and physical joint limits.
They add no network-anchor penalty. The network is an initializer only.

`EMFTrajectorySolver.solve(..., backend=...)` accepts three new choices:

- `causal_prefix`: analytic frame residuals/Jacobians; solve the lower-bidiagonal
  causal stationarity linearization using native CUDA tile Cholesky and an
  affine prefix scan. Linear-solve depth is logarithmic in segment length.
- `frozen_parallel`: block Jacobi; each frame freezes its predecessor from the
  previous outer iteration. Frames run concurrently. Information propagates
  forward across iterations, not backward from future frames.
- `frozen_history`: finish each current frame while holding the preceding raw
  FP32 result fixed. Initialize from that preceding result, run two robot
  stages, then update the raw history and FP32 alpha=.2 output filter.
  This online algorithm has temporal dependence; only independent segments and
  frame-internal arithmetic are parallel. Robot stage caps are first 50/80
  LM updates, later 1/10, distinct from SDK SLSQP evaluation caps.

All three use source-rule causal human reconstruction. A user-supplied initial
trajectory must itself be causal to claim past/present information only. The
built-in network initializer is temporal_causal; source-matched frozen_history
needs no network initializer.

The parallel variants solve G_t = grad phi_t(q_t) + .01(q_t-q_(t-1)) = 0,
with diagonal GN block J_t.T J_t + .01 I and lower block -.01 I. Their
linearization is nonsymmetric and is NOT an arbitrarily masked joint Hessian.
The prefix method evaluates per-frame shifted-history local merit at proposed
predecessor motion; damping/acceptance are independent for each frame. This is
a causal nonlinear root update with a local safeguard, without a global
least-squares monotonicity guarantee. Rejected predecessor steps are corrected
at later outer iterations. Physical bound activity and damping are retained.

The two stages are optimized sequentially over the parallel trajectory. A
finite stage-one budget differs from SDK's per-frame interleaving of stages;
this and stopping rules remain possible sources of SDK disagreement. Sharing
weights or considering history alone does not guarantee identical answers.

Run CPU structural tests:

```bash
python -m pytest -q tests/learning/test_wuji_trajectory_oneway.py
```

`scripts/benchmark_wuji_oneway.py` uses the five existing development records,
full continuous segments, each glove and hand configuration. It compares:

1. Explicit diagnostic common SDK-human targets, isolating robot differences.
2. Source-rule causal human reconstructed from EMF, for pipeline quality.

Parallel robot budgets are 4/16/32/200 updates after four warm-stage updates;
online caps are recorded separately. Each case has two warmups and ten timed
repeats. Robot timings include output filtering but exclude cached network
inference, preprocessing and IO. Causal human times are reported separately;
their sum with robot medians is NOT a measured full-pipeline latency. SDK-human
states are never used in the production-input cases. Both raw recovered-SDK
and exact filtered-SDK errors are saved; inverse-IIR SDK raw is approximate.
A resident GPU monitor enforces 8 GiB, with a 6 GiB Torch allocator cap.
No neural training, heldout selection, or original source modification occurs.

The existing joint `banded_lm` path is already parallel over the trajectory in
both the human and robot stages. Its analytic frame derivatives are batched;
compact CUDA PCR eliminates blocks in logarithmic rounds. It does not wait for
the preceding frame's IK. Its nonlinear iterations, human-to-robot stage
ordering, and CUDA reduction rounds remain dependencies. Joint human modeling
and future access differ from the source's causal algorithm.

An explicit sparse shooting KKT correctness reference is provided in
`trajectory_kkt_reference.py`. With x_t=q_(t-1), u_t=q_t and P the segmented
previous-control shift, its step system is

```
[ H_uu  H_ux  -P^T ] [du]   [-g_u]
[ H_xu  H_xx    I  ] [dx] = [-g_x]
[ -P      I    0  ] [nu]   [  0 ]
```

Eliminating dx=P du yields exactly the block-tridiagonal LM system used by
CUDA PCR. CPU sparse KKT and eliminated PCR steps are tested at FP64 precision
with resets, physical bound activity, damping, zero history weight and one
frame. The explicit KKT reference uses SciPy on CPU; it is **not** advertised
as a newly implemented generic GPU sparse-indefinite solver. Production
storage remains O(T d²), and never constructs a dense (T d)² matrix.

The comparison report separately labels full joint EMF-to-q measurements and
one-way robot-only measurements. Joint LM curves can be shown with the
"future allowed" checkbox. Switching a linear solver alone does not remove
future terms from a joint objective or reproduce SDK's finite SLSQP map.

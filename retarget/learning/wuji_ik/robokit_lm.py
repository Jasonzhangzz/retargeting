"""Robokit's native CUDA LM with GPU residuals and hard joint bounds.

The Cholesky, trust-region accept/reject, and graph execution use Robokit/Warp
CUDA kernels. Residuals and analytic Jacobians use the validated Torch CUDA FK;
no CPU linear algebra or per-frame optimization is used.
"""
from enum import Enum
from types import SimpleNamespace

import torch
import warp as wp
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.var_values import VectorVar, VarValues
from robokit.terms.task import ResidualTask


class Key(Enum):
    Q = 0


@wp.kernel
def _bounded_step(x: wp.array2d(dtype=wp.float32), delta: wp.array2d(dtype=wp.float32),
                  lower: wp.array1d(dtype=wp.float32), upper: wp.array1d(dtype=wp.float32),
                  max_step: float, out: wp.array2d(dtype=wp.float32)):
    b, j = wp.tid()
    out[b, j] = wp.clamp(x[b, j] + wp.clamp(delta[b, j], -max_step, max_step), lower[j], upper[j])


@wp.kernel
def _bounded_step_per_frame(x: wp.array2d(dtype=wp.float32), delta: wp.array2d(dtype=wp.float32),
                            lower: wp.array2d(dtype=wp.float32), upper: wp.array2d(dtype=wp.float32),
                            max_step: float, out: wp.array2d(dtype=wp.float32)):
    b, j = wp.tid()
    out[b, j] = wp.clamp(x[b, j] + wp.clamp(delta[b, j], -max_step, max_step), lower[b, j], upper[b, j])


class BoundedVector(VectorVar):
    def __init__(self, x, lower, upper, max_step=.35):
        super().__init__(x)
        self.lower, self.upper, self.max_step = lower, upper, max_step

    def clone(self):
        return BoundedVector(wp.clone(self.x), self.lower, self.upper, self.max_step)

    def view(self, batch_size, offset=0):
        lower = self.lower[offset:offset+batch_size] if self.lower.ndim == 2 else self.lower
        upper = self.upper[offset:offset+batch_size] if self.upper.ndim == 2 else self.upper
        return BoundedVector(self.x[offset:offset+batch_size], lower, upper, self.max_step)

    def integrate(self, velocity, out=None, tangent_mask=None, weight_decay=0., out_velocity=None):
        if tangent_mask is not None or weight_decay or out_velocity is not None:
            raise ValueError('This bounded LM variable supports direct, unmasked LM steps only')
        if out is None:
            out = self.clone()
        kernel = _bounded_step_per_frame if self.lower.ndim == 2 else _bounded_step
        wp.launch(kernel, self.x.shape,
                  inputs=[self.x, velocity, self.lower, self.upper, self.max_step], outputs=[out.x], device=self.device)
        return out


class GPUResidualTask(ResidualTask):
    var_key = Key.Q
    residual_weight = None

    def __init__(self, objective, initial, lower, upper, compile=True):
        self.objective = objective
        self.lower, self.upper = lower, upper
        # Keep double precision in geometric residuals, float32 in native LM.
        def evaluate(q):
            r, j = objective(q.double())
            g = (j * r[..., None]).sum(1)
            active = ((q <= lower + 1e-7) & (g > 0)) | ((q >= upper - 1e-7) & (g < 0))
            return r.float(), (j * (~active)[:, None]).float()
        self.evaluate = torch.compile(evaluate, fullgraph=True, dynamic=False) if compile else evaluate
        self._r, self._j = self.evaluate(initial.float())
        self._residual_dim = self._r.shape[1]

    @property
    def residual_dim(self):
        return self._residual_dim

    def precompute(self, var_values, need_gradient=False):
        self._r, self._j = self.evaluate(wp.to_torch(var_values.get(Key.Q).x))

    def compute_weighted_residual(self, var_values, out_residual=None, row_offset=0):
        wp.to_torch(out_residual)[:, row_offset:row_offset+self.residual_dim].copy_(self._r)
        return out_residual

    def compute_weighted_jacobian_analytic(self, var_values, out_jacobian=None, row_offset=0):
        wp.to_torch(out_jacobian)[:, row_offset:row_offset+self.residual_dim].copy_(self._j)
        return out_jacobian


_GRAPH_POOLS = {}


class RobokitCUDALM:
    """Persistent full-solve CUDA graph; replay reads fresh initial states/targets."""
    def __init__(self, objective, *, iterations=60, compile=True, graph_mode='full', max_step=.35):
        if iterations < 1 or not 0 < max_step < float('inf'):
            raise ValueError('Positive finite step limit and iteration count required')
        self.objective, self.iterations, self.compile = objective, iterations, compile
        self.graph_mode, self.max_step = graph_mode, max_step
        self.solver = None

    @torch.no_grad()
    def __call__(self, initial, lower, upper):
        if initial.device.type != 'cuda':
            raise ValueError('RobokitCUDALM requires CUDA, with no CPU fallback')
        if lower.shape != upper.shape or lower.shape not in ((initial.shape[1],), initial.shape):
            raise ValueError('Bounds must be shared [D] or independent per frame [N,D]')
        wp.init()
        caller = torch.cuda.current_stream(initial.device)
        if not hasattr(self, 'torch_stream'):
            self.torch_stream = torch.cuda.Stream(device=initial.device)
        self.torch_stream.wait_stream(caller)
        stream = wp.stream_from_torch(self.torch_stream)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(stream):
            if self.solver is None:
                self.initial = initial.clamp(lower, upper).float().contiguous()
                self.lower, self.upper = lower.float().contiguous(), upper.float().contiguous()
                self.var = VarValues.from_var(BoundedVector(wp.from_torch(self.initial),
                    wp.from_torch(self.lower), wp.from_torch(self.upper), self.max_step), Key.Q)
                self.task = GPUResidualTask(self.objective, self.initial, self.lower, self.upper, self.compile)
                config = LMOptimizerConfig(max_iter=self.iterations, early_stopping_interval=0,
                    cuda_graph_mode='none', linear_solver='cholesky', lm_lambda=.01,
                    lambda_min=1e-7, gain_ratio_epsilon=1e-12)
                self.solver = LMOptimizer([self.task], config, device=stream.device)
                # Torch owns the capture pool because the FK/residual kernels allocate
                # Torch intermediates. Warp joins this external capture. Sharing the
                # pool is safe only for these serial, fixed-order solve graphs.
                self.solver.solve(self.var)
                self.torch_stream.synchronize()
                self.initial.copy_(initial.clamp(lower, upper))
                if self.graph_mode == 'full':
                    device_index = initial.device.index
                    if device_index not in _GRAPH_POOLS:
                        _GRAPH_POOLS[device_index] = torch.cuda.graph_pool_handle()
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=self.torch_stream, pool=_GRAPH_POOLS[device_index]):
                        with wp.ScopedCapture(stream=stream, external=True):
                            self.solver.solve(self.var)
                else:
                    self.graph = None
            else:
                if initial.shape != self.initial.shape:
                    raise ValueError('Each graph requires a fixed batch shape')
                if lower.shape != self.lower.shape:
                    raise ValueError('Captured bound shape changed')
                self.lower.copy_(lower); self.upper.copy_(upper)
                self.initial.copy_(initial.clamp(lower, upper))
            initial_cost = .5 * self.objective(initial.double())[0].square().sum(-1)
            if self.graph is not None:
                self.graph.replay()
            else:
                self.solver.solve(self.var)
            q = wp.to_torch(self.var.get(Key.Q).x).double().clone()
            # Return actual FP64 cost, rather than rounded native acceptance costs.
            cost = .5 * self.objective(q)[0].square().sum(-1)
        caller.wait_stream(self.torch_stream)
        return SimpleNamespace(q=q, initial_cost=initial_cost, cost=cost,
            accepted_steps=torch.full_like(cost, float('nan')), converged=torch.zeros_like(cost, dtype=torch.bool),
            history=[dict(backend='robokit_native_cuda_cholesky', cuda_graph_mode=self.graph_mode,
                          iterations=self.iterations, linear_solver='cholesky',capture_owner='torch',warp_external_capture=True)])

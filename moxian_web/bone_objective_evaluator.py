"""Fixed-shape float64 objective and gradient execution for native L-BFGS.

CUDA graph replay removes Python/autograd dispatch from each callback. The
objective, gradient, bounds and stopping criteria remain unchanged.
"""
from __future__ import annotations
import time
import numpy as np
import torch


class ObjectiveEvaluator:
    def __init__(self, objective, start, backend='auto'):
        if backend not in ('auto', 'eager', 'cuda_graph'):
            raise ValueError('无效目标函数执行后端')
        self.objective=objective
        self.input=torch.tensor(start, device=objective.model.device, dtype=torch.float64, requires_grad=True)
        self.term_names=None
        self.graph=None
        self.fallback_reason=None
        self.backend='eager'
        began=time.monotonic()
        if backend!='eager' and self.input.is_cuda:
            try:
                stream=torch.cuda.Stream(device=self.input.device)
                stream.wait_stream(torch.cuda.current_stream(self.input.device))
                with torch.cuda.stream(stream):
                    # Allocate autograd workspaces outside capture.
                    for _ in range(2):self._packed()
                torch.cuda.current_stream(self.input.device).wait_stream(stream)
                self.graph=torch.cuda.CUDAGraph()
                # The live mesh uses a separate model on another thread.
                # Only this evaluator's private tensors enter the capture.
                with torch.cuda.graph(self.graph, stream=stream,capture_error_mode='thread_local'):
                    self.output=self._packed()
                self.backend='cuda_graph'
            except RuntimeError as exc:
                self.graph=None
                self.fallback_reason=str(exc)
                if backend=='cuda_graph':raise
        self.setup_s=time.monotonic()-began

    def _packed(self):
        loss,terms=self.objective.loss(self.input)
        if self.term_names is None:self.term_names=tuple(terms)
        derivative=torch.autograd.grad(loss, self.input)[0]
        # One GPU-to-host transfer per native callback, including diagnostics.
        return torch.cat((loss.reshape(1),derivative,torch.stack([terms[k] for k in self.term_names])))

    def __call__(self, values):
        with torch.no_grad():
            self.input.copy_(torch.as_tensor(values, dtype=torch.float64, device=self.input.device))
        if self.graph is None:output=self._packed()
        else:
            self.graph.replay()
            output=self.output
        packed=output.detach().cpu().numpy().copy()
        if not np.isfinite(packed).all():raise ValueError('标定目标或梯度出现非有限数')
        count=self.input.numel()
        return float(packed[0]),packed[1:count+1],dict(zip(self.term_names,map(float,packed[count+1:])))

    def provenance(self):
        return dict(backend=self.backend,setup_s=self.setup_s,dtype='float64',fallback_reason=self.fallback_reason)

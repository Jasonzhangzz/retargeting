"""AdamW on balanced FA control coordinates."""
from collections.abc import Callable
import numpy as np

def shift_knots(values: np.ndarray, horizon: int, advance: float) -> np.ndarray:
    knots = values.reshape(horizon, -1)
    grid = np.arange(horizon)
    return np.stack([np.interp(grid+advance, grid, knots[:, j])
                     for j in range(knots.shape[1])], axis=1).reshape(-1)

class AdamW:
    def __init__(self, *, learning_rate: float = .3, weight_decay: float = .01,
                 persist: bool = False, betas: tuple = (.9, .999), epsilon: float = 1.e-8,
                 device=None):
        if learning_rate <= 0 or weight_decay < 0 or not np.isfinite([learning_rate, weight_decay]).all():
            raise ValueError('Finite positive AdamW learning rate and nonnegative decay required')
        self.lr, self.decay = learning_rate, weight_decay
        self.persist, self.betas, self.epsilon = persist, betas, epsilon
        self.memory = None
        self.device = device

    def retain_candidate_memory(self, selected):
        """Keep the selected world's moments on their existing device."""
        m,v,count = self._window_memory if self.device is None else self._cuda_window_memory
        copy = (lambda value:value.copy()) if self.device is None else (lambda value:value.clone())
        self.memory = copy(m[selected]),copy(v[selected]),int(count[selected])

    @property
    def _window_memory(self):
        # Preserve the pre-existing NumPy diagnostic/private adapter contract.
        # CUDA solving uses _cuda_window_memory directly and never reads this
        # host view; it is materialized only when a legacy consumer asks for it.
        if self.device is not None and hasattr(self,'_cuda_window_memory'):
            return tuple(value.cpu().numpy().copy() for value in self._cuda_window_memory)
        return self._host_window_memory

    @_window_memory.setter
    def _window_memory(self, value):
        self._host_window_memory = value

    def update(self, variables: np.ndarray, evaluate: Callable, *, iterations: int,
               horizon: int, advance: float, continue_window: bool = False,
               evaluate_on_device: bool = False) -> tuple:
        if iterations < 1:
            raise ValueError('Positive AdamW update budget required')
        if self.device is not None:
            return self._update_device(variables,evaluate,iterations=iterations,horizon=horizon,
                                       advance=advance,continue_window=continue_window,
                                       evaluate_on_device=evaluate_on_device)
        variables = variables.copy()
        m, v = np.zeros_like(variables), np.zeros_like(variables)
        count = np.zeros(len(variables), np.int64)
        restored = (continue_window and hasattr(self, '_window_memory')) or (self.persist and self.memory is not None)
        if continue_window and hasattr(self, '_window_memory'):
            m[:], v[:], count[:] = self._window_memory
        elif restored:
            m[:] = shift_knots(self.memory[0], horizon, advance)
            v[:] = shift_knots(self.memory[1], horizon, advance)
            count[:] = self.memory[2]
        best = variables.copy()
        best_value = np.full(len(variables), np.inf)
        best_m, best_v, best_count = m.copy(), v.copy(), count.copy()
        b1, b2 = self.betas

        def retain(value):
            improved = np.isfinite(value) & (value < best_value)
            best[improved] = variables[improved]
            best_value[improved] = value[improved]
            best_m[improved], best_v[improved] = m[improved], v[improved]
            best_count[improved] = count[improved]

        for _ in range(iterations):
            value, gradient = evaluate(variables, True)
            retain(value)
            eligible = np.isfinite(value) & np.isfinite(gradient).all(axis=1)
            count[eligible] += 1
            m[eligible] = b1*m[eligible]+(1-b1)*gradient[eligible]
            v[eligible] = b2*v[eligible]+(1-b2)*gradient[eligible]**2
            mh = m[eligible]/(1-b1**count[eligible, None])
            vh = v[eligible]/(1-b2**count[eligible, None])
            variables[eligible] *= 1-self.lr*self.decay
            variables[eligible] -= self.lr*mh/(np.sqrt(vh)+self.epsilon)
        value, _ = evaluate(variables, False)
        retain(value)
        evaluate(best, False)
        selected = int(np.argmin(best_value))
        self._window_memory = (best_m.copy(), best_v.copy(), best_count.copy())
        if self.persist and np.isfinite(best_value[selected]):
            self.memory = (best_m[selected].copy(), best_v[selected].copy(), int(best_count[selected]))
        return best, int(np.sum(best_count > 0)), dict(method='AdamW',
            iterations=iterations, gradient_evaluations=iterations,
            learning_rate=self.lr, weight_decay=self.decay, betas=list(self.betas),
            epsilon=self.epsilon, persistent_moments=self.persist, moments_restored=restored,
            continued_within_window=continue_window,
            final_values=best_value.tolist(), selected_candidate=selected,
            finite_difference_gradients=False, export='best_evaluated_iterate')

    def _update_device(self, variables, evaluate, *, iterations, horizon, advance, continue_window,
                       evaluate_on_device):
        """Same update/retention formulas on CUDA, shared by both entry points.

        CUDA callbacks keep parameters, values and derivatives on the device.
        Host callbacks remain supported for independent numerical reference tests.
        """
        import torch
        device = str(self.device)
        if not device.startswith('cuda'):
            raise ValueError('Device AdamW requires CUDA')
        tensor = lambda value: torch.as_tensor(value,dtype=torch.float64,device=device).clone()
        x = tensor(variables)
        m,v = torch.zeros_like(x),torch.zeros_like(x)
        count = torch.zeros(len(x),dtype=torch.int64,device=device)
        restored = (continue_window and hasattr(self,'_cuda_window_memory')) or (self.persist and self.memory is not None)
        def shifted(value):
            knots = tensor(value).reshape(horizon,-1)
            positions = (torch.arange(horizon,dtype=x.dtype,device=device)+advance).clamp(0,horizon-1)
            left = positions.floor().long();right = (left+1).clamp(max=horizon-1)
            alpha = (positions-left)[:,None]
            return ((1-alpha)*knots[left]+alpha*knots[right]).reshape(-1)
        if continue_window and hasattr(self,'_cuda_window_memory'):
            previous = self._cuda_window_memory
            m,v = tensor(previous[0]),tensor(previous[1])
            count = torch.as_tensor(previous[2],dtype=torch.int64,device=device).clone()
        elif restored:
            m[:],v[:] = shifted(self.memory[0]),shifted(self.memory[1])
            count.fill_(int(self.memory[2]))
        best = x.clone();best_value = torch.full((len(x),),torch.inf,dtype=x.dtype,device=device)
        best_m,best_v,best_count = m.clone(),v.clone(),count.clone()
        b1,b2 = (torch.tensor(value,dtype=x.dtype,device=device) for value in self.betas)
        def retain(value):
            improved = torch.isfinite(value)&(value<best_value)
            best[improved],best_value[improved] = x[improved],value[improved]
            best_m[improved],best_v[improved],best_count[improved] = m[improved],v[improved],count[improved]
        with torch.no_grad():
            for _ in range(iterations):
                value,gradient = evaluate(x if evaluate_on_device else x.cpu().numpy(),True)
                value,gradient = tensor(value),tensor(gradient)
                retain(value)
                eligible = torch.isfinite(value)&torch.isfinite(gradient).all(dim=1)
                count[eligible] += 1
                m[eligible] = b1*m[eligible]+(1-b1)*gradient[eligible]
                v[eligible] = b2*v[eligible]+(1-b2)*gradient[eligible]**2
                mh = m[eligible]/(1-b1**count[eligible,None])
                vh = v[eligible]/(1-b2**count[eligible,None])
                x[eligible] *= 1-self.lr*self.decay
                x[eligible] -= self.lr*mh/(vh.sqrt()+self.epsilon)
            value,_ = evaluate(x if evaluate_on_device else x.cpu().numpy(),False)
            retain(tensor(value))
            result = best.cpu().numpy().copy()
            evaluate(best if evaluate_on_device else result,False)
        selected = int(best_value.argmin())
        self._cuda_window_memory = best_m,best_v,best_count
        if self.persist and bool(torch.isfinite(best_value[selected])):
            self.retain_candidate_memory(selected)
        return result,int((best_count>0).sum()),dict(method='AdamW',iterations=iterations,
            gradient_evaluations=iterations,learning_rate=self.lr,weight_decay=self.decay,
            betas=list(self.betas),epsilon=self.epsilon,persistent_moments=self.persist,
            moments_restored=restored,continued_within_window=continue_window,
            final_values=best_value.cpu().tolist(),selected_candidate=selected,
            finite_difference_gradients=False,export='best_evaluated_iterate',parameter_update_backend='torch_cuda',
            device_objective_callback=evaluate_on_device)

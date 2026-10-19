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
                 persist: bool = False, betas: tuple = (.9, .999), epsilon: float = 1.e-8):
        if learning_rate <= 0 or weight_decay < 0 or not np.isfinite([learning_rate, weight_decay]).all():
            raise ValueError('Finite positive AdamW learning rate and nonnegative decay required')
        self.lr, self.decay = learning_rate, weight_decay
        self.persist, self.betas, self.epsilon = persist, betas, epsilon
        self.memory = None

    def update(self, variables: np.ndarray, evaluate: Callable, *, iterations: int,
               horizon: int, advance: float, continue_window: bool = False) -> tuple:
        if iterations < 1:
            raise ValueError('Positive AdamW update budget required')
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

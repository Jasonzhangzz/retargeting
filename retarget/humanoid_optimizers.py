"""Optional CUDA action solvers; the shared run.py AdamW is left intact.

MPPI uses a Gaussian base centred at the current knot sequence (alpha=1 in
Williams et al., arXiv:1707.02342, section III-D2). The existing objective
already includes action reference/smoothness costs. The optional full-objective
incumbent and best-sample safeguard handles nonconvex trajectories whose
weighted mean is worse. Sampling is chunked through MJWarp rollout worlds.
"""
import math

from retarget.fa.adamw import AdamW


class LineSearchAdamW(AdamW):
    """AdamW moments/proposal with per-world Armijo backtracking.

    If momentum or decoupled decay produces a non-descent direction, use the
    current gradient preconditioned by the same second moment. Rejected steps
    keep the controls unchanged; moments still accumulate gradient observations.
    """

    def __init__(self, *, max_trials=6, contraction=.5, armijo=1.e-4, **kwargs):
        super().__init__(**kwargs)
        if (max_trials < 1 or not 0 < contraction < 1 or not 0 < armijo < 1):
            raise ValueError('Invalid Armijo backtracking settings')
        if self.device is None:
            raise ValueError('LineSearchAdamW requires the CUDA objective callback')
        self.max_trials, self.contraction, self.armijo = max_trials, contraction, armijo

    def _update_device(self, variables, evaluate, *, iterations, horizon, advance,
                       continue_window, evaluate_on_device):
        import torch
        device = str(self.device)
        if not device.startswith('cuda') or not evaluate_on_device:
            raise ValueError('LineSearchAdamW requires a CUDA device objective')
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device).clone()
        x = tensor(variables)
        m, v = torch.zeros_like(x), torch.zeros_like(x)
        count = torch.zeros(len(x), dtype=torch.int64, device=device)
        restored = False
        if continue_window and hasattr(self, '_cuda_window_memory'):
            m, v, count = (a.clone() for a in self._cuda_window_memory)
            restored = True
        elif self.persist and self.memory is not None:
            # Same warm shift as AdamW, interpolated on its existing device.
            positions = (torch.arange(horizon, device=device, dtype=x.dtype)+advance).clamp(0, horizon-1)
            left = positions.floor().long(); right = (left+1).clamp(max=horizon-1)
            alpha = (positions-left)[:, None]
            def shifted(a):
                knots = tensor(a).reshape(horizon, -1)
                return ((1-alpha)*knots[left]+alpha*knots[right]).reshape(-1)
            m[:], v[:] = shifted(self.memory[0]), shifted(self.memory[1])
            count.fill_(int(self.memory[2])); restored = True
        b1, b2 = (torch.tensor(a,dtype=x.dtype,device=device) for a in self.betas)
        history = []
        with torch.no_grad():
            for _ in range(iterations):
                value, gradient = evaluate(x, True)
                value, g = tensor(value), tensor(gradient)
                eligible = torch.isfinite(value)&torch.isfinite(g).all(dim=1)
                count[eligible] += 1
                m[eligible] = b1*m[eligible]+(1-b1)*g[eligible]
                v[eligible] = b2*v[eligible]+(1-b2)*g[eligible]**2
                correction = count.clamp(min=1)[:, None]
                mh = m/(1-b1**correction)
                vh = v/(1-b2**correction)
                denom = vh.sqrt()+self.epsilon
                direction = -self.lr*(mh/denom+self.decay*x)
                slope = (g*direction).sum(dim=1)
                fallback = eligible&(slope >= 0)&(g.square().sum(dim=1) > 0)
                direction[fallback] = -self.lr*g[fallback]/denom[fallback]
                slope = (g*direction).sum(dim=1)
                active = eligible&(slope < 0)
                accepted = ~active
                alpha = torch.ones(len(x), dtype=x.dtype, device=device)
                result, result_value = x.clone(), value.clone()
                accepted_alpha = torch.zeros_like(alpha)
                trials = 0
                for trial in range(self.max_trials):
                    if bool(accepted.all()):
                        break
                    candidate = torch.where(accepted[:, None], result, x+alpha[:, None]*direction)
                    candidate_value, _ = evaluate(candidate, False)
                    candidate_value = tensor(candidate_value)
                    ok = ~accepted&torch.isfinite(candidate_value)&(
                        candidate_value <= value+self.armijo*alpha*slope)
                    result[ok], result_value[ok] = candidate[ok], candidate_value[ok]
                    accepted_alpha[ok] = alpha[ok]
                    accepted |= ok
                    alpha[~accepted] *= self.contraction
                    trials = trial+1
                history.append(dict(value_before=value.cpu().tolist(), value_after=result_value.cpu().tolist(),
                                    directional_derivative=slope.cpu().tolist(), alpha=accepted_alpha.cpu().tolist(),
                                    trials=trials, accepted=(accepted&active).cpu().tolist(),
                                    descent_fallback=fallback.cpu().tolist()))
                x = result
            final_value, _ = evaluate(x, False)
        self._cuda_window_memory = m, v, count
        selected = int(final_value.argmin())
        if self.persist and bool(torch.isfinite(final_value[selected])):
            self.retain_candidate_memory(selected)
        return x.cpu().numpy(), sum(any(step['accepted'][i] for step in history) for i in range(len(x))), dict(
            method='AdamW_Armijo', iterations=iterations, gradient_evaluations=iterations,
            learning_rate=self.lr, weight_decay=self.decay, betas=list(self.betas), epsilon=self.epsilon,
            persistent_moments=self.persist, moments_restored=restored,
            continued_within_window=continue_window, parameter_update_backend='torch_cuda',
            device_objective_callback=True, finite_difference_gradients=False,
            max_trials=self.max_trials, contraction=self.contraction, armijo=self.armijo,
            steps=history, final_values=final_value.cpu().tolist(), selected_candidate=selected,
            export='last_accepted_monotone_iterate')


class MPPI:
    """One global nominal, common fixed ADMM contacts, batched forward-only samples."""

    def __init__(self, *, samples=64, noise_std=.05, temperature=1., control_scale=.05,
                 seed=0, device='cuda:0', update_rule='safeguarded'):
        if samples < 2 or not all(math.isfinite(a) and a > 0 for a in (noise_std, temperature, control_scale)):
            raise ValueError('MPPI needs >=2 samples and positive finite noise/temperature/scale')
        import torch
        self.samples, self.noise_std, self.temperature = samples, noise_std, temperature
        if update_rule not in ('safeguarded','weighted-mean'):
            raise ValueError('Unknown MPPI update rule')
        self.update_rule = update_rule
        self.sigma = noise_std/control_scale
        self.device = str(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.persist = False

    def update(self, variables, evaluate, *, iterations, horizon, advance,
               continue_window=False, evaluate_on_device=False):
        import torch
        if iterations < 1 or not evaluate_on_device:
            raise ValueError('MPPI requires a positive update budget and device callback')
        x = torch.as_tensor(variables, dtype=torch.float64, device=self.device).clone()
        if not bool((x == x[0]).all()):
            raise ValueError('MPPI rollout worlds must start from the same nominal')
        worlds = len(x)
        history = []
        with torch.no_grad():
            common=getattr(self,'common_rollout',None)
            if common is not None:common.begin()
            values, _ = evaluate(x, False)
            if not torch.allclose(values, values[:1].expand_as(values), rtol=1.e-5, atol=1.e-5):
                raise ValueError('MPPI samples must share one fixed ADMM objective')
            for _ in range(iterations):
                nominal = x[0].clone()
                before = values[0].clone()
                noise = self.sigma*torch.randn((self.samples, x.shape[1]), dtype=x.dtype,
                                               device=self.device, generator=self.generator)
                samples = nominal[None]+noise
                scores = torch.empty(self.samples, dtype=x.dtype, device=self.device)
                batches = 0
                for start in range(0, self.samples, worlds):
                    count = min(worlds, self.samples-start)
                    batch = nominal[None].expand_as(x).clone()
                    batch[:count] = samples[start:start+count]
                    cost, _ = evaluate(batch, False)
                    scores[start:start+count] = cost[:count]
                    batches += 1
                finite = torch.isfinite(scores)
                weights = torch.zeros_like(scores)
                best, best_value = nominal.clone(), before.clone()
                if bool(finite.any()):
                    # Stabilized Gibbs weights; alpha=1 centred base distribution
                    # means no additional importance control-cost cross term.
                    minimum = scores[finite].min()
                    weights[finite] = torch.softmax(-(scores[finite]-minimum)/self.temperature, dim=0)
                    mean = nominal+(weights[:, None]*noise).sum(dim=0)
                    mean_batch = mean[None].expand_as(x).clone()
                    mean_values, _ = common.value(mean_batch) if common is not None else evaluate(mean_batch, False)
                    mean_value = mean_values[0]
                    if bool(torch.isfinite(mean_value)) and (self.update_rule=='weighted-mean' or bool(mean_value < best_value)):
                        best, best_value = mean, mean_value
                    idx = int(torch.where(finite,scores,torch.inf).argmin())
                    if self.update_rule=='safeguarded' and bool(scores[idx] < best_value):
                        best, best_value = samples[idx].clone(), scores[idx].clone()
                    mean_accepted = bool(torch.equal(best, mean))
                else:
                    mean_value = torch.tensor(torch.inf, device=x.device)
                    mean_accepted = False
                x = best[None].expand_as(x).clone()
                # Always evaluate the exported control batch: cached physics,
                # contact diagnostics and executed states must agree with it.
                values, _ = common.export(x) if common is not None else evaluate(x, False)
                history.append(dict(value_before=float(before), value_after=float(values[0]),
                                    samples=self.samples, forward_batches=batches,
                                    valid_samples=int(finite.sum()),
                                    effective_sample_size=float(1/weights.square().sum()) if bool(finite.any()) else 0.,
                                    weighted_mean_value=float(mean_value) if bool(torch.isfinite(mean_value)) else None,
                                    weighted_mean_accepted=mean_accepted,
                                    best_sample_fallback=bool(not mean_accepted and best_value < before)))
        return x.cpu().numpy(), int(values[0] < history[0]['value_before']), dict(
            method='MPPI', iterations=iterations, gradient_evaluations=0,
            samples_per_iteration=self.samples, sampled_trajectories=self.samples*iterations,
            noise_std_radians=self.noise_std, temperature=self.temperature,
            base_distribution='current_nominal_alpha_1', full_objective_safeguard=self.update_rule=='safeguarded',
            update_rule=self.update_rule,
            parameter_update_backend='torch_cuda', device_objective_callback=True,
            finite_difference_gradients=False, steps=history, final_values=values.cpu().tolist(),
            selected_candidate=0, export=('best_of_incumbent_weighted_mean_and_samples' if self.update_rule=='safeguarded' else 'weighted_mean'))

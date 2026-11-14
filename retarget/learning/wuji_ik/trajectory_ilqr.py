"""Analytic kinematic iLQR using Robokit's native Riccati/box-QP kernels.

Time dependencies in the backward and feedback passes are sequential. Spatial
linearization and trial costs are batched across frames; no finite differences.
"""
from types import SimpleNamespace
import numpy as np
import torch
import warp as wp
from robokit.opt.ilqr_optimizer import ILQROptimizer,_compute_rollout_ctrls_kernel,_STEP_SIZES
from .trajectory import TrajectoryResult
from robokit.opt.ilqr_optimizer import (_compute_value_hess_ab_kernel,_solve_box_qp_kernel,_compute_gains_kernel,_compute_q_gain_kernel,_compute_value_function_kernel)


@wp.kernel
def _full_q_function(t:int,dynamics:wp.array4d(dtype=wp.float64),gradient:wp.array3d(dtype=wp.float64),hessian:wp.array4d(dtype=wp.float64),vg:wp.array2d(dtype=wp.float64),vh_ab:wp.array3d(dtype=wp.float64),qg:wp.array2d(dtype=wp.float64),qh:wp.array3d(dtype=wp.float64)):
    b,i,j=wp.tid();d=vg.shape[1]
    if j==0:
        value=gradient[b,t,i]
        for m in range(d):value+=dynamics[b,t,m,i]*vg[b,m]
        qg[b,i]=value
    value=hessian[b,t,i,j]
    for m in range(d):value+=dynamics[b,t,m,i]*vh_ab[b,m,j]
    qh[b,i,j]=value



class AnalyticILQR(ILQROptimizer):
    def __init__(self,objective,initial,lower,upper,*,fused=True):
        wp.init()
        self.objective=objective;self.n,self.d=initial.shape;self.b=len(objective.starts)
        self.t=max(b-a for a,b in zip(objective.offsets,objective.offsets[1:]));self.time=torch.arange(self.n,device=initial.device)-objective.starts[objective.segment]
        self.device=wp.get_device(str(initial.device));self.keep=[]
        def zeros(*shape,dtype=torch.float64):
            a=torch.zeros(shape,device=initial.device,dtype=dtype);self.keep.append(a);return wp.from_torch(a)
        self._ctrls=zeros(self.b,self.t,self.d,dtype=torch.float32)
        self._value_grad=zeros(self.b,self.d);self._value_hess=zeros(self.b,self.d,self.d)
        self._q_grad=zeros(self.b,2*self.d);self._q_hess=zeros(self.b,2*self.d,2*self.d)
        self._cost_grad=zeros(self.b,self.t+1,2*self.d);self._cost_hess=zeros(self.b,self.t+1,2*self.d,2*self.d)
        self.dynamics_jacobians=zeros(self.b,self.t,self.d,2*self.d);wp.to_torch(self.dynamics_jacobians)[...,self.d:]=torch.eye(self.d,device=initial.device)
        self._value_hess_ab=zeros(self.b,self.d,2*self.d)
        self._backward_ok=zeros(self.b,dtype=torch.int32);self._regularization=zeros(self.b);wp.to_torch(self._regularization).fill_(.01)
        self._qp_factors=zeros(self.b,self.d,self.d);self._qp_free=zeros(self.b,self.d,dtype=torch.int32)
        self._qp_steps=zeros(self.b,self.d);self._qp_directions=zeros(self.b,self.d);self._qp_trials=zeros(self.b,self.d);self._qp_grads=zeros(self.b,self.d);self._qp_quus=zeros(self.b,self.d,self.d);self._qp_qus=zeros(self.b,self.d)
        self._feedforward=zeros(self.b,self.t+1,self.d,dtype=torch.float32);self._feedback=zeros(self.b,self.t,self.d,self.d,dtype=torch.float32)
        self._gain_columns=zeros(self.b,self.d,self.d);self._q_gain=zeros(self.b,self.d,self.d)
        self._ctrl_lower64=wp.from_torch(lower.double().contiguous());self._ctrl_upper64=wp.from_torch(upper.double().contiguous())
        self._lo=wp.from_torch(lower.float().contiguous());self._hi=wp.from_torch(upper.float().contiguous())
        self.alphas=torch.tensor(_STEP_SIZES,device=initial.device,dtype=torch.float32);self.alpha_wp=wp.from_torch(self.alphas);self.k=len(self.alphas)
        self.diffs=zeros(self.b*self.k,self.d,dtype=torch.float32);self.actions=zeros(self.b*self.k,self.d,dtype=torch.float32);self.trials=zeros(self.t+1,self.b*self.k,self.d,dtype=torch.float32)
        self.q=initial.clone();self.lower,self.upper=lower,upper
        from .trajectory_batch import BatchedTrajectoryObjective
        self.many=BatchedTrajectoryObjective(objective,self.k)
        self.graph=None;self.fused=fused
        if fused:
            from .trajectory_ilqr_native import setup
            setup(self)

    def _compute_backward_pass(self):
        if self.fused:
            from .trajectory_ilqr_native import fused_backward
            return fused_backward(self)
        # The upstream helper skips Qxu; its value update reads that block.
        # This analytic adapter fills both mixed blocks explicitly.
        b,h,d=self._ctrls.shape;k=2*d
        wp.copy(self._value_grad,self._cost_grad[:,h,:d]);wp.copy(self._value_hess,self._cost_hess[:,h,:d,:d]);self._feedforward[:,h].zero_();self._backward_ok.fill_(1)
        for t in reversed(range(h)):
            wp.launch(_compute_value_hess_ab_kernel,(b,d,k),inputs=[t,self.dynamics_jacobians,self._value_hess],outputs=[self._value_hess_ab])
            wp.launch(_full_q_function,(b,k,k),inputs=[t,self.dynamics_jacobians,self._cost_grad,self._cost_hess,self._value_grad,self._value_hess_ab],outputs=[self._q_grad,self._q_hess])
            wp.launch(_solve_box_qp_kernel,b,inputs=[t,self._q_grad,self._q_hess,self._ctrls,self._ctrl_lower64,self._ctrl_upper64,self._regularization],outputs=[self._backward_ok,self._qp_factors,self._qp_free,self._qp_steps,self._qp_directions,self._qp_trials,self._qp_grads,self._qp_quus,self._qp_qus,self._feedforward])
            wp.launch(_compute_gains_kernel,(b,d),inputs=[t,self._q_hess,self._backward_ok,self._qp_factors,self._qp_free],outputs=[self._gain_columns,self._feedback])
            wp.launch(_compute_q_gain_kernel,(b,d,d),inputs=[t,self._q_hess,self._feedback],outputs=[self._q_gain])
            wp.launch(_compute_value_function_kernel,(b,d,d),inputs=[t,self._q_grad,self._q_hess,self._q_gain,self._feedforward,self._feedback],outputs=[self._value_grad,self._value_hess])

    def _iteration(self):
        obj=self.objective;d=self.d;r,j,delta=obj.residuals(self.q);g=(j*r[...,None]).sum(1);h=j.transpose(-1,-2)@j
        cost_grad=wp.to_torch(self._cost_grad);cost_hess=wp.to_torch(self._cost_hess);cost_grad.zero_();cost_hess.zero_()
        s,t=obj.segment,self.time;w=obj.weight;eye=torch.eye(d,device=self.q.device,dtype=self.q.dtype)
        cost_grad[s,t,:d]=-w*delta;cost_grad[s,t,d:]=g+w*delta
        cost_hess[s,t,:d,:d]=w*eye;cost_hess[s,t,:d,d:]=-w*eye;cost_hess[s,t,d:,:d]=-w*eye;cost_hess[s,t,d:,d:]=h+w*eye
        ctrls=wp.to_torch(self._ctrls);ctrls.zero_();ctrls[s,t]=self.q.float()
        self._compute_backward_pass()
        trial=wp.to_torch(self.trials).reshape(self.t+1,self.b,self.k,d)
        if self.fused:
            from .trajectory_ilqr_native import fused_rollout
            fused_rollout(self)
        else:
            self._rollout_sequential()
        candidates=trial[self.time,obj.segment].transpose(0,1).double()
        costs=self.many(candidates,False)[0];before=obj.evaluate(self.q,False)
        best=costs.argmin(0);best_cost=costs.gather(0,best[None]).squeeze(0);qnew=candidates[best[obj.segment],torch.arange(self.n,device=self.q.device)]
        ok=torch.isfinite(best_cost)&(best_cost<before)&(wp.to_torch(self._backward_ok).bool())
        self.q.copy_(torch.where(ok[obj.segment,None],qnew,self.q))
        reg=wp.to_torch(self._regularization);reg.copy_(torch.where(ok,reg*.5,reg*2).clamp(1e-7,1e10))

    def _rollout_sequential(self):
        d=self.d;ctrls=wp.to_torch(self._ctrls)
        differences=wp.to_torch(self.diffs).reshape(self.b,self.k,d);differences.zero_()
        for t in range(self.t):
            wp.launch(_compute_rollout_ctrls_kernel,self.b*self.k,inputs=[t,0,self.k,0,0.,self._ctrls,self._feedforward,self._feedback,self.alpha_wp,self._lo,self._hi,self.diffs],outputs=[self.actions,self.trials],device=self.device)
            differences.copy_(wp.to_torch(self.actions).reshape(self.b,self.k,d)-ctrls[:,t,None])

    @torch.no_grad()
    def __call__(self,initial,iterations):
        self.many.refresh()
        self.q.copy_(initial.clamp(self.lower,self.upper));wp.to_torch(self._regularization).fill_(.01)
        caller=torch.cuda.current_stream(initial.device)
        if not hasattr(self,"torch_stream"):self.torch_stream=torch.cuda.Stream(device=initial.device)
        self.torch_stream.wait_stream(caller);stream=wp.stream_from_torch(self.torch_stream)
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(stream):
            if self.graph is None:
                saved=self.q.clone();self._iteration();torch.cuda.synchronize();self.q.copy_(saved);wp.to_torch(self._regularization).fill_(.01)
                self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=self.torch_stream):
                    with wp.ScopedCapture(stream=stream,external=True):self._iteration()
                self.q.copy_(saved);wp.to_torch(self._regularization).fill_(.01)
            for _ in range(iterations):self.graph.replay()
        caller.wait_stream(self.torch_stream)
        return TrajectoryResult(self.q.clone(),self.objective.evaluate(initial,False),self.objective.evaluate(self.q,False),backend='robokit_analytic_ilqr_cuda_fused_sequential_riccati' if self.fused else 'robokit_analytic_ilqr_cuda_sequential_riccati',function_evaluations=(self.k+1)*iterations+2,gradient_evaluations=iterations)

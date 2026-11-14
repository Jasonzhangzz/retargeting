"""Robokit L-BFGS with per-segment padded variables and projected Armijo checks."""
from enum import Enum
import torch
import warp as wp
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.var_values import VectorVar,VarValues
from robokit.terms.task import GradientTask
from robokit.utils.warp_utils import gather
from .trajectory import TrajectoryResult


class Key(Enum):Q=0


@wp.kernel
def _project(x:wp.array2d(dtype=wp.float32),v:wp.array2d(dtype=wp.float32),lo:wp.array2d(dtype=wp.float32),hi:wp.array2d(dtype=wp.float32),out:wp.array2d(dtype=wp.float32),actual:wp.array2d(dtype=wp.float32),record:bool):
    i,j=wp.tid();y=wp.clamp(x[i,j]+v[i,j],lo[i,j],hi[i,j]);out[i,j]=y
    if record:actual[i,j]=y-x[i,j]


class ProjectedVector(VectorVar):
    def __init__(self,x,lo,hi):super().__init__(x);self.lo,self.hi=lo,hi
    def clone(self):return ProjectedVector(wp.clone(self.x),self.lo,self.hi)
    def view(self,batch_size,offset=0):return ProjectedVector(self.x[offset:offset+batch_size],self.lo[offset:offset+batch_size],self.hi[offset:offset+batch_size])
    def gather(self,indices,out=None):
        if out is None:return ProjectedVector(gather(self.x,indices),gather(self.lo,indices),gather(self.hi,indices))
        gather(self.x,indices,out.x);gather(self.lo,indices,out.lo);gather(self.hi,indices,out.hi);return out
    def integrate(self,velocity,out=None,tangent_mask=None,weight_decay=0.,out_velocity=None):
        if tangent_mask is not None or weight_decay:raise ValueError('Unexpected mask or optimizer weight decay')
        if out is None:out=self.clone()
        wp.launch(_project,self.x.shape,inputs=[self.x,velocity,self.lo,self.hi,out.x,out_velocity if out_velocity is not None else velocity,out_velocity is not None],device=self.device)
        return out


class PackedObjective(GradientTask):
    var_key=Key.Q
    residual_dim=1  # unused scratch required by the current Robokit base lifecycle
    parallel_safe=False  # Torch callbacks must remain on the shared Torch/Warp stream
    def compute_weighted_residual(self,*args,**kwargs):raise RuntimeError('Direct-gradient task has no residual dispatch')
    def compute_weighted_jacobian(self,*args,**kwargs):raise RuntimeError('Direct-gradient task has no Jacobian dispatch')
    def __init__(self,objective,d):
        self.objective=objective;self.batch=len(objective.starts);self.d=d;self.length=max(b-a for a,b in zip(objective.offsets,objective.offsets[1:]));self.calls=0;self.grad_calls=0;self.expanded={}
        self.time=torch.arange(objective.offsets[-1],device=objective.reference.device)-objective.starts[objective.segment]
    def pack(self,q):
        out=q.new_zeros((self.batch,self.length,self.d));out[self.objective.segment,self.time]=q;return out.flatten(1)
    def unpack(self,x):return x.reshape(self.batch,self.length,self.d)[self.objective.segment,self.time]
    def compute_weighted_cost_and_gradient(self,var_values,*,out_cost,out_gradient=None):
        from .trajectory_batch import BatchedTrajectoryObjective
        x=wp.to_torch(var_values.get(Key.Q).x);trials=len(x)//self.batch
        if trials not in self.expanded:self.expanded[trials]=BatchedTrajectoryObjective(self.objective,trials)
        expanded=self.expanded[trials];expanded.refresh()
        q=x.reshape(self.batch,trials,self.length,self.d)[self.objective.segment,:,self.time].transpose(0,1).double()
        cost,g=expanded(q,out_gradient is not None)
        self.last_cost64=cost.T.flatten()
        wp.to_torch(out_cost).add_(self.last_cost64.float());self.calls+=trials
        if out_gradient is not None:
            packed=q.new_zeros((self.batch,trials,self.length,self.d));packed[self.objective.segment,:,self.time]=g.transpose(0,1)
            wp.to_torch(out_gradient).add_(packed.flatten(2).reshape(self.batch*trials,-1).float());self.grad_calls+=trials



class ProjectedLBFGS(LBFGSOptimizer):
    @classmethod
    def _build_kernel(cls,tangent_dim,*args):
        # Avoid the native per-thread (history, ceil(T*D/threads)) arrays for
        # long vectors. The short causal frame path keeps Robokit's kernel.
        if tangent_dim>256:return None
        return super()._build_kernel(tangent_dim,*args)
    def _tiled_search_direction(self,var,select_first=False,direction=True):
        if self._search_direction_kernel is not None:
            return super()._tiled_search_direction(var,select_first,direction)
        if self._fold or self.config.line_search!='armijo':raise ValueError('Global-history trajectory path requires unfolded Armijo search')
        from .trajectory_lbfgs_streaming import search_direction,select_best
        if select_first:select_best(self)
        if direction:search_direction(self)
    def _start(self,var):
        super()._start(var);self.costs.zero_();self.best_costs.zero_()
        self._best_improved.fill_(1);self._best_var.view(self.batch_size).accept(self._best_improved,var)
    def _compute_cost_and_gradient_direct(self,var):
        super()._compute_cost_and_gradient_direct(var)
        self.current_cost64=self.terms[0].last_cost64.clone()
        # Native scalar buffers are FP32. Recenter every iteration so tiny
        # improvements are not lost against a large absolute trajectory cost.
        self.costs.zero_();self.best_costs.zero_()
        q=wp.to_torch(var.get(Key.Q).x);lo=wp.to_torch(var.get(Key.Q).lo);hi=wp.to_torch(var.get(Key.Q).hi);g=wp.to_torch(self.gradient)
        active=((q<=lo+1e-7)&(g>0))|((q>=hi-1e-7)&(g<0))
        if hasattr(self,'previous_active'):
            changed=(active!=self.previous_active).any(-1)
            wp.to_torch(self.history_count).masked_fill_(changed,0);wp.to_torch(self.history_start).masked_fill_(changed,0);wp.to_torch(self.rho_history).masked_fill_(changed[:,None],0);wp.to_torch(self.last_step_dq).masked_fill_(changed[:,None],0)
        self.previous_active=active.clone();g.masked_fill_(active,0)
    def _evaluate_line_search_costs_direct(self,expanded_batch):
        super()._evaluate_line_search_costs_direct(expanded_batch)
        # Robokit selection may impose a stricter unprojected-slope test too;
        # this filter guarantees every accepted trial satisfies actual-step Armijo.
        delta=wp.to_torch(self._candidate_velocities).reshape(self.batch_size,self.n_line_search,-1)
        g=wp.to_torch(self.gradient);slope=(delta*g[:,None]).sum(-1)
        costs=wp.to_torch(self._expanded_costs).reshape(self.batch_size,self.n_line_search)
        difference=self.terms[0].last_cost64.reshape(self.batch_size,self.n_line_search)-self.current_cost64[:,None]
        costs.copy_(difference.float())
        costs.masked_fill_(~((slope<0)&(difference<=self.config.wolfe_c1*slope.double())),float('inf'))


class RobokitTrajectoryLBFGS:
    def __init__(self,objective,iterations=16):self.objective,self.iterations=objective,iterations;self.solver=None
    @torch.no_grad()
    def __call__(self,initial,lower,upper):
        wp.init();device=wp.get_device(str(initial.device))
        if self.solver is None:
            self.task=PackedObjective(self.objective,initial.shape[-1]);task=self.task
            packed=task.pack(initial).float();lo=task.pack(lower.expand_as(initial)).float();hi=task.pack(upper.expand_as(initial)).float()
            self.var=VarValues.from_var(ProjectedVector(wp.from_torch(packed),wp.from_torch(lo),wp.from_torch(hi)),Key.Q)
            config=LBFGSOptimizerConfig(max_iter=self.iterations,early_stopping_interval=0,cuda_graph_mode='none',history_len=10,gradient_mode='analytic_gradient',fold_line_search_gradient=False)
            self.solver=ProjectedLBFGS([task],config,device=device)
        task,var,solver=self.task,self.var,self.solver
        from dataclasses import replace
        solver.config=replace(solver.config,max_iter=self.iterations)
        wp.to_torch(var.get(Key.Q).x).copy_(task.pack(initial).float());task.calls=task.grad_calls=0
        stream=wp.get_stream(str(initial.device))
        with wp.ScopedStream(stream):solver.solve(var)
        q=task.unpack(wp.to_torch(var.get(Key.Q).x)).double().clone()
        return TrajectoryResult(q,self.objective.evaluate(initial,False),self.objective.evaluate(q,False),backend='robokit_projected_lbfgs_global_history' if solver._search_direction_kernel is None else 'robokit_projected_lbfgs',function_evaluations=task.calls,gradient_evaluations=task.grad_calls)

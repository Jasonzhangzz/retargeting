"""Evaluate shared MPPI nominal controls once, then broadcast observables.

Sampling worlds have identical states, references and fixed ADMM contacts.
Mean/incumbent/export controls therefore need one physical rollout, not one
copy per sampling slot. MJWarp sampling and gradient rollouts remain intact.
"""
from copy import copy
from dataclasses import replace
from functools import cache

import torch
import warp as wp

from retarget.warp_arrays import device_copy


@cache
def copy_kernel(dtype):
    @wp.kernel(enable_backward=False,module='unique')
    def broadcast(source:wp.array(dtype=dtype),width:int,target:wp.array(dtype=dtype)):
        i=wp.tid()
        target[i]=source[i%width]
    @wp.kernel(enable_backward=False,module='unique')
    def first_world(source:wp.array(dtype=dtype),worlds:int,width:int,target:wp.array(dtype=dtype)):
        i=wp.tid()
        target[i]=source[(i//width)*worlds*width+i%width]
    return broadcast,first_world


class CommonRollout:
    def __init__(self,parent):
        from retarget.humanoid_mjwarp import MJWarpHumanoid
        self.parent=parent
        problem=copy(parent.problem)
        # Keep a cache even when the caller disables its own cache: export
        # needs the list of observable arrays, not differentiation workspace.
        problem.settings=replace(problem.settings,multistarts=1,trajectory_cache=True)
        self.backend=MJWarpHumanoid(problem)
        self.graphs={}
        self.forward_calls=0
        self.observables=parent.observable_arrays()
        assert len(self.observables)==len(self.backend.rollout_cache.arrays)
        for src,dst in zip(self.backend.rollout_cache.arrays,self.observables):
            assert src.shape[0]==1 and dst.shape[0]==parent.worlds
            assert src.shape[1:]==dst.shape[1:] and src.dtype==dst.dtype

    def _launch(self,key,fn):
        if key not in self.graphs:
            fn()
            with wp.ScopedCapture(device=self.parent.device) as capture:fn()
            self.graphs[key]=capture.graph
        wp.capture_launch(self.graphs[key])

    def begin(self):
        """Synchronize one fixed action subproblem, once per ADMM block."""
        p,c=self.parent,self.backend
        if (p.dense_control_base is not None)!=(c.dense_control_base is not None):
            c.use_dense_control_base(None if p.dense_control_base is None else p.dense_control_base.numpy())
            self.graphs.clear()
        def sync():
            if p.dense_control_base is not None:
                device_copy(c.dense_control_base,p.dense_control_base)
                device_copy(c.dense_control_boundary,p.dense_control_boundary)
            for name in ('qpos','qvel','ctrl','position','rotation','com','wrench','object_wrench','hand_position',
                         'mask','local','anchor','stance','penalty'):
                device_copy(getattr(c.targets,name),getattr(p.targets,name))
            for name in ('z','dual','forces','floor_point','floor_force','floor_torque'):
                src,dst=getattr(p.targets,name),getattr(c.targets,name)
                width=dst.size//dst.shape[0]
                wp.launch(copy_kernel(src.dtype)[1],dst.size,
                          inputs=[src.flatten(),p.worlds,width,dst.flatten()],device=p.device)
            for name in ('q0','v0','warmstart0','previous_controls'):
                src,dst=getattr(p,name),getattr(c,name)
                wp.launch(copy_kernel(src.dtype)[0],dst.size,
                          inputs=[src.flatten(),src.size,dst.flatten()],device=p.device)
            for name in ('previous','knot_reference'):
                device_copy(getattr(c,name),getattr(p,name))
            device_copy(c.model.weights,p.model.weights)
            src,dst=p.steps[0].inputs.time,c.steps[0].inputs.time
            wp.launch(copy_kernel(src.dtype)[0],1,
                      inputs=[src.flatten(),src.size,dst.flatten()],device=p.device)
        self._launch('sync',sync)
        c._cached=None;c._snapshot_cache=None;c.dirty=True
        c.rollout_cache.clear()

    def value(self,variables):
        c=self.backend
        before=c.forward_calls
        value,_=c.evaluate(variables[:1],False)
        self.forward_calls+=c.forward_calls-before
        return value.expand(self.parent.worlds).clone(),None

    def export(self,variables):
        p,c=self.parent,self.backend
        value,_=self.value(variables)
        caller=torch.cuda.current_stream(str(p.device))
        if not hasattr(p,'torch_stream'):
            p.torch_stream=torch.cuda.ExternalStream(wp.get_stream(p.device).cuda_stream,device=str(p.device))
        p.torch_stream.wait_stream(caller)
        with torch.no_grad(),torch.cuda.stream(p.torch_stream):
            def restore():
                for src,dst in zip(c.rollout_cache.arrays,self.observables):
                    wp.launch(copy_kernel(src.dtype)[0],dst.size,
                              inputs=[src.flatten(),src.size,dst.flatten()],device=p.device)
                # COM, losses and diagnostics are derived arrays, not cache
                # observables. Rebuild them on the broadcast physical state.
                p._objective()
            self._launch('export',restore)
            wp.to_torch(p.variables).copy_(variables)
            p._cached=variables.clone();p._value=wp.to_torch(p.total).double().clone()
            if p.rollout_cache is not None:
                p.rollout_cache.record(variables)
                p.rollout_cache.workspace_valid=False
        caller.wait_stream(p.torch_stream)
        p._external_rollout=True
        p._snapshot_cache=None;p.dirty=False
        return p._value.clone(),None

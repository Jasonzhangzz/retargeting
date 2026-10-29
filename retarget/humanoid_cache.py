"""Exact per-world reuse of two physical rollouts for AdamW best-iterate export.

Observable MJWarp states and solver warm starts are cached together.
"""
from functools import cache
import warp as wp
from retarget.warp_arrays import device_copy


@cache
def select_kernel(dtype):
    @wp.kernel(enable_backward=False,module='unique')
    def select(first:wp.array(dtype=dtype),second:wp.array(dtype=dtype),
               slot:wp.array(dtype=int),row_width:int,output:wp.array(dtype=dtype)):
        i=wp.tid()
        if slot[i//row_width]==0:output[i]=first[i]
        else:output[i]=second[i]
    return select


class RolloutCache:
    def __init__(self,backend):
        self.backend=backend
        self.arrays=backend.observable_arrays()
        self.buffers=[[wp.empty(a.shape,dtype=a.dtype,device=a.device) for a in self.arrays] for _ in range(2)]
        self.slot=wp.empty(backend.worlds,dtype=int,device=backend.device)
        self.graphs={};self.clear()

    def clear(self):
        self.keys=[None,None];self.next_slot=0;self.workspace_valid=True;self.hits=0

    def _launch(self,key,fn):
        if key not in self.graphs:
            fn()
            with wp.ScopedCapture(device=self.backend.device) as capture:fn()
            self.graphs[key]=capture.graph
        wp.capture_launch(self.graphs[key])

    def record(self,variables):
        slot=self.next_slot;self.next_slot=1-slot
        def save():
            for src,dst in zip(self.arrays,self.buffers[slot]):device_copy(dst,src)
        self._launch(slot,save)
        self.keys[slot]=variables.clone();self.workspace_valid=True

    def restore(self,variables):
        import torch
        if not any(key is not None for key in self.keys):return False
        selected=torch.full((len(variables),),-1,dtype=torch.int32,device=variables.device)
        for slot,key in enumerate(self.keys):
            if key is not None:selected[(variables==key).all(dim=1)]=slot
        if bool((selected<0).any()):return False
        wp.to_torch(self.slot).copy_(selected)
        def restore():
            for a,first,second in zip(self.arrays,*self.buffers):
                wp.launch(select_kernel(a.dtype),a.size,inputs=[first.flatten(),second.flatten(),self.slot,a.size//len(variables),a.flatten()],device=a.device)
        self._launch('restore',restore)
        self.workspace_valid=False;self.hits+=1
        return True

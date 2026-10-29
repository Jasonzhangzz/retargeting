"""Device array copies usable inside Warp graphs."""
from functools import cache
import warp as wp

@cache
def copy_kernel(dtype):
    @wp.kernel(enable_backward=False,module='unique')
    def copy(source:wp.array(dtype=dtype),target:wp.array(dtype=dtype)):
        i=wp.tid();target[i]=source[i]
    return copy


def device_copy(target,source,*args,**kwargs):
    if (not args and not kwargs and target.device.is_cuda and target.device==source.device
            and target.dtype==source.dtype and target.size==source.size and target.is_contiguous and source.is_contiguous):
        if target.size and target.ptr!=source.ptr:
            wp.launch(copy_kernel(source.dtype),target.size,inputs=[source.flatten(),target.flatten()],device=source.device)
        return
    return wp.copy(target,source,*args,**kwargs)



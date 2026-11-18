"""Serialize Warp host dispatch shared by display and calibration threads."""
import functools
import threading

_dispatch_lock=threading.RLock()


def serialized(method):
    @functools.wraps(method)
    def call(self,*args,**kwargs):
        import warp as wp
        import torch
        device=getattr(self,'device',None)
        if device is None and args:device=getattr(args[0],'device',None)
        with _dispatch_lock:
            wp.init()
            if device is not None and str(device).startswith('cuda'):
                # Torch copies and Warp launches use the same stream unless a
                # query explicitly enters its own synchronized capture stream.
                stream=wp.stream_from_torch(torch.cuda.current_stream(device))
                with wp.ScopedDevice(str(device)),wp.ScopedStream(stream):
                    return method(self,*args,**kwargs)
            return method(self,*args,**kwargs)
    return call

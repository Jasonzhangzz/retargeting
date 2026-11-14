"""Sample whole-device usage, including Warp allocations outside PyTorch."""
import threading
import torch


class DeviceMemoryMonitor:
    def __init__(self):
        self.peak=0;self.stop=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set():
            free,total=torch.cuda.mem_get_info();self.peak=max(self.peak,total-free)
            self.stop.wait(.2)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=1)

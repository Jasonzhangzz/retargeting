"""Bounded endpoint mesh diagnostics, independent of CARD candidate scoring."""
from concurrent.futures import ThreadPoolExecutor
import time
import torch


class MeshAudit:
    def __init__(self,collision,device):
        self.collision=collision;self.device=device
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='card-mesh-audit')
        self.pending=None;self.latest=None;self.next_at=0.;self.error=None

    def close(self):self.pool.shutdown(wait=False,cancel_futures=True)

    def run(self,positions,rotations,stamp,session):
        began=time.perf_counter()
        with torch.inference_mode():
            p=torch.as_tensor(positions,device=self.device,dtype=torch.float32)[None]
            r=torch.as_tensor(rotations,device=self.device,dtype=torch.float32)[None]
            depth,valid=self.collision.query_links(p,r,dense=True)
            report=self.collision.report(depth[0],valid[0])
        return dict(timestamp_ms=stamp,solver_session_id=session,report=report,
            processing_ms=(time.perf_counter()-began)*1000)

    def snapshot(self,session):
        if self.pending is not None and self.pending.done():
            try:self.latest=self.pending.result();self.error=None
            except Exception as exc:self.error=str(exc)
            self.pending=None
        if self.latest and self.latest['solver_session_id']==session:
            age=time.time()*1000-self.latest['timestamp_ms']
            if age<2000:return dict(self.latest,age_ms=age)
        return None

    def submit(self,positions,rotations,stamp,session):
        if self.pending is None and time.monotonic()>=self.next_at:
            self.next_at=time.monotonic()+.5
            self.pending=self.pool.submit(self.run,positions,rotations,stamp,session)

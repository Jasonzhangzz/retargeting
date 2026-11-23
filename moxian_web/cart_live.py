"""Measured policy inputs and bounded, timestamped background mesh diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import torch
from .cart_observations import canonical_emf,tactile_input
from .cart_contacts import overlay,transform_contacts


def policy_row(geometry,frame):
    """Same train/runtime packing without an unused nearest-surface search."""
    v,j=np.asarray(frame['vertices_m']),np.asarray(frame['joints_m'])
    root=j[geometry.adapter.indices[0]];_,cj=geometry.canonical(v,j)
    taxels,_=geometry.detector.taxels.evaluate(v)
    tp,ts,tv=tactile_input(frame,(taxels-root)@geometry.adapter.transform,geometry.topology)
    emf,valid=canonical_emf(frame.get('emf_observation'),geometry.adapter,root,frame['timestamp_ms'])
    return dict(timestamp_ms=frame.get('recorded_timestamp_ms',frame['timestamp_ms']),
        stream_epoch=frame.get('input_stream_epoch'),joints_m=cj.tolist(),points=cj[[4,8,12,16,20]].tolist(),
        emf_pose=emf.tolist(),emf_valid=valid.tolist(),tactile_positions_m=tp.tolist(),
        tactile_state=ts.tolist(),tactile_valid=tv.tolist())


class LiveAudit:
    def __init__(self,owner):
        self.owner=owner;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cart-audit')
        self.pending=None;self.next_at=0.;self.latest=None;self.error=None

    def close(self):self.pool.shutdown(wait=False,cancel_futures=True)

    def snapshot(self):
        if self.pending is not None and self.pending.done():
            try:
                value=self.pending.result()
                if value['session_id']==self.owner.session_id:self.latest=value;self.error=None
            except Exception as exc:self.error=str(exc)
            self.pending=None
        if self.latest and self.latest['session_id']==self.owner.session_id:
            age=time.time()*1000-self.latest['timestamp_ms']
            if age<2000:return dict(self.latest,age_ms=age)
        return None

    def submit(self,frame,row,q,old_q):
        if self.pending is None and time.monotonic()>=self.next_at:
            self.next_at=time.monotonic()+.5
            self.pending=self.pool.submit(self.run,frame,row,q.copy(),old_q.copy(),self.owner.session_id)

    def run(self,frame,row,q,old_q,session):
        started=time.perf_counter();owner=self.owner;geometry=owner.geometry
        with torch.inference_mode():
            observed=geometry.detector.detect(frame['vertices_m'],frame['joints_m'],frame)
            root=np.asarray(frame['joints_m'])[geometry.adapter.indices[0]]
            contacts=transform_contacts(observed,geometry.adapter.transform,root)
            sample=dict(row,contacts=contacts)
            collision=owner.cart_predictor.collision;collision.auditing=True
            try:
                _,_,quality=owner.objective.batch(owner.cart_predictor.hand.tensor(q)[None],[sample],
                    owner.cart_predictor.hand.tensor(old_q)[None],audit=True)
            finally:collision.auditing=False
            quality=quality[0]
            metrics=dict(required=quality['contact_count'],max_gap_mm=float(quality['max_gap_m'][0])*1000,
                max_normal_deg=float(quality['max_normal_deg'][0]),bone_direction_deg=float(quality['bone_direction_deg'][0]),
                max_penetration_mm=float(quality['max_penetration_m'][0])*1000)
            pairs=[]
            for pair,(pa,pb) in zip(contacts['self_pairs'],quality['points']):
                pairs.append(dict(regions=pair['regions'],source_depth_mm=pair['depth_m']*1000,
                    robot_gap_mm=float((pa-pb).norm(dim=-1)[0])*1000))
            physical=transform_contacts(contacts,owner.source.basis,root@owner.source.mirror.T,inverse=True)
            source_overlay=overlay(physical,timestamp_ms=frame['timestamp_ms'],geometry_version=frame['geometry_version'],processing_ms=(time.perf_counter()-started)*1000)
            robot_points=[]
            for pa,pb in quality['points']:
                robot_points.extend((point[0].cpu().numpy()@owner.source.basis.T+root@owner.source.mirror.T).tolist() for point in (pa,pb))
            return dict(session_id=session,timestamp_ms=frame['timestamp_ms'],metrics=metrics,
                self_contacts=len(contacts['self_pairs']),object_contacts=len(contacts['object_contacts']),
                tactile_status=contacts['tactile_status'],source_self_pairs=pairs,
                contact_resolution=observed.get('contact_resolution'),feasible=bool(quality['feasible'][0]),
                source_hull=source_overlay,robot_hull=dict(ready=True,contact_points_m=robot_points,patch_points_m=[],
                    timestamp_ms=frame['timestamp_ms'],geometry_version=frame['geometry_version']),
                processing_ms=(time.perf_counter()-started)*1000,kind='independent_physical_contact_audit')

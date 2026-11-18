"""Versioned action contract and speed-independent contact event extraction.

This module deliberately has no GPU/SDK dependency. Reference geometry is
frozen outside the detector; a candidate being trained can never relabel data.
"""
from __future__ import annotations
import math
import numpy as np
from .calibration_targets import DEFINITION,target
from .contact import FINGERS

PROTOCOL='contact_motion_v3'
MOTION_PROTOCOLS=('geort_motion_v2',PROTOCOL,'contact_motion_v4','contact_static_v5')
STAGES=[dict(id=key,seconds=5.,cycles=2,prompt=target(key)['label']+'；自然完成两次贴合、稍作停留后松开，不规定速度或保持时长。')
        for key in DEFINITION['protocols'][PROTOCOL]['stages']]
SCHEDULE_REVISION='contact_motion_v3_spread_v1'
SPREAD_STAGE=dict(id='motion_spread_together',seconds=5.,cycles=2,prompt='保持五指伸直，自然完成两次张开—并拢，稍作停留；不规定速度，勿握拳。')
EXTENDED_STAGES=STAGES+[SPREAD_STAGE]
STAGE_BY_ID={s['id']:s for s in EXTENDED_STAGES}


def schedule(extended=True):
    return [[p,s['id']] for p in ('train','validation') for s in (EXTENDED_STAGES if extended else STAGES)]


def active_fingers(row):
    if row.get('capture_protocol') in (PROTOCOL,'contact_motion_v4','contact_static_v5') and not row.get('initialization'):
        return tuple(target(row.get('stage'))['active_fingers'])
    return None


def observation_mask(row):
    active=active_fingers(row)
    return [float(f in active) for f in FINGERS] if active is not None else None


def region_finger(region):
    return None if region.startswith('palm_mcp_') else FINGERS.index(region.split('_')[0])


def cluster_tree(regions,distances):
    """Frozen reference MST, deterministic thumb preference only for ties."""
    parent=list(range(len(regions)));edges=[]
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    choices=sorted((float(distances[i,j]),0 if i==0 else 1,i,j)
                   for i in range(len(regions)) for j in range(i+1,len(regions)))
    for _,_,i,j in choices:
        a,b=root(i),root(j)
        if a!=b:
            parent[a]=b;edges.append(dict(regions=[regions[i],regions[j]],required=True))
    for r in regions[1:]:
        pair=[regions[0],r]
        if not any(e['regions']==pair for e in edges):edges.append(dict(regions=pair,required=False,weight=.25))
    return edges


def extract_events(rows,limit=2,allow_terminal=False):
    """Two closed excursions, not two clock bins; return actual sample indices.

    The 2mm floor / robust noise prominence measures displacement, never speed
    or duration. Flat minima use their middle sample. Missing/incomplete data
    breaks a trajectory so a stream gap cannot manufacture a release.
    """
    if len(rows)<3:return []
    values=np.array([r.get('reference_gap_m',float('nan')) for r in rows],float)
    stamps=np.array([r['timestamp_ms'] for r in rows],float)
    good=np.isfinite(values)
    events=[];start=0
    breaks=[i for i in range(1,len(rows)) if not good[i] or not good[i-1] or stamps[i]-stamps[i-1]>250]
    for end in breaks+[len(rows)]:
        ids=np.arange(start,end);start=end
        ids=ids[good[ids]]
        if len(ids)<3:continue
        y=values[ids]
        # Median removes isolated spikes without imposing a hold duration.
        sm=np.array([np.median(y[max(0,i-1):i+2]) for i in range(len(y))])
        noise=np.median(np.abs(np.diff(sm,2)))/.67449 if len(sm)>2 else 0.
        prominence=max(.002,4*float(noise))
        peak=sm[0];low=sm[0];minimum=0;descending=False
        for k in range(1,len(sm)):
            if not descending:
                peak=max(peak,sm[k])
                if peak-sm[k]>=prominence:
                    descending=True;low=sm[k];minimum=k
            else:
                if sm[k]<low-1e-9:low=sm[k];minimum=k
                if sm[k]-low>=max(prominence,.35*(peak-low)):
                    # Select a real central plateau sample, using raw minimum
                    # only inside this complete, denoised excursion.
                    left=minimum
                    while left>0 and abs(sm[left-1]-low)<1e-9:left-=1
                    right=minimum
                    while right+1<=k and abs(sm[right+1]-low)<1e-9:right+=1
                    center=(left+right)//2
                    candidates=ids[max(left,center-1):min(right+1,center+2)]
                    chosen=int(min(candidates,key=lambda i:(values[i],abs(i-ids[center]))))
                    events.append(dict(index=chosen,timestamp_ms=float(stamps[chosen]),gap_m=float(values[chosen]),prominence_m=float(peak-low),source='trajectory_assumed_contact'))
                    descending=False;peak=sm[k];low=sm[k]
        # The new non-contact exercise may end with fingers together. Require
        # a visible excursion and a stopped descent in real final samples;
        # contact actions still require their fully observed release.
        if allow_terminal and descending and minimum<len(sm)-1 and len(sm)>=3 and peak-low>=prominence and np.max(np.abs(np.diff(sm[-3:])))<.15*prominence:
            chosen=int(ids[minimum])
            events.append(dict(index=chosen,timestamp_ms=float(stamps[chosen]),gap_m=float(values[chosen]),prominence_m=float(peak-low),source='trajectory_assumed_adduction'))
    return events[:limit]

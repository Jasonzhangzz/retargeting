"""Lightweight recording schema shared by SDK ingestion and geometry workers.

This module deliberately has no torch, Warp, or model imports.
"""
import numpy as np

CAPTURE_SCHEMA = 'cart_capture_wrist_v1'


def binary_taxels(frame, topology):
    contact=frame.get('source_contact') or {}
    cells=contact.get('cell_contact')
    measured=isinstance(cells,(list,tuple)) and len(cells)==460 and all(type(v) in (bool,int) and v in (0,1) for v in cells)
    # Preserve observed bits even when stale; the validity mask prevents them
    # from masquerading as synchronized labels. Raw evidence is also retained.
    state=np.asarray(cells,np.uint8) if measured else np.zeros(460,np.uint8)
    stamps=[frame.get('timestamp_ms'),frame.get('tactile_timestamp_ms'),contact.get('timestamp_ms')]
    synchronized=all(isinstance(t,(int,float)) and np.isfinite(t) for t in stamps)
    synchronized=synchronized and max(stamps)-min(stamps)<=25
    bound=(np.asarray(topology['sid_face_ids'])>=0)&(np.asarray(topology['sid_patch_indices'])>=0)
    valid=bound & bool(measured and synchronized and contact.get('ready') is True)
    return state,valid,bound

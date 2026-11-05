#!/usr/bin/env python3
"""Export fixed SID bindings for the frozen, mirrored primitive hand."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moxian_web.mano_runtime import ManoSurface
from scripts.audit_moxian_calibration import restore
from moxian_web.contact import default_mapping

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    saved=a.output/'saved_left';p=json.loads((saved/'current.json').read_text());m=ManoSurface(saved/'MANO_LEFT.pkl','cpu');restore(m,p)
    mapping=p.get('mapping',{'version':'default-left','assignments':default_mapping('left'),'confirmed':[]});m.binding_overrides=p.get('binding_overrides',{});m.build_binding(mapping)
    topo=m.topology();topo['hand_side']='right';topo['faces']=m.faces[:,[0,2,1]].tolist();topo['sid_barycentric']=m.sid_bary[:,[0,2,1]].tolist();topo['sensor_contact_source']='none: offline page only evaluates geometric contact'
    # Snapshot rest geometry is authoritative; bindings use standard face IDs.
    geo=np.load(saved/'geometry.npz');points=(geo['rest_vertices'][m.faces[m.sid_faces]]*m.sid_bary[:,:,None]).sum(1)
    if len(np.unique(np.round(points,8),axis=0))!=460:raise ValueError('SID positions overlap')
    (a.output/'web/tactile.json').write_text(json.dumps(topo,separators=(',',':')));print(json.dumps({'taxels':460,'binding_version':m.binding_version,'per_frame_reprojection':False}))
if __name__=='__main__':main()

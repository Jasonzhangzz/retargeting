#!/usr/bin/env python3
"""Save an immutable, reloadable local left hand and its exact appearance."""
import sys,json,shutil,hashlib,time,argparse
from pathlib import Path
import numpy as np
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from scripts.audit_moxian_calibration import restore
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--serial',default='2DF5F3B00DA5');args=parser.parse_args();out=args.output;archive=out/'saved_left';archive.mkdir(parents=True,exist_ok=True)
if (archive/'manifest.json').exists():raise RuntimeError('snapshot already saved; keep immutable')
home=ROOT/'output/moxian/hand_model'/hashlib.sha256(args.serial.encode()).hexdigest()[:16];profile=json.loads((home/'current.json').read_text());m=ManoSurface(DEFAULT_MANO);restore(m,profile)
for name in ('current.json','sensor-reference.json','fit-settings.json'):
 if (home/name).exists():shutil.copy2(home/name,archive/name)
shutil.copy2(DEFAULT_MANO,archive/'MANO_LEFT.pkl')
np.savez_compressed(archive/'geometry.npz',rest_vertices=m.rest_np,rest_joints=m.rest_j_np,weights=m.model.weights.cpu().numpy(),parents=np.array(m.parents),faces=m.faces,tip_anchors=m.tip_anchors,dimension_parameters=m.dimension_parameters)
colors={'skin':'#d4ba96','roughness':.75,'metalness':0,'skeleton':'#f7e9ca','contact':'#39ff76','background':'#29342d','finger_palette':['#ed875b','#d0bc74','#a8c794','#7aafa4','#81a0c6']};(archive/'appearance.json').write_text(json.dumps(colors,indent=2))
for name in ('mano-view.js','app.js','style.css'):shutil.copy2(ROOT/'moxian_web/static'/name,archive/name)
try:
 state=json.load(urlopen('http://127.0.0.1:8890/api/state'));top=json.load(urlopen('http://127.0.0.1:8890/api/hand_model/model'));(archive/'live-state.json').write_text(json.dumps(state));(archive/'topology.json').write_text(json.dumps(top))
except Exception as e:print('live snapshot unavailable',e)
d={'saved_at_ms':time.time()*1000,'hand_side':'left','model_revision':profile.get('revision'),'geometry_version':m.shape_version,'purpose':'immutable source; offline mirror creates right-hand comparison, never overwrites live glove calibration','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.iterdir() if p.is_file()}}
(archive/'manifest.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))

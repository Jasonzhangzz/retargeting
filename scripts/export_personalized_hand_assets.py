#!/usr/bin/env python3
"""Export the immutable left hand and its right reflection as standard OBJ/MTL."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def write_obj(path,vertices,faces):
 with path.open('w') as f:
  f.write('mtllib hand.mtl\nusemtl saved_skin\n# units: metres\n')
  for p in vertices:f.write('v '+' '.join(f'{x:.9g}' for x in p)+'\n')
  for face in faces:f.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output;source=root/'saved_left';dest=root/'exports';dest.mkdir(exist_ok=True);geo=np.load(source/'geometry.npz');appearance=json.loads((source/'appearance.json').read_text());color=appearance['skin'].lstrip('#');rgb=[int(color[i:i+2],16)/255 for i in (0,2,4)]
 (dest/'hand.mtl').write_text('newmtl saved_skin\nKd '+' '.join(map(str,rgb))+'\nPr '+str(appearance['roughness'])+'\nPm 0\n')
 write_obj(dest/'left_rest.obj',geo['rest_vertices'],geo['faces']);write_obj(dest/'right_mirrored_rest.obj',geo['rest_vertices']*[-1,1,1],geo['faces'][:,[0,2,1]])
 state=json.loads((source/'live-state.json').read_text());hand=state.get('hand_model',{})
 if hand.get('vertices_m'):write_obj(dest/'left_saved_pose.obj',hand['vertices_m'],geo['faces'])
 d={'units':'m','left_model_sha256':json.loads((source/'manifest.json').read_text())['sha256']['geometry.npz'],'color':appearance['skin'],'right_is_exact_reflection':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file() and p.name!='exports.json'}}
 (dest/'exports.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
if __name__=='__main__':main()

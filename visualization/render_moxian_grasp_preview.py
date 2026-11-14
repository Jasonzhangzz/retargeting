import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

D=json.loads(Path(__file__).with_name('moxian_grasp_samples.json').read_text())
T='24_Tip_Pinch'; row=D['types'][T][1]; names=D['node_order']; P=np.array(row['moxian_skeleton_positions'],float)
chains=[['wrist','thumb_metacarpal','thumb_proximal','thumb_distal','thumb_tip'],['wrist','index_metacarpal','index_proximal','index_intermediate','index_distal','index_tip'],['wrist','middle_metacarpal','middle_proximal','middle_intermediate','middle_distal','middle_tip'],['wrist','ring_metacarpal','ring_proximal','ring_intermediate','ring_distal','ring_tip'],['wrist','little_metacarpal','little_proximal','little_intermediate','little_distal','little_tip']]
idx={n:i for i,n in enumerate(names)}
fig=plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection='3d'); ax.set_facecolor('#0b1020'); fig.patch.set_facecolor('#0b1020')
colors=['#6fe7ff','#ff7dc7','#b58cff','#ffd36e','#85f0a6']
for chain,c in zip(chains,colors):
 q=P[[idx[n] for n in chain]]; ax.plot(q[:,0],q[:,1],q[:,2],'-o',color=c,lw=2.4,ms=4)
for n,p in zip(names,P):
 if n in ('wrist','thumb_tip','index_tip','middle_tip','ring_tip','little_tip'):
  ax.text(*p,n.replace('_',' '),color='white',fontsize=8)
# object proxy box
op=np.array(row['object_pose_world'],float); s=np.array(row['object_scale'],float); c=op[:3]; x,y,z=s/2
verts=np.array([[dx,dy,dz] for dx in (-x,x) for dy in (-y,y) for dz in (-z,z)])+c
faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
ax.add_collection3d(Poly3DCollection([verts[f] for f in faces],facecolors='#ffb65c',edgecolors='#ffd79b',alpha=.18,linewidths=1.2))
ax.scatter(*c,color='#ffb65c',s=45,label='object proxy')
ax.set_xlabel('x (m)',color='white');ax.set_ylabel('y (m)',color='white');ax.set_zlabel('z (m)',color='white');ax.tick_params(colors='#b8c5e2'); ax.set_title(f'Moxian grasp · {T} · sample {row["index"]}\ncase {row["case"]}',color='white')
lims=np.array([P.min(0),P.max(0)]); lo=lims[0]-0.03; hi=lims[1]+0.03; ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]); ax.view_init(elev=24,azim=-62)
plt.tight_layout(); out=Path(__file__).with_name('moxian_grasp_preview.png'); fig.savefig(out,dpi=180,facecolor=fig.get_facecolor()); print(out)

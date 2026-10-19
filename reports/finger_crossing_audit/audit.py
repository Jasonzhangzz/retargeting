from pathlib import Path
import sys,json
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from forceaware.mpc_loss import _fit_collision_capsule
R=Path(__file__).resolve().parents[2];m=mujoco.MjModel.from_xml_path(str(R/'data/example/sequence/scene/wuji_right/sugar_box_mesh.xml'));d=mujoco.MjData(m)
bodies=[];starts=[];ends=[];radii=[];fingers=[];names=[]
for finger in range(1,6):
 for link in range(1,5):
  name=f'right_finger{finger}_link{link}';cloud=np.load(R/'assets/wuji/qp_right'/(name+'.npz'))['local_pos'];a,b,r=_fit_collision_capsule(np.c_[cloud,np.zeros(len(cloud))]);bodies.append(m.body(name).id);starts.append(a);ends.append(b);radii.append(r);fingers.append(finger);names.append(name)
starts=np.array(starts);ends=np.array(ends);radii=np.array(radii);bodies=np.array(bodies)
pairs=np.array([(i,j) for i in range(20) for j in range(i+1,20) if fingers[i]!=fingers[j]])
def pseg(p,a,b):
 seg=b-a;frac=np.clip(np.einsum('ij,ij->i',p-a,seg)/np.maximum(np.sum(seg*seg,axis=1),1e-15),0,1);return np.sum((p-a-frac[:,None]*seg)**2,axis=1)
def gaps(q):
 d.qpos[:len(q)]=q;mujoco.mj_kinematics(m,d)
 mat=d.xmat[bodies].reshape(-1,3,3);p=d.xpos[bodies];a=np.einsum('bij,bj->bi',mat,starts)+p;b=np.einsum('bij,bj->bi',mat,ends)+p
 i,j=pairs.T;al,bl,ar,br=a[i],b[i],a[j],b[j];u,v,w=bl-al,br-ar,al-ar
 dot=lambda x,y:np.einsum('ij,ij->i',x,y)
 uu,uv,vv,uw,vw=dot(u,u),dot(u,v),dot(v,v),dot(u,w),dot(v,w);denom=uu*vv-uv*uv
 s=(uv*vw-vv*uw)/np.maximum(denom,1e-20);t=(uu*vw-uv*uw)/np.maximum(denom,1e-20)
 ds=np.minimum.reduce([pseg(al,ar,br),pseg(bl,ar,br),pseg(ar,al,bl),pseg(br,al,bl)])
 valid=(denom>1e-12*uu*vv)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
 ds=np.where(valid,np.minimum(ds,np.sum((w+s[:,None]*u-t[:,None]*v)**2,axis=1)),ds)
 return (np.sqrt(ds+1e-12)-radii[i]-radii[j])*1000
sources={}
for name in ['projected_zero','all_350','all_600']:
 sources[name+'__reference']=R/'reports/mano_direct_reference/inputs'/name/'contactaware/wuji_right_qpos.npy'
 for method in ['sampled-lm','no-admm']:sources[name+'__'+method]=R/'reports/mano_direct_reference/runs/comfree'/(name+'__'+method)/'rollout.npz'
out={}
for name,path in sources.items():
 q=np.load(path);q=q['qpos_traj'] if isinstance(q,np.lib.npyio.NpzFile) else q
 values=np.array([gaps(pose) for pose in q]);frame,pair=np.unravel_index(np.argmin(values),values.shape);i,j=pairs[pair];d.qpos[:len(q[frame])]=q[frame];mujoco.mj_forward(m,d)
 left=[g for g in range(m.ngeom) if m.geom_bodyid[g]==bodies[i] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]>0]
 right=[g for g in range(m.ngeom) if m.geom_bodyid[g]==bodies[j] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]>0]
 distances=[(float(mujoco.mj_geomDistance(m,d,a,b,1.,None)),a,b) for a in left for b in right]
 mesh=min(distances) if distances else None
 out[name]=dict(frames=len(q),worst_frame=int(frame),pair=[names[i],names[j]],min_capsule_gap_mm=float(values.min()),frames_capsule_overlap_over_1mm=int(np.count_nonzero(np.min(values,axis=1)<-1)),frames_margin_below_2mm=int(np.count_nonzero(np.min(values,axis=1)<2)),worst_pair_convex_mesh_gap_mm=1000*mesh[0] if mesh else None)
 print(name,out[name],flush=True)
(R/'reports/finger_crossing_audit/results.json').write_text(json.dumps(out,indent=2)+'\n')

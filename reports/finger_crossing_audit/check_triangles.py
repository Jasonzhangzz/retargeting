from pathlib import Path
import json,numpy as np,mujoco
R=Path(__file__).resolve().parents[2];row=json.loads((R/'reports/finger_crossing_audit/results.json').read_text())['all_350__sampled-lm'];m=mujoco.MjModel.from_xml_path(str(R/'data/example/sequence/scene/wuji_right/sugar_box_mesh.xml'));d=mujoco.MjData(m);a=np.load(R/'reports/mano_direct_reference/runs/comfree/all_350__sampled-lm/rollout.npz');d.qpos[:]=a['qpos_traj'][row['worst_frame']];mujoco.mj_kinematics(m,d);triangles=[]
for name in row['pair']:
 body=m.body(name).id;g=next(g for g in range(m.ngeom) if m.geom_bodyid[g]==body and m.geom_contype[g]>0 and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH);mesh=m.geom_dataid[g];va,vn=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];fa,fn=m.mesh_faceadr[mesh],m.mesh_facenum[mesh];vertices=m.mesh_vert[va:va+vn]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];triangles.append(vertices[m.mesh_face[fa:fa+fn]])
start=np.concatenate([triangles[0][:,i] for i in range(3)]);end=np.concatenate([triangles[0][:,(i+1)%3] for i in range(3)]);target=triangles[1];edge1=target[:,1]-target[:,0];edge2=target[:,2]-target[:,0]
found=None
for k in range(0,len(start),32):
 p=start[k:k+32];direction=end[k:k+32]-p;h=np.cross(direction[:,None],edge2[None]);denom=np.einsum('tj,etj->et',edge1,h);valid=np.abs(denom)>1e-12;inv=np.where(valid,1/np.where(valid,denom,1),0)
 s=p[:,None]-target[None,:,0];u=np.einsum('etj,etj->et',s,h)*inv;q=np.cross(s,edge1[None]);v=np.einsum('ej,etj->et',direction,q)*inv;t=np.einsum('tj,etj->et',edge2,q)*inv
 valid&=(u>1e-6)&(v>1e-6)&(u+v<1-1e-6)&(t>1e-6)&(t<1-1e-6)
 if valid.any():
  e,face=np.argwhere(valid)[0];point=p[e]+t[e,face]*direction[e];found=dict(source_edge=int(k+e),target_triangle=int(face),edge_fraction=float(t[e,face]),triangle_barycentric=[float(1-u[e,face]-v[e,face]),float(u[e,face]),float(v[e,face])],point_world=point.tolist());break
path=R/'reports/finger_crossing_audit/triangle_mesh_witness.json';out=json.loads(path.read_text());out['triangle_surface_intersection_confirmed']=found is not None;out['intersection_witness']=found;out['winding_depth_is_not_a_certified_penetration_depth']='Meshes are not watertight; only the edge-triangle intersection is conclusive.';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

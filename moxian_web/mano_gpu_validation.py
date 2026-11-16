"""All-frame GPU triangle validation; only side-section diagnostics use CPU."""
import numpy as np
import torch
from .mano_gpu_collision import BatchedSurfaceCollision
from .contact import FINGERS,PATCH_IDS
from .motion_capture import confirmed_touch,bilateral
from retarget.baseline.card.geometry import contact_weights
from .calibration_targets import pinch_patches
from scripts.calibrate_wuji_mano import CHAINS


@torch.no_grad()
def geometry_report(model,rows,poses,roots,check_collision=True):
    v,j,_,_=model.decode(poses,roots)
    collision_frames=[];depth=0.;query_valid=True;cache={}
    for start in range(0,len(rows) if check_collision else 0,32):
        vertices=v[start:start+32];size=len(vertices)
        if size not in cache:cache[size]=BatchedSurfaceCollision(model,size)
        w=cache[size].query(vertices);valid=bool(w['valid']);query_valid &= valid
        values=w['maximum'].cpu().tolist();depth=max(depth,max(values,default=0.))
        collision_frames.extend(dict(stage=row['stage'],timestamp_ms=row['timestamp_ms'],max_penetration_m=d,query_valid=valid,method='GPU_closed_segment_oriented_ray_original_triangles') for row,d in zip(rows[start:start+32],values))
    topology=model.topology();faces=np.asarray(topology['faces']);sid_faces=np.asarray(topology['sid_face_ids']);bc=np.asarray(topology['sid_barycentric']);patches=np.asarray(topology['sid_patch_indices'])
    ids=[];centers=[];stages=[]
    for i,row in enumerate(rows):
        partner=row['stage'][6:] if row['stage'].startswith('pinch_') else None
        if not partner or not confirmed_touch(row) or not bilateral(row,partner):continue
        data=contact_weights(row.get('contact') or {},patches,row['timestamp_ms'],row.get('tactile_timestamp_ms',-1e20))
        desired=list(pinch_patches(row['stage']))
        if not desired:continue
        if not all(data['active'][p] for p in desired):continue
        ids.append(i);centers.append(data['center'][desired]);stages.append(row['stage'])
    records=[]
    if ids:
        t=lambda a,dtype=torch.float32:torch.as_tensor(a,device=model.device,dtype=dtype)
        tri=v[t(ids,torch.long)[:,None,None],t(faces[sid_faces],torch.long)[None]]
        points=(tri*t(bc)[None,:,:,None]).sum(-2)
        normal=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),dim=-1)
        p=torch.einsum('nks,nsd->nkd',t(np.asarray(centers)),points)
        n=torch.nn.functional.normalize(torch.einsum('nks,nsd->nkd',t(np.asarray(centers)),normal),dim=-1)
        gaps=(p[:,0]-p[:,1]).norm(dim=-1).cpu().tolist();dots=(n[:,0]*n[:,1]).sum(-1).cpu().tolist()
        records=[dict(stage=s,gap_m=g,normal_dot=d,evidence=['fresh_bilateral_taxels']*2) for s,g,d in zip(stages,gaps,dots)]
        from .mano_contact_loss import vertex_normals,normal_error_deg
        smooth=vertex_normals(v[t(ids,torch.long)],t(faces,torch.long))
        taxel_normal=torch.nn.functional.normalize((smooth[:,t(faces[sid_faces],torch.long)]*t(bc)[None,:,:,None]).sum(-2),dim=-1)
        smooth_n=torch.nn.functional.normalize(torch.einsum('nks,nsd->nkd',t(np.asarray(centers)),taxel_normal),dim=-1)
        errors=normal_error_deg(n[:,0],n[:,1]).cpu().tolist()
        smooth_errors=normal_error_deg(smooth_n[:,0],smooth_n[:,1]).cpu().tolist()
        for record,error,smooth_error in zip(records,errors,smooth_errors):record.update(normal_error_deg=error,smooth_normal_error_deg=smooth_error)
    # These are original mesh cross-sections, not a proxy contact acceptance.
    side=[];missing=[];distances=[];together={};gaps=[]
    from .side_contact import side_material_pairs
    from .proximal_shape import proximal_pair_gaps
    indices=[i for i,row in enumerate(rows) if row['stage'] in ('together','together_repeat')]
    vv=v[indices].cpu().numpy();jj=j[indices].cpu().numpy();weights=model.model.weights.cpu().numpy()
    for k,i in enumerate(indices):
        row=rows[i];thumb=row.get('thumb_index_side_contact') is True
        try:
            side.extend(dict(timestamp_ms=row['timestamp_ms'],**p) for p in side_material_pairs(model,vv[k],jj[k],include_thumb=thumb,skin=weights))
            sections=proximal_pair_gaps(vv[k],jj[k],model.faces,weights,CHAINS,True,include_thumb=thumb)
            distances.append([p['surface_distance_m']*1000 for p in sections])
            for section in sections:
                a,b=map(int,section['pair'].split('_'));gap=abs(section['gap_m']);gaps.append(gap);together.setdefault(FINGERS[a]+'_'+FINGERS[b],[]).append(gap)
        except ValueError as exc:missing.append(dict(timestamp_ms=row['timestamp_ms'],reason=str(exc)))
    contact=[p['gap_m'] for p in records]
    result=dict(thumb_index_side_contact=any(r.get('thumb_index_side_contact') is True for r in rows),together_material_records=side,max_penetration_m=depth,noncontact_penetration_m=depth,collision_frames=collision_frames,collision_query_valid=query_valid,
                contact_gap_m=float(np.mean(contact)) if contact else 0.,contact_max_gap_m=max(contact,default=0.),contact_material_records=records,contact_observations=len(records),contact_stages=sorted(set(stages)),
                contact_normal_error_deg=float(np.mean([p['normal_error_deg'] for p in records])) if records else None,
                contact_normal_max_error_deg=max((p['normal_error_deg'] for p in records),default=None),
                contact_smooth_normal_error_deg=float(np.mean([p['smooth_normal_error_deg'] for p in records])) if records else None,
                together_gap_m=float(np.mean(gaps)) if gaps else 0.,together_observations=len(gaps),together_pair_gaps_mm={k:float(np.mean(g))*1000 for k,g in together.items()},together_surface_distances_mm=distances,together_missing_sections=missing,fusion=[],relative_response_cost=None)
    if rows and rows[0].get('capture_protocol')=='contact_motion_v3':
        from .contact_templates import region_geometry_report
        result.update(region_geometry_report(model,rows,v))
    return result


def skeleton_report(rows,reports):
    errors={f:[] for f in FINGERS};baseline={f:[] for f in FINGERS}
    from .mano_objective import participation
    masks=participation(rows,training=True)
    for row,report,mask in zip(rows,reports,masks):
        for index,f in enumerate(FINGERS):
            if not mask[index]:continue
            errors[f].append(report['source_tracking_errors'][f]['position_error_mm'])
            baseline[f].append(row.get('skeleton',{}).get('tracking_errors',{}).get(f,{}).get('position_error_mm',float('inf')))
    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values and np.isfinite(values).all() else None
    summary={f:dict(rmse_mm=rmse(e),max_mm=float(max(e)) if e and np.isfinite(e).all() else None,
                    baseline_rmse_mm=rmse(baseline[f])) for f,e in errors.items()}
    # Missing baseline evidence is a failure, not an infinite allowed tolerance.
    valid=all(v['rmse_mm'] is not None and v['baseline_rmse_mm'] is not None and v['rmse_mm']<=15 and v['rmse_mm']<=v['baseline_rmse_mm']*1.05+.5 for v in summary.values())
    return dict(accepted=bool(valid),per_finger=summary,frames=len(rows),source='frozen_geometry_gpu_sensor_predictions')

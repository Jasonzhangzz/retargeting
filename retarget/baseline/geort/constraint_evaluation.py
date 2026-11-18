"""Same-session diagnostic comparison; no paired robot ground truth is assumed."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .adapter import ROOT,FINGERS
from .training import Predictor,atomic_json
from .constraints import synchronized_targets,PAIRS,ContactGeometry
from .diagnostic_queries import SurfacePairs
from moxian_web.warp_runtime import serialized


def pinch_metrics(source_points,robot_points,contacts):
    """True-FK tip separation, separate from physical surface contact success."""
    pairs=[]
    for i,(a,b) in enumerate(PAIRS):
        source_distance=np.linalg.norm(source_points[:,a]-source_points[:,b],axis=-1)
        distance=np.linalg.norm(robot_points[:,a]-robot_points[:,b],axis=-1)*1000
        pinch=source_distance<.015;physical=contacts[:,i]
        pairs.append(dict(fingers=[FINGERS[a],FINGERS[b]],official_pinch_frames=int(pinch.sum()),
            official_pinch_tip_distance_mean_mm=float(distance[pinch].mean()) if pinch.any() else None,
            physical_contact_frames=int(physical.sum()),
            physical_contact_tip_distance_mean_mm=float(distance[physical].mean()) if physical.any() else None))
    return dict(pairs=pairs,tip_sites_are_not_surface_witnesses=True)


class Audit:
    def __init__(self,robot):
        self.device='cuda:0';self.geometry=ContactGeometry(robot,self.device);self.robot=robot
        self.fk=self.geometry.fk;self.basis=self.geometry.basis
        s=self.geometry.surface;self.local=torch.tensor(s.local,device=self.device,dtype=torch.float32)
        self.owners=torch.tensor(s.owners,device=self.device);self.faces=torch.tensor(s.faces,device=self.device)
        owner=s.owners[s.faces[:,0]]
        regions={f+'_distal':np.flatnonzero(owner==4+4*i).tolist() for i,f in enumerate(FINGERS)}
        self.pairs=SurfacePairs(self.local,self.faces,regions,[dict(regions=[FINGERS[a]+'_distal',FINGERS[b]+'_distal']) for a,b in PAIRS])

    def frames(self,q):return self.fk.frames(torch.cat((q.new_zeros((len(q),6)),q),1))

    @torch.no_grad()
    def all_metrics(self,q,joints,bends,flat,lower):
        angle=[];points=[]
        q=torch.tensor(q,device=self.device)
        source=torch.tensor(np.diff(joints[:,1:].reshape(-1,5,4,3),axis=2),device=self.device)
        source=torch.nn.functional.normalize(source[:,:,-1],dim=-1)
        for start in range(0,len(q),512):
            p,r=self.frames(q[start:start+512]);dirs=[];tips=[]
            for f in range(5):
                body=self.robot.body_ids[4+4*f];local=torch.tensor(self.robot.model.site_pos[self.robot.tip_ids[f]],device=self.device,dtype=torch.float32)
                vector=(r[:,body]@local[:,None]).squeeze(-1)
                dirs.append(torch.nn.functional.normalize(vector@self.basis,dim=-1))
                tips.append((vector+p[:,body]-p[:,self.robot.body_ids[0]])@self.basis)
            dirs=torch.stack(dirs,1)
            cos=(dirs*source[start:start+len(dirs)]).sum(-1).clamp(-1,1)
            angle.append(torch.rad2deg(torch.acos(cos)).cpu().numpy());points.append(torch.stack(tips,1).cpu().numpy())
        degrees=np.rad2deg(q.cpu().numpy()).reshape(-1,5,4);error=np.concatenate(angle)
        bend_error=np.abs(degrees[:,:,[2,3]]-np.rad2deg(bends))
        bound=np.rad2deg(np.array(lower)).reshape(5,4)[:,[2,3]]
        return dict(frames=len(q),flat_frames=len(flat),
            distal_direction_error_mean_deg=error.mean(0).tolist(),flat_distal_direction_error_mean_deg=error[flat].mean(0).tolist(),
            pip_dip_signed_angle_error_mean_deg=bend_error.mean(0).tolist(),
            flat_pip_dip_signed_angle_error_mean_deg=bend_error[flat].mean(0).tolist(),
            flat_finger_joint_means_deg=degrees[flat].mean(0).tolist(),
            excessive_extension_fraction=float((degrees[:,:,[2,3]]<bound-1.).mean()),
            flat_excessive_extension_fraction=float((degrees[flat][:,:,[2,3]]<bound-1.).mean()),
            flat_pip_dip_below_minus_10deg_fraction=float((degrees[flat,1:][:,:,[2,3]]<-10.).mean())),np.concatenate(points)

    @torch.no_grad()
    @serialized
    def surface_metrics(self,q,contacts,indices,pinch_contacts=None):
        q=torch.tensor(q[indices],device=self.device);max_depth=[];valid=[];pair_depth=[];gaps=[];normals=[]
        for start in range(0,len(q),16):
            p,r=self.frames(q[start:start+16]);raw,ok=self.geometry.collision.query(p,r,dense=True)
            max_depth.extend(raw.amax((1,2)).cpu().tolist());valid.extend(ok.cpu().tolist())
            pair_depth.extend(torch.stack([torch.maximum(raw[:,1+4*a:5+4*a,1+4*b:5+4*b].amax((1,2)),raw[:,1+4*b:5+4*b,1+4*a:5+4*a].amax((1,2))) for a,b in PAIRS],1).cpu().tolist())
            for k in range(len(p)):
                pos=p[k,self.robot.body_ids];rot=r[k,self.robot.body_ids]
                vertices=(rot[self.owners]@self.local[...,None]).squeeze(-1)+pos[self.owners]
                gap,normal=self.pairs.evaluate(vertices);gaps.append(gap);normals.append(normal)
        gaps=np.array(gaps);normals=np.array(normals);depth=np.array(pair_depth);required=contacts[indices]
        success=(gaps<=2.)&(normals<=30.)&(depth<=.5)&np.array(valid)[:,None]
        per_pair=[]
        for i,(a,b) in enumerate(PAIRS):
            mask=required[:,i];count=int(mask.sum())
            pair=dict(fingers=[FINGERS[a],FINGERS[b]],required_frames=count,
                contact_success_percent=float(success[mask,i].mean()*100) if count else None,
                gap_mean_mm=float(gaps[mask,i].mean()) if count else None,
                gap_p95_mm=float(np.percentile(gaps[mask,i],95)) if count else None,
                normal_error_mean_deg=float(normals[mask,i].mean()) if count else None,
                penetration_max_mm=float(depth[mask,i].max()) if count else None)
            if pinch_contacts is not None:
                pinch=pinch_contacts[indices,i]
                pair.update(official_pinch_frames=int(pinch.sum()),
                    official_pinch_gap_mean_mm=float(gaps[pinch,i].mean()) if pinch.any() else None,
                    official_pinch_success_percent=float(success[pinch,i].mean()*100) if pinch.any() else None)
            per_pair.append(pair)
        return dict(sampled_frames=len(q),indices=indices.tolist(),query_valid=bool(all(valid)),
            max_penetration_mm=float(np.max(max_depth)),mean_max_penetration_mm=float(np.mean(max_depth)),
            p95_max_penetration_mm=float(np.percentile(max_depth,95)),
            frames_penetrating_over_05mm_percent=float((np.array(max_depth)>.5).mean()*100),
            required_contacts=int(required.sum()),required_contact_success_percent=float(success[required].mean()*100) if required.any() else None,
            pairs=per_pair,sampled_not_continuous_intersection_proof=True,
            thresholds=dict(gap_mm=2.,normal_deg=30.,penetration_mm=.5))


def compare(root,baseline,output):
    torch.set_num_threads(1);root=Path(root);baseline=Path(baseline)
    old=json.loads((baseline/'baseline.json').read_text());ids=[d['id'] for d in old['datasets']]
    points,joints,contacts,bends,identity,records,targets=synchronized_targets(root,ids,lambda *a,**k:None,lambda:None)
    model=Predictor(root/'models/current','cuda:0')
    if model.config['source']!=identity or old['source']!=identity:raise ValueError('比较数据源不匹配')
    with torch.no_grad():q=model.robot.unnormalize(model.model(torch.tensor(points,device='cuda:0')).cpu().numpy())
    with np.load(baseline/'baseline_predictions.npz') as data:previous=data['q']
    if previous.shape!=q.shape:raise ValueError('比较帧数不一致')
    score=np.abs(bends[:,1:]).mean((1,2));flat=np.argsort(score)[:min(100,len(score))]
    fixed=np.unique(np.r_[flat,np.linspace(0,len(q)-1,min(192,len(q))).astype(int)])
    pinch=np.stack([np.linalg.norm(points[:,a]-points[:,b],axis=-1)<.015 for a,b in PAIRS],1)
    for i in range(10):
        for mask in (contacts,pinch):
            candidates=np.flatnonzero(mask[:,i])
            if len(candidates):fixed=np.unique(np.r_[fixed,candidates[np.linspace(0,len(candidates)-1,min(12,len(candidates))).astype(int)]])
    audit=Audit(model.robot);limit=model.config['output_domain']['lower']
    old_metrics,old_points=audit.all_metrics(previous,joints,bends,flat,limit);new_metrics,new_points=audit.all_metrics(q,joints,bends,flat,limit)
    old_metrics['point_pinch']=pinch_metrics(points,old_points,contacts)
    new_metrics['point_pinch']=pinch_metrics(points,new_points,contacts)
    old_metrics['surface']=audit.surface_metrics(previous,contacts,fixed,pinch);new_metrics['surface']=audit.surface_metrics(q,contacts,fixed,pinch)
    durations=[];cpu=Predictor(root/'models/current','cpu')
    for p in points[fixed[:100]]:
        t=time.perf_counter();cpu.predict(p);durations.append((time.perf_counter()-t)*1000)
    result=dict(scope='same_training_session_diagnostic_not_independent_validation',paired_robot_ground_truth=False,
        datasets=records,surface_targets=targets,flat_indices=flat.tolist(),source_flat_mean_absolute_bend_deg=float(np.rad2deg(np.abs(bends[flat,1:])).mean()),
        baseline_run_id=old['config']['run_id'],model_run_id=model.config['run_id'],output_domain=model.config['output_domain'],
        baseline=old_metrics,constrained=new_metrics,inference_cpu_ms=dict(mean=float(np.mean(durations)),p95=float(np.percentile(durations,95))))
    atomic_json(output,result);np.savez(Path(output).with_suffix('.npz'),q=q,baseline_q=previous,flat_indices=flat,surface_indices=fixed)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'output/moxian/geort')
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from moxian_web.saved_assets import asset_lock
    with asset_lock(args.root):result=compare(args.root,args.baseline,args.output)
    print(json.dumps({k:result[k] for k in ('model_run_id','source_flat_mean_absolute_bend_deg','baseline','constrained','inference_cpu_ms')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

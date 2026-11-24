"""Native-label audit, independent of learned geometry and policy checkpoints.

Run: python -m moxian_web.cart_geometry_audit --output /path/to/audit
The NPZ retains the actual queries, labels and finite differences for inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def audit_identity():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
        ('cart_geometry_audit.py','cart_collision.py','cart_contact_field.py','cart_q_collision.py','cart_kinematics.py','wuji_surface.py','cart_structured_geometry.py')}


def directional_check(analytic,coarse,fine,*,atol,rtol):
    """Define stability from two numerical probes, independently of the label."""
    scale=torch.maximum(coarse.abs(),fine.abs())
    tolerance=atol+rtol*scale
    if analytic.ndim>1:
        stable=(coarse-fine).norm(dim=-1)<atol+rtol*fine.norm(dim=-1)
        error=(analytic-fine).norm(dim=-1)
        correct=error<atol+rtol*fine.norm(dim=-1)
    else:
        stable=(coarse-fine).abs()<tolerance
        error=(analytic-fine).abs();correct=error<tolerance
    finite=torch.isfinite(analytic)&torch.isfinite(coarse)&torch.isfinite(fine)
    if finite.ndim>1:finite=finite.all(-1)
    stable &= finite
    return dict(samples=len(analytic),stable_fraction=float(stable.float().mean()),
        correct_on_stable_fraction=float(correct[stable].float().mean()) if stable.any() else 0.,
        error_median=float(error[stable].median()) if stable.any() else None,
        error_p95=float(torch.quantile(error[stable],.95)) if stable.any() else None),stable


@torch.no_grad()
def audit_geometry_labels(collision,*,samples=4096,q_samples=512,seed=318241,output=None):
    from .cart_contact_field import ContactTeacher,PAD_OWNERS
    from .cart_q_collision import QCollisionTeacher
    from .cart_pretrained_sdf import geometry_spec
    hand=collision.hand;oracle=collision.oracle;teacher=ContactTeacher(hand)
    devices=[torch.device(hand.device)] if hand.device.startswith('cuda') else []
    arrays={};report=dict(schema='cart_native_label_audit_v1',seed=seed,geometry=geometry_spec(hand),
        contact_surface_policy=hand.surface.contact_policy,contact_links={})
    report['implementation_sha256']=audit_identity()
    def keep(**values):
        arrays.update({k:v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k,v in values.items()})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        x,links=oracle.random_queries(samples);d,n=oracle.label(x,links)
        v=F.normalize(torch.randn_like(x),dim=-1);fds=[]
        for step in (2.e-5,1.e-5):
            dp,_=oracle.label(x+step*v,links);dm,_=oracle.label(x-step*v,links)
            fds.append((dp-dm)/(2*step))
        report['sdf_spatial_gradient'],stable=directional_check((n*v).sum(-1),*fds,atol=.02,rtol=.02)
        report['sdf_normal_norm_max_error']=float((n.norm(dim=-1)-1).abs().max())
        report['sdf_finite']=bool(torch.isfinite(d).all()&torch.isfinite(n).all())
        keep(sdf_points_m=x,sdf_links=links,sdf_distance_m=d,sdf_gradient=n,sdf_probe=v,
             sdf_fd_coarse=fds[0],sdf_fd_fine=fds[1],sdf_stable=stable)
        # Independent float64 solid-angle sum, not ray parity (which can fail
        # for non-watertight or overlapping CAD components).
        import trimesh
        winding=[];winding_ids=[];unsigned_errors=[]
        for owner,mesh in enumerate(hand.surface.meshes):
            ids=(links==owner).nonzero().flatten()[:16]
            queries=x[ids].cpu().numpy().astype(np.float64)
            _,dist,_=trimesh.proximity.closest_point(mesh,queries)
            unsigned_errors.extend(np.abs(dist-d[ids].abs().cpu().numpy()).tolist())
            for index,point in zip(ids.tolist(),queries):
                tri=np.asarray(mesh.triangles,dtype=np.float64)-point
                a,b,c=tri[:,0],tri[:,1],tri[:,2]
                la=np.linalg.norm(a,axis=-1);lb=np.linalg.norm(b,axis=-1);lc=np.linalg.norm(c,axis=-1)
                num=(a*np.cross(b,c)).sum(-1)
                den=la*lb*lc+(a*b).sum(-1)*lc+(b*c).sum(-1)*la+(c*a).sum(-1)*lb
                winding.append(np.arctan2(num,den).sum()/(2*np.pi));winding_ids.append(index)
        winding=np.abs(winding);winding_ids=np.asarray(winding_ids)
        distances=d.cpu().numpy()[winding_ids]
        unambiguous=(np.abs(winding-.5)>.1)&(np.abs(distances)>.0001)
        report.update(independent_signed_samples=int(unambiguous.sum()),
            independent_winding_sign_agreement=float(np.mean((winding[unambiguous]>.5)==(distances[unambiguous]<0))),
            independent_unsigned_distance_p95_mm=float(np.quantile(unsigned_errors,.95)*1000),
            independent_unsigned_distance_max_mm=float(np.max(unsigned_errors)*1000))
        keep(sdf_independent_winding_ids=winding_ids,sdf_independent_winding=winding)

        # A fixed wrist-frame point sees both joint and six-DoF wrist motion.
        count=min(q_samples,samples);joint=hand.low+torch.rand(count,20,device=hand.device)*(hand.high-hand.low)
        state=torch.cat((joint,torch.zeros(count,6,device=hand.device)),-1).requires_grad_()
        owners=links[:count];local=x[:count];row=torch.arange(count,device=hand.device)
        with torch.enable_grad():
            pos,rot=hand.frames(state)[:2]
            world=((rot[row,owners]@local[...,None]).squeeze(-1)+pos[row,owners]).detach()
            query=(rot[row,owners].transpose(-1,-2)@(world-pos[row,owners])[...,None]).squeeze(-1)
            distance,normal=oracle.label(query,owners)
            surrogate=distance+((query-query.detach())*normal).sum(-1)
            gradient=torch.autograd.grad(surrogate.sum(),state)[0]
        probe=F.normalize(torch.randn_like(state),dim=-1);probe[:,20:23]*=.01
        fds=[]
        for step in (5.e-4,2.5e-4):
            values=[]
            for sign in (1,-1):
                pos,rot=hand.frames(state.detach()+sign*step*probe)[:2]
                query=(rot[row,owners].transpose(-1,-2)@(world-pos[row,owners])[...,None]).squeeze(-1)
                values.append(oracle.label(query,owners)[0])
            fds.append((values[0]-values[1])/(2*step))
        report['articulated_point_gradient'],stable=directional_check((gradient*probe).sum(-1),*fds,atol=5.e-5,rtol=.05)
        keep(articulated_state=state,articulated_world_points=world,articulated_links=owners,
            articulated_gradient=gradient,articulated_probe=probe,articulated_fd_coarse=fds[0],articulated_fd_fine=fds[1])

        x,links=teacher.area_queries(samples,sigma=.006);flags=torch.ones_like(links,dtype=torch.bool)
        hit,normal,valid=teacher.label(x,torch.zeros_like(x),links,flags)
        v=F.normalize(torch.randn_like(x),dim=-1)
        analytic,ok=teacher.projection_derivative(x,links,flags,v);fds=[]
        for step in (1.e-5,5.e-6):
            a,_,_=teacher.label(x+step*v,torch.zeros_like(x),links,flags)
            b,_,_=teacher.label(x-step*v,torch.zeros_like(x),links,flags)
            fds.append((a-b)/(2*step))
        report['projection_jacobian'],stable=directional_check(analytic,*fds,atol=.03,rtol=.01)
        distance,_=oracle.label(hit,links)
        report['projection_on_solid_p95_mm']=float(torch.quantile(distance.abs(),.95)*1000)
        report['projection_on_solid_max_mm']=float(distance.abs().max()*1000)
        report['projection_valid_fraction']=float(valid.float().mean())
        report['projection_normal_norm_max_error']=float((normal.norm(dim=-1)-1).abs().max())
        keep(pad_queries_m=x,pad_links=links,pad_hits_m=hit,pad_normals=normal,pad_probe=v,
            pad_jvp=analytic,pad_fd_coarse=fds[0],pad_fd_fine=fds[1],pad_stable=stable)
        for owner in PAD_OWNERS:
            surface=hand.surface.contact_surface(owner)
            ids=np.asarray(surface['face_ids'],dtype='<i8')
            report['contact_links'][str(owner)]=dict(facets=len(ids),area_mm2=float(teacher.areas[owner].sum()*1.e6),
                query_count=int((links==owner).sum()),face_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                solid_watertight=bool(hand.surface.meshes[owner].is_watertight))

        q=hand.low+torch.rand(q_samples,20,device=hand.device)*(hand.high-hand.low)
        qt=QCollisionTeacher(collision)
        labels=qt.label(q,group_sobolev=True,consistent=True)
        v=F.normalize(torch.randn_like(q),dim=-1);fd=[]
        for step in (5.e-4,2.5e-4):
            a=qt.values(hand.frames(q+step*v),return_groups=True)
            b=qt.values(hand.frames(q-step*v),return_groups=True)
            fd.append({k:(a[k]-b[k])/(2*step) for k in ('clearance_norm','cap_norm','smooth_depth','amplitude','cap_amplitude')})
        for key,name in (('gradient','clearance_norm'),('cap_gradient','cap_norm'),('depth_gradient','smooth_depth')):
            stats,stable=directional_check((labels[key]*v).sum(-1),fd[0][name],fd[1][name],atol=.05,rtol=.05)
            active=stable&(labels[key].norm(dim=-1)>1.e-3)
            if active.any():
                relative=((labels[key]*v).sum(-1)-fd[1][name]).abs()/labels[key].norm(dim=-1).clamp_min(1.e-3)
                stats['directional_residual_over_gradient_norm_p95']=float(torch.quantile(relative[active],.95))
            retained=stable&labels[key+'_valid']
            if retained.any():
                checked,_=directional_check((labels[key]*v).sum(-1)[retained],fd[0][name][retained],fd[1][name][retained],atol=.05,rtol=.05)
                stats['correct_on_retained_fraction']=checked['correct_on_stable_fraction']
            stats['retained_fraction']=float(labels[key+'_valid'].float().mean())
            report['q_'+key]=stats
            keep(**{f'q_{key}':labels[key],f'q_{key}_fd_coarse':fd[0][name],f'q_{key}_fd_fine':fd[1][name],f'q_{key}_stable':stable})
        for key,name in (('group_gradient','amplitude'),('cap_group_gradient','cap_amplitude')):
            probe=labels['group_probe']
            a=(fd[0][name]*probe).sum(-1);b=(fd[1][name]*probe).sum(-1)
            stats,stable=directional_check((labels[key]*v).sum(-1),a,b,atol=.05,rtol=.05)
            report['q_'+key]=stats
            keep(**{f'q_{key}':labels[key],f'q_{key}_fd_coarse':a,f'q_{key}_fd_fine':b,f'q_{key}_stable':stable})
        keep(q_rad=q,q_probe=v,q_group_probe=labels['group_probe'],q_native_depth_mm=labels['native_depth_mm'],
            **{'q_label_'+k:v for k,v in labels.items() if k.endswith('_valid')})
        # Differentiate the same potential and then check it on the native mesh.
        for key,name in (('gradient','clearance_norm'),('cap_gradient','cap_norm')):
            before=qt.values(hand.frames(q),return_groups=True)[name]
            moved=(q-5.e-4*F.normalize(labels[key],dim=-1)).clamp(hand.low,hand.high)
            after=qt.values(hand.frames(moved),return_groups=True)[name]
            active=labels[key].norm(dim=-1)>1.e-3
            report['q_'+key+'_native_descent_fraction']=float((after[active]<=before[active]+1.e-4).float().mean())
    reasons=[]
    report['all_saved_labels_finite']=all(np.isfinite(v).all() for v in arrays.values())
    if not report['all_saved_labels_finite']:reasons.append('nonfinite_sample_data')
    for key in ('sdf_spatial_gradient','articulated_point_gradient','projection_jacobian','q_gradient','q_cap_gradient','q_depth_gradient','q_group_gradient','q_cap_group_gradient'):
        if report[key]['stable_fraction']<.9:reasons.append(key+'_insufficient_regular_samples')
        if report[key]['correct_on_stable_fraction']<.98:reasons.append(key+'_incorrect_derivatives')
    if not report['sdf_finite'] or report['sdf_normal_norm_max_error']>.001:reasons.append('invalid_sdf_labels')
    if report['independent_signed_samples']<20 or not np.isfinite(report['independent_winding_sign_agreement']) or report['independent_winding_sign_agreement']<.99:reasons.append('incorrect_native_distance_signs')
    if report['independent_unsigned_distance_p95_mm']>.05:reasons.append('incorrect_native_distance_values')
    if report['projection_valid_fraction']<1 or report['projection_on_solid_p95_mm']>.01:reasons.append('invalid_projection_values')
    for key in ('q_gradient_native_descent_fraction','q_cap_gradient_native_descent_fraction'):
        if not np.isfinite(report[key]) or report[key]<.98:reasons.append(key)
    report.update(accepted=not reasons,rejection_reasons=reasons)
    if output is not None:
        folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'samples.npz',**arrays)
        report['samples_sha256']=hashlib.sha256((folder/'samples.npz').read_bytes()).hexdigest()
        (folder/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    from .cart_wrist import FloatingCartHand
    from .cart_collision import CollisionModel
    from .wuji_surface import COMPLETE_PAD_POLICY
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--samples',type=int,default=4096);parser.add_argument('--q-samples',type=int,default=512)
    args=parser.parse_args();torch.set_num_threads(1)
    hand=FloatingCartHand(args.device,contact_surface_policy=COMPLETE_PAD_POLICY)
    report=audit_geometry_labels(CollisionModel(hand),samples=args.samples,q_samples=args.q_samples,output=args.output)
    print(json.dumps(report,indent=2))
    if not report['accepted']:raise SystemExit(1)


if __name__=='__main__':main()

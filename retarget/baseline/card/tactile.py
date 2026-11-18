"""Tactile pinch hypotheses and current-frame native-joint seed proposals."""
import numpy as np
import torch


def pinch_hypothesis(anchors,native,external_object=False,geometry_source='robokit'):
    """Propose one unambiguous pinch using bilateral pressure + Robokit pose.

    This is an association hypothesis, not a pressure-only contact detector.
    A known external object, distant pads or ambiguous partners suppress it.
    MANO need not have already closed the pinch.
    """
    report=dict(policy=f'bilateral_tactile_{geometry_source}_pinch_v1',candidates=[],selected=None)
    by_region={a['region']:a for a in anchors}
    if external_object or 3 not in by_region or 3 not in native:
        report['reason']='tracked_external_object' if external_object else 'no_bilateral_distal_activity'
        return None,report
    thumb=by_region[3];p,n=native[3]
    for region in (6,9,12,15):
        if region not in by_region or region not in native:continue
        other=by_region[region];q,m=native[region]
        source_gap=float(np.linalg.norm(np.asarray(thumb['position_m'])-other['position_m']))
        gap=float(np.linalg.norm(p-q));facing=float(n@m)
        eligible=bool(max(source_gap,gap)<=.060 and facing<-.1)
        score=gap+.5*source_gap+.010*(1+facing)
        report['candidates'].append(dict(regions=[3,region],source_gap_mm=source_gap*1000,
            robokit_gap_mm=gap*1000,robokit_normal_dot=facing,eligible=eligible,score=score))
    candidates=sorted((a for a in report['candidates'] if a['eligible']),key=lambda a:a['score'])
    if not candidates:
        report['reason']='outside_capture_range_or_not_opposing';return None,report
    if len(candidates)>1 and candidates[1]['score']-candidates[0]['score']<.010:
        report['reason']='ambiguous_partner';return None,report
    pair=tuple(candidates[0]['regions']);report.update(selected=list(pair),reason=f'bilateral_tactile_{geometry_source}_hypothesis')
    return pair,report


@torch.no_grad()
def pinch_seed(fk,initial,body_ids,local,normal,pair,low,high):
    """Small bounded native-joint proposal; CEM still owns final acceptance.

    Only the two tactile-associated branches move. Fixed material CoPs and
    opposite normals enter the residual before CEM, without using old poses.
    """
    t,r,q=[a.detach().cpu().clone() for a in initial];base=q.clone()
    fingers=sorted({(region-1)//3 for region in pair})
    ids=torch.tensor([4*f+k for f in fingers for k in range(4)])
    lo=torch.maximum(low.detach().cpu()[ids],base[ids]-.65)
    hi=torch.minimum(high.detach().cpu()[ids],base[ids]+.65)
    owners=torch.tensor([body_ids[a] for a in pair]);points=torch.stack([local[a].cpu() for a in pair]);ns=torch.stack([normal[a].cpu() for a in pair])
    def residual(values):
        full=base.expand(len(values),-1).clone();full[:,ids]=values
        p,rr=fk.frames(torch.cat((full.new_zeros((len(full),6)),full),-1))
        x=p[:,owners]+(rr[:,owners]@points[...,None]).squeeze(-1)
        n=(rr[:,owners]@ns[...,None]).squeeze(-1)
        return torch.cat(((x[:,0]-x[:,1])/.002,(n[:,0]+n[:,1])*2.,(values-base[ids])*.2),-1)
    values=base[ids].clone();before=float(residual(values[None])[0,:3].norm())*2
    for _ in range(10):
        eye=torch.eye(len(ids))*.001
        batch=residual(torch.cat((values[None],values+eye,values-eye)))
        jac=((batch[1:1+len(ids)]-batch[1+len(ids):])/.002).T
        step=torch.linalg.solve(jac.T@jac+.05*torch.eye(len(ids)),-jac.T@batch[0])
        step*=min(1.,.25/max(float(step.norm()),1e-9))
        trials=(values+torch.tensor([1.,.5,.25,.125])[:,None]*step).clamp(lo,hi)
        costs=residual(trials).square().sum(-1);best=int(costs.argmin())
        if not torch.isfinite(costs[best]) or costs[best]>=batch[0].square().sum():break
        values=trials[best]
    q[ids]=values;after=float(residual(values[None])[0,:3].norm())*2
    proposal=tuple(a.to(initial[0].device) for a in (t,r,q))
    return proposal,dict(applied=bool((q-base).abs().max()>1e-6),regions=list(pair),
        gap_before_mm=before,gap_after_mm=after,max_joint_change_rad=float((q-base).abs().max()),
        correction_limit_rad=.65,
        active_fingers=fingers,uses_history=False,final_acceptance='CEM_and_dense_native_mesh_guard')

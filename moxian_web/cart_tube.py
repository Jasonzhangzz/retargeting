"""Constraint-formed causal CEM tubes; temporal checks precede elite updates."""
from dataclasses import dataclass, asdict
import copy
import numpy as np
import torch
from scipy.optimize import minimize
from retarget.baseline.card.math import CardSettings, joint_elite_cem

TUBE_POLICY = 'fixed_wrist_constraint_formed_cem_tube_v2'


@dataclass(frozen=True)
class TubeSettings:
    population: int = 1024
    iterations: int = 16
    elite_count: int = 51
    neighbors: int = 8
    interpolation_step_rad: float = .025
    local_radius_rad: float = .25
    sigma_rad: float = .05
    velocity: float = .01
    acceleration: float = .001
    seed: int = 714
    stage1_evaluations: int = 50
    stage2_evaluations: int = 80
    max_velocity_rad_s: float = 12.5
    max_acceleration_rad_s2: float = 500.
    audit_batch_size: int = 16

    def __post_init__(self):
        if min(self.population,self.iterations,self.elite_count,self.neighbors,self.audit_batch_size,
               self.stage1_evaluations,self.stage2_evaluations)<1:
            raise ValueError('Tube budgets must be positive')
        positive=(self.interpolation_step_rad,self.local_radius_rad,self.sigma_rad,
                  self.max_velocity_rad_s,self.max_acceleration_rad_s2)
        if not all(np.isfinite(v) and v>0 for v in positive):
            raise ValueError('Tube motion bounds must be finite and positive')
        if not all(np.isfinite(v) and v>=0 for v in (self.velocity,self.acceleration)):
            raise ValueError('Tube regularization weights must be finite and nonnegative')

    def payload(self): return asdict(self)


def signature(row):
    c=row['contacts']
    return (c.get('tactile_status','unknown'),tuple(sorted(tuple(sorted(p['regions'])) for p in c['self_pairs'])),
            tuple(sorted(p['region'] for p in c['object_contacts'])))


def interpolate_row(a,b,alpha):
    """Only interpolate persistent contacts; events form explicit boundaries."""
    if signature(a)!=signature(b):raise ValueError('Cannot interpolate across contact events')
    result=copy.deepcopy(a)
    def lerp(x,y):return ((1-alpha)*np.asarray(x)+alpha*np.asarray(y)).tolist()
    def normals(x,y):
        n=np.asarray(lerp(x,y));n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
        return n.tolist()
    result['joints_m']=lerp(a['joints_m'],b['joints_m'])
    result['timestamp_ms']=(1-alpha)*a['timestamp_ms']+alpha*b['timestamp_ms']
    lookup={tuple(sorted(p['regions'])):p for p in b['contacts']['self_pairs']}
    for c in result['contacts']['self_pairs']:
        other=lookup[tuple(sorted(c['regions']))]
        if c['regions']!=other['regions']:
            other=dict(other,points_m=other['points_m'][::-1],normals=other['normals'][::-1])
        c['points_m']=lerp(c['points_m'],other['points_m']);c['normals']=normals(c['normals'],other['normals'])
    lookup={p['region']:p for p in b['contacts']['object_contacts']}
    for c in result['contacts']['object_contacts']:
        other=lookup[c['region']];c['position_m']=lerp(c['position_m'],other['position_m'])
        c['normal']=normals(c['normal'],other['normal']);c['force']=(1-alpha)*c['force']+alpha*other['force']
    return result


def path_search(nodes, costs, edges, stamps, velocity=.01, acceleration=.001, smooth=.01):
    """Second-order DP. Each disconnected component has its own real start."""
    paths=[];states={};segment=[];gaps=[]
    def finish():
        nonlocal states,segment
        if states: paths.append(min(states.values(),key=lambda item:item[0])[1])
        states={};segment=[]
    for i, current in enumerate(nodes):
        if not len(current):finish();gaps.append(dict(frame=i,reason='no_feasible_elites'));continue
        if not states:
            states={(None,k):(float(costs[i][k]),[(i,k)]) for k in range(len(current))};segment=[i];continue
        dt=(stamps[i]-stamps[i-1])/1000
        next_states={}
        if 0<dt<=.25:
            for a,b in edges[i]:
                for (older,previous),(value,path) in states.items():
                    if previous!=a:continue
                    speed=(current[b]-nodes[i-1][a])/dt
                    penalty=velocity*float(np.mean(speed**2))+.5*smooth*float(np.sum((current[b]-nodes[i-1][a])**2))
                    if len(path)>1:
                        t,k=path[-2];old_dt=(stamps[i-1]-stamps[t])/1000
                        old_speed=(nodes[i-1][a]-nodes[t][k])/old_dt
                        accel=(speed-old_speed)/(.5*(dt+old_dt))
                        penalty+=acceleration*float(np.mean(accel**2))
                    candidate=(value+float(costs[i][b])+penalty,path+[(i,b)])
                    key=(a,b)
                    if key not in next_states or candidate[0]<next_states[key][0]:next_states[key]=candidate
        if next_states:states=next_states;segment.append(i)
        else:
            finish();gaps.append(dict(frame=i,reason='no_reachable_audited_edge'))
            states={(None,k):(float(costs[i][k]),[(i,k)]) for k in range(len(current))};segment=[i]
    finish()
    return paths,gaps


class TubeBuilder:
    def __init__(self,objective,settings=None,check=lambda:None):
        self.objective=objective;self.hand=objective.hand;self.settings=settings or TubeSettings();self.check=check
        self.generator=torch.Generator(device=self.hand.device).manual_seed(self.settings.seed)

    def initialize(self,row):
        h=self.hand;s=self.settings
        q=((h.low+h.high)/2).detach().cpu().numpy()
        traces=[]
        for stage,budget in ((1,s.stage1_evaluations),(2,s.stage2_evaluations)):
            def fun(values):
                self.check()
                x=h.tensor(values)[None].requires_grad_()
                total,_,_=self.objective(x,row,None,stage=stage)
                grad=torch.autograd.grad(total.sum(),x)[0]
                return float(total[0].detach()),grad[0].detach().cpu().numpy().astype(float)
            result=minimize(fun,q,jac=True,method='L-BFGS-B',bounds=list(zip(h.low.cpu().numpy(),h.high.cpu().numpy())),
                            options=dict(maxiter=budget,maxfun=budget,maxls=12,ftol=1e-9))
            q=result.x;traces.append(dict(stage=stage,evaluations=int(result.nfev),success=bool(result.success),cost=float(result.fun)))
        return h.tensor(q),traces

    def search(self,row,centers,reference=None,*,previous=None,previous_row=None):
        """Only current-frame centers; old elites constrain transitions, never proposals."""
        s=self.settings;h=self.hand
        settings=CardSettings(cem_backend='torch',population=s.population,iterations=s.iterations,
            elite_fraction=min(1.,(s.elite_count+.1)/s.population),joint_step_rad=s.local_radius_rad,joint_sigma_rad=s.sigma_rad)
        if previous is not None and (previous_row is None or not previous['q']):
            raise ValueError('Temporal CEM requires an available predecessor layer')
        dt=(row['timestamp_ms']-previous_row['timestamp_ms'])/1000 if previous is not None else None
        if dt is not None and (not 0<dt<=.25 or signature(previous_row)!=signature(row)):
            raise ValueError('Temporal CEM cannot cross a sensor or contact boundary')
        if previous is not None:
            prior=h.tensor(previous['q']);prior_cost=h.tensor(previous['cumulative_costs'])
            prior_velocity=h.tensor(previous['velocity_rad_s']) if previous.get('dt_s') is not None else None
            baseline=prior_cost.min()
        records={};audited=0;swept_edges=0

        def key(q):return q.detach().cpu().numpy().tobytes()

        def transitions(q):
            delta=q[:,None]-prior[None];speed=delta/dt
            limit=min(s.local_radius_rad,s.max_velocity_rad_s*dt)
            step_excess=(delta.abs()-limit).clamp_min(0).amax(-1)
            valid=step_excess<=1e-6
            penalty=s.velocity*speed.square().mean(-1)+.5*self.objective.settings.lambda_smooth*delta.square().sum(-1)
            violation=(step_excess/limit).square()
            if prior_velocity is not None:
                acceleration=(speed-prior_velocity[None])/(.5*(dt+previous['dt_s']))
                excess=(acceleration.abs()-s.max_acceleration_rad_s2).clamp_min(0).amax(-1)
                valid &= excess<=1e-4
                violation=violation+(excess/s.max_acceleration_rad_s2).square()
                penalty=penalty+s.acceleration*acceleration.square().mean(-1)
            return prior_cost[None]-baseline+penalty,valid,violation,speed

        def score(q):
            cost,_,quality=self.objective(q,row,None)
            feasible=quality['feasible']
            if previous is not None:
                temporal,valid,violation,_=transitions(q)
                # Recovery may approach the feasible set, but is never archived.
                recovery=(temporal+1000*violation).amin(-1)
                best=torch.where(valid,temporal,torch.inf).amin(-1)
                cost=cost+torch.where(valid.any(-1),best,recovery)
                feasible=feasible & valid.any(-1)
            return cost,feasible

        def audit(q,cost,feasible):
            nonlocal audited,swept_edges
            accepted=torch.zeros_like(feasible);updated=cost.clone();kept=0
            order=torch.nonzero(feasible & torch.isfinite(cost)).flatten()
            order=order[torch.argsort(cost[order],stable=True)]
            # Audit until enough accepted elites, not merely the first nominees.
            for start in range(0,len(order),s.audit_batch_size):
                self.check()
                if kept>=s.elite_count:break
                ids=order[start:start+s.audit_batch_size];batch=q[ids]
                raw,parts,quality=self.objective(batch,row,None,dense=True);audited+=len(batch)
                dense_ok=quality['feasible'] & torch.isfinite(raw)
                parents={j:[] for j in range(len(batch))}
                if previous is not None:
                    temporal,valid,_,speed=transitions(batch)
                    valid &= dense_ok[:,None]
                    linked=self.connections(previous['q'],batch.detach().cpu().tolist(),previous_row,row,
                        allowed=valid.T,rank=temporal.T,first_feasible=True)
                    swept_edges+=len(linked)
                    for a,b in linked:parents[b].append(a)
                for j,idx in enumerate(ids.tolist()):
                    if not dense_ok[j] or (previous is not None and not parents[j]):continue
                    if kept>=s.elite_count:break
                    parent=min(parents[j],key=lambda k:float(temporal[j,k])) if previous is not None else None
                    value=float(raw[j])
                    cumulative=value if parent is None else value+float(temporal[j,parent]+baseline)
                    updated[idx]=cumulative-(float(baseline) if previous is not None else 0.)
                    accepted[idx]=True;kept+=1
                    records[key(q[idx])]=dict(parent=parent,parents=parents[j],cumulative_cost=cumulative,
                        velocity=None if parent is None else speed[j,parent].cpu().tolist(),cost=value,
                        parts={k:float(v[j]) for k,v in parts.items()},
                        quality={k:v[j].cpu().item() for k,v in quality.items() if torch.is_tensor(v) and v.ndim==1})
            return updated,accepted

        archive=joint_elite_cem(centers,h.low,h.high,score,settings,self.generator,check=self.check,constrain=audit)
        q=archive['values'];entries=[records[key(value)] for value in q]
        parts={k:[entry['parts'][k] for entry in entries] for k in entries[0]['parts']} if entries else {}
        quality={k:[entry['quality'][k] for entry in entries] for k in entries[0]['quality']} if entries else {}
        return dict(q=q.cpu().tolist(),costs=[entry['cost'] for entry in entries],
            graph_costs=[entry['cost'] for entry in entries],parts=parts,quality=quality,
            cumulative_costs=[entry['cumulative_cost'] for entry in entries],
            parents=[entry['parent'] for entry in entries],
            edges=[[a,b] for b,entry in enumerate(entries) for a in entry['parents']],
            velocity_rad_s=[entry['velocity'] for entry in entries],dt_s=dt,
            evaluations=archive['evaluations'],recovery_q=archive['recovery'].cpu().tolist(),
            dense_audited_candidates=audited,swept_accepted_edges=swept_edges,
            constraints_in_cem=True,generation_feasible=archive['generation_feasible'],
            device=archive['device'],gpu_parallel=archive['gpu_parallel'],
            local_only=True,centers=archive['centers'].cpu().tolist(),
            search_reference_q=None,history_used_for_sampling=False,
            history_role='temporal_feasibility_and_cumulative_path_cost' if previous is not None else 'segment_root')

    @torch.no_grad()
    def connections(self,previous,current,a,b,*,allowed=None,rank=None,first_feasible=False):
        """Swept feasibility oracle used inside CEM, before accepting an elite."""
        if not len(previous) or not len(current) or signature(a)!=signature(b):return []
        dt=(b['timestamp_ms']-a['timestamp_ms'])/1000
        if not 0<dt<=.25:return []
        qa=np.asarray(previous);qb=np.asarray(current);s=self.settings
        distance=((qa[:,None]-qb[None])**2).sum(-1)
        admissible=(np.max(abs(qa[:,None]-qb[None]),axis=-1)<=min(s.local_radius_rad,s.max_velocity_rad_s*dt)+1e-6)
        if allowed is not None:admissible &= allowed.detach().cpu().numpy()
        ranking=distance if rank is None else rank.detach().cpu().numpy()
        candidates=[];ranked=[]
        for j in range(len(qb)):
            ids=np.flatnonzero(admissible[:,j])
            choices=ids[np.argsort(ranking[ids,j],kind='stable')[:s.neighbors]]
            ranked.append(choices)
            for i in choices:candidates.append((int(i),j))
        if not candidates:return []
        if first_feasible:
            # A node carries one best incoming history. Check parents in cost
            # order and stop once that optimal feasible parent has been found.
            accepted=[];unresolved=set(range(len(qb)))
            for level in range(s.neighbors):
                pairs=[(int(ranked[j][level]),j) for j in sorted(unresolved) if level<len(ranked[j])]
                if not pairs:break
                valid=self._swept_pairs(qa,qb,pairs,a,b)
                accepted.extend(valid);unresolved.difference_update(j for _,j in valid)
            return accepted
        return self._swept_pairs(qa,qb,candidates,a,b)

    def _swept_pairs(self,qa,qb,candidates,a,b):
        s=self.settings;h=self.hand
        pairs=np.asarray(candidates);accepted=[]
        for offset in range(0,len(pairs),s.audit_batch_size):
            selected=pairs[offset:offset+s.audit_batch_size]
            start=h.tensor(qa[selected[:,0]]);end=h.tensor(qb[selected[:,1]])
            steps=torch.ceil((end-start).abs().amax(-1)/s.interpolation_step_rad).clamp_min(2).long()
            ok=torch.ones(len(selected),device=h.device,dtype=torch.bool)
            for n in torch.unique(steps).tolist():
                ids=torch.nonzero(steps==n).flatten()
                for k in range(1,n):
                    self.check();active=ids[ok[ids]]
                    if not len(active):break
                    alpha=k/n;row=interpolate_row(a,b,alpha)
                    _,_,quality=self.objective(start[active]*(1-alpha)+end[active]*alpha,row,dense=True)
                    ok[active]&=quality['feasible']
            accepted.extend(selected[ok.cpu().numpy()].tolist())
        return accepted

    def build(self,rows,progress=None,*,checkpoint=None,resume=None):
        nodes=[];edges=[];boundaries=[];gaps=[];paths=[]
        trajectory=[None]*len(rows);segments=[None]*len(rows)
        component_end=None
        if resume is not None:
            if resume['policy']!=TUBE_POLICY or resume['settings']!=self.settings.payload() or len(resume['nodes'])>len(rows):
                raise ValueError('Tube 检查点与当前求解设置不一致')
            nodes=copy.deepcopy(resume['nodes']);edges=copy.deepcopy(resume['edges'])
            boundaries=copy.deepcopy(resume['boundaries']);gaps=copy.deepcopy(resume['gaps'])
            state=resume['resume_state'];component_end=state['component_end']
            paths=copy.deepcopy(resume['paths'][:state['finalized_paths_count']])
            for segment,path in enumerate(paths):
                for i,k in path:trajectory[i]=nodes[i]['q'][k];segments[i]=segment
            if state['rng_state'] is not None:
                self.generator.set_state(torch.tensor(state['rng_state'],dtype=torch.uint8))

        def finish():
            nonlocal component_end
            if component_end is None:return
            i=component_end;k=int(np.argmin(nodes[i]['cumulative_costs']));path=[]
            while True:
                path.append((i,k));parent=nodes[i]['parents'][k]
                if parent is None:break
                i-=1;k=parent
            path.reverse();segment=len(paths);paths.append(path)
            for i,k in path:trajectory[i]=nodes[i]['q'][k];segments[i]=segment
            component_end=None

        for i in range(len(nodes),len(rows)):
            row=rows[i];boundary=None;gap=None
            self.check()
            continuous=i>0 and 0<row['timestamp_ms']-rows[i-1]['timestamp_ms']<=250
            same_contacts=i>0 and signature(rows[i-1])==signature(row)
            if i and (not continuous or not same_contacts):
                finish();boundary=dict(frame=i,reason='stream_gap' if not continuous else 'contact_event');boundaries.append(boundary)
            predecessor=nodes[-1] if component_end is not None else None
            # No previous pose enters either initializer stage or proposal centers.
            initial,traces=self.initialize(row)
            node=self.search(row,initial[None],None,previous=predecessor,
                             previous_row=rows[i-1] if predecessor is not None else None)
            node['initializer']=traces
            if not node['q']:
                finish();gap=dict(frame=i,reason='no_feasible_elites' if predecessor is None else 'no_reachable_cem_elites');gaps.append(gap)
            nodes.append(node);edges.append(node['edges'])
            if node['q']:component_end=i
            if checkpoint:checkpoint(i,node,boundary,gap,self.generator.get_state())
            if progress:progress('tube',completed=i+1,total=len(rows),elites=len(node['q']),connections=len(node['edges']),
                constraints_in_cem=True,device=node['device'],gpu_parallel=node['gpu_parallel'])
        finish()
        return dict(policy=TUBE_POLICY,settings=self.settings.payload(),nodes=nodes,edges=edges,paths=paths,
            trajectory=trajectory,segments=segments,gaps=gaps,boundaries=boundaries,
            feasible_frames=sum(q is not None for q in trajectory),frames=len(rows),complete=True,completed_frames=len(rows),total_frames=len(rows),
            construction='causal_constraints_inside_each_cem_generation',posthoc_connection=False,
            history_used_for_sampling=False,device=str(self.hand.device),gpu_parallel=torch.device(self.hand.device).type=='cuda',
            swept_check='adaptive_samples_not_continuous_intersection_proof')

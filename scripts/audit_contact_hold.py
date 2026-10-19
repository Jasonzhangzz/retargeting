#!/usr/bin/env python3
"""Audit force closure and replay terminal-hold ablations without changing defaults."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def closure(points,normals,friction,active,center,radius,edges=16):
    """Independent rank + strictly positive null-vector LP, 16 cone edges."""
    from scipy.optimize import linprog
    from retarget.contact_solvers import contact_basis,grasp_matrix
    basis=contact_basis(normals[None]);n,t1,t2=basis[...,0],basis[...,1],basis[...,2]
    angle=np.arange(edges)*2*np.pi/edges
    primitive=n[...,None]+friction[None,:,0,None,None]*t1[...,None]*np.cos(angle)+friction[None,:,1,None,None]*t2[...,None]*np.sin(angle)
    G=grasp_matrix(points[None],primitive,center,radius)[0].reshape(6,len(points),edges)[:,active].reshape(6,-1)
    count=G.shape[1]
    if not count:return dict(rank=0,force_closure=False,positive_margin=0.)
    rank=np.linalg.matrix_rank(G)
    result=linprog(np.r_[np.zeros(count),-1.],A_ub=np.c_[-np.eye(count),np.ones(count)],b_ub=np.zeros(count),
        A_eq=np.r_[np.c_[G,np.zeros(6)],np.r_[np.ones(count),0.][None]],b_eq=np.r_[np.zeros(6),1.],
        bounds=[(0,None)]*(count+1),method='highs')
    margin=float(result.x[-1]) if result.success else 0.
    return dict(rank=int(rank),force_closure=bool(rank==6 and margin>1e-7),positive_margin=margin)


def device_lm(solver,initial,template,hand,weights,friction,rotation,center,radius):
    """Run the actual default Warp sampler/LM on a static free-space input."""
    from types import SimpleNamespace
    import warp as wp
    from retarget.contact_lm_warp import DeviceLambdaLM
    device=wp.get_device('cuda:0');count=len(initial)
    array=lambda v,t:wp.array(v,dtype=t,device=device)
    points=np.asarray(initial,np.float32)[None,None]
    batch=SimpleNamespace(c=array(points,wp.vec3),x=array(np.asarray(hand)[None,None],wp.vec3),
        dual=array(np.zeros_like(points),wp.vec3),weights=array(np.asarray(weights)[None],float),
        friction=array(np.asarray(friction)[None,None],wp.vec2),rotation=array(rotation[None,None],wp.mat33),
        position=array(np.zeros((1,1,3)),wp.vec3),velocity=array(np.zeros((1,1,6)),wp.spatial_vector),
        goal=array(np.zeros((1,1,6)),wp.spatial_vector))
    inverse=solver._physics(1,radius,None)['inverse'][0]
    physics=dict(inverse=inverse,com=center,gravity=np.array([0.,0.,-9.81]),floor=None,
                 support_vertices=np.zeros((1,3)))
    lm=DeviceLambdaLM(solver,groups=1,contacts=count,worlds=1,physics=physics,device=device)
    lm.bind(batch,np.asarray(template)[None],np.asarray(weights)[None],radius)
    result=lm.solve().numpy();row=lm.job.statistics.numpy()[0]
    return result,dict(method='live_warp_sampled_lm',held_feasible=bool(row[7]),searched=bool(row[8]),
        feasible=bool(row[10]),motion_error_mm=float(row[9]),initial_value=float(row[0]),
        final_value=float(row[1]),selected_candidate=int(row[2]))


def contact_audit(out,source):
    from forceaware.config import load_retarget_config
    from forceaware.runtime import setup_runtime
    from forceaware.workflow import _load_runtime_models,_collision_output_layout
    from forceaware.targets import build_index,load_sequence
    from forceaware.mpc_loss import _grasp_output_properties,_object_grasp_geometry
    from retarget.contact_exploration import ObjectMeshSurface
    from retarget.contact_solvers import SampledComFreeLM,QPContactSolver,skew
    from retarget.trigrasp import quat_wxyz_to_matrix
    import comfree_warp
    root=ROOT/'data/example'
    cfg=load_retarget_config(root/'sequence',root/'forceaware.yaml',contactaware_dir=root/'contactaware')
    setup_runtime(cfg);models=_load_runtime_models(cfg,comfree_warp);m=models.planner_cpu;idx=build_index(m)
    body,enabled=_collision_output_layout(models.planner_device)
    seq=load_sequence(cfg,output_body_ids=body,enabled_output_indices=enabled)
    from comfree_warp.native_adjoint.fast_step import compile_fast_step
    compiled=compile_fast_step(m,models.planner_device)
    collision=compiled.base.collision
    _,_,mu=_grasp_output_properties(m,idx,collision)
    center,radius=_object_grasp_geometry(m,idx,collision)
    surface=ObjectMeshSurface.from_model(m,idx.object_body)
    mass=float(m.body_mass[idx.object_body]);cross=skew(center)
    rotation=quat_wxyz_to_matrix(m.body_iquat[idx.object_body])
    inertia=rotation@np.diag(m.body_inertia[idx.object_body])@rotation.T
    inverse=np.linalg.inv(np.block([[mass*np.eye(3),-mass*cross],[mass*cross,inertia-mass*cross@cross]]))
    report=dict(definition='rank(G)=6 and strictly positive G alpha=0, sum alpha=1; 16 friction edges per contact',
                solver_defaults=dict(samples=8,lm_iterations=3,qp_iterations=96,radius_mm=5),frames={})
    rng=np.random.default_rng(0)
    for frame in (23,40,71):
        template=seq.contact.position_object[frame];weights=seq.contact.mask[frame]*seq.contact.weight[frame]
        friction=mu[seq.contact.output_index[frame]]
        solver=SampledComFreeLM(surface,device='cuda:0',mass=mass);solver.inverse=inverse
        p,n,movable=solver._project(template[None,None],template[None],weights[None]);p,n=p[0,0],n[0,0]
        active=(weights>0)&movable[0]
        world_r=quat_wxyz_to_matrix(seq.object_pose_xyzw[frame][[3,0,1,2]])
        gravity=mass*(world_r.T@np.array(m.opt.gravity))
        context=dict(external=np.r_[gravity,np.cross(center,gravity)])
        before=closure(p,n,friction,active,center,radius)
        # Whole-set, surface-preserving perturbations: radius 4 mm about the
        # projected MANO combination. The perturbed set becomes the solver's
        # input anchor, with the usual independent 5 mm correction radius.
        proposals=np.broadcast_to(p,(16,*p.shape)).copy()
        offsets=rng.normal(size=proposals.shape);offsets*=.004/np.maximum(np.linalg.norm(offsets,axis=-1,keepdims=True),1e-15)
        proposals+=offsets
        perturbed,directions,_=solver._project(proposals[None],p[None],weights[None])
        perturbed,directions=perturbed[0],directions[0]
        rows=[]
        for initial,normal in zip(perturbed,directions):
            start=closure(initial,normal,friction,active,center,radius)
            row=dict(before=start,solvers={})
            for cls in (SampledComFreeLM,QPContactSolver):
                solve=cls(surface,device='cuda:0',mass=mass);solve.inverse=inverse
                if cls is SampledComFreeLM:
                    after,diag=device_lm(solve,initial,initial,p,weights,friction,world_r,center,radius)
                else:
                    after,diag=solve.solve(initial,initial,p,weights,friction,center=center,object_radius=radius,context=context)
                projected,n_after,movable_after=solve._project(after[None,None],initial[None],weights[None])
                row['solvers'][cls.__name__]=dict(after=closure(projected[0,0],n_after[0,0],friction,
                    active&movable_after[0],center,radius),diagnostic=diag)
            rows.append(row)
        report['frames'][str(frame)]=dict(original=before,active_contacts=int(active.sum()),
            friction=np.unique(friction[active]).tolist(),perturbation_mm=4,cases=rows)
    # Targeted surface perturbations, then bisection at a closure boundary.
    # Keep only failures with an independently verified closed neighbor inside
    # the solver's connected 5 mm patch; this excludes impossible repairs.
    recoverable=[]
    for frame in (23,40,71):
        template=seq.contact.position_object[frame];weights=seq.contact.mask[frame]*seq.contact.weight[frame]
        friction=mu[seq.contact.output_index[frame]]
        broad=SampledComFreeLM(surface,radius=.08,device='cuda:0',mass=mass)
        p,n,movable=broad._project(template[None,None],template[None],weights[None]);p=p[0,0]
        active=(weights>0)&movable[0]
        world_r=quat_wxyz_to_matrix(seq.object_pose_xyzw[frame][[3,0,1,2]])
        gravity=mass*(world_r.T@np.array(m.opt.gravity))
        context=dict(external=np.r_[gravity,np.cross(center,gravity)])
        offsets=rng.normal(size=(64,*p.shape))
        offsets*=.04/np.maximum(np.linalg.norm(offsets,axis=-1,keepdims=True),1e-15)
        candidates,normals,_=broad._project((p+offsets)[None],template[None],weights[None])
        for bad,bad_normal in zip(candidates[0],normals[0]):
            if closure(bad,bad_normal,friction,active,center,radius)['force_closure']:continue
            low,high=0.,1.;good=p.copy();broken=bad.copy()
            for _ in range(22):
                mid=(low+high)*.5
                projected,direction,_=broad._project(((1-mid)*p+mid*bad)[None,None],template[None],weights[None])
                point,normal=projected[0,0],direction[0,0]
                if closure(point,normal,friction,active,center,radius)['force_closure']:
                    low=mid;good=point
                else:high=mid;broken=point
            # Stay 2 mm on each side of the boundary instead of using a
            # floating-point-sized break/repair near an LP tolerance.
            step=.002/max(float(np.linalg.norm(bad-p,axis=-1).max()),1e-12)
            path=np.stack(((1-max(0.,high-step))*p+max(0.,high-step)*bad,
                           (1-min(1.,high+step))*p+min(1.,high+step)*bad))
            projected,_,_=broad._project(path[None],template[None],weights[None])
            good,broken=projected[0]
            narrow=SampledComFreeLM(surface,device='cuda:0',mass=mass)
            neighbor,nn,available=narrow._project(good[None,None],broken[None],weights[None])
            neighbor,nn=neighbor[0,0],nn[0,0]
            known=closure(neighbor,nn,friction,active&available[0],center,radius)
            if not known['force_closure'] or known['positive_margin']<1e-6:continue
            row=dict(frame=frame,initial=broken.tolist(),known_neighbor=neighbor.tolist(),known_neighbor_closure=known,
                     known_neighbor_four_edge_closure=closure(neighbor,nn,friction,active&available[0],center,radius,edges=4),
                     max_template_perturbation_mm=float(np.linalg.norm(broken-p,axis=-1).max()*1000),
                     known_neighbor_max_shift_mm=float(np.linalg.norm(neighbor-broken,axis=-1).max()*1000),solvers={})
            for cls in (SampledComFreeLM,QPContactSolver):
                solve=cls(surface,device='cuda:0',mass=mass);solve.inverse=inverse
                before,bn,bm=solve._project(broken[None,None],broken[None],weights[None])
                row['before']=closure(before[0,0],bn[0,0],friction,active&bm[0],center,radius)
                if cls is SampledComFreeLM:
                    after,diag=device_lm(solve,broken,broken,p,weights,friction,world_r,center,radius)
                else:
                    after,diag=solve.solve(broken,broken,p,weights,friction,center=center,object_radius=radius,context=context)
                projected,n_after,available=solve._project(after[None,None],broken[None],weights[None])
                row['solvers'][cls.__name__]=dict(after=closure(projected[0,0],n_after[0,0],friction,
                    active&available[0],center,radius),diagnostic=diag)
            if row['before']['force_closure']:continue
            recoverable.append(row)
            if sum(r['frame']==frame for r in recoverable)>=3:break
    report['recoverable_surface_perturbations']=recoverable
    # Inspect the same native collision geometry used by the hand loss at
    # measured poses in the supplied completed run, without advancing physics.
    from comfree_warp.native_adjoint.observation_adjoint import allocate_workspace,forward
    import warp as wp
    latest=source/'rollout.npz'
    saved=np.load(latest);steps=np.array([237,270,340,440,540])
    observation=allocate_workspace(compiled.base,len(steps),requires_grad=False)
    state=forward(compiled.base,wp.array(saved['qpos_traj'][steps],dtype=float,device='cuda:0'),observation)
    positions,frames=state.contact_position.numpy(),state.contact_frame.numpy()
    distances=state.contact_distance.numpy();enabled=observation.collision.contacts.active.numpy()
    bp,bmat=state.body_position.numpy(),state.body_matrix.numpy()
    force_sign,_,friction=_grasp_output_properties(m,idx,collision)
    rows=collision.output_contact_rows.numpy();valid=rows>=0;indices=rows[valid]
    physical=[]
    for b,step in enumerate(steps):
        rotation=bmat[b,idx.object_body]
        point=(positions[b,indices]-bp[b,idx.object_body])@rotation
        normal=-(force_sign[valid,None]*frames[b,indices,0])@rotation
        active=(enabled[b,indices]>0)&(distances[b,indices]<0.)
        physical.append(dict(time=float(step*.01),active_penetrating_outputs=int(active.sum()),
            closure=closure(point,normal,friction[valid],active,center,radius)))
    report['measured_contact_geometry']=physical
    # A recoverable geometric counterexample: two antipodal locations, with
    # a duplicated third contact, cannot resist torque around their line.
    # Moving the third contact tangentially by 1 mm fixes rank/closure.
    vertices=[[-.05,-.1,-.1],[-.05,.1,-.1],[-.05,.1,.1],[-.05,-.1,.1],
              [.05,-.1,-.1],[.05,.1,-.1],[.05,.1,.1],[.05,-.1,.1]]
    sides=ObjectMeshSurface(vertices,[[0,2,1],[0,3,2],[4,5,6],[4,6,7]])
    initial=np.array([[-.05,0,0],[.05,0,0],[-.05,0,0]],float)
    good=initial.copy();good[2,1]=.001
    normal=np.array([[-1.,0,0],[1.,0,0],[-1.,0,0]])
    friction=np.full((3,2),.6);active=np.ones(3,bool)
    counter=dict(before=closure(initial,normal,friction,active,np.zeros(3),.1),
                 known_neighbor=closure(good,normal,friction,active,np.zeros(3),.1),
                 neighbor_max_shift_mm=1.,solvers={})
    for cls in (SampledComFreeLM,QPContactSolver):
        solve=cls(sides,device='cuda:0')
        if cls is SampledComFreeLM:
            after,diag=device_lm(solve,initial,initial,initial,np.ones(3),friction,np.eye(3),np.zeros(3),.1)
        else:
            after,diag=solve.solve(initial,initial,initial,np.ones(3),friction,center=np.zeros(3),object_radius=.1)
        p,n,_=solve._project(after[None,None],initial[None],np.ones((1,3)))
        counter['solvers'][cls.__name__]=dict(after=closure(p[0,0],n[0,0],friction,active,np.zeros(3),.1),
            shift_mm=float(np.linalg.norm(after-initial,axis=-1).max()*1000),diagnostic=diag)
    report['recoverable_counterexample']=counter
    (out/'closure.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:dict(original=v['original'],broken=sum(not c['before']['force_closure'] for c in v['cases']),
        repaired={name:sum(not c['before']['force_closure'] and c['solvers'][name]['after']['force_closure'] for c in v['cases'])
                  for name in ('SampledComFreeLM','QPContactSolver')}) for k,v in report['frames'].items()},indent=2))
    print('counterexample',json.dumps(counter))


def worker(args):
    from forceaware import workflow
    from retarget.fa import solver as module
    source=np.load(args.source/'rollout.npz');contacts=np.load(args.source/'contacts.npz')
    boundary=int(np.flatnonzero(source['qpos_frames']>=71)[0]);boundary-=boundary%3
    previous_window=boundary//3-1
    original_initial,original_steps=workflow._initial_run,workflow._run_steps
    def initial(*a,**kw):
        result=original_initial(*a,**kw)
        return replace(result,qpos=source['qpos_traj'][boundary],warm=source['raw_warm_history'][boundary],
                       previous_action=source['ctrl_traj'][boundary-1])
    def steps(ctx,**kw):
        ctx=replace(ctx,initial=replace(ctx.initial,qvel=source['qvel_traj'][boundary]))
        ctx.executor.elapsed=boundary*.01
        return original_steps(ctx,**kw)
    workflow._initial_run,workflow._run_steps=initial,steps
    factory=module.optimizer_class
    def build(settings):
        cls=factory(settings)
        class AuditSolver(cls):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                from retarget.contact_memory import allocate
                stages=len(self.backend.objective.points)
                memory=allocate(stages,self.backend.guidance_count,self.backend.device)
                for name,key in [('c','c'),('dual','dual'),('local','hand_local'),('body','body')]:
                    value=contacts[key][previous_window]
                    value=np.concatenate((value[:1],value))
                    if args.worker=='no-dual' and name=='dual':value=np.zeros_like(value)
                    getattr(memory,name).assign(value)
                active=(contacts['mask'][previous_window]>0).astype(np.int32)
                memory.active.assign(np.concatenate((np.zeros_like(active[:1]),active)))
                self._contact_memory,self._contact_memory_shift=memory,3
                if args.worker=='no-seed':self.trigrasp_mpc=None
            def _dual_block(self):
                result=super()._dual_block()
                if args.worker=='no-dual':
                    for p in self.backend.objective.points:p.dual.zero_()
                return result
        return AuditSolver
    module.optimizer_class=build
    import run
    argv=['--adamw','--no-viewer','--experiment',str(args.experiment),'--duration',str(args.duration),
          '--output-base',str(args.output/'runs'),'--case-name',args.worker,'--seed','79']
    if args.worker=='small-lr':argv+=['--adamw-lr','0.03']
    if args.worker=='persist':argv+=['--adamw-persist']
    if args.worker=='single-start':argv+=['--multistarts','1']
    run.main(argv)


def hold_audit(args):
    with tempfile.TemporaryDirectory(prefix='retarget_terminal_audit_') as tmp:
        prepared=Path(tmp);root=ROOT/'data/example'
        for name in ('sequence','contactaware','demonstration_visual.npz'):(prepared/name).symlink_to(root/name)
        config=yaml.safe_load((root/'forceaware.yaml').read_text())
        config['sequence'].update(start=70,hold_frame=71,hold_seconds=args.duration)
        (prepared/'forceaware.yaml').write_text(yaml.safe_dump(config))
        rows={}
        for case in ('baseline','small-lr','persist','single-start','no-seed','no-dual'):
            command=[sys.executable,__file__,'--worker',case,'--experiment',str(prepared),
                     '--source',str(args.source),'--output',str(args.output),'--duration',str(args.duration)]
            with (args.output/(case+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            a=np.load(args.output/'runs/comfree'/case/'rollout.npz')
            q,v=a['qpos_traj'],a['qvel_traj']
            rows[case]=dict(position_error_end_mm=float(a['drift'][-1]*1000),
                rotation_error_end_deg=float(a['rot_err_deg'][-1]),
                position_error_mean_mm=float(a['drift'].mean()*1000),
                rotation_error_mean_deg=float(a['rot_err_deg'].mean()),
                object_linear_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(v[:,26:29]**2,axis=1)))),
                object_angular_speed_rms_rad_s=float(np.sqrt(np.mean(np.sum(v[:,29:32]**2,axis=1)))),
                wrist_rotation_range_deg=np.rad2deg(np.ptp(q[:,3:6],axis=0)).tolist())
            (args.output/'hold_ablations.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(case,json.dumps(rows[case]),flush=True)


def source_audit(source,out):
    saved=np.load(source/'rollout.npz');contacts=np.load(source/'contacts.npz')
    records=[json.loads(line) for line in (source/'windows.jsonl').read_text().splitlines()]
    q,v=saved['qpos_traj'],saved['qvel_traj']
    frames=saved['qpos_frames'];last=float(frames.max())
    first=int(np.flatnonzero(frames>=last)[0]);summary=dict(source=str(source),
        terminal_reference_frame=last,terminal_reference_reached_seconds=first*.01,
        wrist_rotation_range_deg=np.rad2deg(np.ptp(q[:,3:6],axis=0)).tolist(),
        final_position_error_mm=float(saved['drift'][-1]*1000),
        final_rotation_error_deg=float(saved['rot_err_deg'][-1]),intervals=[])
    for begin,end in ((2.4,3.4),(3.4,4.4),(4.4,5.4)):
        index=np.arange(round(begin*100),min(round(end*100)+1,len(q)))
        if len(index)<2:continue
        summary['intervals'].append(dict(begin=begin,end=end,
            position_error_mean_mm=float(saved['drift'][index[:-1]].mean()*1000),
            rotation_error_mean_deg=float(saved['rot_err_deg'][index[:-1]].mean()),
            object_linear_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(v[index,26:29]**2,axis=1)))),
            object_angular_speed_rms_rad_s=float(np.sqrt(np.mean(np.sum(v[index,29:32]**2,axis=1))))))
    active=contacts['mask'][-1]>0
    summary['last_window']=dict(primal_mean_mm=records[-1]['primal_mean_mm'],
        mean_dual_mm=float(np.linalg.norm(contacts['dual'][-1],axis=-1)[active].mean()*1000),
        held_feasible_fraction=float(np.mean([r['held_feasible'] for r in records[-1]['solver']['contact_surface_diagnostics']])))
    (out/'source_run.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/comfree/adamw_20261019_145931_413461')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/contact_hold_audit')
    parser.add_argument('--duration',type=float,default=1.2)
    parser.add_argument('--contacts-only',action='store_true')
    parser.add_argument('--holds-only',action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--experiment',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    source_audit(args.source,args.output)
    if not args.holds_only:contact_audit(args.output,args.source)
    if not args.contacts_only:hold_audit(args)


if __name__=='__main__':main()

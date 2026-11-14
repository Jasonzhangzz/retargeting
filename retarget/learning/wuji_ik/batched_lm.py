"""Independent-frame bounded LM and URDF FK; never consumes SDK solutions."""
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch


def skew(v):
    x,y,z=v.unbind(-1); zero=torch.zeros_like(x)
    return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)


def so3_log(rotation):
    vee=torch.stack((rotation[...,2,1]-rotation[...,1,2],rotation[...,0,2]-rotation[...,2,0],rotation[...,1,0]-rotation[...,0,1]),-1)*.5
    sine=vee.norm(dim=-1);cosine=((rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
    theta=torch.atan2(sine,cosine)
    factor=torch.where(theta<1e-5,1+theta.square()/6+7*theta.pow(4)/360,theta/sine.clamp_min(1e-15))
    value=vee*factor[...,None]
    # Symmetrized Rodrigues: a column of R+Rᵀ-2 cos(theta) I is
    # exactly parallel to the rotation axis. R+I alone has O(pi-theta)
    # axis error and an incorrect derivative near a half turn.
    plus=rotation+rotation.transpose(-1,-2)-2*cosine[...,None,None]*torch.eye(3,device=rotation.device,dtype=rotation.dtype)
    column=plus.square().sum(-2).argmax(-1)
    axis=plus.gather(-1,column[...,None,None].expand(*column.shape,3,1)).squeeze(-1)
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-15)
    axis=axis*torch.where((axis*vee).sum(-1)<0,-1.,1.)[...,None]
    return torch.where((theta>math.pi-1e-5)[...,None],theta[...,None]*axis,value)


def so3_left_inverse(v):
    theta=v.norm(dim=-1);theta2=theta.square();k=skew(v)
    coefficient=torch.where(theta<1e-4,1/12+theta2/720+theta2.square()/30240,
        (1-.5*theta*torch.cos(.5*theta)/torch.sin(.5*theta).clamp_min(1e-15))/theta2.clamp_min(1e-15))
    return torch.eye(3,device=v.device,dtype=v.dtype)-.5*k+coefficient[...,None,None]*(k@k)


class BatchURDF:
    """Exact URDF tree, named joint order, analytic position/angular Jacobians."""
    def __init__(self,path,joint_names,links,*,device='cpu',dtype=torch.float64,root_transform=None):
        from scipy.spatial.transform import Rotation
        self.path=Path(path);self.joint_names=tuple(joint_names);self.links=tuple(links)
        self.device=torch.device(device);self.dtype=dtype;self.nq=len(joint_names)
        self.tensor=lambda x:torch.as_tensor(x,device=self.device,dtype=dtype)
        tree=ET.parse(path).getroot();all_links=[e.get('name') for e in tree.findall('link')]
        index={n:i for i,n in enumerate(joint_names)};records=[];by_child={}
        bounds=np.zeros((self.nq,2));found=set()
        for element in tree.findall('joint'):
            name=element.get('name');kind=element.get('type');origin=element.find('origin')
            xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            axis_element=element.find('axis');axis=np.fromstring(axis_element.get('xyz','1 0 0') if axis_element is not None else '1 0 0',sep=' ');axis/=np.linalg.norm(axis)
            qi=index.get(name)
            if kind not in ('fixed','revolute') or element.find('mimic') is not None or (kind!='fixed' and qi is None):
                raise ValueError('Unmapped or unsupported URDF joint '+name)
            record=dict(name=name,parent=element.find('parent').get('link'),child=element.find('child').get('link'),
                xyz=self.tensor(xyz),rotation=self.tensor(Rotation.from_euler('xyz',rpy).as_matrix()),axis=self.tensor(axis),q=qi)
            records.append(record);by_child[record['child']]=record
            if qi is not None:
                limit=element.find('limit');bounds[qi]=float(limit.get('lower')),float(limit.get('upper'));found.add(qi)
        if found!=set(range(self.nq)):raise ValueError('URDF joint coverage is incomplete')
        roots=set(all_links)-set(by_child)
        if len(roots)!=1:raise ValueError('Expected a single rooted tree')
        self.root=roots.pop();self.ordered=[];available={self.root};paths={self.root:[]}
        while records:
            ready=[j for j in records if j['parent'] in available]
            if not ready:raise ValueError('Disconnected URDF')
            for j in ready:
                self.ordered.append(j);available.add(j['child']);records.remove(j)
                paths[j['child']]=paths[j['parent']]+([] if j['q'] is None else [j['q']])
        self.mask=self.tensor([[int(i in paths[name]) for i in range(self.nq)] for name in links])
        self.lower=self.tensor(bounds[:,0]);self.upper=self.tensor(bounds[:,1])
        transform=np.eye(4) if root_transform is None else np.asarray(root_transform)
        self.root_R=self.tensor(transform[:3,:3]);self.root_p=self.tensor(transform[:3,3])

    def forward(self,q):
        if q.ndim!=2 or q.shape[1]!=self.nq:raise ValueError('Expected batch x named joint order')
        n=len(q);rotations={self.root:self.root_R.expand(n,3,3)};positions={self.root:self.root_p.expand(n,3)}
        axes=[None]*self.nq;origins=[None]*self.nq
        for j in self.ordered:
            parent_R=rotations[j['parent']]
            p=positions[j['parent']]+(parent_R@j['xyz'][...,None]).squeeze(-1)
            r=parent_R@j['rotation'];i=j['q']
            if i is not None:
                axes[i]=(r@j['axis'][...,None]).squeeze(-1);origins[i]=p
                k=skew(j['axis']);angle=q[:,i,None,None]
                r=r@(torch.eye(3,device=q.device,dtype=q.dtype)+torch.sin(angle)*k+(1-torch.cos(angle))*(k@k))
            rotations[j['child']]=r;positions[j['child']]=p
        points=torch.stack([positions[name] for name in self.links],1)
        rotation=torch.stack([rotations[name] for name in self.links],1)
        axis=torch.stack(axes,1);origin=torch.stack(origins,1)
        jp=torch.linalg.cross(axis[:,None].expand(-1,len(self.links),-1,-1),points[:,:,None]-origin[:,None],dim=-1)*self.mask[None,:,:,None]
        jw=axis[:,None]*self.mask[None,:,:,None]
        return points,rotation,jp.transpose(-1,-2),jw.transpose(-1,-2)


class FastFingerURDF(BatchURDF):
    """Vectorize the five independent finger chains as an extra batch axis."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if len(self.links)==5:
            tips=list(self.links);selected=[[link] for link in tips];self.with_root=False
        elif len(self.links)==21:
            selected=[list(self.links[1+4*f:5+4*f]) for f in range(5)]
            tips=[x[-1] for x in selected];self.with_root=True
        else:raise ValueError('Expected five receivers or 21 robot landmarks')
        by_child={j['child']:j for j in self.ordered};chains=[]
        for tip in tips:
            chain=[];node=tip
            while node!=self.root:
                j=by_child[node];chain.append(j);node=j['parent']
            chains.append(list(reversed(chain)))
        depth=max(map(len,chains));origins=[];rotations=[];axes=[];indices=[];active=[]
        joint_locations=[None]*self.nq;outputs=[]
        for f,chain in enumerate(chains):
            slots={j['child']:i for i,j in enumerate(chain)}
            outputs.extend(f*depth+slots[name] for name in selected[f])
            for i,j in enumerate(chain):
                if j['q'] is not None:
                    if joint_locations[j['q']] is not None:raise ValueError('Actuated joints shared between fingers')
                    joint_locations[j['q']]=f*depth+i
        if any(x is None for x in joint_locations):raise ValueError('Finger chains do not cover all joints')
        self.root_location=None
        if self.with_root and self.links[0]!=self.root:
            node=self.links[0];slot=next(i for i,j in enumerate(chains[0]) if j['child']==node)
            if any(j['q'] is not None for j in chains[0][:slot+1]):raise ValueError('Articulated wrist is unsupported')
            self.root_location=slot
        for step in range(depth):
            xyz=[];r=[];axis=[];qi=[];mask=[]
            for chain in chains:
                j=chain[step] if step<len(chain) else None
                xyz.append(j['xyz'] if j else self.tensor([0.,0.,0.]))
                r.append(j['rotation'] if j else torch.eye(3,device=self.device,dtype=self.dtype))
                axis.append(j['axis'] if j else self.tensor([1.,0.,0.]))
                is_active=j is not None and j['q'] is not None
                qi.append(j['q'] if is_active else 0);mask.append(float(is_active))
            origins.append(torch.stack(xyz));rotations.append(torch.stack(r));axes.append(torch.stack(axis))
            indices.append(torch.tensor(qi,device=self.device));active.append(self.tensor(mask))
        self.depth=depth;self.chain_xyz=origins;self.chain_R=rotations;self.chain_axis=axes
        self.chain_index=indices;self.chain_active=active
        self.joint_locations=torch.tensor(joint_locations,device=self.device)
        self.output_locations=torch.tensor(outputs,device=self.device)
        self.eye=torch.eye(3,device=self.device,dtype=self.dtype)

    def forward(self,q):
        n=len(q);r=self.root_R[None,None].expand(n,5,3,3);p=self.root_p[None,None].expand(n,5,3)
        positions=[];rotations=[];all_axes=[];all_origins=[]
        for step in range(self.depth):
            p=p+(r@self.chain_xyz[step][...,None]).squeeze(-1)
            r=r@self.chain_R[step];axis=(r@self.chain_axis[step][...,None]).squeeze(-1)
            all_axes.append(axis);all_origins.append(p)
            angle=(q[:,self.chain_index[step]]*self.chain_active[step])[...,None,None]
            k=skew(self.chain_axis[step]);r=r@(self.eye+torch.sin(angle)*k+(1-torch.cos(angle))*(k@k))
            positions.append(p);rotations.append(r)
        p_all=torch.stack(positions,2).flatten(1,2);r_all=torch.stack(rotations,2).flatten(1,2)
        points=p_all[:,self.output_locations];rotation=r_all[:,self.output_locations]
        if self.with_root:
            root_p=self.root_p.expand(n,3) if self.root_location is None else p_all[:,self.root_location]
            root_r=self.root_R.expand(n,3,3) if self.root_location is None else r_all[:,self.root_location]
            points=torch.cat((root_p[:,None],points),1);rotation=torch.cat((root_r[:,None],rotation),1)
        axis=torch.stack(all_axes,2).flatten(1,2)[:,self.joint_locations]
        origin=torch.stack(all_origins,2).flatten(1,2)[:,self.joint_locations]
        jp=torch.linalg.cross(axis[:,None].expand(-1,len(self.links),-1,-1),points[:,:,None]-origin[:,None],dim=-1)*self.mask[None,:,:,None]
        jw=axis[:,None]*self.mask[None,:,:,None]
        return points,rotation,jp.transpose(-1,-2),jw.transpose(-1,-2)


@dataclass
class LMResult:
    q:torch.Tensor
    initial_cost:torch.Tensor
    cost:torch.Tensor
    accepted_steps:torch.Tensor
    converged:torch.Tensor
    damping:torch.Tensor
    history:list


def _lm_graph_body(objective,initial,lower,upper,iterations,initial_damping,max_step):
    """Capture-safe fixed budget; converged rows freeze without a host sync."""
    q=initial.clamp(lower,upper).clone();n,d=q.shape
    damping=q.new_full((n,),initial_damping);accepted_count=torch.zeros(n,dtype=torch.long,device=q.device)
    converged=torch.zeros(n,dtype=torch.bool,device=q.device);r,j=objective(q)
    cost=.5*r.square().sum(-1);initial_cost=cost.clone();eye=torch.eye(d,device=q.device,dtype=q.dtype)[None]
    for _ in range(iterations):
        h=j.transpose(-1,-2)@j;g=(j.transpose(-1,-2)@r[...,None]).squeeze(-1)
        active=((q<=lower+1e-10)&(g>0))|((q>=upper-1e-10)&(g<0));free=(~active).to(q.dtype)
        converged=converged|((g*free).abs().amax(-1)<1e-7);work=~converged
        diagonal=h.diagonal(dim1=-2,dim2=-1).clamp_min(1e-8)
        matrix=h*free[:,:,None]*free[:,None,:]+eye*(damping[:,None]*diagonal+(1-free))[...,None,:]
        delta=-torch.linalg.solve_ex(matrix,(g*free)[...,None],check_errors=False).result.squeeze(-1)
        delta=delta.clamp(-max_step,max_step)*work[:,None]
        trial=(q+delta).clamp(lower,upper);step=trial-q
        trial_r,trial_j=objective(trial);trial_cost=.5*trial_r.square().sum(-1)
        predicted=-(g*step).sum(-1)-.5*(step*(h@step[...,None]).squeeze(-1)).sum(-1)
        reduction=cost-trial_cost;ratio=reduction/predicted.clamp_min(1e-30)
        accept=work&torch.isfinite(trial_cost)&(reduction>0)&(predicted>0)
        q=torch.where(accept[:,None],trial,q);r=torch.where(accept[:,None],trial_r,r)
        j=torch.where(accept[:,None,None],trial_j,j);old_cost=cost;cost=torch.where(accept,trial_cost,cost)
        damping=torch.where(accept,torch.where(ratio>.75,damping*.3,torch.where(ratio<.25,damping*2,damping)),damping*10).clamp(1e-12,1e12)
        accepted_count=accepted_count+accept.long()
        converged=converged|(accept&((step.abs().amax(-1)<1e-8)|(reduction<1e-10*(1+old_cost))))
    return q,initial_cost,cost,accepted_count,converged,damping


class CUDAGraphLM:
    """Replay solver kernels, never cached solutions; objective buffers stay live.

    Callers update the objective's target tensors in place before each call.
    Initial q is copied anew every invocation. Results are cloned out of the
    graph pool. Neither capture nor replay uses a previous frame's solution.
    """
    def __init__(self,objective,*,iterations=60,initial_damping=1e-2,max_step=.35):
        self.objective=objective;self.iterations=iterations;self.initial_damping=initial_damping;self.max_step=max_step
        self.graph=None

    @torch.no_grad()
    def __call__(self,initial,lower,upper):
        if not initial.is_cuda:raise ValueError('CUDA graph LM needs CUDA tensors')
        if self.graph is None:
            self.input=initial.clone();self.lower=lower.clone();self.upper=upper.clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Initialize compiled code, cuBLAS/solver workspaces, allocators.
                _lm_graph_body(self.objective,self.input,self.lower,self.upper,2,self.initial_damping,self.max_step)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs=_lm_graph_body(self.objective,self.input,self.lower,self.upper,self.iterations,self.initial_damping,self.max_step)
        if initial.shape!=self.input.shape:raise ValueError('Recapture is required when batch/sequence shapes change')
        self.input.copy_(initial);self.graph.replay()
        q,initial_cost,cost,accepted,converged,damping=(value.clone() for value in self.outputs)
        if not torch.isfinite(q).all() or not torch.isfinite(cost).all() or torch.any(cost>initial_cost+1e-10):
            raise FloatingPointError('Invalid CUDA graph LM output')
        history=[dict(iteration=self.iterations,mean_cost=float(cost.mean()),max_cost=float(cost.max()),
                      converged=int(converged.sum()),execution='CUDA graph, fixed budget, converged frames frozen')]
        return LMResult(q,initial_cost,cost,accepted,converged,damping,history)


@torch.no_grad()
def bounded_lm(objective,initial,lower,upper,*,iterations=60,initial_damping=1e-2,max_step=.35,trace_every=10):
    """Per-frame trust-region decisions; no frame reads another optimized q."""
    q=initial.clamp(lower,upper).clone();n,d=q.shape
    damping=q.new_full((n,),initial_damping);accepted_count=torch.zeros(n,dtype=torch.long,device=q.device)
    converged=torch.zeros(n,dtype=torch.bool,device=q.device);r,j=objective(q)
    cost=.5*r.square().sum(-1);initial_cost=cost.clone();history=[]
    eye=torch.eye(d,device=q.device,dtype=q.dtype)[None]
    for iteration in range(iterations):
        h=j.transpose(-1,-2)@j;g=(j.transpose(-1,-2)@r[...,None]).squeeze(-1)
        active=((q<=lower+1e-10)&(g>0))|((q>=upper-1e-10)&(g<0))
        free=(~active).to(q.dtype);projected=g*free
        converged|=projected.abs().amax(-1)<1e-7
        work=~converged
        if iteration%trace_every==0 and not work.any():break
        hfree=h*free[:,:,None]*free[:,None,:]
        diagonal=h.diagonal(dim1=-2,dim2=-1).clamp_min(1e-8)
        matrix=hfree+eye*(damping[:,None]*diagonal+(1-free))[...,None,:]
        delta=-torch.linalg.solve_ex(matrix,(g*free)[...,None],check_errors=False).result.squeeze(-1)
        delta=delta.clamp(-max_step,max_step)*work[:,None]
        trial=(q+delta).clamp(lower,upper);actual_step=trial-q
        trial_r,trial_j=objective(trial);trial_cost=.5*trial_r.square().sum(-1)
        predicted=-(g*actual_step).sum(-1)-.5*(actual_step*(h@actual_step[...,None]).squeeze(-1)).sum(-1)
        reduction=cost-trial_cost;ratio=reduction/predicted.clamp_min(1e-30)
        accept=work&torch.isfinite(trial_cost)&(reduction>0)&(predicted>0)
        q=torch.where(accept[:,None],trial,q)
        r=torch.where(accept[:,None],trial_r,r);j=torch.where(accept[:,None,None],trial_j,j)
        old_cost=cost;cost=torch.where(accept,trial_cost,cost)
        damping=torch.where(accept,torch.where(ratio>.75,damping*.3,torch.where(ratio<.25,damping*2,damping)),damping*10).clamp(1e-12,1e12)
        accepted_count+=accept.long()
        converged|=accept&((actual_step.abs().amax(-1)<1e-8)|(reduction<1e-10*(1+old_cost)))
        if (iteration+1)%trace_every==0 or iteration==iterations-1:
            history.append(dict(iteration=iteration+1,mean_cost=float(cost.mean()),max_cost=float(cost.max()),
                                accepted=int(accept.sum()),converged=int(converged.sum())))
    if not torch.isfinite(q).all() or not torch.isfinite(cost).all():raise FloatingPointError('Nonfinite LM result')
    if torch.any(cost>initial_cost+1e-10):raise AssertionError('LM accepted a cost increase')
    return LMResult(q,initial_cost,cost,accepted_count,converged,damping,history)


class HumanPoseObjective:
    def __init__(self,fk,position,rotation,reference,posture_weights):
        self.fk=fk;self.position=position;self.rotation=rotation
        self.reference=reference;self.posture_weights=posture_weights
        self.eye=torch.eye(fk.nq,device=position.device,dtype=position.dtype)

    def __call__(self,q):
        p,r,jp,jw=self.fk.forward(q)
        log=so3_log(r.transpose(-1,-2)@self.rotation)
        jr=-so3_left_inverse(log)@(r.transpose(-1,-2)@jw)
        residual=torch.cat(((p-self.position)*300.,log*5.729577951308232),-1).flatten(1)
        jac=torch.cat((jp*300.,jr*5.729577951308232),-2).flatten(1,2)
        prior=(q-self.reference)*self.posture_weights
        prior_jac=(self.eye*self.posture_weights[:,None])[None].expand(len(q),-1,-1)
        return torch.cat((residual,prior),-1),torch.cat((jac,prior_jac),1)


class RobotRetargetObjective:
    """SDK spatial topology residuals; temporal q_ref and output IIR excluded."""
    def __init__(self,fk,source,laplacian,differential,edges,directions,*,stage=2):
        self.fk=fk;self.s=source;self.l=laplacian;self.d=differential
        self.edges=edges;self.directions=directions;self.stage=stage
        self.row_weights=source.new_ones(21);self.row_weights[1:4]=0
        self.distal=torch.tensor([i for i in range(20) if i%4 in (2,3)],device=source.device)
        u=source[:,3]-source[:,2];v=source[:,4]-source[:,3]
        u=u/u.norm(dim=-1,keepdim=True).clamp_min(1e-8);v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        cross=torch.linalg.cross(u,v);sine=cross.norm(dim=-1);cosine=(u*v).sum(-1)
        self.normal=cross/sine[:,None].clamp_min(1e-8);self.cosine=cosine;self.sine=sine
        t=((cosine-math.cos(.5))/(math.cos(.35)-math.cos(.5))).clamp(0,1)
        self.thumb_weight=(.5*t.square()*(3-2*t)*(sine>1e-8)).sqrt()

    def __call__(self,q):
        p,_,jp,_=self.fk.forward(q);residuals=[];jacobians=[]
        if self.stage==2:
            weight=(150.*self.row_weights).sqrt()[None,:,None]
            residuals.append(((self.l@p-self.d)*weight).flatten(1))
            jacobians.append((torch.einsum('nij,njck->nick',self.l,jp)*weight[...,None]).flatten(1,2))
        a,b=self.edges[:,0],self.edges[:,1];delta=p[:,b]-p[:,a];length=delta.norm(dim=-1).clamp_min(1e-8)
        unit=delta/length[...,None];dj=jp[:,b]-jp[:,a]
        du=(dj-unit[...,None]*(unit[...,None]*dj).sum(-2,keepdim=True))/length[...,None,None]
        weight=math.sqrt(.1 if self.stage==2 else 1.)
        residuals.append(((unit-self.directions)*weight).flatten(1));jacobians.append((du*weight).flatten(1,2))
        u=p[:,3]-p[:,2];v=p[:,4]-p[:,3];lu=u.norm(dim=-1).clamp_min(1e-8);lv=v.norm(dim=-1).clamp_min(1e-8)
        u=u/lu[:,None];v=v/lv[:,None];ju=jp[:,3]-jp[:,2];jv=jp[:,4]-jp[:,3]
        ju=(ju-u[:,:,None]*(u[:,:,None]*ju).sum(-2,keepdim=True))/lu[:,None,None]
        jv=(jv-v[:,:,None]*(v[:,:,None]*jv).sum(-2,keepdim=True))/lv[:,None,None]
        cosine=(u*v).sum(-1);sine=(self.normal*torch.linalg.cross(u,v)).sum(-1)
        jc=(v[:,:,None]*ju+u[:,:,None]*jv).sum(-2)
        js=(torch.linalg.cross(v,self.normal)[:,:,None]*ju+torch.linalg.cross(self.normal,u)[:,:,None]*jv).sum(-2)
        residuals.append(torch.stack((cosine-self.cosine,sine-self.sine),-1)*self.thumb_weight[:,None])
        jacobians.append(torch.stack((jc,js),1)*self.thumb_weight[:,None,None])
        negative=q[:,self.distal].clamp_max(0)
        residuals.append(-math.sqrt(20)*negative)
        antijac=q.new_zeros((len(q),len(self.distal),20))
        antijac[:,torch.arange(len(self.distal),device=q.device),self.distal]=-math.sqrt(20)*(q[:,self.distal]<0).to(q.dtype)
        jacobians.append(antijac)
        return torch.cat(residuals,-1),torch.cat(jacobians,1)

    def copy_targets_(self,other):
        """Refresh graph-bound targets for new input; preserve storage addresses."""
        for name in ('s','l','d','directions','normal','cosine','sine','thumb_weight'):
            getattr(self,name).copy_(getattr(other,name))


class JointAnchorObjective:
    """Add .5*weight*||q-anchor||^2 without coupling frames or changing targets."""
    def __init__(self, base, anchor, weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Anchor weight must be finite and nonnegative')
        self.base, self.anchor = base, anchor
        self.scale = anchor.new_tensor(math.sqrt(weight))
        self.eye = torch.eye(anchor.shape[-1], device=anchor.device, dtype=anchor.dtype)

    def __call__(self, q):
        residual, jacobian = self.base(q)
        prior = (q-self.anchor)*self.scale
        prior_jacobian = (self.eye*self.scale)[None].expand(len(q),-1,-1)
        return torch.cat((residual,prior),-1), torch.cat((jacobian,prior_jacobian),1)


def parallel_sdk_output_filter(q,alpha=.2):
    """Same alpha=.2 IIR convention, evaluated as a parallel linear operator."""
    n=len(q);index=torch.arange(n,device=q.device);difference=index[:,None]-index[None,:]
    weights=alpha*(1-alpha)**difference.clamp_min(0).to(q.dtype)
    weights=weights*(difference>=0);weights[:,0]=(1-alpha)**index.to(q.dtype)
    return weights@q


def prepare_robot_targets(skeleton,alignment):
    """Batched palm alignment, pinch, graph and directions; no frame loop."""
    # Match the public SDK skeleton float32 boundary before spatial retargeting.
    skeleton=skeleton.float().to(skeleton.dtype)
    s=(skeleton-skeleton[:,:1])@alignment
    ids=torch.tensor([8,12,16,20],device=s.device)
    thumb=s[:,4];fingers=s[:,ids];distance=(fingers-thumb[:,None]).norm(dim=-1)
    denominator=1/(1+math.exp(-.009/.0015))-.5
    weights=((torch.sigmoid((.009-distance)/.0015)-.5)/denominator).clamp(0,1)*(distance<.009)
    center=(weights[...,None]*fingers).sum(1)/weights.sum(1)[:,None].clamp_min(1e-30)
    midpoint=(thumb+center)*.5;thumb_weight=1-(1-weights).prod(1)
    source=s.clone();source[:,4]=thumb+thumb_weight[:,None]*(midpoint-thumb)
    source[:,ids]=fingers+weights[...,None]*(midpoint[:,None]-fingers)
    neighbors=s.new_ones(21);neighbors[1:4]=0
    distances=(source[:,:,None]-source[:,None,:]).norm(dim=-1)
    eye=torch.eye(21,device=s.device,dtype=s.dtype)
    adjacency=torch.exp(-50*distances)*neighbors[None,None,:]*(1-eye)
    laplacian=eye-adjacency/adjacency.sum(-1,keepdim=True).clamp_min(1e-30)
    edges=torch.tensor([(a,b) for f in range(1,5) for a,b in zip([0]+list(range(1+4*f,4+4*f)),range(1+4*f,5+4*f))],device=s.device)
    delta=source[:,edges[:,1]]-source[:,edges[:,0]]
    directions=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    return source,laplacian,laplacian@source,edges,directions

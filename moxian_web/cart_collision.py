"""Native signed-distance labels for the current six-branch geometry teacher."""
from contextlib import contextmanager
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _distance(meshes: wp.array(dtype=wp.uint64), points: wp.array(dtype=wp.vec3),
              links: wp.array(dtype=int), distances: wp.array(dtype=float),
              normals: wp.array(dtype=wp.vec3), valid: wp.array(dtype=int)):
    i=wp.tid(); p=points[i]
    hit=wp.mesh_query_point_sign_winding_number(meshes[links[i]],p,2.)
    if hit.result:
        closest=wp.mesh_eval_position(meshes[links[i]],hit.face,hit.u,hit.v)
        delta=p-closest; length=wp.length(delta)
        distances[i]=hit.sign*length
        n=wp.mesh_eval_face_normal(meshes[links[i]],hit.face)
        if length>1.e-7:n=hit.sign*delta/length
        normals[i]=n;valid[i]=1


@contextmanager
def frozen(model):
    flags=[p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():p.requires_grad_(False)
        yield model
    finally:
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)


class MeshOracle:
    def __init__(self,hand):
        self.hand=hand;self.device=hand.device
        self.mesh_ids=hand.collision.mesh_ids
        self.lower=wp.to_torch(hand.collision.lower)
        self.upper=wp.to_torch(hand.collision.upper)
        self.points,self.owners=(wp.to_torch(x) for x in hand.collision.samples[False])
        self.owners=self.owners.long()
        # Equal material samples per link; all mechanical exclusions stay exact.
        selected=[]
        for link in range(21):
            ids=torch.where(self.owners==link)[0]
            selected.extend(ids[torch.linspace(0,len(ids)-1,8,device=ids.device).long()].tolist())
        ids=torch.as_tensor(selected,device=self.device)
        self.surface=self.points[ids];self.surface_owners=self.owners[ids]
        self.surface_by_link=[self.surface[self.surface_owners==i] for i in range(21)]
        cpu_points=self.surface.cpu().tolist();cpu_owners=self.surface_owners.cpu().tolist()
        self.surface_by_link_cpu=[[point for point,owner in zip(cpu_points,cpu_owners) if owner==i] for i in range(21)]
        self.allowed=wp.to_torch(hand.collision.allowed).bool()
        self.collar=wp.to_torch(hand.collision.collar)
        self.query_sources,self.query_targets=self.allowed[self.surface_owners].nonzero(as_tuple=True)

    @torch.no_grad()
    def label(self,points,links):
        points=points.detach().to(device=self.device,dtype=torch.float32).reshape(-1,3).contiguous()
        links=links.detach().to(device=self.device,dtype=torch.int32).reshape(-1).contiguous()
        n=len(points)
        if not n:return points.new_empty(0),points.new_empty((0,3))
        d=wp.zeros(n,dtype=float,device=self.device)
        normals=wp.zeros(n,dtype=wp.vec3,device=self.device);valid=wp.zeros(n,dtype=int,device=self.device)
        wp.launch(_distance,n,inputs=[self.mesh_ids,wp.from_torch(points,dtype=wp.vec3),wp.from_torch(links),d,normals,valid],device=self.device)
        valid=wp.to_torch(valid).bool()
        if not valid.all():raise ValueError('真实 mesh SDF 查询失败，不能作为训练标签')
        return wp.to_torch(d).clone(),wp.to_torch(normals).clone()

    @torch.no_grad()
    def random_queries(self,n):
        links=torch.randint(21,(n,),device=self.device)
        lower,upper=self.lower[links]-.03,self.upper[links]+.03
        points=lower+torch.rand(n,3,device=self.device)*(upper-lower)
        ids=torch.randint(len(self.points),(n,),device=self.device)
        near=torch.rand(n,device=self.device)<.7
        points[near]=self.points[ids[near]]+torch.randn(int(near.sum()),3,device=self.device)*.003
        links[near]=self.owners[ids[near]]
        return points,links

    def self_queries(self,frames):
        p,r=frames[:2];owners=self.surface_owners
        world=(r[:,owners]@self.surface[...,None]).squeeze(-1)+p[:,owners]
        a,b=self.query_sources,self.query_targets
        source=owners[a];points=world[:,a]
        local=(r[:,b].transpose(-1,-2)@(points-p[:,b])[...,None]).squeeze(-1)
        collar=((source==0)&(self.collar[b]>0))[None]&(local.norm(dim=-1)<=self.collar[b][None])
        reverse=((b==0)&(self.collar[source]>0))[None]&(self.surface[a].norm(dim=-1)<=self.collar[source])[None]
        return local,b.expand(len(p),-1),~(collar|reverse)

    def active_clouds(self,frames,rows):
        p=frames[0];clouds=[];counts=[]
        for i,row in enumerate(rows):
            if '_cart_active_points' in row:points=row['_cart_active_points']
            else:
                positions=torch.as_tensor(row['tactile_positions_m'],device=p.device,dtype=p.dtype)
                mask=torch.as_tensor(row['tactile_valid'],device=p.device,dtype=torch.bool)&torch.as_tensor(row['tactile_state'],device=p.device,dtype=torch.bool)
                points=positions[mask]
            clouds.append(points);counts.append(len(points))
        return clouds,counts

    def tactile_queries(self,frames,rows):
        """Only confirmed active taxels; every target link including the palm."""
        local,links,counts=self.flat_tactile_queries(frames,rows)
        return list(zip(local.split(counts),links.split(counts)))

    def flat_tactile_queries(self,frames,rows):
        p,r=frames[:2];clouds,counts=self.active_clouds(frames,rows)
        points=torch.cat(clouds)
        # One FK gather/transform for all ragged point clouds, instead of a
        # sequence of tiny CUDA launches for every window and time step.
        ids=torch.arange(len(rows),device=p.device).repeat_interleave(torch.as_tensor(counts,device=p.device))
        if len(points):local=(r[ids].transpose(-1,-2)@(points[:,None]-p[ids])[...,None]).squeeze(-1)
        else:local=p.new_empty((0,21,3))
        links=torch.arange(21,device=p.device).expand(len(points),-1)
        return local,links,counts

    def sampled_queries(self,frames,rows,budget):
        """Uniform detached label candidates, sampled before dense expansion."""
        p,r=frames[:2];a,b=self.query_sources,self.query_targets
        flat=torch.randint(len(p)*len(a),(2*budget,),device=p.device)
        pose=flat//len(a);selected=flat%len(a);source=a[selected];target=b[selected]
        owners=self.surface_owners[source];material=self.surface[source]
        world=(r[pose,owners]@material[...,None]).squeeze(-1)+p[pose,owners]
        local=(r[pose,target].transpose(-1,-2)@(world-p[pose,target])[...,None]).squeeze(-1)
        collar=(owners==0)&(self.collar[target]>0)&(local.norm(dim=-1)<=self.collar[target])
        reverse=(target==0)&(self.collar[owners]>0)&(material.norm(dim=-1)<=self.collar[owners])
        valid=~(collar|reverse)
        result=[(local[valid][:budget],target[valid][:budget])]
        clouds,counts=self.active_clouds(frames,rows);points=torch.cat(clouds)
        if len(points):
            # Sampling a point and target independently is uniform over the
            # same all-link query set used by the penetration objective.
            ids=torch.randint(len(points),(budget,),device=p.device)
            poses=torch.searchsorted(torch.as_tensor(counts,device=p.device).cumsum(0),ids,right=True)
            links=torch.randint(21,(budget,),device=p.device)
            local=(r[poses,links].transpose(-1,-2)@(points[ids]-p[poses,links])[...,None]).squeeze(-1)
            result.append((local,links))
        return result

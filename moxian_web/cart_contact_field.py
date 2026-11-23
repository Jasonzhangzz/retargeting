"""Continuous contact correspondences; native facets are supervision only."""
import numpy as np
import torch
import warp as wp
from torch import nn
from torch.nn import functional as F

CONTACT_FIELD_VERSION = 'directional_surface_field_v1'


@wp.kernel(enable_backward=False)
def _projection_derivative(skins:wp.array(dtype=wp.uint64),solids:wp.array(dtype=wp.uint64),
        points:wp.array(dtype=wp.vec3),links:wp.array(dtype=int),palmar:wp.array(dtype=int),
        probes:wp.array(dtype=wp.vec3),derivatives:wp.array(dtype=wp.vec3),valid:wp.array(dtype=int)):
    i=wp.tid();mesh=solids[links[i]]
    if palmar[i]==1:mesh=skins[links[i]]
    hit=wp.mesh_query_point_no_sign(mesh,points[i],2.)
    if hit.result:
        a=wp.mesh_eval_position(mesh,hit.face,1.,0.)
        b=wp.mesh_eval_position(mesh,hit.face,0.,1.)
        c=wp.mesh_eval_position(mesh,hit.face,0.,0.)
        u=hit.u;v=hit.v;w=1.-u-v
        du=int(u>1.e-5);dv=int(v>1.e-5);dw=int(w>1.e-5)
        count=du+dv+dw;probe=probes[i];out=wp.vec3(0.)
        if count==3:
            n=wp.mesh_eval_face_normal(mesh,hit.face)
            out=probe-n*wp.dot(n,probe)
        elif count==2:
            edge=b-a
            if du==0:edge=c-b
            elif dv==0:edge=c-a
            out=edge*(wp.dot(edge,probe)/wp.max(wp.dot(edge,edge),1.e-20))
        derivatives[i]=out;valid[i]=1


class ContactField(nn.Module):
    """Predict a local surface displacement and outward normal, without selection.

    The escape direction specifies the intended side, including deep inside
    queries. A fixed semantic flag distinguishes palmar pads from full meshes.
    A zero direction requests the ordinary nearest point, with no side retraction.
    Link embeddings are shared with the signed distance head.
    """
    def __init__(self,width=128,depth=3):
        super().__init__()
        layers=[nn.Linear(23,width),nn.Softplus(beta=10)]
        for _ in range(depth-1):layers.extend((nn.Linear(width,width),nn.Softplus(beta=10)))
        self.network=nn.Sequential(*layers,nn.Linear(width,6))

    def forward(self,point,direction,palmar,embedding):
        direction=F.normalize(direction,dim=-1,eps=1.e-8)
        values=self.network(torch.cat((point/.05,direction,palmar.to(point.dtype)[...,None],embedding),-1))
        hit=point+.05*values[...,:3]
        # Smooth normalization remains defined even at a vanishing prediction.
        normal=direction+values[...,3:]
        normal=normal/torch.sqrt(normal.square().sum(-1,keepdim=True)+1.e-12)
        return hit,normal


@wp.kernel(enable_backward=False)
def _contact_labels(skins: wp.array(dtype=wp.uint64), solids: wp.array(dtype=wp.uint64),
                    points: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
                    links: wp.array(dtype=int), palmar: wp.array(dtype=int),
                    hits: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3), valid: wp.array(dtype=int)):
    i=wp.tid();link=links[i];skin=solids[link]
    if palmar[i] == 1:skin=skins[link]
    point=points[i];direction=directions[i]
    if wp.length(direction)<1.e-8:
        nearest=wp.mesh_query_point_no_sign(skin,point,2.)
        if nearest.result:
            hits[i]=wp.mesh_eval_position(skin,nearest.face,nearest.u,nearest.v)
            normals[i]=wp.mesh_eval_face_normal(skin,nearest.face)
            valid[i]=1
        return
    direction=wp.normalize(direction)
    nearest=wp.mesh_query_point_no_sign(skin,point,2.)
    signed=wp.mesh_query_point_sign_winding_number(solids[link],point,2.)
    if not nearest.result or not signed.result:return
    hit=wp.mesh_eval_position(skin,nearest.face,nearest.u,nearest.v)
    normal=wp.mesh_eval_face_normal(skin,nearest.face)
    probe=point
    solid_hit=wp.mesh_eval_position(solids[link],signed.face,signed.u,signed.v)
    if (signed.sign<0. and wp.length(point-solid_hit)>.0005) or wp.dot(normal,direction)<=.1:
        # Trace in the trusted direction: outside wrong-side queries enter and
        # then leave; inside queries leave immediately. Never follow a back face.
        ray=wp.mesh_query_ray(solids[link],point,direction,2.)
        if not ray.result:return
        probe=point+(ray.t+.00001)*direction
        if wp.dot(ray.normal,direction)<=.1:
            ray=wp.mesh_query_ray(solids[link],probe,direction,2.)
            if not ray.result or wp.dot(ray.normal,direction)<=.1:return
            probe=probe+(ray.t+.00001)*direction
        outside=wp.mesh_query_point_sign_winding_number(solids[link],probe,2.)
        if not outside.result or outside.sign<0.:return
        nearest=wp.mesh_query_point_no_sign(skin,probe,2.)
        if not nearest.result:return
        hit=wp.mesh_eval_position(skin,nearest.face,nearest.u,nearest.v)
        normal=wp.mesh_eval_face_normal(skin,nearest.face)
        if wp.length(probe-hit)>.0005:return
    if wp.dot(normal,direction)<=.1:return
    hits[i]=hit;normals[i]=normal;valid[i]=1


class ContactTeacher:
    """Direction-preserving real-mesh labels, never part of a policy graph."""
    def __init__(self,hand):
        self.hand=hand;self.skins=list(hand.full_meshes);self.pad_owners=[];clouds=[];normals=[];owners=[]
        # Only anatomical contact links have palmar semantic surfaces.
        for owner in [0]+[2+4*f+k for f in range(5) for k in range(3)]:
            surface=hand.surface.contact_surface(owner);tri=np.asarray(surface['triangles'],np.float32)
            if not len(tri):continue
            if owner not in hand.pad_meshes:
                hand.pad_meshes[owner]=wp.Mesh(points=wp.array(tri.reshape(-1,3),dtype=wp.vec3,device=hand.device),
                    indices=wp.array(np.arange(tri.size//3),dtype=int,device=hand.device))
            self.skins[owner]=hand.pad_meshes[owner];self.pad_owners.append(owner)
            clouds.append(hand.tensor(tri.mean(1)));normals.append(hand.tensor(surface['normals']))
            owners.append(torch.full((len(tri),),owner,device=hand.device,dtype=torch.long))
        self.mesh_ids=wp.array([mesh.id for mesh in self.skins],dtype=wp.uint64,device=hand.device)
        self.points=torch.cat(clouds);self.normals=torch.cat(normals);self.owners=torch.cat(owners)

    @torch.no_grad()
    def label(self,points,directions,links,palmar):
        device=self.hand.device;n=len(points)
        points=points.detach().to(device=device,dtype=torch.float32).contiguous()
        directions=directions.detach().to(device=device,dtype=torch.float32).contiguous()
        links=links.to(device=device,dtype=torch.int32).contiguous();palmar=palmar.to(device=device,dtype=torch.int32).contiguous()
        hits=wp.zeros(n,dtype=wp.vec3,device=device);normals=wp.zeros(n,dtype=wp.vec3,device=device);valid=wp.zeros(n,dtype=int,device=device)
        if n:wp.launch(_contact_labels,n,inputs=[self.mesh_ids,self.hand.collision.mesh_ids,
            wp.from_torch(points,dtype=wp.vec3),wp.from_torch(directions,dtype=wp.vec3),
            wp.from_torch(links),wp.from_torch(palmar),hits,normals,valid],device=device)
        return wp.to_torch(hits).clone(),wp.to_torch(normals).clone(),wp.to_torch(valid).bool().clone()

    @torch.no_grad()
    def projection_derivative(self,points,links,palmar,probes):
        """Exact local Jacobian on the central nearest feature, label-only.

        Finite differences of a nonconvex surface can cross two different
        nearest features with identical normals; they are not Jacobian labels.
        """
        n=len(points);device=self.hand.device
        x=points.detach().to(device=device,dtype=torch.float32).contiguous()
        probes=probes.detach().to(device=device,dtype=torch.float32).contiguous()
        links=links.to(device=device,dtype=torch.int32).contiguous();palmar=palmar.to(device=device,dtype=torch.int32).contiguous()
        values=wp.zeros(n,dtype=wp.vec3,device=device);valid=wp.zeros(n,dtype=int,device=device)
        if n:wp.launch(_projection_derivative,n,inputs=[self.mesh_ids,self.hand.collision.mesh_ids,
            wp.from_torch(x,dtype=wp.vec3),wp.from_torch(links),wp.from_torch(palmar),wp.from_torch(probes,dtype=wp.vec3),values,valid],device=device)
        return wp.to_torch(values).clone(),wp.to_torch(valid).bool().clone()

    @torch.no_grad()
    def random_queries(self,oracle,n):
        # Half full-mesh and half semantic-pad examples, with their true normals.
        full=n//2;ids=torch.randint(len(oracle.points),(full,),device=self.hand.device)
        points=oracle.points[ids];links=oracle.owners[ids]
        _,normal=oracle.label(points,links)
        ids=torch.randint(len(self.points),(n-full,),device=self.hand.device)
        points=torch.cat((points,self.points[ids]));links=torch.cat((links,self.owners[ids]))
        normal=torch.cat((normal,self.normals[ids]))
        direction=F.normalize(normal+.15*torch.randn_like(normal),dim=-1)
        # Include deep crossings; direction, rather than the closest facet,
        # supplies the intended side. Global/far points are added by rollouts.
        points=points+normal*torch.randn(n,1,device=points.device)*.006+torch.randn_like(points)*.001
        palmar=torch.arange(n,device=points.device)>=full
        return points,direction,links,palmar

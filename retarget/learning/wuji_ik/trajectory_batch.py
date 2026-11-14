"""Explicit parallel trial axis; avoids vmap/index-add capture ambiguities."""
import copy
import torch


class BatchedTrajectoryObjective:
    def __init__(self,objective,trials,compile=True):
        self.obj=objective;self.k=trials;raw=getattr(objective.spatial,'raw',objective.spatial);self.raw=copy.copy(raw);self.fields=[]
        if hasattr(raw,'base'):
            self.raw.base=copy.copy(raw.base)
            for name in ['position','rotation']:self.clone_field(self.raw.base,raw.base,name)
        elif hasattr(raw,'s'):
            for name in ['s','l','d','directions','normal','cosine','sine','thumb_weight']:self.clone_field(self.raw,raw,name)
        elif hasattr(raw,'target'):self.clone_field(self.raw,raw,'target')
        else:raise TypeError('No explicit trial expansion for '+type(raw).__name__)
        self.fn=torch.compile(self.compute,fullgraph=True,dynamic=False) if compile else self.compute

    def clone_field(self,dest,source,name):
        original=getattr(source,name);expanded=original.repeat((self.k,)+(1,)*(original.ndim-1));setattr(dest,name,expanded);self.fields.append((expanded,original))

    def refresh(self):
        for dest,original in self.fields:dest.copy_(original.repeat((self.k,)+(1,)*(original.ndim-1)))

    def compute(self,q,gradient):
        obj=self.obj;k,n,d=q.shape;r,j=self.raw(q.reshape(k*n,d));r=r.reshape(k,n,-1)
        previous=q[:,obj.previous].clone();previous[:,obj.starts]=obj.reference;delta=q-previous
        values=.5*r.square().sum(-1)+.5*obj.weight*delta.square().sum(-1)
        cost=q.new_zeros((k,len(obj.starts))).scatter_add(1,obj.segment[None].expand(k,-1),values)
        if not gradient:return cost,None
        j=j.reshape(k,n,-1,d);g=(j*r[...,None]).sum(-2)+obj.weight*delta
        correction=torch.cat((delta[:,1:]*obj.link[None,1:,None],torch.zeros_like(delta[:,:1])),1)
        return cost,g-obj.weight*correction

    def __call__(self,q,gradient=False):return self.fn(q,gradient)

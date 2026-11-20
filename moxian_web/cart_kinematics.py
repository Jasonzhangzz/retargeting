"""Exact fixed-wrist Wuji2 FK with the five fingers evaluated together."""
import torch


class FixedWristFK:
    def __init__(self,reference,body_ids):
        self.reference=reference;self.ids=body_ids
        model=reference.model;device=reference.device
        self.moving=torch.tensor(body_ids[1:],device=device,dtype=torch.long)
        if len(body_ids)!=21:raise ValueError('CART FK requires the native five four-joint chains')
        moving=body_ids[1:];tips=[]
        for f in range(5):
            chain=moving[4*f:4*f+4]
            if [reference.parents[b] for b in chain]!=[body_ids[0],*chain[:-1]]:
                raise ValueError('Unexpected Wuji2 finger topology')
            if any(len(reference.joints[b])!=1 or reference.kinds[reference.joints[b][0]]!=3 for b in chain):
                raise ValueError('CART FK requires one native hinge per finger link')
            children=[b for b,p in enumerate(reference.parents) if p==chain[-1]]
            if len(children)!=1 or reference.joints[children[0]]:raise ValueError('Unexpected Wuji2 fixed fingertip')
            tips.append(children[0])
        joints=[reference.joints[b][0] for b in moving]
        if [reference.addresses[j] for j in joints]!=list(range(6,26)):
            raise ValueError('Unexpected Wuji2 joint ordering')
        # Prove that every other body is fixed below the fixed root or palm.
        known=set(range(body_ids[0]+1))|set(moving)|set(tips)
        for b in range(1,model.nbody):
            if b not in known and (reference.parents[b]!=body_ids[0] or reference.joints[b]):
                raise ValueError('Unaccounted moving body in Wuji2 FK')
        self.tips=torch.tensor(tips,device=device,dtype=torch.long)
        with torch.no_grad():
            self.base_p,self.base_r=[v.detach() for v in reference.frames(torch.zeros(1,model.nq,device=device,dtype=reference.dtype))]
        self.position=reference.position[self.moving];self.rotation=reference.rotation[self.moving]
        self.pivot=reference.pivot[joints];axis=reference.axis[joints]
        a,b,c=axis.unbind(-1);z=torch.zeros_like(a)
        self.cross=torch.stack((z,-c,b,c,z,-a,-b,a,z),-1).reshape(20,3,3)
        self.cross2=self.cross@self.cross;self.eye=torch.eye(3,device=device,dtype=reference.dtype)
        self.zero=reference.reference[6:26]
        self.tip_p=reference.position[self.tips];self.tip_r=reference.rotation[self.tips]

    def frames(self,q):
        x=q-self.zero
        turn=self.eye+x.sin()[...,None,None]*self.cross+(1-x.cos())[...,None,None]*self.cross2
        local_p=self.position+(self.rotation@(self.pivot-(turn@self.pivot[...,None]).squeeze(-1))[...,None]).squeeze(-1)
        local_r=self.rotation@turn
        local_p=local_p.reshape(len(q),5,4,3);local_r=local_r.reshape(len(q),5,4,3,3)
        p=self.base_p[:,self.ids[0],None].expand(len(q),5,3)
        r=self.base_r[:,self.ids[0],None].expand(len(q),5,3,3)
        positions=[];rotations=[]
        for depth in range(4):
            p=p+(r@local_p[:,:,depth,...,None]).squeeze(-1)
            r=r@local_r[:,:,depth];positions.append(p);rotations.append(r)
        all_p=torch.stack(positions,2).reshape(len(q),20,3)
        all_r=torch.stack(rotations,2).reshape(len(q),20,3,3)
        full_p=self.base_p.expand(len(q),-1,-1).index_copy(1,self.moving,all_p)
        full_r=self.base_r.expand(len(q),-1,-1,-1).index_copy(1,self.moving,all_r)
        tip_p=p+(r@self.tip_p[...,None]).squeeze(-1);tip_r=r@self.tip_r
        return full_p.index_copy(1,self.tips,tip_p),full_r.index_copy(1,self.tips,tip_r)

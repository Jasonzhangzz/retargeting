"""Causal CART network, separate from unmodified official GeoRT models."""
import torch
from torch import nn
from .cart_protocol import ACTION_COUNT

class CartNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(84,256),nn.LeakyReLU(),nn.LayerNorm(256),
                                   nn.Linear(256,256),nn.LeakyReLU())
        self.memory=nn.GRU(256,128,batch_first=True)
        self.action=nn.Linear(128,ACTION_COUNT);self.phase=nn.Linear(128,4)
        self.decoder=nn.Sequential(nn.Linear(128+ACTION_COUNT+4+30,256),nn.LeakyReLU(),nn.Linear(256,256),nn.LeakyReLU())
        self.angles=nn.Linear(256,20);self.pose=nn.Linear(256,9)
        nn.init.zeros_(self.pose.weight);nn.init.zeros_(self.pose.bias)
        with torch.no_grad():self.pose.bias[3]=1;self.pose.bias[7]=1
    def encode(self,x,hidden=None):
        context,hidden=self.memory(self.encoder(x),hidden)
        return context,self.action(context),self.phase(context),hidden
    def decode(self,context,action,phase,hull):
        z=self.decoder(torch.cat((context,action,phase,hull),-1))
        q=self.angles(z).tanh();p=self.pose(z)
        a=torch.nn.functional.normalize(p[...,3:6],dim=-1,eps=1e-6)
        b=p[...,6:9]-(p[...,6:9]*a).sum(-1,keepdim=True)*a
        b=torch.nn.functional.normalize(b,dim=-1,eps=1e-6)
        rotation=torch.stack((a,b,torch.linalg.cross(a,b)),-1)
        return q,hull[...,:3]+p[...,:3].tanh()*.15,rotation
    def forward(self,x,hull,action=None,phase=None,hidden=None):
        context,logits,phase_logits,hidden=self.encode(x,hidden)
        q,t,r=self.decode(context,logits.softmax(-1) if action is None else action,
                          phase_logits.softmax(-1) if phase is None else phase,hull)
        return dict(q=q,translation=t,rotation=r,action=logits,phase=phase_logits,hidden=hidden)

def hull_descriptor(target):
    if not target.get('ready'):return [0.]*30
    return [*target['center_m'],*target['support_m'],1.]

"""Independent closure oracle distinguishes rank loss from positive preload."""
import numpy as np
from scripts.audit_contact_hold import closure


def test_closure_requires_full_wrench_rank_and_detects_a_local_repair():
    points=np.array([[-.05,0,0],[.05,0,0],[-.05,0,0]])
    normals=np.array([[-1.,0,0],[1.,0,0],[-1.,0,0]])
    args=(normals,np.full((3,2),.6),np.ones(3,bool),np.zeros(3),.1)
    before=closure(points,*args)
    assert before['positive_margin']>0 and before['rank']==5
    assert not before['force_closure']
    points[2,1]=.001
    after=closure(points,*args)
    assert after['force_closure'] and after['rank']==6


def test_all_upward_contacts_can_balance_gravity_without_force_closure():
    points=np.array([[0.,0,0],[.01,0,0],[0,.01,0]])
    normals=np.tile([0.,0.,-1.],(3,1))
    result=closure(points,normals,np.full((3,2),.6),np.ones(3,bool),np.zeros(3),.1)
    assert not result['force_closure']

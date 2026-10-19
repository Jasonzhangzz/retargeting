"""Selected contact basins survive MPC shifts, with correspondence checks."""
import numpy as np
import warp as wp
from retarget.contact_memory import allocate,remember,restore


def test_contact_memory_shifts_selected_start_and_resets_invalid_correspondences():
    device='cpu';stages=3;contacts=5
    memory=allocate(stages,contacts,device)
    array=lambda x,t:wp.array(x,dtype=t,device=device)
    local=array(np.zeros((contacts,3)),wp.vec3)
    body=array(np.arange(contacts)+1,int)
    mask=array(np.ones(contacts),float);weight=array(np.ones(contacts),float)
    for stage in range(stages):
        values=np.zeros((2,contacts,3),np.float32)
        values[1,:,0]=.001*(stage+1)
        c=array(values,wp.vec3);dual=array(values*.1,wp.vec3)
        wp.launch(remember,contacts,inputs=[c,dual,local,body,mask,weight,1,stage,1,memory],device=device)
    shape=(2,contacts,3)
    anchor=np.zeros((contacts,3),np.float32);anchor[3,0]=.02
    c=array(np.broadcast_to(anchor,shape).copy(),wp.vec3)
    z=array(np.broadcast_to(anchor,shape).copy(),wp.vec3)
    last=array(np.broadcast_to(anchor,shape).copy(),wp.vec3)
    dual=array(np.zeros(shape),wp.vec3)
    # slot 1 changed body, slot 2 changed local point, slot 3 outside radius,
    # slot 4 inactive: only slot 0 may inherit the previous selected solution.
    body.assign(np.array([1,99,3,4,5],np.int32))
    new_local=np.zeros((contacts,3),np.float32);new_local[2,0]=.001
    local.assign(new_local);mask.assign(np.array([1,1,1,1,0],np.float32))
    wp.launch(restore,(2,contacts),inputs=[c,z,last,dual,array(anchor,wp.vec3),local,body,
        mask,weight,1,0,1,.005,memory],device=device)
    expected=np.broadcast_to(anchor,shape).copy();expected[:,0,0]=.002
    for output in (c,z,last):np.testing.assert_allclose(output.numpy(),expected,atol=1e-9)
    expected_dual=np.zeros(shape);expected_dual[:,0,0]=.0002
    np.testing.assert_allclose(dual.numpy(),expected_dual,atol=1e-9)

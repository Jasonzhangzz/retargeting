from types import SimpleNamespace
import numpy as np
import pytest
from retarget.env.grasp.native_model_assets import body_permutation


def test_serialized_mocap_reordering_preserves_body_and_equality_ownership():
    names=['world','hand','object','mocap']
    model=SimpleNamespace(nbody=4,body=lambda i:SimpleNamespace(name=names[i]),
        body_parentid=np.zeros(4,dtype=int),geom_bodyid=np.array([1,2]),jnt_bodyid=np.array([1]),
        jnt_type=np.array([3]),jnt_qposadr=np.array([0]),jnt_dofadr=np.array([0]),geom_dataid=np.array([0,1]),
        eq_obj1id=np.array([3]),eq_obj2id=np.array([1]))
    frozen={k:np.array(getattr(model,k)) for k in ['body_parentid','jnt_bodyid','jnt_type','jnt_qposadr','jnt_dofadr','geom_dataid','eq_obj2id']}
    frozen.update(geom_bodyid=np.array([1,3]),eq_obj1id=np.array([2]),eq_objtype=np.array([1]))
    permutation,index=body_permutation(model,frozen,{'body_names':['world','hand','mocap','object']})
    np.testing.assert_array_equal(permutation,[0,1,3,2])
    old_mass=np.array([0.,2.,0.,3.])
    np.testing.assert_array_equal(old_mass[permutation],[0.,2.,3.,0.])
    assert index['object']==2
    model.eq_obj1id[0]=2
    with pytest.raises(ValueError,match='equality body binding'):
        body_permutation(model,frozen,{'body_names':['world','hand','mocap','object']})

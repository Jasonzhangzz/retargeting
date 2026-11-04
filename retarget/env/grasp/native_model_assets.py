"""Load frozen native 3.3.3 physical assets into the isolated mjwarp runtime."""
from pathlib import Path
import json
import numpy as np
import mujoco


PHYSICAL_FIELDS={
    'body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat','body_gravcomp',
    'jnt_axis','jnt_pos','jnt_range','jnt_margin','jnt_solimp','jnt_solref','jnt_stiffness',
    'geom_pos','geom_quat','geom_size','geom_solimp','geom_solref','geom_margin','geom_gap',
    'geom_friction','geom_condim','geom_contype','geom_conaffinity','eq_data','eq_solimp','eq_solref',
    'dof_frictionloss','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','actuator_gear',
    'actuator_ctrlrange','actuator_forcerange','qpos0','qpos_spring'}


def body_permutation(model,frozen,metadata):
    old_names=metadata['body_names'];new_names=[model.body(i).name for i in range(model.nbody)]
    if len(set(old_names))!=len(old_names) or set(old_names)!=set(new_names):
        raise ValueError('Compiled body identities changed')
    old_index={n:i for i,n in enumerate(old_names)};new_index={n:i for i,n in enumerate(new_names)}
    new_to_old=np.array([old_index[n] for n in new_names])
    old_to_new=np.array([new_index[n] for n in old_names])
    np.testing.assert_array_equal(model.body_parentid,old_to_new[frozen['body_parentid']][new_to_old])
    for name in ('geom_bodyid','jnt_bodyid'):
        np.testing.assert_array_equal(getattr(model,name),old_to_new[frozen[name]])
    for name in ('jnt_type','jnt_qposadr','jnt_dofadr','geom_dataid'):
        np.testing.assert_array_equal(getattr(model,name),frozen[name])
    for i,kind in enumerate(frozen['eq_objtype']):
        if kind!=int(mujoco.mjtObj.mjOBJ_BODY):raise ValueError('Unverified non-body equality binding')
        for name in ('eq_obj1id','eq_obj2id'):
            if getattr(model,name)[i]!=old_to_new[frozen[name][i]]:
                raise ValueError('Compiled equality body binding changed')
    return new_to_old,new_index


def load_model(root,stem):
    root=Path(root);model=mujoco.MjModel.from_xml_path(str(root/(stem+'.xml')))
    metadata=json.loads((root/(stem+'.json')).read_text())
    frozen=dict(np.load(root/(stem+'.npz')))
    permutation,body_index=body_permutation(model,frozen,metadata)
    metadata['object_body']=body_index[metadata['body_names'][metadata['object_body']]]
    for category,values in metadata['flags'].items():
        own=mujoco.mjtDisableBit if category=='disable' else mujoco.mjtEnableBit
        other=mujoco.mjtEnableBit if category=='disable' else mujoco.mjtDisableBit
        prefix='mjDSBL_' if category=='disable' else 'mjENBL_'
        other_prefix='mjENBL_' if category=='disable' else 'mjDSBL_'
        attr='disableflags' if category=='disable' else 'enableflags'
        opposite='enableflags' if category=='disable' else 'disableflags'
        for name,on in values.items():
            if hasattr(own,prefix+name):
                bit=int(getattr(own,prefix+name));old=int(getattr(model.opt,attr))
                setattr(model.opt,attr,old|bit if on else old&~bit)
            elif hasattr(other,other_prefix+name):
                bit=int(getattr(other,other_prefix+name));old=int(getattr(model.opt,opposite))
                setattr(model.opt,opposite,old&~bit if on else old|bit)
            elif on:raise ValueError('Missing native option '+name)
    for key,value in metadata['numeric_options'].items():
        if isinstance(value,list):getattr(model.opt,key)[:]=value
        else:setattr(model.opt,key,value)
    for name in PHYSICAL_FIELDS:
        if name not in frozen:continue
        target=getattr(model,name)
        if name=='eq_data' and target.shape==(len(frozen[name]),12) and frozen[name].shape==(len(frozen[name]),11):
            if not np.all(model.eq_type==mujoco.mjtEq.mjEQ_WELD):
                raise ValueError('Only native body-weld eq_data layout migration has been verified')
            # Weld anchors, relative quaternion and torquescale remain at 0:11.
            # 3.17 adds a final field; retain the compiler value for that field.
            target[:,:11]=frozen[name]
            continue
        if target.shape!=frozen[name].shape:raise ValueError('Physical field layout changed: '+name)
        target[:]=frozen[name][permutation] if name.startswith('body_') else frozen[name]
    model.body_simple[:]=0;model.dof_simplenum[:]=0
    model.body_sameframe[:]=0;model.geom_sameframe[:]=0
    mujoco.mj_setConst(model,mujoco.MjData(model))
    return model,frozen,metadata

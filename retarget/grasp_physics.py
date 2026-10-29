"""Resolve grasp contact inputs from run.py's reference scene, with overrides."""
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT/'data/example/sequence/scene/wuji_right/sugar_box_sdf.xml'


@lru_cache(maxsize=8)
def reference_physics(path):
    """Compile only the CPU collision XML; do not allocate a reference GPU model."""
    from comfree_warp.gaussian_loader import _preprocess_xml
    prepared = _preprocess_xml(str(Path(path).resolve()))
    model = mujoco.MjModel.from_xml_string(prepared.collision_xml,assets=prepared.assets)
    joint = model.joint('obj_joint').id
    body = model.jnt_bodyid[joint]
    ids = np.flatnonzero((model.geom_bodyid==body)&((model.geom_contype|model.geom_conaffinity)!=0))
    if len(ids)!=1:
        raise ValueError('Physics reference must have one colliding object geom')
    geom = ids[0]
    if model.opt.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError('The shared native engine requires pyramidal friction cones')
    return dict(impratio=float(model.opt.impratio),friction=model.geom_friction[geom].tolist(),
                solref=model.geom_solref[geom].tolist(),solimp=model.geom_solimp[geom].tolist(),
                condim=int(model.geom_condim[geom]))


def resolve_physics_args(args):
    explicit_reference = args.physics_reference is not None
    args.physics_reference = (args.physics_reference or DEFAULT_REFERENCE).expanduser().resolve()
    reference = reference_physics(str(args.physics_reference))
    overrides = dict(reference) if explicit_reference else {}
    for field,key in (('friction','friction'),('impratio','impratio'),
                      ('contact_solref','solref'),('contact_solimp','solimp')):
        value = getattr(args,field)
        if value is None:
            value = reference[key][0] if key=='friction' else reference[key]
            # Five fingertip contacts need enough tangential impedance to
            # suppress regularization creep after a real preload is built.
            # Archived replays and explicit reference scenes retain theirs.
            if (key=='impratio' and getattr(args,'load_conditioning',False)
                    and not getattr(args,'replay',None) and not explicit_reference):
                value = 10.
        else:
            overrides[key] = [value,*reference['friction'][1:]] if key=='friction' else value
        setattr(args,field,value)
    args.physics_overrides = overrides


def apply_object_physics(tree,values):
    """Apply before loading the GPU model so compiled contact rows agree."""
    root = tree.getroot() if hasattr(tree,'getroot') else tree
    option = root.find('option')
    if 'impratio' in values:
        option.set('impratio',str(values['impratio']))
    obj = root.find('.//body[@name="grasp_object"]')
    geom = obj.find('./geom[@name="object_collision"]')
    for name in ('friction','solref','solimp'):
        if name in values:
            geom.set(name,' '.join(map(str,values[name])))
    if 'condim' in values:
        geom.set('condim',str(values['condim']))


def physics_snapshot(cpu,device,index):
    geom = cpu.geom('object_collision').id
    collisions = device.gaussian_collision
    batches = collisions if isinstance(collisions,tuple) else (collisions,)
    return dict(impratio=float(cpu.opt.impratio),
        gpu_impratio_invsqrt=device.opt.impratio_invsqrt.numpy().tolist(),
        object_mass_kg=float(cpu.body_mass[index.object_body]),
        object_inertia_kg_m2=cpu.body_inertia[index.object_body].tolist(),
        finger_kp=cpu.actuator_gainprm[index.hand_ctrl[6:],0].tolist(),
        finger_kv=(-cpu.actuator_biasprm[index.hand_ctrl[6:],2]).tolist(),
        finger_torque_limits_Nm=cpu.actuator_forcerange[index.hand_ctrl[6:]].tolist(),
        friction=cpu.geom_friction[geom].tolist(),solref=cpu.geom_solref[geom].tolist(),
        solimp=cpu.geom_solimp[geom].tolist(),condim=int(cpu.geom_condim[geom]),
        compiled_contact_friction=np.unique(np.concatenate([b.contact_friction.numpy() for b in batches]),axis=0).tolist(),
        compiled_contact_solref=np.unique(np.concatenate([b.contact_solref.numpy() for b in batches]),axis=0).tolist(),
        compiled_contact_solimp=np.unique(np.concatenate([b.contact_solimp.numpy() for b in batches]),axis=0).tolist())

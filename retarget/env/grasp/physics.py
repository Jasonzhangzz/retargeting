"""Resolve grasp contact inputs from run.py's reference scene, with overrides."""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from retarget.config.run import ROOT

DEFAULT_REFERENCE = (
    ROOT / "data/dexycb/prepared/sugar_box/sequence/scene/wuji_right/sugar_box_sdf.xml"
)


@lru_cache(maxsize=8)
def reference_physics(path):
    """Compile only the CPU collision XML; do not allocate a reference GPU model."""
    model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
    joint = model.joint("obj_joint").id
    body = model.jnt_bodyid[joint]
    ids = np.flatnonzero(
        (model.geom_bodyid == body)
        & ((model.geom_contype | model.geom_conaffinity) != 0)
    )
    if len(ids) != 1:
        raise ValueError("Physics reference must have one colliding object geom")
    geom = ids[0]
    if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError("The shared native engine requires pyramidal friction cones")
    return dict(
        impratio=float(model.opt.impratio),
        friction=model.geom_friction[geom].tolist(),
        solref=model.geom_solref[geom].tolist(),
        solimp=model.geom_solimp[geom].tolist(),
        condim=int(model.geom_condim[geom]),
    )


def apply_object_physics(tree, values):
    """Apply before loading the GPU model so compiled contact rows agree."""
    root = tree.getroot() if hasattr(tree, "getroot") else tree
    option = root.find("option")
    if "impratio" in values:
        option.set("impratio", str(values["impratio"]))
    obj = root.find('.//body[@name="grasp_object"]')
    geom = obj.find('./geom[@name="object_collision"]')
    for name in ("friction", "solref", "solimp"):
        if name in values:
            geom.set(name, " ".join(map(str, values[name])))
    if "condim" in values:
        geom.set("condim", str(values["condim"]))


def physics_snapshot(cpu, device, index):
    geom = cpu.geom("object_collision").id
    return dict(
        impratio=float(cpu.opt.impratio),
        gpu_impratio_invsqrt=device.opt.impratio_invsqrt.numpy().tolist(),
        object_mass_kg=float(cpu.body_mass[index.object_body]),
        object_inertia_kg_m2=cpu.body_inertia[index.object_body].tolist(),
        finger_kp=cpu.actuator_gainprm[index.hand_ctrl[6:], 0].tolist(),
        finger_kv=(-cpu.actuator_biasprm[index.hand_ctrl[6:], 2]).tolist(),
        finger_torque_limits_Nm=cpu.actuator_forcerange[index.hand_ctrl[6:]].tolist(),
        friction=cpu.geom_friction[geom].tolist(),
        solref=cpu.geom_solref[geom].tolist(),
        solimp=cpu.geom_solimp[geom].tolist(),
        condim=int(cpu.geom_condim[geom]),
    )

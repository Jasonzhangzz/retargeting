"""Environment parameters and task-query layout, independent of ComFree law."""
import numpy as np
from retarget.env.grasp.hand import HAND_BODIES


def _configure_comfree_physics(cfg, cpu_model, device_model, *, timestep):
    cpu_model.opt.timestep = timestep
    device_model.opt.timestep.fill_(timestep)


def _collision_output_layout(cpu_model):
    bodies = np.array(sorted(cpu_model.body(name).id for name in HAND_BODIES), np.int32)
    return bodies, np.arange(len(bodies), dtype=np.int32)

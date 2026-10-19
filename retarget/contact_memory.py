"""Shift the selected surface solution and multipliers across MPC windows."""
import warp as wp


@wp.struct
class ContactMemory:
    c: wp.array2d(dtype=wp.vec3)
    dual: wp.array2d(dtype=wp.vec3)
    local: wp.array2d(dtype=wp.vec3)
    body: wp.array2d(dtype=int)
    active: wp.array2d(dtype=int)


@wp.kernel(enable_backward=False)
def remember(c: wp.array2d(dtype=wp.vec3), dual: wp.array2d(dtype=wp.vec3),
             local: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
             mask: wp.array(dtype=float), weight: wp.array(dtype=float),
             enabled: int, stage: int, selected: int, memory: ContactMemory):
    slot=wp.tid()
    memory.c[stage,slot]=c[selected,slot]
    memory.dual[stage,slot]=dual[selected,slot]
    memory.local[stage,slot]=local[slot]
    memory.body[stage,slot]=body[slot]
    memory.active[stage,slot]=wp.where(enabled == 1 and mask[slot]*weight[slot] > 0.,1,0)


@wp.kernel(enable_backward=False)
def restore(c: wp.array2d(dtype=wp.vec3), z: wp.array2d(dtype=wp.vec3),
            last_c: wp.array2d(dtype=wp.vec3), dual: wp.array2d(dtype=wp.vec3),
            anchor: wp.array(dtype=wp.vec3), local: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
            mask: wp.array(dtype=float), weight: wp.array(dtype=float), enabled: int,
            stage: int, shift: int, radius: float, memory: ContactMemory):
    world,slot=wp.tid()
    source=wp.min(stage+shift,memory.c.shape[0]-1)
    point=memory.c[source,slot]
    # Changed hand correspondence, inactive contacts, and points outside the
    # NEW anchor's trust region start afresh. The surface solver validates
    # the new connected patch before evaluating this warm incumbent.
    if (enabled == 1 and mask[slot]*weight[slot] > 0. and memory.active[source,slot] == 1
            and body[slot] == memory.body[source,slot]
            and wp.length(local[slot]-memory.local[source,slot]) < 1.e-6
            and wp.length(point-anchor[slot]) <= radius):
        c[world,slot]=point
        z[world,slot]=point
        last_c[world,slot]=point
        dual[world,slot]=memory.dual[source,slot]


def allocate(stages,contacts,device):
    memory=ContactMemory()
    for name in ('c','dual','local'):
        setattr(memory,name,wp.zeros((stages,contacts),dtype=wp.vec3,device=device))
    for name in ('body','active'):
        setattr(memory,name,wp.zeros((stages,contacts),dtype=int,device=device))
    return memory

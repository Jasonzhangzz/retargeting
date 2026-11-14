"""Lazy backend selection. Optimizers never import one another's objective."""
def create_retarget(backend, topology, cache=None, device='cpu', **kwargs):
    if backend == 'slsqp':
        from .wuji_contact_slsqp import ContactSLSQPRetarget
        return ContactSLSQPRetarget(topology, cache, device, **kwargs)
    if backend == 'cem':
        from .wuji_retarget import TactileRetarget
        return TactileRetarget(topology, cache, device, **kwargs)
    raise ValueError('Unknown Wuji2 solver: '+str(backend))

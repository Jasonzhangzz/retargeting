"""The five supported Wuji2 solvers; optimizer imports stay lazy."""
DEFAULT_SOLVER = 'slsqp_graph'
SLSQP_PROFILES = {'slsqp_graph': 'contact_graph'}
SOLVERS = ('cem', 'slsqp_graph', 'wuji-retargeting', 'geort', 'cart')


def create_retarget(backend, topology, cache=None, device='cpu', **kwargs):
    if backend == 'cart':
        from .cart_runtime import CartRetarget
        return CartRetarget(topology, cache, device, **kwargs)
    if backend == 'geort':
        from retarget.baseline.geort.runtime import GeoRTRetarget
        return GeoRTRetarget(topology, cache, device, **kwargs)
    if backend == 'wuji-retargeting':
        from retarget.baseline.wuji_retargeting.runtime import WujiReferenceRetarget
        return WujiReferenceRetarget(topology, cache, device, **kwargs)
    if backend in SLSQP_PROFILES:
        from .wuji_contact_slsqp import ContactSLSQPRetarget
        kwargs['objective_profile'] = SLSQP_PROFILES[backend]
        return ContactSLSQPRetarget(topology, cache, device, **kwargs)
    if backend == 'cem':
        from retarget.baseline.card.retarget import TactileRetarget
        return TactileRetarget(topology, cache, device, **kwargs)
    raise ValueError('Unknown Wuji2 solver: ' + str(backend))

"""CPU4 named-array bridge to the original Robokit simulation cost terms."""
import numpy as np
import warp as wp
from robokit.terms import SensorTask, ControlTask, SupportTask
from robokit.terms.simulation import SimContext

_CACHE = {}


class RobokitCost:
    def __init__(self, model, horizon: int, descriptors: list):
        self.model, self.horizon = model, horizon
        self.clock = wp.zeros(3, dtype=int, device='cpu')
        self.sensor = wp.zeros((1, model.nsensordata), dtype=float, device='cpu')
        self.ctrl = wp.zeros((1, model.nu), dtype=float, device='cpu')
        self.ref = wp.zeros((1, horizon + 1, model.nsensordata), dtype=float, device='cpu')
        self.ctrl_ref = wp.zeros((1, horizon + 1, model.nu), dtype=float, device='cpu')
        self.capacity = max(64, model.ngeom * 8)
        self.count = wp.zeros(1, dtype=int, device='cpu')
        self.world = wp.zeros(self.capacity, dtype=int, device='cpu')
        self.geoms = wp.full(self.capacity, [-1, -1], dtype=wp.vec2i, device='cpu')
        self.distance = wp.zeros(self.capacity, dtype=float, device='cpu')
        self.position = wp.zeros(self.capacity, dtype=wp.vec3, device='cpu')
        context = {SimContext.CLOCK: self.clock, SimContext.NUM_INSTANCES: 1,
            SimContext.WORLDS_PER_INSTANCE: 1, SimContext.SENSORDATA: self.sensor,
            SimContext.CTRL: self.ctrl, SimContext.SENSORS: [
                (model.sensor(i).name, int(model.sensor_adr[i]), int(model.sensor_dim[i])) for i in range(model.nsensor)],
            SimContext.GEOMS: [model.geom(i).name for i in range(model.ngeom)],
            SimContext.CONTACTS: (self.count, self.world, self.geoms, self.distance, self.position)}
        self.terms = []
        residual_modes = {0: 'linear', 1: 'angle', 5: 'heading'}
        for item in descriptors:
            kind = item['type']
            if kind == 'SensorTask' and model.nsensordata == 0:
                continue
            if kind == 'ControlTask' and model.nu == 0:
                continue
            if kind == 'SensorTask':
                term = SensorTask(self.ref, item['weight'], norm=item.get('norm', False),
                    residual=residual_modes[item.get('residual', 0)])
            elif kind == 'ControlTask':
                term = ControlTask(item['weight'], self.ctrl_ref, norm=item.get('norm', False))
            elif kind == 'SupportTask':
                term = SupportTask(item['obj'], item['groups'], item['need'], item['weight'],
                    item['margin'], item['free'])
            else:
                raise TypeError(f'CPU4 bridge does not support source term {kind}')
            term.resolve(None, context)
            self.terms.append((term, item.get('running_scale', 1.), item.get('terminal_scale', 1.)))
        self.out = wp.zeros(len(self.terms), dtype=float, device='cpu')

    def evaluate(self, data, packet: dict, step: int, terminal: bool, contacts: list) -> dict:
        if len(contacts) > self.capacity:
            raise OverflowError('CPU4 Robokit cost bridge contact capacity exceeded')
        self.clock.assign([0, step, self.horizon])
        self.sensor.assign(data.sensordata[None]); self.ctrl.assign(data.ctrl[None])
        self.ref.assign(packet['ref'][None]); self.ctrl_ref.assign(packet['ctrl_ref'][None])
        self.count.assign([len(contacts)])
        pairs = np.full((self.capacity, 2), -1, np.int32)
        distances = np.zeros(self.capacity, np.float32)
        for i, contact in enumerate(contacts):
            pairs[i] = [contact['geom1'], contact['geom2']]
            distances[i] = contact['geometric_distance']
        self.geoms.assign(pairs); self.distance.assign(distances)
        self.out.zero_()
        for i, (term, _, _) in enumerate(self.terms):
            term.compute_weighted_cost(None, out_cost=self.out[i:i + 1])
        costs = self.out.numpy()
        result = dict(task=0., control=0.)
        for value, (term, running, final) in zip(costs, self.terms):
            result['control' if isinstance(term, ControlTask) else 'task'] += float(value) * (final if terminal else running)
        return result


def source_cost(model, data, packet: dict, step: int, terminal: bool, contacts: list) -> dict:
    descriptors = packet.get('robokit_terms')
    if descriptors is None:
        # Generic test/query packets still use source terms, not a second cost
        # implementation. Real migrated tasks always supply their source config.
        descriptors = [dict(type='SensorTask', weight=packet['weight'][None],
            running_scale=1., terminal_scale=1.), dict(type='ControlTask', weight=packet['ctrl_weight'])]
        if packet.get('heading', -1) >= 0:
            sensor = next(model.sensor(i).name for i in range(model.nsensor)
                          if int(model.sensor_adr[i]) == packet['heading'])
            descriptors.append(dict(type='SensorTask', weight={sensor: packet['heading_weight']},
                residual=5, terminal_scale=packet['heading_terminal']))
    key = (id(model), len(packet['ref']) - 1, repr(descriptors))
    if key not in _CACHE:
        _CACHE[key] = RobokitCost(model, len(packet['ref']) - 1, descriptors)
    return _CACHE[key].evaluate(data, packet, step, terminal, contacts)

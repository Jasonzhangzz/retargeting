"""Current CART version and retained, inactive v3 primitive helpers.

The v5 differentiable policies use no action or phase labels. The definitions
below remain solely for inspecting old primitive assets, never new supervision.
"""
import numpy as np

LEGACY_PROTOCOL = 'cart_contact_v3'
TUBE_PROTOCOL = 'cart_contact_tube_v4'
PROTOCOL = 'cart_diff_v5'
SUPPORTED_RECORDINGS = (PROTOCOL, TUBE_PROTOCOL, LEGACY_PROTOCOL)
ARCHITECTURES = {'attention': 'cart_cross_attention128_delta_v2', 'fsq': 'cart_future_fsq8_delta_v2'}
POLICY_OUTPUT = 'normalized_delta_q_v1'
ARCHITECTURE = ARCHITECTURES['attention']
LOOKAHEAD = 10
HISTORY_LENGTH = 10
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
LABELS = ('拇指', '食指', '中指', '无名指', '小指')
PHASES = ('open', 'closing', 'hold', 'release')
DIRECTIONS = np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1)
                       for z in (-1,0,1) if x or y or z], float)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)
STAGES = []

def _add(key, category, label, fingers, patches, pairs=(), envelope=None):
    STAGES.append(dict(id=key,index=len(STAGES),category=category,label=label,
        fingers=list(fingers),patches=list(patches),pairs=[list(p) for p in pairs],
        envelope=list(envelope or patches),cycles=2))

for f,label in zip(FINGERS[1:],LABELS[1:]):
    _add('pinch_'+f,1,'对指 · 拇指对'+label,[0,FINGERS.index(f)],['thumb_distal',f+'_distal'],
         [('thumb_distal',f+'_distal')])
_add('precision_grasp',2,'Precision grasp · 拇指参与指尖对向抓握',range(5),
     [f+'_distal' for f in FINGERS])
_add('power_grasp',3,'Power grasp · 五指全指节',range(5),
     ['thumb_proximal','thumb_distal']+
     [f+'_'+p for f in FINGERS[1:] for p in ('proximal','intermediate','distal')],
     [(f+'_distal','palm_mcp_'+f) for f in FINGERS[1:]])
_add('wrap',4,'Wrap · 拇指参与包围卷曲',range(1,5),
     [f+'_'+p for f in FINGERS[1:] for p in ('proximal','intermediate','distal')],
     [(f+'_distal',f+'_proximal') for f in FINGERS[1:]])
_add('prismatic',5,'Prismatic · 拇指与二至四指对向夹持',range(1,5),
     [f+'_'+p for f in FINGERS[1:] for p in ('proximal','distal')],
     [(f+'_distal',f+'_proximal') for f in FINGERS[1:]],
     [f+'_'+p for f in FINGERS[1:] for p in ('proximal','intermediate','distal')])
BY_ID = {s['id']:s for s in STAGES}
CATEGORIES = ('opposition','precision_grasp','power_grasp','wrap','prismatic')
NEUTRAL_INDEX = len(STAGES)
ACTION_COUNT = NEUTRAL_INDEX+1
# Selected Dexonomy families; source hand geometry determines their dimensions.
PRIMITIVE_TYPES = {2:(6,7,8,13),3:(3,5,11,26,28),4:(3,5,26,28),5:(6,7,8)}
for _spec in STAGES:
    _spec.update(category_name=CATEGORIES[_spec['category']-1],
        primitive_types=list(PRIMITIVE_TYPES.get(_spec['category'],())),
        initial_shapes=sorted({'sphere' if n in (11,13,26,28) else 'cylinder'
                               for n in PRIMITIVE_TYPES.get(_spec['category'],())}),
        patch_weights={p:2. if _spec['category']>=4 and not p.endswith('_intermediate') else 1.
                       for p in _spec['patches']},
        shape_weight=.1 if _spec['category']>=3 else 1.,
        opposition_weight=2. if _spec['id']=='prismatic' else 1.)


def stage(value):
    if value not in BY_ID: raise ValueError('请选择有效的 CART 子动作')
    return BY_ID[value]

class MotionTracker:
    """Distance hysteresis, full close/release cycles, no timed bins."""
    def __init__(self): self.reset()
    def reset(self):
        self.closed=False;self.cycles=0;self.last=None;self.peak=None;self.low=None
        self.armed=False;self.descending=False;self.phase='open';self.last_gap=None
    def update(self, stamp, gap):
        if not np.isfinite(gap):return dict(phase=self.phase,cycles=self.cycles,closed=self.closed)
        if self.last is not None and not 0<stamp-self.last<=250:
            # A missing release may never complete a cycle.
            self.closed=False;self.armed=False;self.peak=None;self.low=None;self.descending=False
        old=self.last_gap;self.last=stamp;self.last_gap=float(gap)
        self.peak=max(gap,self.peak if self.peak is not None else gap)
        if self.descending:self.low=min(gap,self.low if self.low is not None else gap)
        else:self.low=gap
        if gap>.004:self.armed=True
        if gap<=.002 and self.armed:self.closed=True
        # Excursion completion also counts nonzero-volume grasps.  A real
        # closing excursion and >=80% release are required, never a timer.
        if self.peak-gap>=.002:self.descending=True
        released=self.descending and gap-self.low>=max(.002,.8*(self.peak-self.low))
        if released and self.armed:
            self.cycles+=1;self.armed=False;self.closed=False
            self.peak=gap;self.low=gap;self.descending=False
        if self.closed and gap>.004:self.closed=False
        velocity=0 if old is None else gap-old
        self.phase='hold' if self.closed or self.descending and abs(velocity)<.0003 else (
            'closing' if velocity<-.0001 else 'release' if velocity>.0001 else 'open')
        return dict(phase=self.phase,cycles=self.cycles,closed=self.closed,gap_m=float(gap))

def features(joints, dt):
    """84 features: 21 palm-local points, 10 internal angles, 10 gaps, dt."""
    j=np.asarray(joints,float).reshape(21,3)
    chains=j[1:].reshape(5,4,3);v=np.diff(chains,axis=1)
    v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)
    angles=np.arccos(np.clip((v[:,:-1]*v[:,1:]).sum(-1),-1,1))
    tips=j[[4,8,12,16,20]]
    gaps=[np.linalg.norm(tips[i]-tips[k]) for i in range(5) for k in range(i+1,5)]
    return np.r_[j.ravel(),angles.ravel(),gaps,np.clip(dt,0,.25)].astype(np.float32)

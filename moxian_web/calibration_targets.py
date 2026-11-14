"""One reviewed action/region contract shared by capture, loss and diagrams."""
import json
from pathlib import Path
from .contact import PATCH_IDS

DEFINITION=json.loads((Path(__file__).parent/'static/calibration-contact-targets.json').read_text())
REVISION=DEFINITION['revision']

def target(stage):
    return DEFINITION['stages'].get(stage,dict(kind='none',label='无指定接触目标',patches=[]))

def pinch_patches(stage):
    spec=target(stage)
    return tuple(PATCH_IDS.index(p) for p in spec['patches']) if spec['kind']=='opposing_pads' else ()


def contact_edges(stage):
    """New regions are geometry IDs, independent of the 20 tactile patches."""
    return target(stage).get('edges',[])

"""Load the byte-preserved remote EMF pipeline; no algorithm is reimplemented here."""
from pathlib import Path
import sys

REFERENCE_ROOT = Path(__file__).resolve().parent / 'upstream'
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from emf_retarget import EmfFrame, EmfRetargetPipeline  # noqa: E402
from retarget_reference import RetargetSession  # noqa: E402

# Fail explicitly if another project has already imported identically named
# packages. Silently mixing two retarget_reference versions breaks parity.
for _name in ('emf_retarget', 'human_ik_reference', 'retarget_reference'):
    _module = sys.modules[_name]
    if not Path(_module.__file__).resolve().is_relative_to(REFERENCE_ROOT):
        raise ImportError(f'{_name} was loaded outside {REFERENCE_ROOT}')

import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import solve_dataset_mano_graspadmm as campaign


def test_ragged_output_keeps_failed_row_and_contact_offsets():
    values=[dict(final_qpos=np.ones(27),hand_slots=np.ones((4,6))),{},dict(hand_slots=np.ones((2,6)))]
    packed=campaign.pack_arrays(values)
    assert packed['final_qpos'].shape==(3,27)
    assert np.isnan(packed['final_qpos'][1:]).all()
    np.testing.assert_array_equal(packed['hand_slots_offsets'],[0,4,4,6])
    assert packed['hand_slots'].shape==(6,6)
    assert all(v.dtype!=object for v in packed.values())


def test_chunking_never_drops_short_final_chunk_or_repeats_pilot_type():
    groups=[dict(type='7_Test',part=0,samples=130),dict(type='7_Test',part=1,samples=64),dict(type='1_Test',part=0,samples=7)]
    jobs=list(campaign.jobs_for(groups,64))
    assert sum(end-start for _,start,end,_ in jobs)==201
    assert [(start,end) for _,start,end,_ in jobs[:3]]==[(0,64),(64,128),(128,130)]
    assert len({j[3] for j in jobs})==len(jobs)
    pilot=list(campaign.jobs_for(groups,64,pilot=True))
    assert len(pilot)==2 and all(end-start==1 for _,start,end,_ in pilot)

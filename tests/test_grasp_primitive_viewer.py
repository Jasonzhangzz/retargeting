"""The size control follows thickness while preserving historical geometries."""
from pathlib import Path

from retarget.viewer.runtime import filter_rows, primary_primitive_rows, size_scale


def test_thickness_size_control_does_not_select_fixed_radial_scale():
    rows=[dict(task='grasp',method='graspadmm',hand='wuji2',template='16_Lateral',
        name=f'size{scale}',batch=Path('/output/new'),radius_scale=1.5,
        primitive=dict(kind='plate',radial_scale=1.5,size_axis='thickness',size_scale=scale))
        for scale in [.5,.75,1,1.25,1.5]]
    assert [size_scale(row) for row in rows]==[.5,.75,1,1.25,1.5]
    selected=filter_rows(rows,size='0.5×')
    assert len(selected)==1 and selected[0]['primitive']['size_axis']=='thickness'


def test_historical_size_without_new_metadata_keeps_legacy_scale():
    assert size_scale(dict(radius_scale=.75))==.75


def test_reevaluated_old_type9_disk_is_visible_with_its_real_geometry():
    base=dict(task='grasp',method='mppi',template='9_Palmar_Pinch',primitive=dict(kind='disk'))
    old=dict(base,historical_reevaluation=True,name='old real disk')
    new=dict(base,primitive=dict(kind='cylinder'),method='graspadmm',name='new cylinder')
    assert primary_primitive_rows([old,new])==[old,new]
    assert primary_primitive_rows([dict(base,name='legacy unselected')])==[]

"""Primitive semantics are observations, separate from native acceptance."""
import copy

import numpy as np
import pytest

from retarget.env.grasp.orientation import primitive_contact_diagnostics


def test_type9_reports_side_cap_and_rim_with_measured_cap_distance():
    primitive=dict(kind='cylinder',radius_m=.018,height_m=.12)
    contacts=dict(bn1=['right_finger1_link4','right_finger2_link4','right_palm_link'],
                  pos=[[.018,0,0],[0,0,.06],[.018,0,.06]],normal=[[1,0,0],[0,0,1],[1,0,0]])
    before=copy.deepcopy(contacts)
    report=primitive_contact_diagnostics(9,primitive,contacts)
    observed=report['native_contacts']
    assert [c['surface_region'] for c in observed['contacts']]==['side','cap','rim']
    assert observed['contacts'][0]['cap_distance_m']==.06
    assert observed['minimum_cap_distance_m']==0
    assert observed['side_contact_count']==1 and not observed['all_observed_contacts_on_side']
    assert contacts==before and report['diagnostic_only'] and report['affects_task_success'] is False
    assert 'task_success' not in report


def test_type11_anchors_do_not_become_physical_palm_contact():
    observed=dict(bn1=['right_finger1_link4'],pos=[[0,0,0]],normal=[[1,0,0]])
    initial=dict(hand_cbody=['right_palm_link'],hand_cpn_w=[[0,0,0,0,1,0]])
    report=primitive_contact_diagnostics(11,dict(kind='sphere'),observed,initial_hand_contacts=initial)
    assert not report['native_contacts']['palm_contact_present']
    assert report['initial_hand_anchors']['palm_contact_present']
    assert report['initial_hand_anchors']['source']=='initial_hand_anchors_not_physical_contacts'
    observed['bn1'].append('right_palm_link');observed['pos'].append([1,0,0]);observed['normal'].append([-1,0,0])
    assert primitive_contact_diagnostics(11,dict(kind='sphere'),observed)['native_contacts']['palm_contact_present']


@pytest.mark.parametrize('number',[16,22])
def test_thumb_thickness_angle_is_unsigned_and_records_missing_thumb(number):
    ho=dict(bn1=['right_finger1_link4','right_finger1_link3'],pos=[[0,0,0]]*2,
            normal=[[0,0,2],[0,0,-3]])
    report=primitive_contact_diagnostics(number,dict(kind='plate'),ho)
    observed=report['native_contacts']
    assert observed['thumb_normal_axis_unsigned_angle_deg']==[0.,0.]
    assert observed['thumb_axis_angle_available']
    ho=dict(bn1=['right_finger2_link4'],pos=[[0,0,0]],normal=[[0,1,0]])
    observed=primitive_contact_diagnostics(number,dict(kind='plate'),ho)['native_contacts']
    assert observed['thumb_normal_axis_unsigned_angle_deg'] is None
    assert observed['reason']=='no_thumb_contact' and not observed['thumb_axis_angle_available']


def test_native_contact_absence_is_available_with_zero_contacts_but_missing_record_is_not():
    report=primitive_contact_diagnostics(11,dict(kind='sphere'),dict(bn1=[],pos=[],normal=[]))
    assert report['native_contacts']['available'] and report['native_contacts']['contact_count']==0
    assert report['native_contacts']['palm_contact_present'] is False
    missing=primitive_contact_diagnostics(11,dict(kind='sphere'))['native_contacts']
    assert not missing['available'] and missing['reason']=='not_recorded'


def test_initial_thumb_angles_are_reported_without_substituting_for_real_contacts():
    initial=dict(hand_cbody=['right_finger1_link4'],hand_cpn_w=[[0,0,0,1,0,1]])
    report=primitive_contact_diagnostics(16,dict(kind='plate'),initial_hand_contacts=initial)
    assert report['initial_hand_anchors']['thumb_normal_axis_unsigned_angle_deg']==pytest.approx([45.])
    assert not report['native_contacts']['available']
    assert 'no added semantic constraint' in report['initialization_policy']

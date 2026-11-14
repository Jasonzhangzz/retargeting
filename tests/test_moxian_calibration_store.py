import json
from pathlib import Path

import pytest

from moxian_web.calibration_store import archive_calibration,restore_calibration
from moxian_web.native_skeleton import joint_limit_diagnostics


def test_new_official_capture_preserves_records_and_can_restore_both_versions(tmp_path):
    glove=tmp_path/'glove';glove.mkdir();(glove/'record').write_bytes(b'old calibration and raw capture')
    (tmp_path/'mano.json').write_text('unchanged hand shape')
    archive=archive_calibration(tmp_path)
    assert not glove.exists()
    assert (archive/'glove/record').read_bytes()==b'old calibration and raw capture'
    glove.mkdir();(glove/'record').write_bytes(b'new partial capture')
    restore_calibration(tmp_path)
    assert (glove/'record').read_bytes()==b'old calibration and raw capture'
    restore_calibration(tmp_path)
    assert (glove/'record').read_bytes()==b'new partial capture'
    assert (tmp_path/'mano.json').read_text()=='unchanged hand shape'


def test_restore_missing_or_outside_history_does_not_move_active_records(tmp_path):
    with pytest.raises(ValueError):restore_calibration(tmp_path)
    glove=tmp_path/'glove';glove.mkdir();(glove/'record').write_text('keep')
    (tmp_path/'previous-official-calibration.json').write_text(json.dumps({'archive':'../outside'}))
    with pytest.raises(ValueError):restore_calibration(tmp_path)
    assert (glove/'record').read_text()=='keep'


def test_thumb_joint_saturation_is_diagnostic_and_does_not_modify_angles():
    values=[-.5,.2,1.]
    names=['left_thumb_cmc_flexion','left_thumb_ip_flexion','left_index_pip_flexion']
    limits={names[0]:(-.5,.5),names[1]:(-.3,1.5),names[2]:(-.4,1.)}
    result=joint_limit_diagnostics(names,values,limits)
    assert [(r['joint'],r['bound']) for r in result]==[(names[0],'lower'),(names[2],'upper')]
    assert values==[-.5,.2,1.]

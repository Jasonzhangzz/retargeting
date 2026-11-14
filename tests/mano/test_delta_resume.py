"""Frozen template reuse and resume provenance without held-out selection."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from moxian_web.contact_templates import reuse_templates


def test_retained_templates_are_reused_exactly_and_do_not_read_validation_pose():
    path=Path('output/moxian/debug/delta-resume-v1/parent-checkpoint/candidate.json')
    if not path.exists():pytest.skip('Retained trained checkpoint required')
    c=json.loads(path.read_text());artifact=c['contact_templates'];before=copy.deepcopy(artifact)
    rows=[dict(stage=a['stage'],capture_protocol='contact_motion_v3',partition=p,contact_confirmed=touch,
               fingers=None if p=='validation' else {}) for a in artifact['actions'] for p in ('train','validation') for touch in (True,False)]
    model=SimpleNamespace(faces=[None]*1538)
    actual=reuse_templates(model,rows,artifact)
    assert actual==before and artifact==before
    edges={a['stage']:a['edges'] for a in artifact['actions']}
    for row in rows:assert row['contact_edges']==(edges[row['stage']] if row['contact_confirmed'] else [])
    rows[0]['contact_edges'][0]['weight']=999
    assert artifact==before and actual==before
    for key,value in (('validation_used',True),('source','validation'),('actions',[])):
        bad=copy.deepcopy(artifact);bad[key]=value
        with pytest.raises(ValueError):reuse_templates(model,rows,bad)


def test_resume_rejects_different_capture_before_any_model_is_built():
    from moxian_web.mano_fit import fit_capture
    import threading
    with pytest.raises(ValueError,match='完整采集'):
        fit_capture(None,{},None,[],lambda _:None,threading.Event(),initial_candidate={'report':{'source_capture_hash':'different'}})

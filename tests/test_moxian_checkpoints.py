import copy,hashlib,json
from types import SimpleNamespace
import pytest
from moxian_web.checkpoints import CheckpointCatalog,comparison_profile
from moxian_web.hand_anatomy import GEOMETRY_REVISION
from moxian_web.hand_model import MANO_FIT_REVISION


def candidate():
    return dict(serial='device',side='right',weights_hash='weights',geometry_revision=GEOMETRY_REVISION,
                report=dict(fit_revision=MANO_FIT_REVISION,accepted=False,method='parameters',contact_gap_mm=22.),
                calibration_fingerprint='f',skeleton_runtime_revision='r',mapping={'assignments':[1,2]},calibrated=True)


def test_catalog_only_exposes_complete_trials_and_scopes_device(tmp_path):
    directory=tmp_path/'debug'/'benchmark';directory.mkdir(parents=True)
    records=[]
    for cache in ('cold','warm'):
        for run in (1,2,3):
            records.append(dict(method='parameters',cache=cache,run=run,accepted=False))
            folder=directory/f'parameters-{cache}-{run}';folder.mkdir();(folder/'candidate.json').write_text(json.dumps(candidate()))
    summary=directory/'summary.json';summary.write_text(json.dumps(records[:5]))
    catalog=CheckpointCatalog(tmp_path);assert catalog.list('device','right')==[]
    summary.write_text(json.dumps(records));catalog.cache.clear()
    items=catalog.list('device','right',dict(calibration_fingerprint='f',skeleton_runtime_revision='r'))
    assert len(items)==1 and items[0]['method']=='parameters' and all(i['compatible'] and not i['accepted'] for i in items)
    assert catalog.list('another-device','right')==[] and catalog.list('device','left')==[]
    metadata,weight=catalog.get(items[0]['id'],'device','right');weight['shape']='modified'
    assert 'shape' not in catalog.get(items[0]['id'],'device','right')[1]
    with pytest.raises(ValueError):catalog.get('../../secret','device','right')
    assert all(not i['compatible'] for i in catalog.list('device','right',dict(calibration_fingerprint='changed')))
    (directory/'DIAGNOSTIC_ONLY.json').write_text('{}');catalog.cache.clear();assert catalog.list('device','right')==[]


def test_checkpoint_symlinks_cannot_escape_output(tmp_path):
    output=tmp_path/'output';key=hashlib.sha256(b'device').hexdigest()[:16]
    home=output/'hand_model'/key;home.mkdir(parents=True)
    outside=tmp_path/'outside.json';outside.write_text(json.dumps(candidate()));(home/'candidate.json').symlink_to(outside)
    assert CheckpointCatalog(output).list('device','right')==[]


def test_catalog_keeps_only_metadata_and_loads_full_weights_for_preview(tmp_path):
    home=tmp_path/'hand_model'/hashlib.sha256(b'device').hexdigest()[:16]
    home.mkdir(parents=True);path=home/'candidate.json';weight=candidate()
    weight['report']['runtime_solver_frames']=[{'residual':[1.,2.,3.]} for _ in range(100)]
    path.write_text(json.dumps(weight));catalog=CheckpointCatalog(tmp_path)
    item=catalog.list('device','right')[0]
    assert 'runtime_solver_frames' not in repr(catalog.cache)
    assert 'runtime_solver_frames' not in repr(catalog.file_cache)
    assert catalog.get(item['id'],'device','right')[1]==weight
    weight['report']['runtime_solver_frames'].append({'changed':True});path.write_text(json.dumps(weight))
    with pytest.raises(ValueError,match='更新'):
        catalog.get(item['id'],'device','right')


def test_least_squares_discovery_needs_no_numeric_solver_dependencies(tmp_path,monkeypatch):
    import builtins
    weight=candidate()
    weight.update(surface_backend='qnbot',delta_policy=None,report=dict(method='least_squares',
        fit_revision='qnbot_least_squares_v1',accepted=False,frozen_forward=True,online_optimization=False))
    home=tmp_path/'hand_model'/hashlib.sha256(b'device').hexdigest()[:16]/'qnbot/methods/least_squares'
    home.mkdir(parents=True);(home/'candidate.json').write_text(json.dumps(weight))
    original_import=builtins.__import__
    def sdk_import(name,*args,**kwargs):
        if name.split('.')[0] in ('numpy','torch','scipy') or 'qnbot_least_squares' in name:
            raise ModuleNotFoundError(name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',sdk_import)
    items=CheckpointCatalog(tmp_path).list('device','right')
    assert len(items)==1 and items[0]['compatible'] and not items[0]['accepted']


def test_failed_candidates_can_be_compared_without_promoting_acceptance():
    weight=candidate();before=copy.deepcopy(weight)
    model=SimpleNamespace(weights_hash='weights');frame=dict(calibration_fingerprint='f',skeleton_runtime_revision='r')
    preview=comparison_profile(weight,model,frame,{'assignments':[1,2]},'device','right')
    assert not preview['calibrated'] and not preview['report']['accepted'] and preview['revision'] is None
    assert preview['mounting_calibrated'] and weight==before
    for change in (dict(calibration_fingerprint='changed'),dict(skeleton_runtime_revision='changed')):
        with pytest.raises(ValueError):comparison_profile(weight,model,{**frame,**change},{'assignments':[1,2]},'device','right')
    with pytest.raises(ValueError):comparison_profile(weight,model,frame,{'assignments':[3,4]},'device','right')

import copy,hashlib,json,threading
import pytest
from moxian_web.mano_calibration import prepare_rows


def test_only_missing_frames_replay_and_cache_is_content_validated(tmp_path,monkeypatch):
    serial='test';key=hashlib.sha256(serial.encode()).hexdigest()[:16]
    home=tmp_path/'hand_model'/key;home.mkdir(parents=True)
    config=tmp_path/'ik'/key/'runtime'/'kinematic-abcdefghijklmnop.yaml';config.parent.mkdir(parents=True);config.write_text('unchanged config')
    p=dict(serial=serial,side='right',artifact_home=str(home),calibration_fingerprint='abcdefghijklmnop-more',skeleton_runtime_revision='runtime')
    rows=[dict(timestamp_ms=1.,calibration_fingerprint=p['calibration_fingerprint'],skeleton_runtime_revision='runtime',skeleton={'valid':True},skeleton_targets={'valid':True}),dict(timestamp_ms=2.,calibration_fingerprint=p['calibration_fingerprint'])]
    before=copy.deepcopy(rows);calls=[]
    def fake(command,cancel,progress):
        source=command[command.index('--input')+1];out=command[command.index('--output')+1]
        data=json.load(open(source));calls.append(len(data['rows']))
        for r in data['rows']:r.update(skeleton={'valid':True},skeleton_targets={'valid':True},skeleton_runtime_revision='runtime')
        open(out,'w').write(json.dumps(data))
    monkeypatch.setattr('moxian_web.fit_process.run_replay',fake)
    result,meta=prepare_rows(p,rows,lambda _:None,threading.Event())
    assert calls==[1] and meta['replayed_frames']==1 and result[0]==rows[0] and rows==before
    cached,meta=prepare_rows(p,rows,lambda _:None,threading.Event());assert meta['cache_hit'] and cached==result and calls==[1]
    target=next((home/'gpu-cache').glob('*/prepared.json'));target.write_text('{}')
    _,meta=prepare_rows(p,rows,lambda _:None,threading.Event());assert not meta['cache_hit'] and calls==[1,1]
    config.write_text('changed config')
    _,meta=prepare_rows(p,rows,lambda _:None,threading.Event());assert not meta['cache_hit'] and calls==[1,1,1]
    _,meta=prepare_rows(p,rows,lambda _:None,threading.Event(),use_cache=False);assert not meta['cache_hit'] and calls==[1,1,1,1]
    folder=home/'gpu-cache'/meta['key'];(folder/'index.json').write_text('{broken')
    _,meta=prepare_rows(p,rows,lambda _:None,threading.Event());assert not meta['cache_hit'] and len(calls)==5
    cached=json.loads((folder/'prepared.json').read_text());cached['rows'][1].pop('skeleton_targets')
    content=json.dumps(cached).encode();(folder/'prepared.json').write_bytes(content)
    (folder/'index.json').write_text(json.dumps({'sha256':hashlib.sha256(content).hexdigest()}))
    _,meta=prepare_rows(p,rows,lambda _:None,threading.Event());assert not meta['cache_hit'] and len(calls)==6


def test_cancel_before_any_preparation():
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):prepare_rows({},[],lambda _:None,cancel)


def test_shared_geometry_keeps_same_upstream_ik_seed():
    from moxian_web.hand_model import tracking_registration_matches
    profile=dict(calibration_fingerprint='f',skeleton_runtime_revision='r',runtime_backend='gpu_mano_v1',skeleton_model={'geometry_hash':'new'},skeleton_seed_model={'geometry_hash':'seed'})
    frame=dict(calibration_fingerprint='f',skeleton_runtime_revision='r',skeleton={'personalized_geometry_hash':'seed'})
    assert tracking_registration_matches(profile,frame)
    frame['skeleton']['personalized_geometry_hash']='new'
    assert not tracking_registration_matches(profile,frame)
    profile['skeleton_seed_model']=None;frame['skeleton'].pop('personalized_geometry_hash')
    assert tracking_registration_matches(profile,frame)
    frame['skeleton_runtime_revision']='other'
    assert not tracking_registration_matches(profile,frame)


def test_seed_hash_does_not_hide_a_runtime_policy_change():
    from moxian_web.hand_model import tracking_registration_matches,tracking_geometry_hash
    saved=dict(calibration_fingerprint='f',skeleton_runtime_revision='before',runtime_backend='gpu_mano_v1',skeleton_seed_model={'geometry_hash':'seed'},skeleton_model={'geometry_hash':'fitted'})
    frame=dict(calibration_fingerprint='f',skeleton_runtime_revision='after',skeleton={'personalized_geometry_hash':'seed'})
    assert tracking_geometry_hash(saved)=='seed'
    assert not tracking_registration_matches(saved,frame)
    assert not tracking_registration_matches(saved,frame,pending=True)
    saved['skeleton_runtime_revision']='after'
    assert tracking_registration_matches(saved,frame)

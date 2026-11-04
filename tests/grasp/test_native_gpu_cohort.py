import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_fixed_cohort_preserves_every_input_once_and_reuses_only_committed_rows(tmp_path,monkeypatch):
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    import run_native_mano_gpu_cohort as campaign
    source=tmp_path/'source';source.mkdir()
    (source/'hand').mkdir();(source/'upstream').mkdir();(source/'mapping.json').write_text('{}')
    inputs=[]
    for i in range(11):
        path=source/f'{i}.npy'
        np.save(path,dict(scene_path=f'scene_{i%2}',hand_cbody=['index3'],required_cbody=[['index3']]))
        inputs.append(dict(path=str(path),sha256=campaign.task.sha(path)))
    (source/'initialization.json').write_text(json.dumps(dict(inputs=inputs)))
    first=tmp_path/'first';first.mkdir()
    (first/'inputs.json').write_text(json.dumps([dict(input=r['path'],sha256=r['sha256']) for r in inputs[:3]]))
    (first/'solve_receipt.json').write_text(json.dumps(dict(completed=3,finite=3,native_output=2,projected_closure=1)))
    (first/'MANIFEST.json').write_text(json.dumps(dict(source_sha256={})))
    np.savez(first/'poses.npz',final_qpos=np.zeros((3,27)))
    root=tmp_path/'cohort';root.mkdir()
    campaign.prepare(SimpleNamespace(root=root,source=source,first=first,batch_size=4))
    jobs=json.loads((root/'jobs.json').read_text());manifest=json.loads((root/'MANIFEST.json').read_text())
    paths=[r['path'] for j in jobs for r in j['inputs']]
    assert len(paths)==len(set(paths))==8
    assert set(paths)=={r['path'] for r in inputs[3:]}
    assert manifest['samples']==11 and manifest['reused_samples']==3
    assert all(j['samples']<=4 for j in jobs)
    assert manifest['production_launch_allowed'] is False

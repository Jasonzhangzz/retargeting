from pathlib import Path
import json
import pytest
from scripts import run_mano_then_wuji as pipeline


@pytest.mark.parametrize('failed',[False,True])
def test_wuji_starts_only_after_complete_matching_mano_stage(tmp_path,monkeypatch,failed):
    root=tmp_path
    (root/'results').mkdir()
    files=['retarget/solver/cem.py','retarget/env/grasp/closure_types.py','retarget/env/grasp/projected_qp.py','retarget/env/grasp/pose_gpu.py','retarget/env/grasp/pose_inputs.py','retarget/env/grasp/qp_projection.py','retarget/env/grasp/pose_objective.py',
        'retarget/env/grasp/mano_adapter.py','retarget/env/grasp/mano_solver.py','scripts/pack_dexonomy_batches.py',
        'scripts/retarget_dexonomy_mano.py','scripts/retarget_mano_wuji2.py','scripts/run_mano_then_wuji.py']
    for name in files:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('pinned implementation')
    monkeypatch.setattr(pipeline,'ROOT',root)
    monkeypatch.setattr(pipeline.signal,'signal',lambda *a:None)
    launches=[]
    class Process:
        returncode=0
        def __init__(self,argv,**kwargs):
            self.pid=100+len(launches);launches.append(argv[2])
            output=Path(argv[argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
            if argv[2].endswith('pack_dexonomy_batches.py'):status=dict(state='completed',samples=2048)
            elif argv[2].endswith('retarget_dexonomy_mano.py'):
                assert launches==['scripts/pack_dexonomy_batches.py','scripts/retarget_dexonomy_mano.py']
                status=dict(state='completed',samples=2047 if failed else 2048,failed_shards=0)
            else:
                assert launches==['scripts/pack_dexonomy_batches.py','scripts/retarget_dexonomy_mano.py','scripts/retarget_mano_wuji2.py']
                status=dict(state='completed')
            (output/'status.json').write_text(json.dumps(status))
        def poll(self):return 0
    monkeypatch.setattr(pipeline.subprocess,'Popen',Process)
    arguments=['pipeline','--output',str(root/'controller'),'--raw-source',str(root/'raw'),
        '--packed',str(root/'packed'),'--mano',str(root/'mano'),'--wuji',str(root/'wuji')]
    monkeypatch.setattr('sys.argv',arguments)
    if failed:
        with pytest.raises(RuntimeError,match='completion gate'):pipeline.main()
        assert len(launches)==2
    else:
        pipeline.main()
        assert len(launches)==3
    status=json.loads((root/'controller/status.json').read_text())
    assert status['stage']==('failed' if failed else 'completed')

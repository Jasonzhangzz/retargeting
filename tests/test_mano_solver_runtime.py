"""Regression tests for cache lifetimes, frame-safe warm starts and early exit."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import torch,pytest
from scipy.spatial.transform import Rotation
from retarget.env.grasp.mano_solver import *
from scripts.retarget_dexonomy_mano import GeometryLoader,interleaved_order
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def fixture():
    torch.set_num_threads(1)
    raw=json.loads((ROOT/'output/dexonomy_shadow_mano_mapping_20261103/data.json').read_text())
    source=ROOT/'results/dexonomy_shadow_grasp_full/7_Prismatic_3_Finger/shards/0000000'
    entries=json.loads((source/'manifest.json').read_text())['cases'][:2]
    with np.load(source/'poses.npz') as a:q=a['grasp_qpos'][:2];op=a['object_pose_world'][:2];scale=a['scene_scale'][:2]
    model=ManoDecoder(ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl');model.configure_contact_vertices(raw['mappings'])
    skeleton=ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml')
    mapping=raw['mappings']['7_Prismatic_3_Finger'];points,normals=skeleton.contacts(q,mapping)
    loader=GeometryLoader(ROOT/'data/dexonomy');geos=[];ids=[];keys={}
    for e in entries:
        k,g=loader.get(e)
        if k not in keys:keys[k]=len(geos);geos.append(g)
        ids.append(keys[k])
    return dict(model=model,target=skeleton(q),points=points,normals=normals,fixed=FixedContacts(model,mapping),
        geo=geos,ids=ids,op=op,scale=scale)
def surface(f,cache=None):return SurfaceBatch(f['geo'],f['ids'],f['op'],f['scale'],'cpu',mesh_cache=cache)
def run(f,settings,**kw):
    return optimize(f['model'],f['target'],surface(f),settings,17,f['fixed'],f['points'],f['normals'],**kw)

def test_so3_log_including_pi():
    angles=np.array([[0,0,0],[np.pi,0,0],[0,np.pi,0],[0,0,np.pi],[.2,-.8,2.8]],np.float32)
    matrices=Rotation.from_rotvec(angles).as_matrix().astype(np.float32)
    result=matrix_to_rotvec(torch.tensor(matrices)).numpy()
    np.testing.assert_allclose(Rotation.from_rotvec(result).as_matrix(),matrices,atol=5e-7)

def test_mesh_cache_reuse_and_lifetime(fixture):
    f=fixture;cache=MeshCache('cpu',capacity=1);a=surface(f,cache);builds=cache.builds
    b=surface(f,cache)
    assert cache.builds==builds and cache.hits>0 and a.meshes[0].id==b.meshes[0].id
    # Active batches retain their meshes after unrelated LRU entries are evicted.
    changed=dict(f['geo'][0]);changed['sha256']='different';cache.get(changed)
    p=torch.tensor(f['points'],dtype=torch.float32);idx=torch.arange(2)
    torch.testing.assert_close(a.query(p,idx)[2],b.query(p,idx)[2])

def test_warm_start_object_frames_and_size_gate():
    settings=Settings(warm_size_ratio=1.2);bank=WarmStartBank('cpu',settings)
    r=Rotation.from_rotvec([.2,-.1,.3]).as_matrix().astype(np.float32)
    surf=type('S',(),dict(origin=torch.tensor([[.1,.2,.3]]),rotation=torch.tensor(r[None]),scale=torch.tensor([.5])))()
    variables=torch.zeros(1,30);variables[0,:3]=torch.tensor([.12,.18,.35]);variables[0,3:6]=torch.tensor([.4,.3,-.1]);variables[0,6:]=.2
    target=torch.randn(1,21,3,generator=torch.Generator().manual_seed(2))*.02+surf.origin[:,None]
    _,_,feature=bank.lookup(['same'],target,surf);bank.add(['same'],variables,feature,surf)
    r2=Rotation.from_rotvec([-.4,.2,-.3]).as_matrix().astype(np.float32)
    other=type('S',(),dict(origin=torch.tensor([[-.2,.3,.1]]),rotation=torch.tensor(r2[None]),scale=torch.tensor([.55])))()
    newtarget=(feature*other.scale[:,None,None])@other.rotation.transpose(-1,-2)+other.origin[:,None]
    warm,valid,_=bank.lookup(['same'],newtarget,other);assert valid.item()
    expected=((variables[:,:3]-surf.origin)@surf.rotation[0])/surf.scale[:,None]*other.scale[:,None]@other.rotation[0].T+other.origin
    torch.testing.assert_close(warm[:,:3],expected)
    torch.testing.assert_close(rotmat(warm[:,3:6]),other.rotation@surf.rotation.transpose(-1,-2)@rotmat(variables[:,3:6]),atol=1e-6,rtol=1e-5)
    other.scale*=2
    assert not bank.lookup(['same'],newtarget,other)[1].item()
    assert not bank.lookup(['different'],newtarget,surf)[1].item()


def test_per_row_early_stop_is_not_a_success_label(fixture):
    settings=Settings(population=8,iterations=12,ik_steps=12,candidate_chunk=16,
        search_min_iterations=4,search_check_every=2,search_patience=1,
        stop_absolute_tol=1.e9,warm_start=False)
    out=run(fixture,settings)
    assert (out['ik_iterations']<=4).all() and (out['cem_iterations']<=4).all()
    assert set(out['stop_reason']).issubset({1,2}) and 2 in out['stop_reason']
    assert out['history'].shape==(12,2) and (np.diff(out['history'],axis=0)<=1e-5).all()
    assert (out['final_cost']<=out['initial_cost']+1e-5).all()
    assert np.isfinite(out['vertices']).all()

def test_round_robin_batches_keep_every_original_index():
    order=interleaved_order([0,0,0,1,1,2,2,2])
    np.testing.assert_array_equal(order,[0,3,5,1,4,6,2,7])
    np.testing.assert_array_equal(np.arange(8)[order][np.argsort(order)],np.arange(8))

def test_invalid_policy_settings():
    for kwargs in ({'search_check_every':0},{'search_patience':0},{'warm_size_ratio':.9},{'stop_relative_tol':-1}):
        with pytest.raises(ValueError):Settings(**kwargs)

def test_verified_reuse_and_viewer_index(tmp_path):
    from scripts.retarget_dexonomy_mano import reused_shards
    from retarget.viewer.mano_results import SolvedIndex
    full=ROOT/'results/dexonomy_mano_mppi_full_20261103'
    baseline=tmp_path/'baseline';baseline.mkdir()
    config=json.loads((full/'configuration.json').read_text())
    (baseline/'configuration.json').write_text(json.dumps(config))
    target=baseline/'7_Prismatic_3_Finger/shards';target.mkdir(parents=True)
    (target/'0000000').symlink_to(full/'7_Prismatic_3_Finger/shards/0000000',target_is_directory=True)
    refs=reused_shards([baseline],config['source'],config['model_sha256'],config['mapping_fingerprint'])
    assert len(refs)==1 and refs[0]['samples']==260
    viewer=tmp_path/'viewer';viewer.mkdir();(viewer/'reused_shards.json').write_text(json.dumps(refs))
    index=SolvedIndex(viewer).refresh();rows,total=index.page()
    assert len(rows)==100 and total==260
    with pytest.raises(ValueError,match='Incompatible'):
        reused_shards([baseline],config['source'],'wrong',config['mapping_fingerprint'])

def test_reuse_checks_artifacts_and_flattens_references(tmp_path):
    from scripts.retarget_dexonomy_mano import reused_shards,sha
    source=tmp_path/'source';case=source/'1_Test/shards/0000000';case.mkdir(parents=True)
    (case/'poses.npz').write_bytes(b'immutable source')
    (case/'manifest.json').write_text(json.dumps(dict(cases=[dict(name='one')])))
    first=tmp_path/'first';out=first/'1_Test/shards/0000000';out.mkdir(parents=True)
    config=dict(source=str(source),model_sha256='model',mapping_fingerprint='mapping')
    (first/'configuration.json').write_text(json.dumps(config))
    (out/'payload.bin').write_bytes(b'cached result')
    (out/'checksums.json').write_text(json.dumps({'payload.bin':sha(out/'payload.bin')}))
    (out/'verification.json').write_text(json.dumps(dict(success=True,samples=1,checks={'finite':True},
        source_poses_sha256=sha(case/'poses.npz'))))
    refs=reused_shards([first],source,'model','mapping')
    second=tmp_path/'second';second.mkdir();(second/'configuration.json').write_text(json.dumps(config))
    (second/'reused_shards.json').write_text(json.dumps(refs))
    flattened=reused_shards([first,second],source,'model','mapping')
    assert len(flattened)==1 and flattened[0]['path']==str(out)
    (out/'payload.bin').write_bytes(b'corrupted result')
    with pytest.raises(ValueError,match='checksum mismatch'):
        reused_shards([second],source,'model','mapping')

def test_cli_has_only_production_algorithm(monkeypatch):
    import scripts.retarget_dexonomy_mano as cli
    for arguments in (['--ik-method','adam_full'],['--search-method','elite'],['--population','512'],['--cem-noise-schedule','hybrid']):
        monkeypatch.setattr('sys.argv',['retarget_dexonomy_mano.py',*arguments])
        with pytest.raises(SystemExit) as error:cli.main()
        assert error.value.code==2

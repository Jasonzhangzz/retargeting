import unittest,io,json,importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
from retarget.env.grasp.dexonomy_grasp_static import ROOT,StaticModels,decode_record,static_sample
class GraspOnlyTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').is_file(), 'Requires external Dexonomy dataset/cache')
    def test_grasp_only_record_without_other_stages(self):
        plan=json.loads((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').read_text());item=plan[0]
        p=Path(item['path']);p=p if p.is_absolute() else ROOT/p;original=decode_record(p.read_bytes())
        record={k:original[k] for k in ['grasp_qpos','scene_path','scene_scale']}
        buffer=io.BytesIO();np.save(buffer,record);record=decode_record(buffer.getvalue())
        with patch('mujoco.mj_step',side_effect=AssertionError('No integration')),patch('mujoco.mj_forward',side_effect=AssertionError('No dynamics')),patch('mujoco.mj_contactForce',side_effect=AssertionError('No forces')):
            sample=static_sample(StaticModels(ROOT/'data/dexonomy'),record,item['row'])
        self.assertEqual(sample['stage_qpos'].shape,(1,36));self.assertEqual(len(sample['contacts']),1)
        np.testing.assert_array_equal(sample['source_stage_hand_qpos'][0],original['grasp_qpos'][item['row']])

    @unittest.skipUnless((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').is_file(), 'Requires external Dexonomy dataset/cache')
    def test_preview_cache_contains_only_grasp(self):
        root=ROOT/'results/dexonomy_shadow_grasp_preview_20261103';total=0
        for path in root.glob('*/shards/*/manifest.json'):
            manifest=json.loads(path.read_text());self.assertEqual(manifest['stage_names'],['grasp'])
            with np.load(path.parent/'poses.npz',allow_pickle=False) as d:
                self.assertNotIn('pregrasp_qpos',d.files);self.assertNotIn('squeeze_qpos',d.files)
                self.assertEqual(d['source_stage_hand_qpos'].shape[1],1)
                np.testing.assert_array_equal(d['source_stage_hand_qpos'][:,0],d['grasp_qpos'])
            with np.load(path.parent/'contacts.npz',allow_pickle=False) as c:
                self.assertEqual(c['stage_names'].tolist(),['grasp'])
                self.assertEqual(len(c['offsets']),len(manifest['cases'])+1)
            total+=len(manifest['cases'])
        self.assertEqual(total,100)

    @unittest.skipUnless((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').is_file(), 'Requires external Dexonomy dataset/cache')
    def test_migration_preserves_original_grasp_contacts(self):
        old=ROOT/'results/dexonomy_shadow_static_full';new=ROOT/'results/dexonomy_shadow_grasp_full'
        marker=next(new.glob('*/shards/*/verification.json'));source=old/marker.parent.relative_to(new)
        with np.load(source/'poses.npz',allow_pickle=False) as a,np.load(marker.parent/'poses.npz',allow_pickle=False) as b:
            np.testing.assert_array_equal(a['grasp_qpos'],b['grasp_qpos'])
        with np.load(source/'contacts.npz',allow_pickle=False) as a,np.load(marker.parent/'contacts.npz',allow_pickle=False) as b:
            points=a['points_world'];offsets=a['offsets'];chunks=[points[int(offsets[3*i+1]):int(offsets[3*i+2])] for i in range((len(offsets)-1)//3)]
            np.testing.assert_array_equal(np.concatenate(chunks),b['points_world'])

    @unittest.skipUnless((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').is_file(), 'Requires external Dexonomy dataset/cache')
    def test_full_resume_does_not_double_count(self):
        import tempfile,tarfile
        from retarget.env.grasp.dexonomy_grasp_static import run_static_full
        plan=json.loads((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').read_text());item=plan[0]
        source=ROOT/item['path'];record=decode_record(source.read_bytes());row=item['row']
        small={'grasp_qpos':record['grasp_qpos'][row:row+1],'scene_path':record['scene_path'],'scene_scale':record['scene_scale'][row:row+1]}
        buffer=io.BytesIO();np.save(buffer,small);raw=buffer.getvalue()
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);dataset=temp/'dataset';dataset.mkdir()
            for name in ['extracted','shadow_assets','download_manifest.json']:
                (dataset/name).symlink_to(ROOT/'data/dexonomy'/name,target_is_directory=name!='download_manifest.json')
            member='/'.join(source.parts[source.parts.index('Dexonomy_GRASP_shadow'):])
            with tarfile.open(dataset/'Dexonomy_GRASP_shadow.tar.gz','w:gz') as tar:
                info=tarfile.TarInfo(member);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
            first=run_static_full(dataset,temp/'cache',workers=1,records_per_shard=1)
            second=run_static_full(dataset,temp/'cache',workers=1,records_per_shard=1)
            self.assertEqual(first['samples_cached'],1);self.assertEqual(second['samples_cached'],1)
            self.assertEqual(second['samples_reused'],1);self.assertEqual(second['shards_completed'],1)

    @unittest.skipUnless((ROOT/'results/dexonomy_shadow_grasp_full').is_dir(), 'Requires external full Dexonomy cache')
    def test_viewer_pages_full_cache(self):
        p=ROOT/'scripts/serve_dexonomy_grasp.py';spec=importlib.util.spec_from_file_location('grasp_viewer_for_test',p);viewer=importlib.util.module_from_spec(spec);spec.loader.exec_module(viewer)
        page=viewer.listing('full','',0,12)
        self.assertGreater(page['total'],100);self.assertEqual(len(page['rows']),12)
        self.assertEqual(page['stage'],'grasp')
        self.assertFalse(viewer.status()['physical_success_evaluated'])
if __name__=='__main__':unittest.main()

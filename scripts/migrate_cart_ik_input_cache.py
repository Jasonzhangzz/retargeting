"""Reuse verified v5 geometry/IK bytes after the policy-only v6 input change.

Only metadata is migrated, with backups. Raw recordings, prepared targets and
IK arrays are never rewritten. All old fingerprints and file hashes must match.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.cart_data import file_hash,manifest,input_digest,contact_fingerprint,preprocessing_current
from moxian_web.cart_capture import capture_identity,capture_current
from moxian_web.cart_capture_schema import CAPTURE_SCHEMA
from moxian_web.cart_initialization import identity,RetargetSession
from retarget.baseline.geort.adapter import RobotAdapter,digest
from retarget.baseline.geort.training import atomic_json

LEGACY_COMMIT='f1d1c44a9ef0bc97ac85018adf452823f9671f37'


def migrate(root,key):
    from moxian_web import cart_observations,cart_capture,cart_data
    from moxian_web.cart_protocol import PROTOCOL
    folder,m=manifest(root,key)
    paths=[folder/name for name in ('cart_capture_meta.json','cart_preprocess.json','cart_ik.json')]
    capture,prepared,ik=[json.loads(p.read_text()) for p in paths]
    robot=RobotAdapter();session=RetargetSession.for_hand('WujiHand2','right')
    if capture_current(folder,m) and preprocessing_current(folder,m) and ik['fingerprint']==identity(folder,robot,session):
        if ik['sha256']!=file_hash(folder/'cart_ik.npz'):raise ValueError('IK cache checksum mismatch')
        return dict(dataset=key,migrated=False,already_current=True)
    old=subprocess.run(['git','show',LEGACY_COMMIT+':moxian_web/cart_observations.py'],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,check=True).stdout
    old_hash=hashlib.sha256(old).hexdigest()
    def capture_functions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
            if isinstance(n,ast.FunctionDef) and n.name in ('capture_emf','canonical_emf','tactile_input')}
    if capture_functions(old)!=capture_functions(Path(cart_observations.__file__).read_bytes()):
        raise ValueError('Capture geometry changed; full preparation is required')
    here=Path(cart_capture.__file__).parent
    implementations={name:file_hash(here/name) for name in ('cart_capture.py','cart_capture_schema.py','cart_data.py',
        'cart_contacts.py','cart_contact_witness.py','cart_observations.py','qnbot_surface.py','hand_anatomy.py','tactile_surface.py')}
    implementations['cart_observations.py']=old_hash
    old_capture=digest(dict(schema=CAPTURE_SCHEMA,
        source={name:file_hash(folder/name) for name in ('profile.json','topology.json')},
        implementation=implementations,linked_emf=file_hash(folder/'cart_emf.jsonl') if (folder/'cart_emf.jsonl').exists() else None))
    if capture['identity']!=old_capture or capture['raw_sha256']!=file_hash(folder/'frames.jsonl') or capture['sha256']!=file_hash(folder/'cart_capture.jsonl'):
        raise ValueError('Original geometry cache does not match the reviewed legacy implementation')
    files=('motion.npz','timestamps.npy','profile.json','topology.json','points.npy','frames.jsonl','cart_emf.jsonl','cart_capture.jsonl','cart_capture_meta.json')
    old_prepared=digest(dict(protocol='cart_diff_v5',source=m['source'],full=True,
        contact=contact_fingerprint(folder),observation=old_hash,adapter=file_hash(cart_data.__file__),
        files={n:file_hash(folder/n) if (folder/n).exists() else None for n in files}))
    if prepared['protocol']!='cart_diff_v5' or prepared['profile']!='full' or prepared['fingerprint']!=old_prepared or prepared['targets_sha256']!=file_hash(folder/'cart_targets.jsonl'):
        raise ValueError('Original prepared observations changed; refusing cache migration')
    if ik['fingerprint']!=identity(folder,robot,session) or ik['sha256']!=file_hash(folder/'cart_ik.npz'):
        raise ValueError('Original IK cache changed; refusing cache migration')
    before={p.name:file_hash(p) for p in paths}
    for path in paths:
        backup=path.with_name(path.stem+'.before_ik_input_v6.json')
        if backup.exists():raise ValueError('Metadata backup exists; inspect an interrupted migration')
        backup.write_bytes(path.read_bytes())
    capture['identity']=capture_identity(folder)
    atomic_json(paths[0],capture)
    prepared.update(protocol=PROTOCOL,fingerprint=input_digest(folder,m,True))
    atomic_json(paths[1],prepared)
    ik['fingerprint']=identity(folder,robot,session)
    atomic_json(paths[2],ik)
    report=dict(dataset=key,migrated=True,legacy_commit=LEGACY_COMMIT,metadata_before=before,
        metadata_after={p.name:file_hash(p) for p in paths},targets_sha256=prepared['targets_sha256'],
        ik_sha256=ik['sha256'],reason='Policy packing changed; capture geometry functions and all data bytes verified unchanged')
    atomic_json(folder/'ik_input_cache_migration.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--datasets',nargs='+',required=True)
    args=parser.parse_args()
    for key in args.datasets:print(json.dumps(migrate(args.root,key)),flush=True)

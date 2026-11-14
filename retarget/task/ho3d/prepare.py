"""Recover all contiguous labelled train segments, resumably, one NPZ per segment."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .source import Source,contiguous_segments,recover_segment


def prepare(source_path,mano_path,output,*,sequences=None,fps=30.,max_frames=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=Source(source_path);selected=sequences or source.sequences()
    mano_hash=hashlib.sha256(Path(mano_path).read_bytes()).hexdigest()
    manifest={'source':str(Path(source_path).resolve()),'split':'train','mano_sha256':mano_hash,'fps_assumed':fps,
              'coordinate_policy':'OpenGL camera to Z-up, constant first-object recenter; camera Y assumed gravity-up',
              'tip_vertices':[744,333,444,555,672],'segments':[],'started_at':time.time()}
    try:
        for sequence in selected:
            for k,rows in enumerate(contiguous_segments(source.annotations(sequence))):
                if len(rows)<2:continue
                if max_frames:rows=rows[:max_frames]
                stem=f'{sequence}_{rows[0][0]:04d}_{rows[-1][0]:04d}'
                path=out/(stem+'.npz')
                fingerprint=dict(sequence=sequence,start_frame=rows[0][0],end_frame=rows[-1][0],frames=len(rows),object=rows[0][1]['objName'],mano_sha256=mano_hash,fps=fps)
                receipt=out/(stem+'.json')
                if path.exists() and receipt.exists():
                    old=json.loads(receipt.read_text())
                    if all(old.get(n)==v for n,v in fingerprint.items()):
                        manifest['segments'].append(old);continue
                    raise ValueError(f'Existing segment has different provenance: {path}')
                motion=recover_segment(rows,mano_path,fps)
                with (out/(stem+'.npz.partial')).open('wb') as f:np.savez_compressed(f,**motion)
                (out/(stem+'.npz.partial')).replace(path)
                info=dict(fingerprint,path=str(path.resolve()),source=str(Path(source_path).resolve()),split='train',fk_joint_rmse_m=float(motion.get('fk_joint_rmse_m',np.nan)),
                          coordinate_policy=manifest['coordinate_policy'])
                receipt.write_text(json.dumps(info,indent=2));manifest['segments'].append(info)
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
                print(json.dumps(info),flush=True)
    finally:source.close()
    manifest['finished_at']=time.time();manifest['frames']=sum(x['frames'] for x in manifest['segments'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

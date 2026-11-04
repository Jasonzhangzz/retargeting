#!/usr/bin/env python3
"""Join reviewed MANO slots to WUJI2 surface anchors via immutable source annotations."""
from pathlib import Path
import sys,json,argparse,csv,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from retarget.task.grasp.dexonomy import load_template,map_contacts,wuji_body
from retarget.env.grasp.hand import hand_tree,query_path
from scripts.retarget_dexonomy_mano import read,write,sha


def prepare(source,output):
    raw=read(source);mappings=raw['mappings'];model=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'))
    result={};assets={};distances=[]
    for name,records in mappings.items():
        shadow,_,contacts,annotation=load_template(ROOT/'data/grasp/templates',name)
        if len(contacts)!=len(records):raise ValueError('Contact slot count changed: '+name)
        bodies,points,normals,robot=map_contacts(shadow,contacts,model)
        joined=[]
        for index,(record,(body,local),anchor) in enumerate(zip(records,contacts,robot)):
            if body!=record['shadow_body']:raise ValueError('Slot order changed: '+name)
            np.testing.assert_allclose(local[:3],record['shadow_local_point'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(local[3:],record['shadow_local_normal'],atol=1e-12,rtol=0)
            assert anchor['wuji_body']==wuji_body(body)
            qp=query_path(model,anchor['wuji_body']);assets[str(qp.relative_to(ROOT))]=sha(qp)
            with np.load(qp,allow_pickle=False) as pool:
                np.testing.assert_array_equal(pool['local_pos'][anchor['query_index']],points[index])
            joined.append(dict(slot_id=record['id'],finger=record['finger'],mano_face_id=record['mano_face_id'],
                mano_barycentric=record['mano_barycentric'],**anchor,mano_mapping_needs_review=record.get('needs_review',False)))
            distances.append(anchor['mapping_distance_mm'])
        assets[str(annotation.relative_to(ROOT))]=sha(annotation);result[name]=joined
    assets['data/grasp/assets/wuji2/right.xml']=sha(ROOT/'data/grasp/assets/wuji2/right.xml')
    payload=dict(version=1,mappings=result,mano_mapping_path=str(Path(source).resolve()),mano_mapping_sha256=sha(source),
        mano_model_sha256=raw['model_sha256'],assets_sha256=assets,
        semantics='Same original Shadow annotation slot -> reviewed MANO triangle and same anatomical WUJI2 link surface query; object-independent; no inferred cross-finger nearest neighbour.',
        audit=dict(types=len(result),slots=sum(map(len,result.values())),same_source_slot=True,same_finger_segment=True,query_anchor_exact=True,
                   max_wuji_mapping_distance_mm=max(distances),wuji_mapping_distance_p95_mm=float(np.percentile(distances,95))))
    write(output,payload)
    write(Path(output).with_name('audit.json'),payload['audit'])
    with Path(output).with_name('contact_mapping.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['grasp_type','slot','finger','mano_face','wuji_body','wuji_query','mapping_distance_mm','mano_needs_review'])
        for name,records in result.items():
            for r in records:writer.writerow([name,r['slot_id'],r['finger'],r['mano_face_id'],r['wuji_body'],r['query_index'],r['mapping_distance_mm'],r['mano_mapping_needs_review']])
    print(json.dumps(payload['audit']),flush=True)
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mano-mapping',type=Path,default=ROOT/'output/dexonomy_shadow_mano_mapping_20261103/data.json');p.add_argument('--output',type=Path,default=ROOT/'output/mano_wuji2_mapping_20261104/mapping.json');a=p.parse_args();prepare(a.mano_mapping,a.output)

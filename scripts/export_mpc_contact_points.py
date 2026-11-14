#!/usr/bin/env python3
"""Overlay actual-object anchors and matched hand points on the paired replay."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/ho3d_mpc_contact_20261112'))
    a = p.parse_args()
    import numpy as np
    import mujoco
    import warp as wp
    from robokit.sim import MjWarpSim
    from retarget.env.dynamic.contact_reference import load_contact_reference
    status = json.loads((a.root/'status.json').read_text())
    for item in status['results']:
        case = Path(item['case'])
        viewer = a.root/item['object']/'viewer'
        packet = json.loads((viewer/'replay.json').read_text())
        model = mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
        ref = load_contact_reference(case, model)
        with np.load(case/'mpc_contact/rollout.npz') as z:
            states = z['qpos'][9::10].astype(np.float32)
        n = len(states)
        with wp.ScopedDevice('cuda:0'):
            sim = MjWarpSim(model, n, mujoco.MjData(model), nconmax=1, njmax=1, device='cuda:0')
            with wp.ScopedStream(sim.stream):
                sim.data.qpos.assign(states)
                sim.kinematics()
                wp.synchronize_stream(sim.stream)
                pos, mat = sim._d.xpos.numpy(), sim._d.xmat.numpy()
        ids = np.arange(10, 10*n+1, 10)
        body, rows = ref.body[ids], np.arange(n)[:, None]
        hand = pos[rows, body] + np.einsum('ncij,ncj->nci', mat[rows, body], ref.hand_local[ids])
        obj = model.body('tracked_object').id
        anchors = pos[:, obj, None] + np.einsum('nij,ncj->nci', mat[:, obj], ref.object_local[ids])
        packet['contact_points'] = dict(mask=(ref.weight[ids]>0).tolist(), hand=hand.round(7).tolist(), anchor=anchors.round(7).tolist())
        (viewer/'replay.json').write_text(json.dumps(packet, separators=(',', ':')))
        html = (viewer/'index.html').read_text()
        if 'function drawContacts' in html:
            continue
        html = html.replace('</header>', '<label><input id="contacts" type="checkbox" checked>接触点：红锚点／绿手指点</label></header>')
        html = html.replace('鼠标拖动旋转，滚轮缩放。', '红点是随实际物体运动的参考锚点，绿点是对应手指材料点，连线表示分离误差。鼠标拖动旋转，滚轮缩放。')
        javascript = '''
const contactGroup=new THREE.Group();scene.add(contactGroup);
const pointGeometry=new THREE.SphereGeometry(.0017,8,6);
const contactMarkers=data.contact_points.mask[0].map(()=>{
 const hand=new THREE.Mesh(pointGeometry,new THREE.MeshBasicMaterial({color:0x46ffa1,depthTest:false}));
 const anchor=new THREE.Mesh(pointGeometry,new THREE.MeshBasicMaterial({color:0xff485b,depthTest:false}));
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(6),3));
 const line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:0xffcc5c,depthTest:false}));
 hand.renderOrder=10;anchor.renderOrder=10;line.renderOrder=9;contactGroup.add(hand,anchor,line);return {hand,anchor,line};
});
function drawContacts(frame){
 contactGroup.visible=el('contacts').checked&&el('robot').checked;
 const points=data.contact_points;
 contactMarkers.forEach((m,c)=>{
  const active=points.mask[frame][c];m.hand.visible=active;m.anchor.visible=active;m.line.visible=active;
  if(active){const h=points.hand[frame][c],a=points.anchor[frame][c];m.hand.position.set(...h);m.anchor.position.set(...a);m.line.geometry.attributes.position.array.set([...h,...a]);m.line.geometry.attributes.position.needsUpdate=true;}
 });
}
'''
        html = html.replace('let playing=true,', javascript+'\nlet playing=true,')
        html = html.replace("['human','ghost','robot','base']", "['human','ghost','robot','base','contacts']")
        html = html.replace("el('seek').value=frame;el('status')", "drawContacts(frame);el('seek').value=frame;el('status')")
        (viewer/'index.html').write_text(html)
        print(json.dumps(dict(object=item['object'], frames=n, active_samples=int((ref.weight[ids]>0).sum()))), flush=True)


if __name__ == '__main__':
    main()

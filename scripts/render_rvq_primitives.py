"""Export one fixed MANO primitive illustration per RVQ class (offline only)."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/datasets/dex_grasping_primitives_20261115')
    parser.add_argument('--output', type=Path, default=ROOT/'moxian_web/rvq_primitives')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.dataset/'source-manifest.json').read_text())
    with np.load(args.dataset/'source-mano_geometry.npz', allow_pickle=False) as geometry:
        faces = geometry['faces'].copy()
    catalog = {}
    for entry in source['entries']:
        path = args.dataset/'wuji2'/entry['file']
        with np.load(path, allow_pickle=False) as data:
            valid = np.flatnonzero(data['valid'] & np.isfinite(data['mano_vertices']).all(axis=(1, 2)))
            if not len(valid):
                raise ValueError('No finite valid primitive: '+entry['type'])
            row = int(valid[np.argmin(np.abs(data['size_scale'][valid]-1.0))])
            points = data['mano_joints'][row].astype(float)
            z = points[9]-points[0]; z /= np.linalg.norm(z)
            x = np.cross(points[5]-points[13], z); x /= np.linalg.norm(x)
            basis = np.stack((x, np.cross(z, x), z), axis=1)
            vertices = (data['mano_vertices'][row]-points[0])@basis
            radius, height = float(data['radius_m'][row]), float(data['height_m'][row])
            obj = (trimesh.creation.icosphere(subdivisions=2, radius=radius) if entry['kind']=='sphere'
                   else trimesh.creation.cylinder(radius=radius, height=height, sections=40))
            pose = data['object_pose_world'][row]
            obj_vertices = ((obj.vertices@Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix().T+pose[:3])-points[0])@basis
            scale = float(data['size_scale'][row])
        fig = plt.figure(figsize=(3.2, 2.3), dpi=120, facecolor='#f4f7f3')
        ax = fig.add_axes([0, 0, 1, 1], projection='3d', facecolor='#f4f7f3')
        # Plot anatomical lateral/depth/forward axes, with fingers pointing up.
        v = vertices[:, [1, 0, 2]]; ov = obj_vertices[:, [1, 0, 2]]
        triangles=v[faces]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        light=np.array([.3,-.5,1.]);light/=np.linalg.norm(light)
        colors=np.array([.94,.84,.69])*(.78+.22*np.abs(normals@light))[:,None]
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none',
            linewidths=0, antialiased=False))
        ax.add_collection3d(Poly3DCollection(ov[obj.faces], facecolors='#64aaa5', linewidths=0,
            alpha=.35, shade=True, lightsource=matplotlib.colors.LightSource(azdeg=125, altdeg=50)))
        bounds = np.concatenate([v, ov]); center = (bounds.min(0)+bounds.max(0))/2
        span = max(np.ptp(bounds, axis=0))*.57
        ax.set(xlim=(center[0]-span, center[0]+span), ylim=(center[1]-span, center[1]+span),
               zlim=(center[2]-span, center[2]+span))
        ax.set_box_aspect((1, 1, 1), zoom=1.27)
        ax.view_init(elev=16, azim=-65); ax.set_proj_type('ortho'); ax.set_axis_off()
        filename = entry['type']+'.png'
        fig.savefig(args.output/filename, dpi=120); plt.close(fig)
        catalog[entry['type']] = dict(file=filename, source=path.relative_to(ROOT).as_posix(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), row=row, size_scale=scale,
            image_sha256=hashlib.sha256((args.output/filename).read_bytes()).hexdigest())
    (args.output/'catalog.json').write_text(json.dumps(dict(schema='rvq_primitive_illustrations_v1',
        description='Fixed source MANO templates; valid row nearest unit size; not live reconstruction',
        primitives=catalog), ensure_ascii=False, indent=2)+'\n')
    print(f'Exported {len(catalog)} static primitive illustrations to {args.output}')


if __name__ == '__main__':
    main()

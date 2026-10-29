#!/usr/bin/env python3
"""Render real saved poses for every supported type, without synthetic images."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_rows(paths):
    rows = []
    for path in paths:
        if not path.exists():
            continue
        files = [path] if path.is_file() else sorted(path.glob('type_*/summary.json'))
        for source in files:
            rows.extend(json.loads(source.read_text()))
    return rows


def render(row, width, height):
    import mujoco
    import numpy as np
    from comfree_warp.gaussian_loader import _preprocess_xml
    case = Path(row['case'])
    report = json.loads((case/'result.json').read_text())
    start = report['selected']['start']
    source = case/f'input_{start:02d}/sequence/scene/wuji_right/primitive_sdf.xml'
    prepared = _preprocess_xml(str(source))
    model = mujoco.MjModel.from_xml_string(prepared.model_xml, assets=prepared.assets)
    data = mujoco.MjData(model)
    with np.load(case/'grasp.npz', allow_pickle=False) as saved:
        data.qpos[:] = saved['qpos']
    mujoco.mj_forward(model, data)
    obj = data.xpos[model.body('grasp_object').id]
    palm = data.xpos[model.body('right_palm_link').id]
    direction = obj-palm
    camera = mujoco.MjvCamera()
    camera.lookat[:] = .7*obj+.3*palm
    camera.distance = max(.29, 1.9*(float(np.linalg.norm(direction))+.065))
    camera.azimuth = np.degrees(np.arctan2(direction[1], direction[0]))+200
    camera.elevation = -np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2])))
    model.vis.headlight.ambient[:] = .5
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 0, 0, 0, 0]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        pixels = renderer.render().copy()
    return pixels


def main():
    from PIL import Image, ImageDraw, ImageFont
    from retarget.grasp_primitives import catalog, CATALOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    types = [row for row in catalog(ROOT/'third_party/Dexonomy') if row['supported']]
    maps = []
    for rows in (read_rows([args.legacy]), read_rows(args.gpu)):
        chosen = {}
        for row in rows:
            primitive = row.get('primitive') or {}
            number = int(row['template'].split('_')[0])
            if (math.isclose(primitive.get('radial_scale', 0), 1.)
                    and math.isclose(primitive.get('axial_scale', 0), 1.)
                    and primitive.get('kind') == CATALOG[number][0][0]):
                chosen[number] = row
        maps.append(chosen)
    tile_w, tile_h, label_h, header_h = 300, 212, 55, 82
    block_w, block_h = 2*tile_w+12, tile_h+label_h+15
    canvas = Image.new('RGB', (2*block_w+24, math.ceil(len(types)/2)*block_h+header_h+12), '#edf0f3')
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 14)
    title_font = ImageFont.truetype(font_path, 20)
    draw.text((12, 10), '31 grasp types: Legacy serial vs GPU nodes', font=title_font, fill='#17212b')
    draw.text((12, 39), 'Actual saved poses | primary shape, size factor 1.0 | green: FC + geometry valid', font=font, fill='#34404c')
    draw.text((12, 59), 'Gray GPU tiles are pending. Static closure does not certify physical retention.', font=font, fill='#596571')
    rendered = []
    for index, item in enumerate(types):
        x = 12+(index % 2)*block_w
        y = header_h+(index//2)*block_h
        number = item['type_id']
        draw.text((x+3, y+2), item['template'].replace('_', ' '), font=font, fill='#17212b')
        for method, label in enumerate(('Legacy', 'GPU')):
            tx, ty = x+method*tile_w, y+24
            row = maps[method].get(number)
            if row is None:
                draw.rectangle((tx, ty, tx+tile_w-2, ty+tile_h), fill='#ccd3da')
                draw.text((tx+85, ty+90), 'PENDING', font=title_font, fill='#607080')
                draw.text((tx+6, ty+tile_h+5), label+' | not finished', font=font, fill='#596571')
            else:
                pixels = render(row, tile_w-2, tile_h)
                canvas.paste(Image.fromarray(pixels), (tx, ty))
                fc = row['force_closure']['force_closure']
                valid = row['feasible']
                color = '#08793e' if valid else '#b63732'
                text = f'{label} | FC {"YES" if fc else "NO"} | valid {"YES" if valid else "NO"}'
                draw.text((tx+4, ty+tile_h+5), text, font=font, fill=color)
                rendered.append(dict(type_id=number, method=label, case=row['case'], force_closure=fc, feasible=valid))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        generated_at=datetime.now().isoformat(), primary_size_only=True,
        legacy_types=len(maps[0]), gpu_types=len(maps[1]), rendered=rendered), indent=2))
    print(json.dumps(dict(image=str(args.output), legacy_types=len(maps[0]), gpu_types=len(maps[1]),
                         width=canvas.width, height=canvas.height)))


if __name__ == '__main__':
    main()

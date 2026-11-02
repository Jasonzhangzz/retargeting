#!/usr/bin/env python3
"""Build a portable inspection HTML from a small Plotly-based data contract."""

import argparse
import json
from pathlib import Path


def validate(data):
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def identified(items, where):
        require(isinstance(items, list) and bool(items), f'{where} must be a nonempty list')
        ids = [item.get('id') for item in items]
        require(all(isinstance(v, str) and v.strip() for v in ids), f'{where} needs string IDs')
        require(len(ids) == len(set(ids)), f'{where} has duplicate IDs')

    require(isinstance(data, dict), 'Root must be an object')
    require(isinstance(data.get('title'), str) and bool(data['title'].strip()), 'title is required')
    views = data.get('views')
    require(isinstance(views, list) and bool(views), 'views must be a nonempty list')
    require(all(isinstance(v.get('label'), str) for v in views), 'Each view needs a label')
    identified(data.get('cases'), 'cases')
    for case in data['cases']:
        identified(case.get('variants'), f"{case['id']}.variants")
        for variant in case['variants']:
            where = f"{case['id']}/{variant['id']}"
            figures = variant.get('figures')
            require(isinstance(figures, list) and len(figures) == len(views), f'{where}: figure/view counts differ')
            if variant.get('rows'):
                identified(variant['rows'], f'{where}.rows')
            for figure in figures:
                require(isinstance(figure.get('data'), list), f'{where}: figure.data must be a list')
                require(isinstance(figure.get('layout', {}), dict), f'{where}: layout must be an object')
                for trace in figure['data']:
                    if trace.get('type') in ('scatter3d', 'mesh3d'):
                        arrays = [trace.get(k) for k in ('x', 'y', 'z')]
                        require(all(isinstance(v, list) for v in arrays), f'{where}: 3D x/y/z must be arrays')
                        require(len({len(v) for v in arrays}) == 1, f'{where}: 3D coordinate lengths differ')
                        require(all(x is None or isinstance(x, (int, float)) for arr in arrays for x in arr), f'{where}: invalid 3D coordinate')
                        if 'customdata' in trace:
                            require(len(trace['customdata']) == len(arrays[0]), f'{where}: customdata length differs')
                    if trace.get('type') == 'mesh3d':
                        faces = [trace.get(k) for k in ('i', 'j', 'k')]
                        require(all(isinstance(v, list) for v in faces), f'{where}: mesh i/j/k must be arrays')
                        require(len({len(v) for v in faces}) == 1, f'{where}: mesh face lengths differ')
                        require(all(isinstance(x, int) and 0 <= x < len(trace['x']) for arr in faces for x in arr), f'{where}: mesh index out of range')
    # Also rejects nested NaN/Infinity in arbitrary traces and layouts.
    json.dumps(data, allow_nan=False)


def demo():
    cases = []
    for number in range(2):
        variants = []
        for stage, offset in [('raw', .016), ('adjusted', .004)]:
            figures = []
            for side in range(2):
                x = [-.04 + offset*side, offset*side, .04 + offset*side]
                z = [.08, .11 + .01*number, .09]
                ids = ['p1', 'p2', 'p3']
                points = dict(type='scatter3d', mode='markers+text', x=x, y=[0]*3, z=z,
                              customdata=ids, text=['1', '2', '3'], textposition='top center',
                              marker=dict(size=7, color=['#0077c8', '#dc8b09', '#00a080']), name='对应点')
                surface = dict(type='mesh3d', x=[-.06,.06,.06,-.06], y=[-.008]*4,
                               z=[0,0,.15,.15], i=[0,0], j=[1,2], k=[2,3],
                               color='#a9b8b1', opacity=.85, meta=dict(role='surface'), hoverinfo='skip')
                normals = dict(type='scatter3d', mode='lines', x=[x[0],x[0],None,x[1],x[1],None,x[2],x[2]],
                               y=[0,.025,None,0,.025,None,0,.025],
                               z=[z[0],z[0],None,z[1],z[1],None,z[2],z[2]],
                               line=dict(color='#006e65',width=4), meta=dict(role='normals'), hoverinfo='skip')
                figures.append(dict(data=[surface,points,normals], layout=dict(showlegend=False,
                    scene=dict(xaxis=dict(title='x / m'), yaxis=dict(title='y / m'), zaxis=dict(title='z / m')))))
            variants.append(dict(id=stage, label='原始对照' if stage=='raw' else '调整后对照',
                note='演示数据：用于检查页面交互，非真实测量或模型结果。',
                metrics=[dict(label='对应点数', value=3),dict(label='演示偏移',value=offset*1000,unit='mm')],
                rows=[dict(id=f'p{i+1}',point=str(i+1),offset=f'{offset*1000:.1f}') for i in range(3)], figures=figures))
        cases.append(dict(id=f'demo-{number+1}',label=f'演示案例 {number+1}',variants=variants))
    return dict(title='模型对照 · 演示数据', eyebrow='RESULTS / INSPECTOR',
        subtitle='复用核对页风格，演示案例切换、阶段对照、同步视角与逐点高亮。',
        views=[dict(label='源视图'),dict(label='目标视图')],
        columns=[dict(key='point',label='编号'),dict(key='offset',label='演示偏移 / mm')],
        legend=[dict(label=label,color=color) for label,color in [('点 1','#0077c8'),('点 2','#dc8b09'),('点 3','#00a080')]],
        footer='此页仅为生成器示例；几何单位 m。表格和指标明确标为演示值。',
        sources=['数据来源：skill 内置合成示例。'], cases=cases)


def build(data, output, plotly_js=None):
    validate(data)
    if plotly_js:
        library = Path(plotly_js).read_text(encoding='utf-8')
    else:
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as error:
            raise RuntimeError('Use a Python environment with plotly, or supply --plotly-js.') from error
        library = get_plotlyjs()
    template = (Path(__file__).resolve().parents[1]/'assets/inspector.html').read_text(encoding='utf-8')
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    for character, escaped in [('&', r'\u0026'), ('<', r'\u003c'), ('>', r'\u003e')]:
        serialized = serialized.replace(character, escaped)
    for token in ('/*__DATA__*/null', '/*__PLOTLY__*/'):
        if template.count(token) != 1:
            raise ValueError(f'Template token missing or repeated: {token}')
    # Script closing tags are case insensitive in HTML, even inside JS strings.
    import re
    library = re.sub(r'</script', r'<\\/script', library, flags=re.IGNORECASE)
    replacements = {'/*__DATA__*/null': serialized, '/*__PLOTLY__*/': library}
    html = re.sub(r'/\*__(?:DATA|PLOTLY)__\*/(?:null)?',
                  lambda match: replacements[match.group()], template)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    (output.parent/f'{output.stem}.data.json').write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    print(f'Created {output.resolve()} ({len(data["cases"])} cases; offline single HTML)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path)
    source.add_argument('--demo',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plotly-js',type=Path)
    args=parser.parse_args()
    build(demo() if args.demo else json.loads(args.data.read_text(encoding='utf-8')),args.output,args.plotly_js)

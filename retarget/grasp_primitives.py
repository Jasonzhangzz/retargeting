"""Auditable GRASP shape families and explicit primitive size experiments.

Dexonomy does NOT ship a per-type primitive table. The associations below are
benchmark choices based on the GRASP taxonomy shape descriptions, not upstream
Dexonomy labels. All physical dimensions and scale grids are our test settings.
Scissors and paired chopsticks are kept as special categories, not silently
replaced with one cylinder.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

SOURCES = dict(
    dexonomy='https://arxiv.org/html/2504.18829v2#S4.SS1',
    taxonomy='https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf',
    shape_catalog='https://www.cs.cmu.edu/~jialiu1/database.html')

# id: primitive families; r=radius, h=full height, xyz=full box dimensions (m).
CATALOG = {
    1: [('cylinder', .035, .10), ('cuboid', .035, .10)],
    2: [('cylinder', .012, .10)], 3: [('cylinder', .025, .10)],
    4: [('cylinder', .016, .10), ('cuboid', .016, .10)],
    5: [('cylinder', .005, .10)], 6: [('cylinder', .020, .08)],
    7: [('cylinder', .012, .08)], 8: [('cylinder', .009, .08)],
    9: [('disk', .018, .004)], 10: [('disk', .040, .015)],
    11: [('sphere', .035, .070)],
    12: [('disk', .045, .008), ('plate', .045, .008)],
    13: [('sphere', .020, .040), ('cuboid', .020, .040)],
    14: [('sphere', .015, .030)], 15: [('cylinder', .015, .10)],
    16: [('plate', .025, .003)], 17: [('cylinder', .025, .10)],
    18: [('plate', .040, .008)], 19: [('scissors', None, None)],
    20: [('cylinder', .004, .12)], 21: [('paired_sticks', None, None)],
    22: [('plate', .035, .008)],
    23: [('cylinder', .005, .08), ('plate', .020, .003)],
    24: [('cylinder', .006, .05)],
    25: [('cylinder', .007, .06), ('sphere', .012, .024)],
    26: [('sphere', .030, .060)], 27: [('sphere', .020, .040)],
    28: [('sphere', .025, .050)], 29: [('cylinder', .007, .10)],
    30: [('disk', .030, .025)], 31: [('cylinder', .025, .10)],
    32: [('cylinder', .008, .10)],
    33: [('sphere', .008, .016), ('plate', .015, .003)],
}


def catalog(checkout):
    paths = Path(checkout).expanduser().resolve().glob('assets/hand/shadow/raw_anno/*.yaml')
    names = sorted((p.stem for p in paths if p.stem.partition('_')[0].isdigit()),
                   key=lambda name: int(name.partition('_')[0]))
    return [dict(template=name, type_id=int(name.partition('_')[0]),
                 families=[x[0] for x in CATALOG[int(name.partition('_')[0])]],
                 source=SOURCES, mapping_status='taxonomy_based_benchmark_association_not_Dexonomy_metadata',
                 supported=int(name.partition('_')[0]) not in (19, 21)) for name in names]


def primitive_geometry(kind, radius, height):
    if kind in ('cuboid', 'plate'):
        size = np.array([2*radius, 1.4*radius, height])
        vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])*size
    elif kind in ('cylinder', 'disk'):
        angle = np.arange(48)*2*np.pi/48
        vertices = np.array([[radius*np.cos(a), radius*np.sin(a), z]
                             for z in (-height/2, height/2) for a in angle])
    elif kind == 'sphere':
        z = 1-2*(np.arange(194)+.5)/194
        angle = np.arange(len(z))*np.pi*(3-np.sqrt(5))
        vertices = radius*np.c_[np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z]
    else:
        raise ValueError(f'Not a single rigid primitive: {kind}')
    faces = ConvexHull(vertices).simplices.copy()
    for i, face in enumerate(faces):
        a, b, c = vertices[face]
        if np.dot(np.cross(b-a, c-a), (a+b+c)/3) < 0:
            faces[i] = face[[0, 2, 1]]
    return vertices, faces


def primitive_mesh(path, kind, radius, height):
    vertices, faces = primitive_geometry(kind, radius, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join('v '+' '.join(map(str, v))+'\n' for v in vertices)
                    + ''.join('f '+' '.join(map(str, f+1))+'\n' for f in faces))
    return path


def jobs(output, templates, args):
    """Each shape/dimension combination is a separate object instance."""
    rows, skipped = [], []
    for template in templates:
        number = template.partition('_')[0]
        if not number.isdigit() or int(number) not in CATALOG:
            raise ValueError(f'No taxonomy primitive association for {template}; use --object for extra templates')
        families = CATALOG[int(number)]
        if getattr(args, 'primary_shape_only', False):
            families = families[:1]
        for kind, r, h in families:
            if args.primitive and kind != args.primitive:
                continue
            if r is None:
                skipped.append(dict(template=template, category=kind,
                                    reason='requires an articulated or multi-object asset; not approximated by a single primitive'))
                continue
            for radial in args.radius_scales:
                # Sphere has only one independent size; do not duplicate it by height.
                for axial in ([1.] if kind == 'sphere' else args.height_scales):
                    radius = (args.radius if args.radius is not None else r)*radial
                    height = (args.height if args.height is not None else h)*axial
                    if kind == 'sphere': height = 2*radius
                    tag = f'{template}_{kind}_r{radius:.7f}_h{height:.7f}'
                    path = primitive_mesh(output / 'meshes' / f'{tag}.obj', kind, radius, height)
                    rows.append((path, template, dict(kind=kind, radius_m=radius,
                                 height_m=2*radius if kind == 'sphere' else height,
                                 radial_scale=radial, axial_scale=axial, sources=SOURCES,
                                 dimensions_status='explicit_benchmark_setting_not_Dexonomy_annotation')))
    return rows, skipped

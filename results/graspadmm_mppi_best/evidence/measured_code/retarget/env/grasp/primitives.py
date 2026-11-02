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
    dexonomy="https://arxiv.org/html/2504.18829v2#S4.SS1",
    taxonomy="https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf",
    shape_catalog="https://www.cs.cmu.edu/~jialiu1/database.html",
)

# id: primitive families; r=radius, h=full height, xyz=full box dimensions (m).
CATALOG = {
    1: [("cylinder", 0.035, 0.12), ("cuboid", 0.035, 0.12)],
    2: [("cylinder", 0.012, 0.12)],
    3: [("cylinder", 0.025, 0.12)],
    4: [("cylinder", 0.016, 0.12), ("cuboid", 0.016, 0.12)],
    5: [("cylinder", 0.005, 0.12)],
    6: [("cylinder", 0.020, 0.12)],
    7: [("cylinder", 0.012, 0.12)],
    8: [("cylinder", 0.009, 0.12)],
    9: [("cylinder", 0.018, 0.12)],
    10: [("disk", 0.040, 0.015)],
    11: [("sphere", 0.035, 0.070)],
    12: [("disk", 0.045, 0.016), ("plate", 0.045, 0.008)],
    13: [("sphere", 0.020, 0.040), ("cuboid", 0.020, 0.040)],
    14: [("sphere", 0.015, 0.030)],
    15: [("cylinder", 0.015, 0.20)],
    16: [("plate", 0.025, 0.003)],
    17: [("cylinder", 0.025, 0.20)],
    18: [("plate", 0.040, 0.008)],
    19: [("scissors", None, None)],
    20: [("cylinder", 0.004, 0.12)],
    21: [("paired_sticks", None, None)],
    22: [("plate", 0.035, 0.008)],
    23: [("cylinder", 0.005, 0.08), ("plate", 0.020, 0.003)],
    24: [("cylinder", 0.006, 0.05)],
    25: [("cylinder", 0.007, 0.06), ("sphere", 0.012, 0.024)],
    26: [("sphere", 0.030, 0.060)],
    27: [("sphere", 0.020, 0.040)],
    28: [("sphere", 0.025, 0.050)],
    29: [("cylinder", 0.007, 0.20)],
    30: [("disk", 0.030, 0.008)],
    31: [("cylinder", 0.025, 0.10)],
    32: [("cylinder", 0.008, 0.20)],
    33: [("sphere", 0.008, 0.016), ("plate", 0.015, 0.003)],
}

SIZE_POLICY_VERSION = 2
DEFAULT_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)


def size_specifications(type_id, kind=None, *, radius_scales=None,
                        height_scales=None, radius=None, height=None,
                        explicit=False, cylinder_length=0.18):
    """Return the shared, physical per-type experiment dimensions.

    ``explicit`` retains the historical radius/height Cartesian grid. It must
    come from the original configuration, before generic defaults are injected.
    ``size_scale`` describes the axis being swept; ``radial_scale`` always
    describes the actual radius, including the fixed 1.5x plate widths.
    """
    type_id = int(type_id)
    if type_id not in CATALOG:
        raise ValueError(f"Unknown grasp type {type_id}")
    if not np.isfinite(cylinder_length) or cylinder_length <= 0:
        raise ValueError("Cylinder length must be finite and positive")
    rows = []
    for family, base_radius, base_height in CATALOG[type_id]:
        if kind is not None and family != kind or base_radius is None:
            continue
        if explicit:
            radial = list([1.] if radius_scales is None else radius_scales)
            axial = list([1.] if height_scales is None else height_scales)
            if not radial or not axial or any(not np.isfinite(v) or v <= 0 for v in radial + axial):
                raise ValueError("Primitive size grids must be finite and positive")
            base_r = base_radius if radius is None else float(radius)
            base_h = base_height if height is None else float(height)
            samples = [(r, a, r, "radius") for r in radial
                       for a in ([1.] if family == "sphere" else axial)]
        else:
            base_r, base_h = base_radius, base_height
            if family == "cylinder" and type_id <= 6:
                base_h = cylinder_length
            scales = DEFAULT_SCALES
            if type_id == 10:
                scales = np.linspace(1., 1.5, 11)
            elif type_id in (11, 28):
                scales = np.linspace(1., 1.5, 5)
            elif type_id == 12:
                scales = np.linspace(.5, 1., 5)
            elif type_id == 26:
                scales = np.linspace(.75, 1.5, 5)
            elif type_id == 33 and family == "sphere":
                scales = np.linspace(1., 2., 5)
            if type_id in (16, 22) and family == "plate":
                samples = [(1.5, s, s, "thickness") for s in scales]
            elif type_id == 30 and family == "disk":
                samples = [(1., s, s, "thickness") for s in scales]
            else:
                samples = [(s, 1., s, "radius") for s in scales]
        for radial, axial, scale, axis in samples:
            r, h = base_r * radial, base_h * axial
            if family == "sphere":
                h = 2 * r
            if not np.isfinite([r, h]).all() or min(r, h) <= 0:
                raise ValueError("Primitive dimensions must be finite and positive")
            rows.append(dict(kind=family, radius_m=float(r), height_m=float(h),
                radial_scale=float(radial), axial_scale=float(axial),
                size_scale=float(scale), size_axis=axis,
                dimensions_m=([float(2*r), float(1.4*r), float(h)]
                    if family in ("plate", "cuboid") else [float(2*r), float(2*r), float(h)]),
                size_policy_version=SIZE_POLICY_VERSION, sources=SOURCES,
                dimensions_status="explicit_benchmark_setting_not_Dexonomy_annotation"))
    return rows


def catalog(checkout):
    paths = (
        Path(checkout).expanduser().resolve().glob("assets/hand/shadow/raw_anno/*.yaml")
    )
    names = sorted(
        (p.stem for p in paths if p.stem.partition("_")[0].isdigit()),
        key=lambda name: int(name.partition("_")[0]),
    )
    return [
        dict(
            template=name,
            type_id=int(name.partition("_")[0]),
            families=[x[0] for x in CATALOG[int(name.partition("_")[0])]],
            source=SOURCES,
            mapping_status="taxonomy_based_benchmark_association_not_Dexonomy_metadata",
            supported=int(name.partition("_")[0]) not in (19, 21),
        )
        for name in names
    ]


def primitive_geometry(kind, radius, height):
    if kind in ("cuboid", "plate"):
        size = np.array([2 * radius, 1.4 * radius, height])
        vertices = (
            np.array(
                [
                    [x, y, z]
                    for x in (-0.5, 0.5)
                    for y in (-0.5, 0.5)
                    for z in (-0.5, 0.5)
                ]
            )
            * size
        )
    elif kind in ("cylinder", "disk"):
        angle = np.arange(48) * 2 * np.pi / 48
        vertices = np.array(
            [
                [radius * np.cos(a), radius * np.sin(a), z]
                for z in (-height / 2, height / 2)
                for a in angle
            ]
        )
    elif kind == "sphere":
        z = 1 - 2 * (np.arange(194) + 0.5) / 194
        angle = np.arange(len(z)) * np.pi * (3 - np.sqrt(5))
        vertices = (
            radius
            * np.c_[
                np.sqrt(1 - z * z) * np.cos(angle),
                np.sqrt(1 - z * z) * np.sin(angle),
                z,
            ]
        )
    else:
        raise ValueError(f"Not a single rigid primitive: {kind}")
    faces = ConvexHull(vertices).simplices.copy()
    for i, face in enumerate(faces):
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3) < 0:
            faces[i] = face[[0, 2, 1]]
    return vertices, faces


def primitive_mesh(path, kind, radius, height):
    vertices, faces = primitive_geometry(kind, radius, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join("v " + " ".join(map(str, v)) + "\n" for v in vertices)
        + "".join("f " + " ".join(map(str, f + 1)) + "\n" for f in faces)
    )
    return path


def jobs(output, templates, args):
    """Each shape/dimension combination is a separate object instance."""
    rows, skipped = [], []
    for template in templates:
        number = template.partition("_")[0]
        if not number.isdigit() or int(number) not in CATALOG:
            raise ValueError(
                f"No taxonomy primitive association for {template}; use --object for extra templates"
            )
        families = CATALOG[int(number)]
        if getattr(args, "primary_shape_only", False):
            families = families[:1]
        for kind, r, h in families:
            if args.primitive and kind != args.primitive:
                continue
            if r is None:
                skipped.append(
                    dict(
                        template=template,
                        category=kind,
                        reason="requires an articulated or multi-object asset; not approximated by a single primitive",
                    )
                )
                continue
            # Direct legacy callers without provenance flags keep explicit-grid
            # behavior. The runner records whether YAML actually supplied one.
            explicit = getattr(args, "primitive_grid_explicit", True)
            specifications = size_specifications(int(number), kind,
                radius_scales=getattr(args, "radius_scales", None),
                height_scales=getattr(args, "height_scales", None),
                radius=getattr(args, "radius", None), height=getattr(args, "height", None),
                explicit=explicit)
            for primitive in specifications:
                radius, height = primitive["radius_m"], primitive["height_m"]
                tag = f"{template}_{kind}_r{radius:.7f}_h{height:.7f}"
                path = primitive_mesh(output / "meshes" / f"{tag}.obj", kind, radius, height)
                rows.append((path, template, primitive))
    return rows, skipped

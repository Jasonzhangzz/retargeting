"""Material (face, barycentric) taxel coordinates, bound once in the rest mesh.

This is an anatomical initialization, not a measured glove installation. It is
fixed to the material surface, including across articulation and size changes.
"""
import numpy as np

BINDING_POLICY = 'material_uv_rows_v2'


def bind_grid(triangles, sids, axis, lateral):
    tri = np.asarray(triangles, float)
    uv = np.stack((tri @ lateral, tri @ axis), -1)
    ids = np.asarray(sids)
    rows, cols = ids // 20, ids % 20
    height = np.ptp(rows) + 1
    y = uv[..., 1].min() + (rows - rows.min() + .5) / height * np.ptp(uv[..., 1])
    targets = []
    for sid, row_y in zip(ids, y):
        # Intersect the material surface with this sensor row. Using its actual
        # lateral extent avoids collapsing the corner cells onto a rounded tip.
        crossings = [[] for _ in tri]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            start, end = uv[:, a], uv[:, b]
            dy = end[:, 1] - start[:, 1]
            valid = (abs(dy) > 1e-12) & ((start[:, 1] - row_y) * (end[:, 1] - row_y) <= 0)
            t = (row_y - start[valid, 1]) / dy[valid]
            for face, x in zip(np.flatnonzero(valid), start[valid, 0] + t * (end[valid, 0] - start[valid, 0])):
                crossings[face].append(float(x))
        intervals=sorted((min(c),max(c)) for c in crossings if len(c)>=2 and max(c)-min(c)>1e-12)
        merged=[]
        for lo,hi in intervals:
            if merged and lo<=merged[-1][1]+1e-10:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        if not merged:
            raise ValueError('Taxel row does not intersect its anatomical patch')
        row_cols = cols[rows == sid // 20]
        u = (sid % 20 - row_cols.min() + .5) / (np.ptp(row_cols) + 1)
        remaining=u*sum(hi-lo for lo,hi in merged)
        for lo,hi in merged:
            if remaining<=hi-lo:break
            remaining-=hi-lo
        targets.append([lo+remaining, row_y])
    targets = np.asarray(targets)
    a, b, c = uv.transpose(1, 0, 2)
    ab, ac = b - a, c - a
    cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    det = cross(ab, ac)
    safe = np.where(abs(det) > 1e-14, det, 1.)
    v = cross(targets[:, None] - a, ac) / safe
    w = cross(ab, targets[:, None] - a) / safe
    bary = np.stack((1 - v - w, v, w), -1)
    inside = (bary.min(-1) >= -1e-9) & (abs(det)[None] > 1e-14)
    candidates = [bary]
    scores = [np.where(inside, 0., np.inf)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start, edge = uv[:, i], uv[:, j] - uv[:, i]
        t = np.clip(((targets[:, None] - start) * edge).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-20), 0, 1)
        bc = np.zeros_like(bary); bc[..., i] = 1 - t; bc[..., j] = t
        candidates.append(bc)
        scores.append(((start + t[..., None] * edge - targets[:, None]) ** 2).sum(-1))
    scores = np.stack(scores, -1)
    flat = scores.reshape(len(ids), -1).argmin(-1)
    face, choice = flat // 4, flat % 4
    result = np.stack(candidates, -2)[np.arange(len(ids)), face, choice]
    result = np.clip(result, 0, 1); result /= result.sum(-1, keepdims=True)
    return face, result.astype(np.float32)


def surface_points(vertices, faces, face_ids, barycentric):
    return (np.asarray(vertices)[np.asarray(faces)[face_ids]] * np.asarray(barycentric)[..., None]).sum(-2)

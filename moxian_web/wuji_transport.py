"""Compact visualization/IPC geometry; optimization and qpos remain float64."""
import array
import base64
import struct
import sys

GEOMETRY_FIELDS = ('source_vertices_m', 'source_taxel_positions_m', 'taxel_positions_m',
    'source_taxel_normals', 'taxel_normals', 'link_rotations', 'link_positions_m', 'source_joints_m')


def pack_result_geometry(message):
    import numpy as np
    result = dict(message); fields = {}; chunks = []; offset = 0
    for key in GEOMETRY_FIELDS:
        if key not in result: continue
        values = np.asarray(result.pop(key), dtype='<f4')
        expected = 3 if key == 'link_rotations' else 2
        if values.ndim != expected or values.shape[-1] != 3 or not np.isfinite(values).all():
            raise ValueError('Invalid Wuji geometry: '+key)
        data = values.tobytes(); fields[key] = dict(offset=offset, shape=list(values.shape))
        chunks.append(data); offset += len(data)
    result['geometry_f32'] = dict(format='wuji-f32-v1', fields=fields,
        data=base64.b64encode(b''.join(chunks)).decode('ascii'))
    return result


def unpack_result_geometry(message):
    descriptor = message.get('geometry_f32')
    if descriptor is None: return dict(message)
    if descriptor.get('format') != 'wuji-f32-v1': raise ValueError('Unknown Wuji geometry format')
    data = base64.b64decode(descriptor['data'], validate=True); values = array.array('f'); values.frombytes(data)
    if sys.byteorder != 'little': values.byteswap()
    result = {k:v for k,v in message.items() if k != 'geometry_f32'}
    for key, field in descriptor['fields'].items():
        shape = field['shape']; offset = field['offset']; count = 1
        if key not in GEOMETRY_FIELDS or len(shape) != (3 if key == 'link_rotations' else 2) or shape[-1] != 3:
            raise ValueError('Invalid Wuji geometry field')
        for n in shape:
            if not isinstance(n, int) or n < 0: raise ValueError('Invalid Wuji geometry shape')
            count *= n
        if offset < 0 or offset % 4 or offset+4*count > len(data): raise ValueError('Invalid Wuji geometry range')
        flat = values[offset//4:offset//4+count]
        rows = [flat[i:i+3].tolist() for i in range(0, count, 3)]
        result[key] = [rows[i:i+3] for i in range(0, len(rows), 3)] if len(shape) == 3 else rows
    return result


def decode_source_frame(frame):
    packed = frame.get('mesh_f32_b64')
    if packed is None: return frame
    import numpy as np
    data = base64.b64decode(packed, validate=True)
    if len(data) < 128: raise ValueError('Incomplete Moxian source mesh')
    magic, version, sequence, nv, nj, _, stamp = struct.unpack_from('<4sIIIIId', data)
    geometry = data[32:64].split(b'\0')[0].decode(); binding = data[64:128].split(b'\0')[0].decode()
    if (magic != b'QNB1' or version != 1 or len(data) != 128+12*(nv+nj)
            or geometry != frame.get('geometry_version') or binding != frame.get('binding_version')
            or abs(stamp-frame['timestamp_ms']) > .01):
        raise ValueError('Moxian mesh/frame identity mismatch')
    points = np.frombuffer(data, dtype='<f4', offset=128).reshape(-1, 3)
    return {**{k:v for k,v in frame.items() if k != 'mesh_f32_b64'},
            'vertices_m':points[:nv], 'joints_m':points[nv:], 'mesh_sequence':sequence}

import base64
import copy
import struct
import numpy as np
import pytest

from moxian_web.wuji_transport import pack_result_geometry, unpack_result_geometry, decode_source_frame


def test_compact_result_geometry_preserves_raw_float64_qpos_and_independent_metadata():
    original=dict(qpos=np.linspace(.0000000001,1.0000000001,26).tolist(),source_timestamp_ms=1234.,
        solver_backend='slsqp',solver_session_id='session',source_vertices_m=np.arange(30).reshape(10,3).tolist(),
        link_rotations=[np.eye(3).tolist()],quality={'processing_ms':2.})
    packed=pack_result_geometry(original)
    assert 'source_vertices_m' not in packed and 'link_rotations' not in packed
    assert packed['qpos']==original['qpos'] and 'source_vertices_m' in original
    decoded=unpack_result_geometry(packed)
    np.testing.assert_array_equal(decoded['source_vertices_m'],original['source_vertices_m'])
    np.testing.assert_array_equal(decoded['link_rotations'],original['link_rotations'])
    assert decoded['qpos']==original['qpos'] and decoded['source_timestamp_ms']==1234.
    bad=copy.deepcopy(packed);bad['geometry_f32']['fields']['source_vertices_m']['offset']=999999
    with pytest.raises(ValueError,match='range'):unpack_result_geometry(bad)


def test_moxian_binary_source_decodes_exact_same_frame_without_json_vertex_expansion():
    vertices=np.arange(15,dtype='<f4').reshape(5,3)*.001;joints=np.arange(9,dtype='<f4').reshape(3,3)*.001
    header=struct.pack('<4sIIIIId',b'QNB1',1,7,5,3,0,1000.)+b'geometry'.ljust(32,b'\0')+b'binding'.ljust(64,b'\0')
    frame=dict(timestamp_ms=1000.,geometry_version='geometry',binding_version='binding',
        mesh_f32_b64=base64.b64encode(header+vertices.tobytes()+joints.tobytes()).decode())
    decoded=decode_source_frame(frame)
    assert decoded['mesh_sequence']==7 and isinstance(decoded['vertices_m'],np.ndarray)
    np.testing.assert_array_equal(decoded['vertices_m'],vertices)
    np.testing.assert_array_equal(decoded['joints_m'],joints)
    assert 'mesh_f32_b64' not in decoded
    with pytest.raises(ValueError,match='identity mismatch'):decode_source_frame({**frame,'timestamp_ms':1001.})
    separate = {**frame, 'timestamp_ms':1020., 'source_display_timestamp_ms':1000., 'emf_input':{'seq':9}}
    # Raw-EMF headers can no longer bypass the current skeleton/mesh identity.
    with pytest.raises(ValueError,match='identity mismatch'):
        decode_source_frame(separate)

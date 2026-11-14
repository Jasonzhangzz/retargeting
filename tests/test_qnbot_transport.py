"""Packed geometry is same-frame, bounded and absent from polling JSON."""
import base64,json,os,struct,threading,time
from collections import deque
from pathlib import Path
from moxian_web.hand_model import HandModelBridge
from moxian_web.recording import SnapshotRecorder


class Process:
    def poll(self):return None


def packet(seq=1):
    vertices=11598;joints=34;payload=struct.pack('<f',(seq*.001))*(vertices+joints)*3
    message=dict(type='frame',status='default',surface_backend='qnbot',ready=True,hand_side='right',coordinate_frame='wrist_local',timestamp_ms=1000.+seq,
        geometry_version='geometry',binding_version='binding',mesh_frame=dict(sequence=seq,vertex_count=vertices,joint_count=joints),mesh_f32_b64=base64.b64encode(payload).decode())
    return message


def test_reader_binary_projection_and_exact_sequence_retention(tmp_path):
    bridge=HandModelBridge(Path.cwd(),tmp_path);bridge.side='right';bridge.proc=Process()
    r,w=os.pipe();thread=threading.Thread(target=bridge._read,args=(r,bridge.proc,bridge.generation));thread.start()
    with os.fdopen(w,'w') as stream:
        for seq in (1,2):stream.write(json.dumps(packet(seq))+'\n')
    thread.join(timeout=3);assert not thread.is_alive()
    state=bridge.snapshot(include_audit=False,mesh_transport=True)
    assert state['ready'] and 'vertices_m' not in state and state['mesh_frame']['sequence']==2
    first=bridge.mesh_frame(sequence=1);latest=bridge.mesh_frame()
    assert len(first)==128+(11598+34)*12 and first[:4]==b'QNB1'
    assert struct.unpack_from('<I',first,8)[0]==1 and struct.unpack_from('<I',latest,8)[0]==2
    assert bridge.mesh_frame(after=2)==b''
    full=bridge.snapshot(include_audit=False)
    assert len(full['vertices_m'])==11598 and len(full['joints_m'])==34
    assert abs(full['vertices_m'][0][0]-.002)<1e-7


def test_background_recording_expands_same_frame_float32_without_mutating_snapshot(tmp_path):
    descriptor=packet(3);raw=base64.b64decode(descriptor.pop('mesh_f32_b64'));header=struct.pack('<4sIIIIId',b'QNB1',1,3,11598,34,0,1003.)+b'geometry'.ljust(32,b'\0')+b'binding'.ljust(64,b'\0')
    descriptor['_mesh_f32']=header+raw;descriptor['ready']=True
    recorder=SnapshotRecorder(tmp_path);recorder.start();recorder.submit(dict(hand_model=descriptor));recorder.stop();recorder.close()
    row=json.loads(recorder.path.read_text());assert len(row['hand_model']['vertices_m'])==11598
    assert '_mesh_f32' not in row['hand_model'] and '_mesh_f32' in descriptor

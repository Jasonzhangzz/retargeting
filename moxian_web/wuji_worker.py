"""Independent WUJI2 worker: no glove SDK, serial device or robot commands."""
import argparse
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import traceback
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from moxian_web.wuji_backends import create_retarget
from moxian_web.card_geometry import TrackedMesh
from moxian_web.wuji_transport import pack_result_geometry, decode_source_frame


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--event-fd',type=int,required=True);ap.add_argument('--home',type=Path,required=True)
    configured_solver=os.environ.get('MOXIAN_WUJI_SOLVER','slsqp_edge')
    if configured_solver == 'slsqp':
        configured_solver='slsqp_edge'
    if configured_solver not in ('slsqp_cem','slsqp_edge','slsqp_graph','cem'):
        configured_solver='slsqp_edge'
    ap.add_argument('--solver',choices=('slsqp','slsqp_cem','slsqp_edge','slsqp_graph','cem'),default=configured_solver);args=ap.parse_args()
    args.home.mkdir(parents=True,exist_ok=True)
    stream=os.fdopen(args.event_fd,'w',buffering=1);stop=threading.Event()
    emit_lock=threading.Lock()
    def emit(m):
        if m.get('type')=='frame':m=pack_result_geometry(m)
        with emit_lock:stream.write(json.dumps(m,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    frames=queue.Queue(maxsize=1);controls=queue.Queue(maxsize=32)
    def reader():
        heartbeat=0.;previous_received=None;previous_stamp=None;epoch=0
        try:
            for line in sys.stdin:
                m=json.loads(line)
                if m.get('type')=='frame':
                    now=time.monotonic();stamp=m['frame'].get('timestamp_ms')
                    if previous_received is not None and (now-previous_received>.250 or stamp<=previous_stamp):epoch+=1
                    previous_received=now;previous_stamp=stamp
                    m['frame'].setdefault('input_stream_epoch',epoch)
                    m['frame']['input_received_wall_ms']=time.time()*1000
                    try:frames.get_nowait()
                    except queue.Empty:pass
                    frames.put_nowait(m['frame'])
                    if time.monotonic()-heartbeat>.1:
                        emit({'type':'state','input_received_wall_ms':time.time()*1000,'input_timestamp_ms':m['frame'].get('timestamp_ms')});heartbeat=time.monotonic()
                else:controls.put(m)
        finally:stop.set()
    threading.Thread(target=reader,daemon=True).start();torch.set_num_threads(1)
    backend='slsqp_edge' if args.solver=='slsqp' else args.solver;device='cuda:0' if torch.cuda.is_available() else 'cpu'
    solver=None;topology=None;rebuild=False;version=None;object_sdf=None;last_output_wall_ms=None
    while not stop.is_set():
        try:m=controls.get_nowait()
        except queue.Empty:m=None
        if m:
            if m['type']=='topology':
                topology=m['model'];new=(topology['geometry_version'],topology['binding_version'],topology.get('model_hash'))
                if new!=version:solver=None;version=new
            else:
                action=m.get('action');error=None
                if action=='object_set':
                    data=m.get('data') or {}
                    try:object_sdf=TrackedMesh(data['vertices_m'],data['faces'],device,data.get('transform'));solver=None
                    except Exception as exc:error=str(exc)
                elif action=='object_clear':object_sdf=None;solver=None
                elif action=='rebuild':rebuild=True;solver=None
                elif action=='reset':
                    if solver:solver.reset()
                elif action=='solver_set':
                    selected=(m.get('data') or {}).get('solver')
                    if selected=='slsqp':selected='slsqp_edge'
                    if selected not in ('slsqp_cem','slsqp_edge','slsqp_graph','cem'):error='Unknown Wuji2 solver'
                    else:backend=selected;solver=None
                else:error='Unknown WUJI2 control'
                if error is None:emit({'type':'state','status':'initializing','ready':False,'solution_feasible':False,'display_available':False,
                    'requested_solver_backend':backend,'active_solver_backend':backend,'active_solver_session_id':None})
                emit({'type':'ack','id':m.get('id'),'result':{'scheduled':True},'error':error})
        if topology is None:stop.wait(.02);continue
        if solver is None:
            try:
                emit({'type':'state','status':'mapping','ready':False,'display_available':False,'error':None,
                      'requested_solver_backend':backend,'active_solver_backend':backend,'active_solver_session_id':None})
                internal_backend='slsqp' if backend in ('slsqp_cem','slsqp_edge','slsqp_graph') else backend
                profile=('cem_contact' if backend == 'slsqp_cem' else
                         'contact_graph' if backend == 'slsqp_graph' else 'edge_laplacian')
                extra={'realtime':True,'objective_profile':profile} if internal_backend=='slsqp' else {}
                solver=create_retarget(internal_backend,topology,args.home/'mappings',device,rebuild=rebuild,object_sdf=object_sdf,**extra);rebuild=False
                model={**solver.topology(),'solver_backend':backend}
                emit({'type':'topology','model':model})
                emit({'type':'state','status':'initializing','ready':False,'mapping_ready':solver.mapping['ready'],
                      'failed_sids':solver.mapping['failed_sids'],'requested_solver_backend':backend,
                      'active_solver_backend':backend,'active_algorithm':model.get('algorithm'),
                      'active_solver_session_id':solver.session_id,'error':None})
            except Exception as exc:
                traceback.print_exc();emit({'type':'state','status':'error','ready':False,'error':str(exc)})
                topology=None;continue
        try:frame=frames.get(timeout=.03)
        except queue.Empty:continue
        if (frame.get('geometry_version'),frame.get('binding_version'),frame.get('model_hash'))!=version:continue
        try:
            emit({'type':'state','status':'solving','processing_source_timestamp_ms':frame['timestamp_ms']})
            frame=decode_source_frame(frame)
            processing_started=time.time()*1000
            result=solver.step(frame)
            result['quality'].update(input_queue_ms=processing_started-frame['input_received_wall_ms'],
                input_source_age_ms=processing_started-frame['timestamp_ms'])
            published=time.time()*1000
            result['quality']['output_hz']=1000/(published-last_output_wall_ms) if last_output_wall_ms and published>last_output_wall_ms else None
            last_output_wall_ms=published
            emit({'type':'frame',**result,'solver_backend':('slsqp' if backend.startswith('slsqp') else backend),
                'requested_solver_backend':backend,'active_solver_backend':backend,
                'active_algorithm':result['algorithm'],'active_solver_session_id':result['solver_session_id'],'error':None})
        except Exception as exc:
            traceback.print_exc();emit({'type':'state','status':'error','ready':False,'error':str(exc)})


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise

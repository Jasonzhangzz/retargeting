#!/usr/bin/env python3
"""Moxian surface replay: envelope SLSQP, legacy graph SLSQP, or CARD/CEM."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.wuji_backends import create_retarget


def replay(source,destination,*,cache=None,topology=None,device='cpu',contact=True,object_sdf=None,solver='slsqp_edge',source_model=None,limit=None):
    if solver not in ('slsqp','slsqp_cem','slsqp_edge','slsqp_graph','cem'):raise ValueError('Unknown solver')
    requested_solver=solver
    if solver=='slsqp': solver='slsqp_edge'
    if limit is not None and limit<1:raise ValueError('limit must be positive')
    if Path(source).resolve()==Path(destination).resolve():raise ValueError('Output must differ from input')
    if Path(destination).exists():raise FileExistsError(destination)
    model=json.loads(Path(topology).read_text()) if topology else None
    reconstructor=None
    if source_model:
        from moxian_web.moxian_replay import MoxianReplayReconstructor
        reconstructor=MoxianReplayReconstructor(json.loads(Path(source_model).read_text()))
    rows=[];segments=[];current=[];identity=None;last=-float('inf');skipped=0;duplicates={};recorded_object=None;recorded_mode=None
    with Path(source).open() as stream:
        for line in stream:
            if not line.strip():continue
            if limit is not None and len(rows)>=limit:break
            row=json.loads(line);row_index=len(rows);rows.append(row)
            if row.get('hand_model_topology'):model=row['hand_model_topology']
            if row.get('wuji2_topology'):
                recorded_object=row['wuji2_topology'].get('object_sdf')
            if (row.get('wuji2') or {}).get('sdf'):recorded_mode=row['wuji2']['sdf']['type']
            hand=dict(row.get('hand_model') or {})
            observation=(row.get('hand_model_input') or {}) if row.get('hand_model') else row
            if reconstructor and observation.get('fingers'):
                hand=reconstructor.frame(observation);model=reconstructor.topology
                row['hand_model']=hand;row['hand_model_topology']=model
                recorded_object=None;recorded_mode=None
            if not hand.get('vertices_m'):skipped+=1;continue
            if not model or not model.get('rest_vertices_m'):raise ValueError('录制缺少源曲面绑定；请提供 --source-topology，或用 --source-model 重建原始EMF')
            key=(hand.get('geometry_version'),hand.get('binding_version'))
            if key!=(model['geometry_version'],model['binding_version']):raise ValueError('Recorded source geometry/binding version differs from topology')
            object_key=recorded_object.get('version') if recorded_object else None
            segment_key=(*key,object_key,(row.get('wuji2') or {}).get('solver_session_id'))
            if segment_key==identity and hand.get('timestamp_ms',0)<=last:
                if hand['timestamp_ms']==last and current:duplicates[row_index]=current[-1][0]
                skipped+=1;continue
            if not hand.get('source_contact'):
                matched=row.get('hand_model_input') or {}
                if contact and abs(matched.get('timestamp_ms',-1e20)-hand['timestamp_ms'])>25:raise ValueError('录制缺少与源mesh同帧的触觉数据')
                hand.update(source_contact=matched.get('contact'),tactile_timestamp_ms=matched.get('tactile_timestamp_ms'))
            if segment_key!=identity or (last>-float('inf') and hand['timestamp_ms']-last>250):
                if current:segments.append((segment_topology,current,segment_object))
                current=[];identity=segment_key;segment_topology=model;segment_object=recorded_object
                if object_sdf is None and recorded_mode=='tracked_mesh' and not segment_object:raise ValueError('Tracked mesh recording requires embedded object geometry or --object-mesh')
            current.append((row_index,hand));last=hand['timestamp_ms']
    if current:segments.append((segment_topology,current,segment_object))
    if not segments:raise ValueError('录制没有有效Moxian曲面帧；原始EMF需要 --source-model 指定匹配标定')
    solved=0
    for model,segment,descriptor in segments:
        provider=object_sdf
        if provider is None and descriptor:
            from moxian_web.card_geometry import TrackedMesh
            provider=TrackedMesh(descriptor['vertices_m'],descriptor['faces'],device,descriptor.get('transform'))
        internal_solver='slsqp' if solver in ('slsqp_cem','slsqp_edge','slsqp_graph') else solver
        engine=create_retarget(internal_solver,model,cache,device,object_sdf=provider,
                               **({'objective_profile':('cem_contact' if solver=='slsqp_cem' else 'contact_graph' if solver=='slsqp_graph' else 'edge_laplacian')} if internal_solver=='slsqp' else {}))
        frames=[f for _,f in segment]
        if solver=='cem':
            from moxian_web.card_offline import prepare_clip,filter_clip
            metadata=prepare_clip(engine,frames,contact)
            raw=[engine.step(f,contact_enabled=contact) for f in frames];results=filter_clip(engine,frames,raw)
        else:
            metadata=dict(policy='causal_current_frame',source_surface='moxian_original_skn')
            results=[engine.step(f,contact_enabled=contact) for f in frames]
        for k,((index,frame),result) in enumerate(zip(segment,results)):
            result['offline_preprocessing']=metadata;result['solver_backend']=solver;result['solver_profile']=('cem_contact' if solver=='slsqp_cem' else 'contact_graph' if solver=='slsqp_graph' else 'edge_laplacian') if internal_solver=='slsqp' else None;rows[index]['wuji2']=result
            rows[index]['wuji2_topology']=engine.topology() if k==0 else None
            solved+=1
    # Reuse only newly solved results for duplicate source snapshots; never old LM.
    indices={i for _,segment,_ in segments for i,_ in segment}
    with Path(destination).open('x') as output:
        for i,row in enumerate(rows):
            if i not in indices:
                row['wuji2']=rows[duplicates[i]]['wuji2'] if i in duplicates else None
                row['wuji2_topology']=None
            output.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
    return dict(solved_frames=solved,duplicate_or_missing_frames=skipped,segments=len(segments),output=str(destination),contact_enabled=contact,solver=requested_solver,algorithm=results[-1]['algorithm'],postprocessing='none' if internal_solver=='slsqp' else 'sg15_order3_validated')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source-topology',type=Path);ap.add_argument('--source-model',type=Path,help='Moxian骨长/外参模型，用原始EMF重建源mesh')
    ap.add_argument('--solver',choices=('slsqp_cem','slsqp_edge','slsqp_graph','cem'),default='slsqp_edge');ap.add_argument('--limit',type=int)
    ap.add_argument('--cache',type=Path,default=Path('output/moxian/wuji2/offline/mappings'));ap.add_argument('--device',default='cpu');ap.add_argument('--pose-only',action='store_true');ap.add_argument('--object-mesh',type=Path)
    a=ap.parse_args();provider=None
    if a.object_mesh:
        import trimesh
        from moxian_web.card_geometry import TrackedMesh
        mesh=trimesh.load(a.object_mesh,force='mesh');provider=TrackedMesh(mesh.vertices,mesh.faces,a.device)
    print(json.dumps(replay(a.input,a.output,cache=a.cache,topology=a.source_topology,device=a.device,contact=not a.pose_only,object_sdf=provider,solver=a.solver,source_model=a.source_model,limit=a.limit),ensure_ascii=False))

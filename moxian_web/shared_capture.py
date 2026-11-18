"""Reuse the official, user-confirmed eight gestures; never invent ADC samples."""
from __future__ import annotations
import bisect,copy,hashlib,json,math,re
from pathlib import Path

PROTOCOL='official_eight_v1'
STAGE_MAP={'close':'together','open':'open','mcp90':'mcp90','mcp90_pip90':'fist','pinch_first':'pinch_index','pinch_middle':'pinch_middle','pinch_ring':'pinch_ring','pinch_little':'pinch_pinky'}
FINGERS=('thumb','index','middle','ring','pinky')

class OfficialSensorRecorder:
    """Serial-owner sidecar; records raw tactile data during the same SDK session."""
    def __init__(self,folder):self.folder=Path(folder);self.stream=None;self.session=None;self.count=0
    def close(self):
        if self.stream:self.stream.close();self.stream=None
    def feed(self,frame,capture,mapping,tare):
        if not capture or capture.get('session_state')!='running':self.close();return
        session=capture.get('session_id')
        if not session or not re.fullmatch(r'[a-zA-Z0-9_-]+',session):return
        if session!=self.session:
            self.close();self.folder.mkdir(parents=True,exist_ok=True);self.stream=(self.folder/(session+'.jsonl')).open('a');self.session=session;self.count=0
        elif self.stream is None:self.stream=(self.folder/(session+'.jsonl')).open('a')
        if self.count>=24000:return
        row={**frame,'mapping':mapping,'tare_profile':tare,'official_stage_id':capture.get('stage',{}).get('stage_id')}
        self.stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');self.count+=1
        if self.count%20==0:self.stream.flush()


def official_capture_document(home,fingerprint):
    home=Path(home);artifacts=[]
    for artifact in home.glob('glove/calibrations/*/artifacts/hand_system.yaml'):
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=fingerprint:continue
        record=(artifact.parent.parent/'record.yaml').read_text()
        match=re.search(r'^\s*capture_payload_sha256:\s*([a-f0-9]{64})\s*$',record,re.M)
        if match:artifacts.append(match[1])
    if not artifacts:raise ValueError('当前官方标定没有可验证的八动作采集来源')
    selected=None
    for path in home.glob('glove/captures/*.json'):
        doc=json.loads(path.read_text())
        if doc.get('payload_sha256') in artifacts:selected=(path,doc);break
    if selected is None:raise ValueError('未找到当前骨架标定对应的八动作原始数据')
    return selected


def load_official_capture(home,fingerprint,mapping,side="left"):
    home=Path(home);path,doc=official_capture_document(home,fingerprint);p=doc['payload']
    if p.get('side')!=side or p.get('capture_profile')!='qnbot.hand.capture.eight.v1':raise ValueError('八动作手侧或采集协议不匹配')
    groups={g['stage_id']:g['frames'] for g in p['stage_samples']}
    requirements={r['stage_id']:r['required_sample_count'] for r in p['requirements'] if not r.get('optional')}
    if not set(STAGE_MAP)<=set(groups) or any(len(groups[s])<max(6,requirements.get(s,30)) for s in STAGE_MAP):raise ValueError('官方八动作尚未完整采集')
    from .input_quality import capture_input_quality
    quality=capture_input_quality(doc)
    if not quality['accepted']:raise ValueError(quality['message'])
    stamps=[float(f['payload']['timestamp_ms']) for frames in groups.values() for f in frames]
    observations={}
    for sidecar in (home/'shared-captures').glob('*.jsonl'):
        for line in sidecar.open():
            try:r=json.loads(line);t=float(r['timestamp_ms'])
            except (ValueError,KeyError):continue
            if min(stamps)-25<=t<=max(stamps)+25:observations[t]=r
    times=sorted(observations);rows=[];matched=0
    labels_path=home/'gesture-constraints.json';labels={}
    if labels_path.exists():
        annotated=json.loads(labels_path.read_text())
        if (annotated.get('schema_version')==1 and annotated.get('source_payload_sha256')==doc['payload_sha256']
            and annotated.get('calibration_fingerprint')==fingerprint and annotated.get('side')==side):
            labels=annotated.get('together',{})
    for stage,target in STAGE_MAP.items():
        for sample in groups[stage]:
            data=sample['payload'];stamp=float(data['timestamp_ms']);fingers=copy.deepcopy(data['fingertip_local'])
            if not math.isfinite(stamp) or set(fingers)!=set(FINGERS):raise ValueError('八动作含无效时间戳或手指')
            for f in FINGERS:
                node=fingers[f];v=node['position'];q=node['quaternion_xyzw']
                if len(v)!=3 or len(q)!=4 or not all(math.isfinite(x) for x in v+q) or abs(sum(x*x for x in q)-1)>.05:raise ValueError('八动作含无效位姿')
            index=bisect.bisect_left(times,stamp);near=min(times[max(0,index-1):index+1],key=lambda t:abs(t-stamp),default=None)
            sensor=observations[near] if near is not None and abs(near-stamp)<=25 else None
            if sensor and (sensor.get('hand_side',side)!=side or abs(sensor.get('tactile_timestamp_ms',float('inf'))-stamp)>25 or sensor.get('mapping',{}).get('assignments')!=mapping['assignments']):sensor=None
            row={'timestamp_ms':stamp,'hand_side':side,'fingers':fingers,'stage':target,'official_stage_id':stage,'capture_protocol':PROTOCOL,
                 'calibration_fingerprint':fingerprint,'raw':None,'tactile_timestamp_ms':None,'contact':{'ready':False,'status':'unknown','patches':[]},'tare_profile':None,'mapping':copy.deepcopy(mapping),
                 'guided_contact':target.startswith('pinch_'),'contact_constraint_source':'user_confirmed_official_gesture' if target.startswith('pinch_') else None,
                 'load_role':'onset' if target.startswith('pinch_') else 'stable','tactile_matched':False}
            if sensor:
                for key in ('raw','tactile_timestamp_ms','contact','tare_profile','mapping'):row[key]=copy.deepcopy(sensor.get(key))
                row['tactile_matched']=True;matched+=1
            if target=='together' and labels.get('thumb_index_side_contact') is True:
                row['thumb_index_side_contact']=True
                row['side_contact_evidence']='operator_confirmed_calibration_gesture'
            rows.append(row)
    if len({r['timestamp_ms'] for r in rows})!=len(rows):raise ValueError('八动作包含重复时间戳')
    return {'schema_version':1,'protocol':PROTOCOL,'side':side,'rows':rows,'completed_groups':list(STAGE_MAP.values()),'source_payload_sha256':doc['payload_sha256'],
            'source_file':str(path),'calibration_fingerprint':fingerprint,'total':8,'tactile_matched_frames':matched,'frames':len(rows),
            'validation_scope':'held_out_frames_within_eight_gestures; no_repeat_or_approach_capture',
            'quality_note':'复用官方八动作，已匹配同期触觉' if matched else '复用官方八动作；本次无同期 ADC，厚度仅由引导自接触及先验约束，不是实测厚度'}

class SharedHandCapture:
    def __init__(self):self.rows=[];self.complete_groups=[];self.status='waiting_official';self.metadata={};self.error=None
    def install(self,data):self.rows=data['rows'];self.complete_groups=data['completed_groups'];self.status='complete';self.metadata={k:v for k,v in data.items() if k!='rows'};self.error=None
    def snapshot(self):
        return {'state':self.status,'stage_id':None,'index':len(self.complete_groups)-1,'total':8,'request_id':None,'samples':len(self.rows),'progress':1. if self.status=='complete' else 0.,'cycles':None,
                'error':self.error,'completed_groups':self.complete_groups,'expected_patches':[],'source':'official_eight','metadata':self.metadata}

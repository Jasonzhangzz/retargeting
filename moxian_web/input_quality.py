"""Inspect raw captures without assuming channels are already registered.

Raw cross-channel distances are advisory. They cannot validate anatomy until
the channel and sensor-mount transforms have been calibrated.
"""
import math
import statistics

PINCH_PARTNERS={'pinch_first':'index','pinch_middle':'middle','pinch_ring':'ring','pinch_little':'pinky'}
MAX_PINCH_SENSOR_DISTANCE_M=.15


def capture_input_quality(document):
    groups={g['stage_id']:g['frames'] for g in document.get('payload',{}).get('stage_samples',[])}
    measurements=[]
    for stage,finger in PINCH_PARTNERS.items():
        distances=[]
        for sample in groups.get(stage,[]):
            try:
                tips=sample['payload']['fingertip_local'];a=tips['thumb']['position'];b=tips[finger]['position']
                if len(a)!=3 or len(b)!=3 or not all(math.isfinite(v) for v in a+b):continue
                distances.append(math.dist(a,b))
            except (KeyError,TypeError,ValueError):continue
        if not distances:
            return {'accepted':False,'status':'unknown','message':'无法核验四组对指的原始 EMF 输入；原标定记录保留。','pairs':measurements}
        median=statistics.median(distances)
        measurements.append({'stage':stage,'partner':finger,'samples':len(distances),'median_sensor_distance_mm':median*1000,
                             'grossly_inconsistent':median>MAX_PINCH_SENSOR_DISTANCE_M})
    failed=[p for p in measurements if p['grossly_inconsistent']]
    message='四组对指输入可读取；原始传感器间距不作为骨架精度或硬件故障判据。'
    if failed:
        values='、'.join(str(round(p['median_sensor_distance_mm'])) for p in failed)
        message=f'原始对指通道间距为 {values} mm；需要核验固定通道及安装变换，不能据此判定 EMF 异常。'
    return {'accepted':True,'status':'registration_required' if failed else 'passed','message':message,'pairs':measurements,
            'threshold_m':MAX_PINCH_SENSOR_DISTANCE_M,'geometry_modified':False,'pose_correction':False}

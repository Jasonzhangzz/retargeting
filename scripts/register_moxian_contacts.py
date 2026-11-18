"""Fit measured SID material coordinates; never infer them from pinch closure."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.contact_registration import context,fit_registration


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--topology',type=Path,help='Saved /api/hand_model/model JSON; defaults to live API')
    ap.add_argument('--url',default='http://127.0.0.1:8890')
    ap.add_argument('--serial',required=True)
    ap.add_argument('--measurements',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--template',action='store_true',help='Write an empty measured-reference input template')
    ap.add_argument('--apply',action='store_true',help='Apply only a candidate passing measured holdout checks')
    args=ap.parse_args()
    topology=json.loads(args.topology.read_text()) if args.topology else json.load(urllib.request.urlopen(args.url+'/api/hand_model/model',timeout=15))
    if args.template:
        if args.apply or args.measurements:ap.error('--template does not accept --apply or --measurements')
        result=dict(schema_version=1,context=context(topology,args.serial),coordinate_frame='mano_wrist_local',units='m',measurement_source='',frames=[])
    else:
        if not args.measurements:ap.error('--measurements is required')
        result=fit_registration(topology,args.serial,json.loads(args.measurements.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n');tmp.replace(args.output)
    if not args.template and not result['accepted']:
        raise SystemExit('Measured holdout failed; candidate saved for diagnosis, active binding unchanged')
    if args.apply:
        body=json.dumps(dict(command='mano_registration_import',artifact=result)).encode()
        request=urllib.request.Request(args.url+'/api/command',data=body,headers={'Content-Type':'application/json'})
        reply=json.load(urllib.request.urlopen(request,timeout=15))
        if not reply.get('ok'):raise SystemExit(reply)
    print(json.dumps(dict(output=str(args.output),template=args.template,registered_sids=len(result.get('records',{})),complete=result.get('complete',False),applied=args.apply)))


if __name__=='__main__':main()

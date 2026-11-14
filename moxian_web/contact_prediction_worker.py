"""Isolated CPU inference. No serial port, RGB or tactile model input."""
import argparse,json,os,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.contact_prediction import load_predictor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--event-fd',type=int,required=True);parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args();out=os.fdopen(args.event_fd,'w',buffering=1)
    def emit(value):out.write(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    engine=load_predictor(args.checkpoint);topology=None
    emit(dict(status='waiting_input',ready=False,error=None,checkpoint=Path(args.checkpoint).name))
    for line in sys.stdin:
        try:
            message=json.loads(line)
            if message.get('type')=='control':
                engine.clear_metrics();emit(dict(type='ack',id=message['id'],result={'ok':True}));continue
            if message.get('topology'):topology=message['topology']
            if not topology:raise ValueError('等待 SID 材料坐标')
            result=engine.step(message['frame'],topology,message.get('history_length',8))
            emit(dict(result,error=None))
        except Exception as exc:
            traceback.print_exc();emit(dict(ready=False,status='error',error=str(exc)))


if __name__=='__main__':main()

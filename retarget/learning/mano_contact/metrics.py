"""Micro metrics including natural contact-free frames; no sklearn dependency."""
import numpy as np


def binary_metrics(probability, target, threshold=.52):
    p=np.asarray(probability); y=np.asarray(target,dtype=bool); predicted=p>=threshold
    tp=int((predicted&y).sum());fp=int((predicted&~y).sum());fn=int((~predicted&y).sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return dict(precision=precision,recall=recall,f1=2*tp/max(2*tp+fp+fn,1),tp=tp,fp=fp,fn=fn)


def average_precision(probability,target):
    p=np.asarray(probability).ravel();y=np.asarray(target,dtype=bool).ravel()
    if not y.any():return None
    order=np.argsort(-p,kind='stable');p=p[order];y=y[order]
    # Group tied scores, matching threshold-based AP.
    ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1]
    positives=np.cumsum(y)[ends];precision=positives/(ends+1)
    recall=positives/y.sum()
    return float((np.diff(np.r_[0.,recall])*precision).sum())


def contact_metrics(probability,target,frame_probability,threshold=.52):
    p=np.asarray(probability);y=np.asarray(target,dtype=bool);fp=np.asarray(frame_probability).ravel()
    result=binary_metrics(p,y,threshold);result['ap']=average_precision(p,y)
    frame=y.any(-1);fm=binary_metrics(fp,frame,.5)
    result.update(frame_f1=fm['f1'],frame_precision=fm['precision'],frame_recall=fm['recall'],
                  frame_false_positive_rate=float((fp[~frame]>=.5).mean()) if (~frame).any() else None,
                  vertex_any_false_positive_rate=float((p[~frame]>=threshold).any(-1).mean()) if (~frame).any() else None,
                  frames=len(y),contact_free_frames=int((~frame).sum()),threshold=threshold)
    return result


def calibrated_threshold(probability,target):
    candidates=np.linspace(.05,.95,91)
    values=[binary_metrics(probability,target,t)['f1'] for t in candidates]
    return float(candidates[int(np.argmax(values))])

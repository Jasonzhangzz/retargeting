"""Cancellable immutable capture copy with observable, real frame counts."""
import copy,time


def copy_rows(rows,cancel,progress,chunk_size=64):
    total=len(rows);result=[];began=time.monotonic();last=began
    def report():
        done=len(result);elapsed=time.monotonic()-began
        progress(dict(state='running',phase='copy',progress=.01*done/max(total,1),completed=done,total=total,
                      message='准备训练数据快照',estimated_remaining_s=elapsed*(total-done)/done if done else None,eta_scope='stage'))
    report()
    for start in range(0,total,chunk_size):
        if cancel.is_set():raise InterruptedError('训练数据准备已取消')
        result.extend(copy.deepcopy(rows[start:start+chunk_size]))
        if time.monotonic()-last>=.1 or len(result)==total:report();last=time.monotonic()
        time.sleep(0)
    return result

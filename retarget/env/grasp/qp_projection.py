"""GPU batch adapter of historical capped-simplex pressure projection."""
import warp as wp
import torch

@wp.func
def simplex(v:wp.vec4)->wp.vec4:
    positive=wp.max(v,wp.vec4(0.))
    if positive[0]+positive[1]+positive[2]+positive[3]<=10.:return positive
    ordered=v
    for i in range(4):
        for j in range(3-i):
            if ordered[j]<ordered[j+1]:
                a=ordered[j];ordered[j]=ordered[j+1];ordered[j+1]=a
    cumulative=float(0.);theta=float(0.)
    for i in range(4):
        cumulative+=ordered[i]
        threshold=(cumulative-10.)/float(i+1)
        if ordered[i]>threshold:theta=threshold
    return wp.max(v-wp.vec4(theta),wp.vec4(0.))

@wp.kernel(enable_backward=False)
def project(value:wp.array4d(dtype=float),output:wp.array4d(dtype=float),contacts:int):
    b,s=wp.tid()
    total=float(0.);maximum=float(0.)
    for c in range(contacts):
        v=wp.vec4(value[b,s,c,0],value[b,s,c,1],value[b,s,c,2],value[b,s,c,3])
        p=simplex(v)
        for k in range(4):
            total+=p[k];maximum=wp.max(maximum,wp.abs(v[k]));output[b,s,c,k]=p[k]
    if total<1.:
        low=float(0.);high=maximum+12.
        for iteration in range(32):
            mid=.5*(low+high);mass=float(0.)
            for c in range(contacts):
                v=wp.vec4(value[b,s,c,0],value[b,s,c,1],value[b,s,c,2],value[b,s,c,3])+wp.vec4(mid)
                p=simplex(v)
                for k in range(4):mass+=p[k]
            if mass<1.:low=mid
            else:high=mid
        for c in range(contacts):
            p=simplex(wp.vec4(value[b,s,c,0],value[b,s,c,1],value[b,s,c,2],value[b,s,c,3])+wp.vec4(high))
            for k in range(4):output[b,s,c,k]=p[k]

@wp.kernel(enable_backward=False)
def project_masked(value:wp.array4d(dtype=float),mask:wp.array2d(dtype=wp.bool),
                   output:wp.array4d(dtype=float),contacts:int):
    b,s=wp.tid()
    total=float(0.);maximum=float(0.);count=int(0)
    for c in range(contacts):
        p=wp.vec4(0.)
        if mask[b,c]:
            count+=1
            v=wp.vec4(value[b,s,c,0],value[b,s,c,1],value[b,s,c,2],value[b,s,c,3])
            p=simplex(v)
            for k in range(4):maximum=wp.max(maximum,wp.abs(v[k]))
        for k in range(4):
            total+=p[k];output[b,s,c,k]=p[k]
    if total<1. and count>0:
        low=float(0.);high=maximum+12.
        for iteration in range(32):
            mid=.5*(low+high);mass=float(0.)
            for c in range(contacts):
                if mask[b,c]:
                    p=simplex(wp.vec4(value[b,s,c,0],value[b,s,c,1],value[b,s,c,2],value[b,s,c,3])+wp.vec4(mid))
                    for k in range(4):mass+=p[k]
            if mass<1.:low=mid
            else:high=mid
        for c in range(contacts):
            if mask[b,c]:
                p=simplex(wp.vec4(value[b,s,c,0],value[b,s,c,1],value[b,s,c,2],value[b,s,c,3])+wp.vec4(high))
                for k in range(4):output[b,s,c,k]=p[k]


def project_cuda(value,contact_mask=None):
    wp.init()
    result=torch.empty_like(value)
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(value.device)),sync_enter=False,sync_exit=False):
        if contact_mask is None:
            wp.launch(project,value.shape[:2],inputs=[wp.from_torch(value),wp.from_torch(result),value.shape[-2]],device=str(value.device))
        else:
            wp.launch(project_masked,value.shape[:2],inputs=[wp.from_torch(value),wp.from_torch(contact_mask.contiguous()),wp.from_torch(result),value.shape[-2]],device=str(value.device))
    return result

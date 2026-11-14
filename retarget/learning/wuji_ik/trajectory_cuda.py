"""Native Warp/CUDA compact block PCR. Reuses Robokit's tile-Cholesky idiom."""
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _pcr_round(stride:int,segment:wp.array(dtype=wp.int64),
    diagonal:wp.array3d(dtype=wp.float32),left:wp.array3d(dtype=wp.float32),right:wp.array3d(dtype=wp.float32),b:wp.array3d(dtype=wp.float32),
    next_diagonal:wp.array3d(dtype=wp.float32),next_left:wp.array3d(dtype=wp.float32),next_right:wp.array3d(dtype=wp.float32),next_b:wp.array3d(dtype=wp.float32)):
    i=wp.tid();n=diagonal.shape[0]
    a=wp.tile_load(diagonal[i],shape=(32,32));v=wp.tile_load(b[i],shape=(32,1))
    al=wp.tile_zeros(shape=(32,32),dtype=wp.float32);ar=wp.tile_zeros(shape=(32,32),dtype=wp.float32)
    if i>=stride:
        k=i-stride
        if segment[i]==segment[k]:
            factor=wp.tile_cholesky(wp.tile_load(diagonal[k],shape=(32,32)))
            l=wp.tile_load(left[i],shape=(32,32))
            y=wp.tile_lower_solve(factor,wp.tile_transpose(l))
            z=wp.tile_upper_solve(wp.tile_transpose(factor),y)
            multiplier=-wp.tile_transpose(z)
            wp.tile_matmul(multiplier,wp.tile_load(right[k],shape=(32,32)),a)
            wp.tile_matmul(multiplier,wp.tile_load(b[k],shape=(32,1)),v)
            wp.tile_matmul(multiplier,wp.tile_load(left[k],shape=(32,32)),al)
    if i+stride<n:
        k=i+stride
        if segment[i]==segment[k]:
            factor=wp.tile_cholesky(wp.tile_load(diagonal[k],shape=(32,32)))
            r=wp.tile_load(right[i],shape=(32,32))
            y=wp.tile_lower_solve(factor,wp.tile_transpose(r))
            z=wp.tile_upper_solve(wp.tile_transpose(factor),y)
            multiplier=-wp.tile_transpose(z)
            wp.tile_matmul(multiplier,wp.tile_load(left[k],shape=(32,32)),a)
            wp.tile_matmul(multiplier,wp.tile_load(b[k],shape=(32,1)),v)
            wp.tile_matmul(multiplier,wp.tile_load(right[k],shape=(32,32)),ar)
    wp.tile_store(next_diagonal[i],a);wp.tile_store(next_left[i],al);wp.tile_store(next_right[i],ar);wp.tile_store(next_b[i],v)


@wp.kernel(enable_backward=False)
def _finish(diagonal:wp.array3d(dtype=wp.float32),rhs:wp.array3d(dtype=wp.float32),out:wp.array3d(dtype=wp.float32)):
    i=wp.tid();factor=wp.tile_cholesky(wp.tile_load(diagonal[i],shape=(32,32)))
    b=wp.tile_load(rhs[i],shape=(32,1));v=wp.tile_lower_solve(factor,b)
    x=wp.tile_upper_solve(wp.tile_transpose(factor),v);wp.tile_store(out[i],x)


@torch.no_grad()
def native_pcr(a,lower,rhs,segment,max_length=None):
    if a.device.type!='cuda' or a.shape[-1]>32:raise ValueError('Native block PCR supports CUDA blocks up to 32 DOFs')
    n,d=rhs.shape
    diagonal=torch.eye(32,device=a.device).expand(n,-1,-1).clone();diagonal[:,:d,:d]=a.float()
    l=torch.zeros_like(diagonal);l[:,:d,:d]=lower.float()
    r=torch.cat((l[1:].transpose(-1,-2),torch.zeros_like(l[:1])),0).contiguous()
    b=torch.zeros((n,32,1),device=a.device);b[:,:d,0]=rhs.float()
    inputs=[diagonal,l,r,b];outputs=[torch.empty_like(x) for x in inputs]
    wp.init();stream=wp.get_stream(str(a.device));stride=1
    while stride<(n if max_length is None else max_length):
        wp.launch_tiled(_pcr_round,dim=n,inputs=[stride,wp.from_torch(segment.contiguous())]+[wp.from_torch(x) for x in inputs],outputs=[wp.from_torch(x) for x in outputs],block_dim=128,stream=stream)
        inputs,outputs=outputs,inputs;stride*=2
    wp.launch_tiled(_finish,dim=n,inputs=[wp.from_torch(inputs[0]),wp.from_torch(inputs[3])],outputs=[wp.from_torch(outputs[3])],block_dim=128,stream=stream)
    return outputs[3][:,:d,0].to(a.dtype)

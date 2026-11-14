"""Native CUDA affine prefix solve of block lower-bidiagonal systems."""
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _factor(a:wp.array3d(dtype=wp.float32),l:wp.array3d(dtype=wp.float32),
            b:wp.array3d(dtype=wp.float32),m:wp.array3d(dtype=wp.float32),v:wp.array3d(dtype=wp.float32)):
    i=wp.tid()
    c=wp.tile_cholesky(wp.tile_load(a[i],shape=(32,32)))
    lm=wp.tile_lower_solve(c,-wp.tile_load(l[i],shape=(32,32)))
    mm=wp.tile_upper_solve(wp.tile_transpose(c),lm)
    bv=wp.tile_lower_solve(c,wp.tile_load(b[i],shape=(32,1)))
    vv=wp.tile_upper_solve(wp.tile_transpose(c),bv)
    wp.tile_store(m[i],mm);wp.tile_store(v[i],vv)


@wp.kernel(enable_backward=False)
def _scan(stride:int,segment:wp.array(dtype=wp.int64),m:wp.array3d(dtype=wp.float32),
          v:wp.array3d(dtype=wp.float32),out_m:wp.array3d(dtype=wp.float32),out_v:wp.array3d(dtype=wp.float32)):
    i=wp.tid();value=wp.tile_load(v[i],shape=(32,1))
    product=wp.tile_zeros(shape=(32,32),dtype=wp.float32)
    if i>=stride:
        k=i-stride
        if segment[i]==segment[k]:
            current=wp.tile_load(m[i],shape=(32,32))
            wp.tile_matmul(current,wp.tile_load(v[k],shape=(32,1)),value)
            wp.tile_matmul(current,wp.tile_load(m[k],shape=(32,32)),product)
    wp.tile_store(out_m[i],product);wp.tile_store(out_v[i],value)


def native_lower_scan(a,lower,rhs,segment,max_length):
    if a.device.type!='cuda' or a.shape[-1]>32: raise ValueError('CUDA blocks <=32 required')
    n,d=rhs.shape
    aa=torch.eye(32,device=a.device).expand(n,-1,-1).clone();aa[:,:d,:d]=a.float()
    ll=torch.zeros_like(aa);ll[:,:d,:d]=lower.float()
    bb=torch.zeros(n,32,1,device=a.device);bb[:,:d,0]=rhs.float()
    m=torch.empty_like(aa);v=torch.empty_like(bb);om=torch.empty_like(m);ov=torch.empty_like(v)
    stream=wp.get_stream(str(a.device));wp.launch_tiled(_factor,dim=n,
        inputs=[wp.from_torch(x) for x in (aa,ll,bb)],outputs=[wp.from_torch(m),wp.from_torch(v)],block_dim=128,stream=stream)
    stride=1
    while stride<max_length:
        wp.launch_tiled(_scan,dim=n,inputs=[stride,wp.from_torch(segment)]+[wp.from_torch(x) for x in (m,v)],
            outputs=[wp.from_torch(om),wp.from_torch(ov)],block_dim=128,stream=stream)
        m,om=om,m;v,ov=ov,v;stride*=2
    return v[:,:d,0].to(a.dtype)

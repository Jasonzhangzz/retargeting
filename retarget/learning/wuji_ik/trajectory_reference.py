"""Small FP64 CPU references, independent of production CUDA/PCR execution."""
import torch


def block_ldlt(diagonal,lower,rhs,segment):
    """Segmented block LDLᵀ factorization and solve, with O(T d²) storage.

    This deliberately sequential CPU implementation is a correctness reference.
    `lower[i]` is H[i,i-1]; it must vanish at segment boundaries.
    """
    if diagonal.device.type!='cpu' or diagonal.dtype!=torch.float64:
        raise ValueError('The block LDL reference requires CPU FP64')
    if rhs.shape!=diagonal.shape[:2] or lower.shape!=diagonal.shape:
        raise ValueError('Incompatible block system shapes')
    n,d=rhs.shape
    if len(segment)!=n or n<1:raise ValueError('Empty or mismatched segment index')
    pivots=[];factors=[];forward=[]
    for i in range(n):
        link=i>0 and bool(segment[i]==segment[i-1])
        if not link and torch.count_nonzero(lower[i]):
            raise ValueError('Nonzero coupling across a segment boundary')
        factor=torch.linalg.solve(pivots[-1],lower[i].T).T if link else torch.zeros_like(lower[i])
        pivot=diagonal[i]-factor@lower[i].T
        # A Cholesky check avoids silently accepting an indefinite reference.
        torch.linalg.cholesky(pivot)
        pivots.append(pivot);factors.append(factor)
        forward.append(rhs[i]-factor@forward[-1] if link else rhs[i].clone())
    result=torch.stack([torch.linalg.solve(a,b) for a,b in zip(pivots,forward)])
    for i in range(n-2,-1,-1):
        if segment[i]==segment[i+1]:result[i]-=factors[i+1].T@result[i+1]
    return result

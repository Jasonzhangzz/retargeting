"""CUDA batched pressure QP; same capped-simplex FISTA as the CPU reference."""
import numpy as np
import warp as wp

Vec32=wp.types.vector(length=32,dtype=float)
Vec12=wp.types.vector(length=12,dtype=float)


@wp.func
def capped_simplex4(value: wp.vec4) -> wp.vec4:
    positive=wp.max(value,wp.vec4(0.))
    result=positive
    if positive[0]+positive[1]+positive[2]+positive[3]>1.:
        ordered=wp.vec4(value)
        for i in range(4):
            for j in range(3-i):
                if ordered[j]<ordered[j+1]:
                    swap=ordered[j];ordered[j]=ordered[j+1];ordered[j+1]=swap
        cumulative=float(0.)
        theta=float(0.)
        for i in range(4):
            cumulative+=ordered[i]
            threshold=(cumulative-1.)/float(i+1)
            if ordered[i]>threshold:theta=threshold
        result=wp.max(value-wp.vec4(theta),wp.vec4(0.))
    return result


@wp.kernel(enable_backward=False)
def solve_pressure_qp(matrix: wp.array3d(dtype=float), target: wp.array2d(dtype=float),
                      iterations: int, output: wp.array2d(dtype=float), cost: wp.array(dtype=float)):
    batch=wp.tid()
    x=Vec32(0.);y=Vec32(0.);moment=float(1.)
    lipschitz=float(1.e-6)
    for r in range(12):
        for c in range(32):lipschitz+=matrix[batch,r,c]*matrix[batch,r,c]
    lipschitz=wp.max(lipschitz,1.01e-6)
    for iteration in range(iterations):
        residual=Vec12(0.)
        for r in range(12):
            value=-target[batch,r]
            for c in range(32):value+=matrix[batch,r,c]*y[c]
            residual[r]=value
        proposal=Vec32(0.)
        for c in range(32):
            gradient=1.e-6*y[c]
            for r in range(12):gradient+=matrix[batch,r,c]*residual[r]
            proposal[c]=y[c]-gradient/lipschitz
        next_x=Vec32(0.)
        for contact in range(8):
            j=contact*4
            projected=capped_simplex4(wp.vec4(proposal[j],proposal[j+1],proposal[j+2],proposal[j+3]))
            for edge in range(4):next_x[j+edge]=projected[edge]
        next_moment=.5*(1.+wp.sqrt(1.+4.*moment*moment))
        y=next_x+(moment-1.)/next_moment*(next_x-x)
        x=next_x;moment=next_moment
    value=float(0.)
    for r in range(12):
        final_residual=-target[batch,r]
        for c in range(32):final_residual+=matrix[batch,r,c]*x[c]
        value+=final_residual*final_residual
    for c in range(32):output[batch,c]=x[c]
    cost[batch]=value


class ContactQP:
    def __init__(self,device):
        self.device=device
        self.batch=None

    def solve(self,matrix,target,iterations):
        batch=matrix.shape[0]
        if batch!=self.batch:
            self.matrix=wp.empty((batch,12,32),dtype=float,device=self.device)
            self.target=wp.empty((batch,12),dtype=float,device=self.device)
            self.output=wp.empty((batch,32),dtype=float,device=self.device)
            self.cost=wp.empty(batch,dtype=float,device=self.device)
            self.batch=batch
        self.matrix.assign(np.asarray(matrix,np.float32));self.target.assign(np.asarray(target,np.float32))
        wp.launch(solve_pressure_qp,batch,inputs=[self.matrix,self.target,iterations,self.output,self.cost],device=self.device)
        return self.output.numpy(),self.cost.numpy()

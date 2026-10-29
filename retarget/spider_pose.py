"""Differentiable SPIDER quaternion log and unsquared norm primitives.

The quaternion convention is xyzw inside Warp, wxyz in MuJoCo qpos.
SPIDER's quat_sub computes Log(inverse(reference) * actual).
"""
import warp as wp


@wp.func
def zero_safe_sqrt(value: float):
    return wp.sqrt(wp.max(value, 0.))


@wp.func_grad(zero_safe_sqrt)
def zero_safe_sqrt_grad(value: float, adj_ret: float):
    if value > 0.:
        wp.adjoint[value] += .5*adj_ret/wp.sqrt(value)


@wp.func
def quat_log(q: wp.quat):
    axis=wp.vec3(q[0],q[1],q[2])
    length=zero_safe_sqrt(wp.dot(axis,axis))
    result=wp.vec3(0.)
    if length > 1.e-8:
        angle=2.*wp.atan2(length,q[3])
        if angle > wp.pi:angle-=2.*wp.pi
        result=axis*(angle/length)
    else:
        # Exact first derivative of Log at either quaternion representative.
        result=axis*(2./wp.where(q[3]>=0.,1.,-1.))
    return result


@wp.func
def quat_exp(vector: wp.vec3):
    length=zero_safe_sqrt(wp.dot(vector,vector))
    scale=float(.5)
    if length > 1.e-8:scale=wp.sin(.5*length)/length
    return wp.quat(scale*vector[0],scale*vector[1],scale*vector[2],wp.cos(.5*length))


@wp.func
def quaternion_error_squared(actual: wp.quat, reference: wp.quat):
    error=quat_log(wp.quat_inverse(reference)*actual)
    return wp.dot(error,error)


@wp.func
def cross_matrix(v: wp.vec3):
    return wp.mat33(0.,-v[2],v[1],v[2],0.,-v[0],-v[1],v[0],0.)


@wp.func
def right_jacobian(v: wp.vec3):
    theta2=wp.dot(v,v)
    a=float(.5);b=float(1./6.)
    if theta2 > 1.e-6:
        theta=wp.sqrt(theta2)
        a=(1.-wp.cos(theta))/theta2
        b=(theta-wp.sin(theta))/(theta2*theta)
    else:
        a-=theta2/24.;b-=theta2/120.
    skew=cross_matrix(v)
    return wp.identity(n=3,dtype=float)-a*skew+b*(skew*skew)


@wp.func
def inverse_right_jacobian(v: wp.vec3):
    theta2=wp.dot(v,v)
    a=float(1./12.)
    if theta2 > 1.e-6:
        theta=wp.sqrt(theta2)
        a=(1.-.5*theta*wp.cos(.5*theta)/wp.sin(.5*theta))/theta2
    else:a+=theta2/720.
    skew=cross_matrix(v)
    return wp.identity(n=3,dtype=float)+.5*skew+a*(skew*skew)

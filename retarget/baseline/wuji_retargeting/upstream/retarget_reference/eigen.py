"""Real 3x3 specialization of nalgebra 0.33.2 symmetric eigendecomposition.

Ported from https://github.com/dimforge/nalgebra/tree/v0.33.2/src/linalg
files symmetric_eigen.rs, symmetric_tridiagonal.rs, householder.rs, givens.rs,
qr.rs and corresponding base/blas.rs and geometry/reflection.rs operations.
Upstream license: Apache-2.0; see LICENSE-nalgebra.txt and NOTICE.txt.
Changes: Python/NumPy real-scalar 3x3 specialization; SDK-observed epsilon;
explicit scalar evaluation ordering; no general complex or dynamic API.
"""
import math
import numpy as np

SDK_EIGEN_EPS = 1.1102230246251565e-15


def _dot(a, b):
    total = 0.
    for x, y in zip(a, b):
        total += float(x) * float(y)
    return total


def _rotate_columns(q, i, c, s):
    for row in range(3):
        a, b = float(q[row, i]), float(q[row, i+1])
        q[row, i] = a*c + s*b
        q[row, i+1] = -s*a + b*c


def symmetric_eigen3(matrix, eps=SDK_EIGEN_EPS):
    """Return unsorted eigenvalues and column eigenvectors, preserving ties."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    amax = float(np.max(np.abs(m)))
    if amax != 0.:
        for row in range(3):
            for col in range(3):
                m[row, col] = float(m[row, col]) / amax
    signed_off = np.zeros(2)
    for i in range(2):
        axis = m[i+1:, i]
        sq = _dot(axis, axis)
        norm = math.sqrt(sq)
        first = float(axis[0])
        signed_norm = -norm if first < -first else norm
        factor = (sq + abs(first)*norm)*2.
        axis[0] = first + signed_norm
        if factor != 0.:
            denom = math.sqrt(factor)
            for j in range(len(axis)):
                axis[j] = float(axis[j])/denom
            denom = math.sqrt(_dot(axis, axis))
            for j in range(len(axis)):
                axis[j] = float(axis[j])/denom
            signed_off[i] = -signed_norm
            size = len(axis)
            block = m[i+1:, i+1:]
            # hegemv(2, lower-triangle block, axis, 0), column update order.
            p = np.zeros(size)
            first_factor = float(axis[0])*2.
            for row in range(size):
                p[row] = first_factor*float(block[row, 0])
            p[0] += _dot(block[1:, 0], axis[1:])*2.
            for col in range(1, size):
                column_factor = float(axis[col])*2.
                for row in range(col, size):
                    p[row] += column_factor*float(block[row, col])
                p[col] += _dot(block[col+1:, col], axis[col+1:])*2.
            dot = _dot(axis, p)
            # Three separate hegerc calls, only the lower triangle is written.
            for col in range(size):
                for row in range(col, size):
                    block[row, col] -= float(p[row])*float(axis[col])
            for col in range(size):
                f = -float(p[col])
                for row in range(col, size):
                    block[row, col] += f*float(axis[row])
            dot *= 2.
            for col in range(size):
                f = float(axis[col])*dot
                for row in range(col, size):
                    block[row, col] += f*float(axis[row])
        else:
            signed_off[i] = signed_norm
    diag = np.diag(m).copy()
    q = np.eye(3)
    for i in (1, 0):
        axis = m[i+1:, i]
        sign = math.copysign(1., float(signed_off[i]))
        for col in range(i, 3):
            factor = (_dot(axis, q[i+1:, col])-0.) * (sign*-2.)
            for row in range(i+1, 3):
                q[row, col] = factor*float(axis[row-i-1]) + sign*float(q[row, col])
    off = np.abs(signed_off)

    def delimit(end):
        n = end
        while n > 0:
            if abs(float(off[n-1])) > eps*(abs(float(diag[n])) + abs(float(diag[n-1]))):
                break
            n -= 1
        if n == 0:
            return 0, 0
        start = n-1
        while start > 0:
            value = float(off[start-1])
            if value == 0. or abs(value) <= eps*(abs(float(diag[start])) + abs(float(diag[start-1]))):
                off[start-1] = 0.
                break
            start -= 1
        return start, n

    start, end = delimit(2)
    while start != end:
        if end-start+1 > 2:
            a, b, t = float(diag[end-1]), float(diag[end]), float(off[end-1])
            sq = t*t
            shift = b
            if sq != 0.:
                d = (a-b)*.5
                shift = b - sq/(d + math.copysign(1.,d)*math.sqrt(d*d+sq))
            vx, vy = float(diag[start])-shift, float(off[start])
            for i in range(start, end):
                if vy == 0.:
                    break
                denom = math.sqrt(abs(vx)*abs(vx)+vy*vy)
                norm = -denom if vx < 0. else denom
                c, s = abs(vx)/denom, -vy/norm
                if i > start:
                    off[i-1] = norm
                a, b, t = float(diag[i]), float(diag[i+1]), float(off[i])
                cc, ss, cs = c*c, s*s, c*s
                twice = (cs+cs)*t
                diag[i] = (cc*a+ss*b)-twice
                diag[i+1] = (ss*a+cc*b)+twice
                off[i] = cs*(a-b)+t*(cc-ss)
                if i != end-1:
                    vx, vy = float(off[i]), -s*float(off[i+1])
                    off[i+1] *= c
                _rotate_columns(q, i, c, -s)
            if abs(float(off[end-1])) <= eps*(abs(float(diag[end-1]))+abs(float(diag[end]))):
                end -= 1
        else:
            a, b, t = float(diag[start]), float(diag[start+1]), float(off[start])
            d = (a-b)*.5
            root = math.sqrt(d*d+t*t)
            mean = (a+b)*.5
            high, low = mean+root, mean-root
            vx, vy = high-b, t
            diag[start], diag[start+1] = high, low
            denom = math.sqrt(abs(vx)*abs(vx)+vy*vy)
            if denom > eps:
                norm = -denom if vx < 0. else denom
                _rotate_columns(q, start, abs(vx)/denom, vy/norm)
            end -= 1
        start, end = delimit(end)
    return diag*amax, q


def _qr_diagonal(matrix):
    """Householder QR's signed diagonal; Q is not needed by the SDK normal."""
    m = matrix.copy()
    diagonal = np.empty(3)
    for i in range(3):
        axis = m[i:, i]
        sq = _dot(axis, axis)
        norm = math.sqrt(sq)
        first = float(axis[0])
        signed_norm = -norm if first < -first else norm
        factor = (sq+abs(first)*norm)*2.
        axis[0] = first+signed_norm
        if factor != 0.:
            denom = math.sqrt(factor)
            for j in range(len(axis)):
                axis[j] = float(axis[j])/denom
            denom = math.sqrt(_dot(axis, axis))
            for j in range(len(axis)):
                axis[j] = float(axis[j])/denom
            diagonal[i] = -signed_norm
            sign = math.copysign(1., -signed_norm)
            for col in range(i+1, 3):
                f = (_dot(axis, m[i:, col])-0.)*(sign*-2.)
                for row in range(i, 3):
                    m[row, col] = f*float(axis[row-i])+sign*float(m[row, col])
        else:
            diagonal[i] = signed_norm
    return diagonal


def palm_svd(centered_palm):
    """SDK Gram-eigen, signed column-sort, QR singular proxy and V columns."""
    c = np.asarray(centered_palm, dtype=np.float64).reshape(3, 3)
    covariance = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            covariance[i,j] = ((float(c[0,i])*float(c[0,j]) + float(c[1,i])*float(c[1,j]))
                               + float(c[2,i])*float(c[2,j]))
    _, vectors = symmetric_eigen3(covariance)
    product = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            product[i,j] = ((float(c[i,0])*float(vectors[0,j]) + float(c[i,1])*float(vectors[1,j]))
                            + float(c[i,2])*float(vectors[2,j]))
    norms = [_dot(product[:,j],product[:,j]) for j in range(3)]
    for a,b in ((0,1),(0,2),(1,2)):
        if norms[b] > norms[a]:
            norms[a],norms[b] = norms[b],norms[a]
            product[:,[a,b]] = product[:,[b,a]]
            vectors[:,[a,b]] = vectors[:,[b,a]]
            product[:,b] *= -1.
            vectors[:,b] *= -1.
    return np.abs(_qr_diagonal(product)), vectors


def palm_normal(centered_palm):
    """V column at first minimum abs(QR diagonal), retaining native tie rules."""
    values, vectors = palm_svd(centered_palm)
    best = 0
    for i in (1, 2):
        if values[best] > values[i]:
            best = i
    return vectors[:, best].copy()

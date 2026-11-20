"""Direction-preserving contact queries after virtual depenetration.

Retraction changes only the query used to choose a material witness. Measured
geometry and true signed distances remain untouched; callers differentiate the
selected witness through the original FK pose.
"""
import warp as wp

DEEP_PENETRATION_M = .0005


@wp.struct
class ContactWitness:
    point: wp.vec3
    normal: wp.vec3
    face: int
    u: float
    v: float
    valid: int
    retracted: int


@wp.func
def contact_witness(skin: wp.uint64, solid: wp.uint64, point: wp.vec3,
                    escape: wp.vec3, threshold: float):
    result = ContactWitness()
    nearest = wp.mesh_query_point_no_sign(skin, point, 2.)
    signed = wp.mesh_query_point_sign_winding_number(solid, point, 2.)
    if not nearest.result or not signed.result:
        return result
    hit = wp.mesh_eval_position(skin, nearest.face, nearest.u, nearest.v)
    normal = wp.mesh_eval_face_normal(skin, nearest.face)
    if signed.sign < 0.:
        solid_hit = wp.mesh_eval_position(solid, signed.face, signed.u, signed.v)
        # An inside nearest face pointing against the known escape direction is
        # also ambiguous, even if the point is already close to the wrong side.
        if wp.length(point-solid_hit) > threshold or wp.dot(normal, escape) < 0.:
            result.retracted = 1
            if wp.length(escape) < 1.e-8:
                return result
            direction = wp.normalize(escape)
            exit = wp.mesh_query_ray(solid, point, direction, 2.)
            if not exit.result or wp.dot(exit.normal, direction) <= .1:
                return result
            probe = point + (exit.t + .00001)*direction
            outside = wp.mesh_query_point_sign_winding_number(solid, probe, 2.)
            if not outside.result or outside.sign < 0.:
                return result
            nearest = wp.mesh_query_point_no_sign(skin, probe, 2.)
            if not nearest.result:
                return result
            hit = wp.mesh_eval_position(skin, nearest.face, nearest.u, nearest.v)
            normal = wp.mesh_eval_face_normal(skin, nearest.face)
            # A closure cap / unrelated semantic patch is not a contact surface.
            if wp.length(probe-hit) > threshold or wp.dot(normal, direction) <= .1:
                return result
    result.point = hit
    result.normal = normal
    result.face = nearest.face
    result.u = nearest.u
    result.v = nearest.v
    result.valid = 1
    return result

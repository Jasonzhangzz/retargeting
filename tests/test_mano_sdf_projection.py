"""Independent analytic-box check of the experiment's nearest SDF projection."""
import numpy as np
from scipy.spatial import ConvexHull
from retarget.contact_exploration import ObjectMeshSurface
from scripts.benchmark_mano_sdf_perturbation import nearest_sdf


def test_mesh_sdf_projection_matches_analytic_box_inside_outside_and_edges():
    vertices=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    faces=ConvexHull(vertices).simplices.copy()
    for i,face in enumerate(faces):
        a,b,c=vertices[face]
        if np.dot(np.cross(b-a,c-a),(a+b+c)/3)<0:faces[i]=face[[0,2,1]]
    surface=ObjectMeshSurface(vertices,faces)
    rng=np.random.default_rng(0)
    points=np.r_[rng.uniform(-2.,2.,(100,3)),[[0.,0.,0.],[1.,.3,.2],[2.,2.,2.]]]
    projected,normal,phi=nearest_sdf(surface,points)
    q=np.abs(points)-1.
    expected=np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(np.max(q,axis=1),0.)
    np.testing.assert_allclose(phi,expected,atol=1e-10)
    np.testing.assert_allclose(np.max(np.abs(projected),axis=1),1.,atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(projected-points,axis=1),np.abs(expected),atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(normal,axis=1),1.,atol=1e-10)

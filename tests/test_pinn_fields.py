"""Units, geometry derivatives and initial-state boundaries of the PINN fields."""

import numpy as np
import pytest

from grouting_pino.pinn_mechanics import NeuralBasis, annulus_points, COMPONENTS


@pytest.mark.parametrize("domain,radii",[("soil",[7.,12.,25.]),
                                         ("paste",[5.83,5.9,5.99]),
                                         ("lining",[5.31,5.5,5.71])])
def test_cartesian_derivatives_and_reflection(domain,radii):
    basis=NeuralBasis.create(domain,6,4,np.random.default_rng(17))
    xy=annulus_points(np.array(radii),np.array([.4,1.3,2.7]))
    eps=1e-5
    for comp in COMPONENTS:
        value,dx,dy=basis.evaluate(xy,comp)
        for axis,derivative in [(0,dx),(1,dy)]:
            shift=np.zeros_like(xy)
            shift[:,axis]=eps
            difference=(basis.evaluate(xy+shift,comp)[0]-basis.evaluate(xy-shift,comp)[0])/(2*eps)
            np.testing.assert_allclose(difference,derivative,rtol=2e-4,atol=3e-5)
        mirrored=xy.copy()
        mirrored[:,0]*=-1
        sign=-1 if comp in ("ux","sxy") else 1
        np.testing.assert_allclose(basis.evaluate(mirrored,comp)[0],sign*value,atol=1e-12)


def test_soil_displacement_constraints_are_hard():
    basis=NeuralBasis.create("soil",6,4,np.random.default_rng(17))
    side=np.c_[np.full(17,30),np.linspace(-60,0,17)]
    base=np.c_[np.linspace(-30,30,17),np.full(17,-60)]
    np.testing.assert_array_equal(basis.evaluate(side,"ux")[0],0)
    for comp in ("ux","uy"):
        np.testing.assert_array_equal(basis.evaluate(base,comp)[0],0)

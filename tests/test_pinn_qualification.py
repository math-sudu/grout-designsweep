"""Boundary calculations must retain history, spatial control and time extrema."""

import numpy as np
import pytest
from scipy.interpolate import BSpline, PPoly

from grouting_pino.pinn_qualification import (
    latest_required_increments, milestone_demand_polygon, polynomial_range,
    polynomial_sublevel_intervals, spline_polynomial,
)


def test_time_range_finds_an_interior_rise_and_an_isolated_feasible_time():
    # The endpoints hide a peak at t=1/2.
    polynomial = PPoly(np.array([[-4.], [4.], [0.]]), [0., 1.])
    low, high = polynomial_range(polynomial)
    assert low == 0 and high == 1
    assert polynomial_range(polynomial.derivative())[1] == 4
    np.testing.assert_allclose(polynomial_sublevel_intervals(polynomial, .75),
                               [[0., .25], [.75, 1.]])
    trough = PPoly(-polynomial.c, polynomial.x)
    assert polynomial_sublevel_intervals(trough, -1.) == [[.5, .5]]


def test_polynomial_conversion_preserves_vector_readouts_at_c0_knots():
    knots = [0., 0., 0., 0., 1., 1., 1., 2., 2., 2., 2.]
    coefficients = np.arange(14).reshape(7, 2) ** 2 / 20
    spline = BSpline(knots, coefficients, 3)
    polynomial = spline_polynomial(spline)
    times = [0., .37, .9999, 1., 1.0001, 1.73, 2.]
    np.testing.assert_allclose(polynomial(times), spline(times), atol=1e-13)


def test_joint_demands_can_exceed_the_budget_when_each_alone_qualifies():
    # Final response is 1 + a + b; the two individual tasks use 1.4 and 1.5.
    fields = np.array([[3., 2.], [2., 1.5], [1., .5]])
    polygon = milestone_demand_polygon(fields, -1.6, 1.6)
    assert np.max(polygon.sum(axis=1)) == pytest.approx(.6)
    increments = latest_required_increments([.2, .5, 1.])
    np.testing.assert_allclose(increments, [.2, .3, .5])
    assert (latest_required_increments([.2, .2, 1.]) @ fields).max() == pytest.approx(1.4)
    assert (latest_required_increments([0., .5, 1.]) @ fields).max() == pytest.approx(1.5)
    assert (increments @ fields).max() == pytest.approx(1.7)


def test_boundary_keeps_spatial_and_intermediate_constraints():
    # An off-axis point controls the first-stage uplift, while the final sum cancels it.
    fields = np.array([[2., -4.], [1.5, 0.], [1., 4.]])
    polygon = milestone_demand_polygon(fields, -1., 2.)
    assert len(polygon)
    assert polygon[:, 0].max() == pytest.approx(.25)
    for a, b in polygon:
        stages = np.cumsum(np.array([a, b-a, 1-b])[:, None] * fields, axis=0)
        assert stages.min() >= -1. - 1e-12
        assert stages.max() <= 2. + 1e-12


def test_redundant_cumulative_requirement_does_not_unload():
    np.testing.assert_allclose(latest_required_increments([.4, .2, .7]), [.4, 0., .6])
    with pytest.raises(ValueError):
        latest_required_increments([.4, 1.2])

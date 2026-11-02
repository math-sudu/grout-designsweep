"""Checks of load memory and spatial constraints used by the history design."""

import numpy as np
import pytest

from grouting_pino.support_history import cumulative_response, two_stage_fraction_interval


def test_holding_and_unloading_a_solidifying_bar():
    # Unit bar: stiffness doubles while a unit force is held. Stress-free
    # growth preserves the strain, then unloading recovers only half of it.
    history = cumulative_response([1., .5, .5], [1., 0., -1.])
    np.testing.assert_allclose(history, [1., 1., .5])
    np.testing.assert_allclose(cumulative_response([1., 1.], [1., -1.]), [1., 0.])


def test_peak_is_taken_after_accumulating_fields():
    fields = [[4., 1.], [1., 4.]]
    history = cumulative_response(fields, [.5, .5])
    np.testing.assert_allclose(history[-1], [2.5, 2.5])
    assert two_stage_fraction_interval(*fields, 2.5) == pytest.approx((.5, .5))
    assert two_stage_fraction_interval(*fields, 2.4) is None


def test_intermediate_stage_and_flat_constraints_are_enforced():
    # Final unloading/uplift cannot excuse an excessive earlier settlement.
    assert two_stage_fraction_interval([4.], [-4.], 1.) == pytest.approx((0., .25))
    assert two_stage_fraction_interval([2.], [2.], 2.) == (0., 1.)
    assert two_stage_fraction_interval([2.], [2.], 1.) is None


def test_prescribed_scenarios_share_one_loading_fraction():
    # A feasible interval for each scenario need not overlap.
    assert two_stage_fraction_interval([[4.], [1.]], [[1.], [4.]], 2.5) == (.5, .5)
    assert two_stage_fraction_interval([[4.], [1.]], [[1.], [4.]], 2.4) is None


def test_existing_displacement_consumes_the_same_nodal_budget():
    initial = np.array([1., .25])
    early, late = np.array([4., 1.]), np.array([1., 4.])
    history = cumulative_response([early, late], [.5, .5], initial_response=initial)
    np.testing.assert_allclose(history[-1], [3.5, 2.75])
    assert two_stage_fraction_interval(early, late, 3., initial_response=initial) is None
    assert two_stage_fraction_interval(early, late, 3.5, initial_response=initial) == pytest.approx((.25, .5))
    assert two_stage_fraction_interval([-1.], [-1.], .5, initial_response=[1.]) is None

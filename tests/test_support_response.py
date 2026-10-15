"""Common-target decisions, table inversion and reference-section geometry."""

from pathlib import Path

import numpy as np
import pytest

from grouting_pino.support_response import (
    SectionResponse,
    first_attainment,
    meets_deadline,
)


def test_shared_target_does_not_follow_candidate_endpoint():
    time = np.array([0.0, 2.0, 4.0])
    # The second material completes its own small response change earlier,
    # but never provides the support required of both candidates.
    first = first_attainment(time, np.array([10.0, 6.0, 2.0]), 5.0)
    second = first_attainment(time, np.array([10.0, 8.0, 8.0]), 5.0)
    assert first.time_h == pytest.approx(2.5)
    assert first.status == "attained"
    assert second.time_h is None
    assert second.status == "not_reached_in_window"
    assert meets_deadline(time, np.array([10.0, 6.0, 2.0]), 5.0, 3.0)
    assert not meets_deadline(time, np.array([10.0, 8.0, 8.0]), 5.0, 3.0)


def test_initial_and_last_sample_attainment():
    time = np.array([0.0, 3.0, 7.0])
    initial = first_attainment(time, np.array([5.0, 4.0, 2.0]), 5.0)
    end = first_attainment(time, np.array([9.0, 7.0, 5.0]), 5.0)
    assert (initial.status, initial.time_h) == ("initially_met", 0.0)
    assert (end.status, end.time_h) == ("attained", 7.0)


def test_attainment_is_independent_of_later_window_endpoint():
    short = first_attainment(np.array([0.0, 4.0]), np.array([10.0, 2.0]), 5.0)
    long = first_attainment(np.array([0.0, 4.0, 8.0]), np.array([10.0, 2.0, 1.0]), 5.0)
    assert short == long


def test_deadline_response_can_differ_from_first_attainment():
    time = np.array([0.0, 2.0, 4.0])
    response = np.array([8.0, 4.0, 8.0])
    assert first_attainment(time, response, 5.0).time_h == pytest.approx(1.5)
    assert not meets_deadline(time, response, 5.0, 4.0)
    with pytest.raises(ValueError, match="within the sampled history"):
        meets_deadline(time, response, 5.0, 5.0)


def test_inversion_uses_log_modulus_and_reports_table_boundaries():
    section = SectionResponse(np.array([1e6, 100e6]), np.array([9.0, 1.0]), {})
    assert section.required_modulus_pa(5.0) == pytest.approx(10e6)
    assert section.required_modulus_pa(10.0) == pytest.approx(1e6)
    assert section.required_modulus_pa(0.5) is None
    assert section.compliance(10e6) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="outside the response table"):
        section.compliance(200e6)


def test_nonmonotone_response_retains_disconnected_feasible_intervals():
    section = SectionResponse(np.array([1.0, 10.0, 100.0, 1000.0]),
                              np.array([8.0, 2.0, 8.0, 2.0]), {})
    intervals = section.admissible_modulus_intervals(5.0)
    np.testing.assert_allclose(intervals,
                               [(10 ** 0.5, 10 ** 1.5), (10 ** 2.5, 1000.0)])
    with pytest.raises(ValueError, match="nonmonotone"):
        section.required_modulus_pa(5.0)


def test_plateau_and_isolated_admissible_point():
    plateau = SectionResponse(np.array([1.0, 2.0, 3.0, 4.0]),
                              np.array([8.0, 5.0, 5.0, 2.0]), {})
    assert plateau.admissible_modulus_intervals(5.0) == [(2.0, 4.0)]
    point = SectionResponse(np.array([1.0, 2.0, 3.0]), np.array([8.0, 5.0, 8.0]), {})
    assert point.admissible_modulus_intervals(5.0) == [(2.0, 2.0)]


def test_reference_geometry_matches_the_excavation_and_erected_ring():
    root = Path(__file__).resolve().parents[1]
    section = SectionResponse.load(root / "results/support_design/reference_response.npz")
    geometry = section.geometry
    assert geometry["excavation_diameter_m"] == pytest.approx(12.04)
    assert geometry["lining_outer_diameter_m"] == pytest.approx(11.60)
    assert geometry["lining_inner_diameter_m"] == pytest.approx(10.50)
    assert geometry["grout_thickness_m"] == pytest.approx(0.22)
    assert geometry["axis_depth_m"] == pytest.approx(25.52)
    # At equal reference modulus, section-specific reference targets invert to
    # the same requirement; changing a section's load response cannot restore
    # a mechanical geometry effect on that performance-equivalence criterion.
    other = SectionResponse(section.modulus_pa, 2 * section.compliance_m_per_pa, {})
    for response in (section, other):
        target = response.compliance(20e6)
        assert response.required_modulus_pa(target) == pytest.approx(20e6)

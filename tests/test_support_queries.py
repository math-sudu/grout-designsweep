"""Censored deadlines and disagreements across model assumptions."""

import numpy as np
import pytest
from pathlib import Path
import sys

from grouting_pino.support_response import (
    Attainment, classify_scenarios, deadline_responses, first_attainment, loading_event,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_support_queries import summarize_designs
from run_support_events import summarize_events


def test_event_can_be_scheduled_before_a_censored_late_deadline():
    hit = first_attainment([0, 1, 2], [3, 1, 4], 2)
    event = loading_event(hit, 2, 3)
    assert event.status == "available"
    assert event.earliest_time_h == pytest.approx(0.5)
    assert not np.isfinite(deadline_responses([0, 1, 2], [3, 1, 4], [3])[0])


def test_loading_update_retains_the_clock_and_does_not_schedule_in_the_past():
    hit = first_attainment([0, 1], [3, 1], 2)
    event = loading_event(hit, 1, 3, state_time_h=2)
    assert event.remaining_time_h == 0.5
    assert event.earliest_time_h == 2.5
    assert event.status == "available"
    assert loading_event(hit, 1, 2.4, state_time_h=2).status == "unavailable"
    already_sufficient = first_attainment([0, 1], [1, 1], 2)
    assert loading_event(already_sufficient, 1, 3, state_time_h=2).earliest_time_h == 2


def test_unobserved_event_is_unknown_only_when_the_loading_window_is_incomplete():
    missing = Attainment(None, "not_reached_in_window")
    assert loading_event(missing, 2, 3).status == "unresolved"
    assert loading_event(missing, 2, 2).status == "unavailable"
    assert loading_event(Attainment(0, "initially_met"), 2, 0).status == "available"


def test_one_missed_scenario_excludes_a_candidate_from_the_all_scenario_choice():
    rows = [dict(case="reference", mix_id=m, target_um_kpa=2.2, time_h=t,
                 status="attained" if t is not None else "not_reached_in_window",
                 supported_end_h=end)
            for m, t, end in [("W40", 1, 2), ("W48", None, 2)]]
    masses = {"W40": 1400, "W48": 1200}
    choice = summarize_events(rows, [3], masses, "thermal")[0]
    assert choice["least_cement_all_evaluated_mix"] == "W40"
    assert not choice["least_cement_choice_resolved"]
    rows.append(dict(rows[-1], supported_end_h=4))
    choice = summarize_events(rows, [3], masses, "thermal")[0]
    assert choice["least_cement_choice_resolved"]
    assert choice["infeasible_some_evaluated"] == "W48"


def test_observed_attainment_does_not_fill_a_censored_deadline():
    times = [0, 1, 2]
    response = [3e-9, 2e-9, 1e-9]
    hit = first_attainment(times, response, 2.5e-9)
    assert hit.time_h == pytest.approx(0.5)
    queried = deadline_responses(times, response, [0, 1.5, 2, 3])
    np.testing.assert_allclose(queried[:3], [3e-9, 1.5e-9, 1e-9])
    assert np.isnan(queried[-1])


@pytest.mark.parametrize("responses, expected", [
    ([1, 2], "feasible_all_evaluated"),
    ([3, 4], "infeasible_all_evaluated"),
    ([1, 3], "assumption_sensitive"),
    ([1, 3, np.nan], "assumption_sensitive"),
    ([1, np.nan], "domain_limited"),
    ([3, np.nan], "domain_limited"),
    ([np.nan, np.nan], "domain_limited"),
])
def test_agreement_requires_every_evaluated_scenario(responses, expected):
    state = classify_scenarios(responses, 2)
    assert state["status"] == expected
    assert sum(state[k] for k in ("feasible_count", "infeasible_count", "outside_domain_count")) == len(responses)


def test_an_unattained_demand_has_no_invented_time():
    assert first_attainment([0, 1, 2], [3, 2.5, 2], 1).time_h is None
    assert first_attainment([0, 1, 2], [3, 2.5, 2], 4).time_h == 0


def test_invalid_inputs_are_distinct_from_censoring():
    with pytest.raises(ValueError):
        deadline_responses([0, 1], [3, 2], [-1])
    with pytest.raises(ValueError):
        classify_scenarios([np.inf], 2)


def test_censored_cheaper_candidate_prevents_a_resolved_optimum():
    rows = [dict(case="reference", deadline_h=3.0, mix_id=mix, response_um_kpa=value)
            for mix, value in [("W32", 1.0), ("W40", 1.0), ("W48", np.nan)]]
    _, choices = summarize_designs(rows, [2e-9], {"W32": 1500, "W40": 1400, "W48": 1200})
    assert choices[0]["least_cement_all_evaluated_mix"] == "W40"
    assert not choices[0]["least_cement_choice_resolved"]
    rows[-1]["response_um_kpa"] = 3.0
    _, choices = summarize_designs(rows, [2e-9], {"W32": 1500, "W40": 1400, "W48": 1200})
    assert choices[0]["least_cement_choice_resolved"]


def test_spatial_probe_exposes_a_choice_with_too_little_margin():
    rows = [dict(case="reference", deadline_h=3.0, mix_id=mix, response_um_kpa=value)
            for mix, value in [("W32", 1.0), ("W40", 1.99), ("W48", 3.0)]]
    _, choices = summarize_designs(rows, [2e-9], {"W32": 1500, "W40": 1400, "W48": 1200}, 0.02)
    assert choices[0]["least_cement_all_evaluated_mix"] == "W32"
    assert choices[0]["assumption_sensitive"] == "W40"
    assert not choices[0]["least_cement_choice_resolved"]

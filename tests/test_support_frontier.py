"""Decision boundaries of the continuous cement-time frontier."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_support_events import adjacent_tradeoffs, frontier_segments


def event(mix, time, end=6.0, scenario="nominal"):
    return dict(case="reference", target_um_kpa=2.2, mix_id=mix,
                scenario=scenario, time_h=time, supported_end_h=end,
                status="attained" if time is not None else "not_reached_in_window")


def test_frontier_switches_at_the_latest_scenario_and_skips_dominated_material():
    rows = [event("A", 1), event("A", 1.2, scenario="cold"),
            event("B", 3), event("B", 4, scenario="cold"),
            event("C", 2), event("C", 3, scenario="cold")]
    result = frontier_segments(rows, {"A": 1500, "B": 1400, "C": 1200}, "thermal")
    choices = [r for r in result if r["cement_kg_m3"] is not None]
    assert [(r["least_cement_all_evaluated_mix"], r["start_h"])
            for r in choices] == [("A", 1.2), ("C", 3.0), ("C", 4.0)]
    trade = adjacent_tradeoffs(result)
    assert len(trade) == 1
    assert trade[0]["from_mix"] == "A" and trade[0]["to_mix"] == "C"
    assert trade[0]["extra_time_h"] == pytest.approx(1.8)
    assert trade[0]["cement_saved_kg_m3"] == 300


def test_censored_cheaper_candidate_changes_resolution_after_its_endpoint():
    result = frontier_segments([event("A", 1), event("B", None, end=2)],
                               {"A": 1500, "B": 1200}, "nominal")
    resolved = [r for r in result if r["cement_kg_m3"] and r["least_cement_choice_resolved"]]
    assert len(resolved) == 1
    assert resolved[0]["start_h"] == 1 and resolved[0]["start_inclusive"]
    assert resolved[0]["end_h"] == 2 and resolved[0]["end_inclusive"]
    assert result[-1]["start_h"] == 2 and not result[-1]["start_inclusive"]
    assert not result[-1]["least_cement_choice_resolved"]
    assert adjacent_tradeoffs(result) == []


def test_initially_met_lowest_cement_remains_available_after_history_end():
    result = frontier_segments([event("A", 0, end=1), event("B", None, end=2)],
                               {"A": 1200, "B": 1500}, "nominal")
    assert all(r["least_cement_all_evaluated_mix"] == "A" for r in result)
    assert all(r["least_cement_choice_resolved"] for r in result)
    assert result[0]["start_h"] == 0 and result[-1]["end_h"] == 6

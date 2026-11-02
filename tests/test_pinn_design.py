"""Consequential selection cases: disconnected windows and surface extrema."""

import numpy as np
import pytest
import json
from pathlib import Path
import sys
import torch

from grouting_pino.pinn_design import (compare_plans, feasible_intervals,
                                      solve_prescribed_plan, surface_maximum, surface_response)
from grouting_pino.pinn_mechanics import PRESSURE


def test_disconnected_windows_keep_both_crossing_directions():
    # Feasible on [0.5, 1.5] and [2.5, 3.5], not after the first crossing forever.
    f = lambda t: (t - .5) * (t - 1.5) * (t - 2.5) * (t - 3.5)
    result = feasible_intervals(f, np.linspace(0, 4, 10), .002)
    assert len(result["intervals_h"]) == 2
    for actual, expected in zip(result["intervals_h"], [[.5, 1.5], [2.5, 3.5]]):
        np.testing.assert_allclose(actual, expected, atol=.002)
        assert f(actual[0]) <= 0 and f(actual[1]) <= 0
    assert [row["direction"] for row in result["crossings"]] == ["enter", "leave", "enter", "leave"]


def test_material_selection_uses_remaining_window_and_checks_reference():
    materials = [{"mix_id": name, "cement_kg_m3": cement,
                  "handover": {"intervals_h": intervals}}
                 for name, cement, intervals in [
                     ("W32", 1500, [[1., 4.]]), ("W40", 1400, [[2.5, 4.]]),
                     ("W48", 1200, [[.5, 1.5], [3.5, 3.8]])]]
    fields = [{"mix_id": mix, "planned_handover_h": plan, "status": status}
              for plan, statuses in [(2., ["feasible", "displacement_infeasible", "displacement_infeasible"]),
                                     (4., ["feasible", "feasible", "outside_material_domain"]),
                                     (0., ["displacement_infeasible"] * 3)]
              for mix, status in zip(["W32", "W40", "W48"], statuses)]
    rows = compare_plans(materials, fields, [2., 4., 0.], 4., "W32")
    assert rows[0]["B1_mix"] == "W32" and rows[0]["B2_mix"] == "W48"
    assert rows[0]["adjusted_handover_h"] == 3.5
    assert rows[1]["B2_mix"] == "W40" and rows[1]["added_cycle_time_min"] == 0
    assert rows[2]["B0_status"] == "reference_infeasible"
    assert rows[2]["material_saving_kg_m3"] is None


def test_surface_history_is_summed_before_off_axis_peak_refinement():
    class Field:
        def __init__(self, center):
            self.center = center

        def query(self, domain, xy):
            return {"uy": -np.exp(-((xy[:, 0] - self.center) / 2) ** 2) * PRESSURE / 1e9}

    a, b = Field(-7.123), Field(7.123)
    result = surface_maximum(lambda x: surface_response([a, b], [.3, .7], x))
    assert result["peak_x_m"] == pytest.approx(7.123, abs=1e-5)
    assert result["peak_um_kpa"] == pytest.approx(.7)
    assert result["peak_um_kpa"] < 1.  # Sum of the separate maxima would be wrong.


def test_empty_and_full_feasible_domains():
    assert feasible_intervals(lambda t: 1., [0., 4.])["intervals_h"] == []
    assert feasible_intervals(lambda t: -1., [0., 4.])["intervals_h"] == [[0., 4.]]


class PrescribedTestFamily:
    def __init__(self, response):
        from types import SimpleNamespace
        self.thermal = SimpleNamespace(config={"mix_id": "W40"})
        self.response = response

    def mean_pressure_fraction(self, coefficients):
        return coefficients["uniform"] + coefficients["sector"] * 62.5 / 360

    def event(self, time, coefficients):
        response = self.response

        class Field:
            def query(self, domain, xy):
                return {"uy": -response(time, coefficients, xy[:, 0]) * PRESSURE / 1e9}

        return Field()


def test_complete_plan_satisfies_early_requirement_and_final_pressure():
    family = PrescribedTestFamily(lambda t, p, x: p["uniform"] * (3. - 2. * t) * np.exp(-x**2))
    plan = solve_prescribed_plan(family, [0., 1.], [.5, 1.], {"uniform": 1.}, upper=2.1)
    assert plan["status"] == "feasible"
    np.testing.assert_allclose([row["remaining_fraction_increment"] for row in plan["events"]], [.5, .5])
    assert plan["minimum_margin_um_kpa"] == pytest.approx(.1)
    assert plan["stages"][-1]["remaining_pressure"] == {"uniform": 0., "sector": 0.}


def test_uplift_and_intermediate_constraints_cannot_be_hidden_by_final_settlement():
    family = PrescribedTestFamily(lambda t, p, x: (4. if t == 0 else -4.) * np.exp(-((x - 8.137) / 2)**2))
    plan = solve_prescribed_plan(family, [0., 1.], [.7, 1.], {"uniform": 1., "sector": -.5})
    assert plan["status"] == "displacement_infeasible"
    assert plan["solution_role"] == "minimum_violation_witness"
    assert plan["stages"][1]["minimum_margin_um_kpa"] == pytest.approx(-.6)
    assert plan["stages"][-1]["minimum_margin_um_kpa"] > 0
    plan = solve_prescribed_plan(family, [1.], [1.], {"uniform": 1., "sector": -.5})
    assert plan["status"] == "displacement_infeasible"
    assert plan["stages"][-1]["uplift_x_m"] == pytest.approx(8.137, abs=1e-5)
    assert plan["minimum_margin_um_kpa"] == pytest.approx(-1.8)


def test_loaded_plan_subtracts_saved_pressure_and_includes_displacement_once():
    from types import SimpleNamespace
    family = PrescribedTestFamily(lambda t, p, x: p["uniform"] * (2. if t == 1. else 1.) * np.exp(-x**2))
    initial = SimpleNamespace(thermal=family.thermal, time_h=.5,
        load_coefficients={"uniform": .2, "sector": 0.},
        compatible_family=lambda candidate: candidate,
        query=lambda domain, xy: {"uy": -.6 * np.exp(-xy[:, 0]**2) * PRESSURE / 1e9})
    plan = solve_prescribed_plan(family, [1., 4.], [.4, 1.], {"uniform": 1.}, initial=initial)
    assert plan["status"] == "feasible"
    assert plan["events"][0]["remaining_fraction_increment"] == pytest.approx(.25)
    assert plan["stages"][-1]["peak_um_kpa"] == pytest.approx(1.6)
    assert plan["stages"][-1]["cumulative_fraction"] == pytest.approx(1.)
    assert initial.time_h == .5 and initial.load_coefficients["uniform"] == .2
    with pytest.raises(ValueError, match="saved material clock"):
        solve_prescribed_plan(family, [0.], [1.], {"uniform": 1.}, initial=initial)
    with pytest.raises(ValueError, match="nonnegative inside and outside"):
        solve_prescribed_plan(family, [1.], [1.], {"uniform": 1., "sector": -1.1})


def test_plan_refines_the_active_position_between_surface_samples():
    family = PrescribedTestFamily(lambda t, p, x: 3. * np.exp(-((x - .137) / .2)**2))
    plan = solve_prescribed_plan(family, [1.], [1.], {"uniform": 1.},
                                 upper=2., x=np.array([-1., 0., 1.]))
    assert plan["status"] == "displacement_infeasible"
    assert plan["minimum_margin_um_kpa"] == pytest.approx(-1.)
    assert plan["stages"][-1]["peak_x_m"] == pytest.approx(.137, abs=1e-5)
    assert plan["linear_solves"] > 1


@pytest.mark.parametrize("solver_status, expected", [(2, "load_requirements_infeasible"), (4, "solver_failure")])
def test_plan_does_not_confuse_solver_failure_with_displacement_infeasibility(monkeypatch, solver_status, expected):
    from types import SimpleNamespace
    import grouting_pino.pinn_design as design
    monkeypatch.setattr(design, "linprog", lambda *args, **kwargs:
                        SimpleNamespace(success=False, status=solver_status, message="test condition"))
    family = PrescribedTestFamily(lambda t, p, x: np.zeros(len(x)))
    result = solve_prescribed_plan(family, [1.], [1.], {"uniform": 1.})
    assert result["status"] == expected and result["events"] == []


@pytest.mark.parametrize("identifier", ["early_requirement_W32", "continuation_k1_W40",
                                       "continuation_k0.5_W40", "initial_envelope_W48"])
def test_saved_complete_events_replay_surface_stress_and_loaded_state(identifier):
    from threadpoolctl import threadpool_limits
    from grouting_pino.pinn_design import replay_prescribed_plan
    from grouting_pino.pinn_history import PINNHistory
    from grouting_pino.pinn_pressure import PressureEventPINN
    from grouting_pino.pinn_thermal import load_thermal
    root = Path(__file__).resolve().parents[1]
    output = root / "results/pinn_support/design/complete"
    if not (output / "plans.json").exists():
        pytest.skip("Retained complete plans are required")
    plans = json.loads((output / "plans.json").read_text())["plans"]
    plan = next(row for row in plans if row["id"] == identifier)
    initial = None
    with threadpool_limits(limits=2), np.load(output / "fields.npz") as fields:
        if "initial_checkpoint" in plan:
            initial = PINNHistory.load(root / plan["initial_checkpoint"], root / "data/public_paste_benchmark")
            family = initial.family
            original = initial.query("paste", fields["paste__xy_m"])
        else:
            thermal = load_thermal(root / f"results/pinn_support/{plan['mix_id']}/thermal_age/model.pt",
                                   root / "data/public_paste_benchmark")
            family = PressureEventPINN.load(root / plan["family"], thermal)
        x = fields["surface_x_m"]
        for i, history in enumerate(replay_prescribed_plan(family, plan, initial), start=1):
            actual = -history.query("soil", np.c_[x, np.zeros_like(x)])["uy"] * 1e9 / PRESSURE
            np.testing.assert_allclose(actual, fields[identifier + "__surface_um_kpa"][i], atol=1e-9)
            stage = plan["stages"][i]
            assert stage["cumulative_fraction"] >= stage["minimum_cumulative_fraction"] - 1e-9
        assert history.load_coefficients == pytest.approx(plan["target_pressure"])
        for domain in ("soil", "paste", "lining"):
            for key, actual in history.query(domain, fields[domain + "__xy_m"]).items():
                np.testing.assert_allclose(actual, fields[identifier + "__" + domain + "__" + key], atol=1e-9)
        if initial is not None:
            assert initial.time_h == 1. and initial.load_coefficients == {"uniform": .2, "sector": 0.}
            for key, value in initial.query("paste", fields["paste__xy_m"]).items():
                np.testing.assert_array_equal(value, original[key])


def test_temperature_boundary_ends_the_valid_query_domain():
    from grouting_pino.pinn_thermal import ThermalPINN
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from run_pinn_design import valid_time_domain

    config = json.loads((root / "configs/pinn_support.json").read_text())
    model = ThermalPINN(config, root / "data/public_paste_benchmark").double()
    with torch.no_grad():
        model.networks["paste"][-1].bias[0] = 2.
    domain = valid_time_domain(model, 4.)
    assert domain["end_reason"] == "temperature_scope"
    assert 0 < domain["end_h"] < 4
    model.query("paste", 5.91, domain["end_h"])
    with pytest.raises(ValueError, match="Temperature"):
        model.query("paste", 5.91, domain["boundary_bracket_h"][1])


@pytest.mark.parametrize("mix", ["W32", "W40", "W48"])
def test_retained_design_decisions_reproduce_from_actual_fields(mix):
    from threadpoolctl import threadpool_limits
    root = Path(__file__).resolve().parents[1]
    result_path = root / f"results/pinn_support/design/{mix}.json"
    if not result_path.exists():
        pytest.skip("Retained material design fields are required")
    sys.path.insert(0, str(root / "scripts"))
    from check_pinn_mechanics import load_section
    from run_pinn_design import EventFields
    result = json.loads(result_path.read_text())
    paths = {row["time_h"]: root / row["directory"] for row in result["events"]}
    limit = result["limit_um_kpa"]
    with threadpool_limits(limits=2):
        first = result["handover"]["intervals_h"][0][0]
        event = load_section(paths[first])
        peak = surface_maximum(lambda x: surface_response([event], [1.], x))["peak_um_kpa"]
        assert peak <= limit
        series = EventFields(mix)
        bounds = series.fraction_interval(0., 4., limit)["fraction_interval"]
        if bounds is not None:
            initial = series.event(0.)
            finish = series.event(4.)
            peak = surface_maximum(lambda x: surface_response([initial, finish],
                                    [bounds[1], 1 - bounds[1]], x))["peak_um_kpa"]
            assert peak <= limit + 1e-8
        crossing = result["handover"]["crossings"][0]
        previous = load_section(paths[crossing["lower_h"]])
        assert surface_maximum(lambda x: surface_response([previous], [1.], x))["peak_um_kpa"] > limit
        for plan in result["plans"]:
            event = load_section(paths[plan["planned_handover_h"]])
            peak = surface_maximum(lambda x: surface_response([event], [1.], x))["peak_um_kpa"]
            assert peak == pytest.approx(plan["peak_um_kpa"], abs=1e-8)
            assert (peak <= limit) == (plan["status"] == "feasible")
        early = load_section(paths[result["half_load"]["intervals_h"][0][0]])
        late = load_section(paths[4.])
        peak = surface_maximum(lambda x: surface_response([early, late], [.5, .5], x))["peak_um_kpa"]
        assert peak <= limit
